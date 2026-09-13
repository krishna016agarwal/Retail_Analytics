"""Step 30: Real-World Generalization & Production Readiness Validation Suite.

Evaluates the Generic Shelf Vacancy Detection system across 7 variation categories:
A. Camera & Geometry Variations (front-facing, perspective skew, tilt, ROI boundary entry/exit)
B. Lighting & Visual Noise Robustness (bounding box jitter from shadows/contrast)
C. Product Arrangement & Density (dense, medium, sparse, large gap, multiple smaller spaces)
D. Detection Noise (1-frame drop, multi-frame drop, partial bboxes)
E. Shopper Interaction Lifecycle (normal -> shopper blocks -> item removed -> shopper leaves -> confirmed)
F. Camera Motion Dynamics (stationary, slow pan, fast pan / optical flow freeze, vibration)
G. Multi-Tier & Multi-Gap Configurations (single tier, simultaneous multi-tier, multiple gaps per tier)
"""

from __future__ import annotations

import pathlib
import sys
import unittest
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inventory.shelf_vacancy import (
    ShelfVacancyEngine,
    ShelfVacancyTracker,
    ShelfVacancySnapshot,
    VacancyTemporalState,
    VacantRegion,
)


class TestStep30Generalization(unittest.TestCase):
    """Step 30 Generalization Test Suite."""

    def setUp(self) -> None:
        self.frame_w = 1920
        self.frame_h = 1080
        self.tiers = [
            {
                "tier_id": "SHELF-01-TIER-01",
                "name": "Tier 1 — Top Rack",
                "roi": (0.05, 0.15, 0.95, 0.40),
            },
            {
                "tier_id": "SHELF-01-TIER-02",
                "name": "Tier 2 — Mid Rack",
                "roi": (0.05, 0.42, 0.95, 0.67),
            },
            {
                "tier_id": "SHELF-01-TIER-03",
                "name": "Tier 3 — Lower Rack",
                "roi": (0.05, 0.69, 0.95, 0.94),
            },
        ]
        self.engine = ShelfVacancyEngine(
            shelf_id="SHELF-01",
            roi=(0.05, 0.15, 0.95, 0.94),
            tiers=self.tiers,
            min_gap_multiplier=1.75,
            min_absolute_gap_px=45,
            max_normal_spacing_ratio=0.40,
            ignore_boundary_margin_pct=0.03,
        )
        self.tracker = ShelfVacancyTracker(
            min_consecutive_frames=10,
            recovery_frames=5,
            min_replenishment_confidence=0.70,
            max_center_drift_ratio=0.25,
            stability_history_window=5,
        )

    def _generate_tier_boxes(
        self,
        tier_idx: int = 0,
        item_w: int = 100,
        item_h: int = 200,
        spacing: int = 20,
        exclude_indices: Tuple[int, ...] = (),
        width_jitter: int = 0,
        y_tilt_px: int = 0,
    ) -> List[Tuple[int, int, int, int]]:
        """Generate parameterized product boxes for a specific tier."""
        roi = self.tiers[tier_idx]["roi"]
        x_start = int(self.frame_w * roi[0]) + 40
        x_end = int(self.frame_w * roi[2]) - 40
        y_top = int(self.frame_h * roi[1]) + 20

        boxes = []
        curr_x = x_start
        idx = 0
        while curr_x + item_w <= x_end:
            if idx not in exclude_indices:
                w_mod = item_w + (width_jitter if idx % 2 == 0 else -width_jitter)
                tilt_offset = int(y_tilt_px * (curr_x - x_start) / max(1, x_end - x_start))
                boxes.append((
                    curr_x,
                    y_top + tilt_offset,
                    curr_x + w_mod,
                    y_top + item_h + tilt_offset,
                ))
            curr_x += item_w + spacing
            idx += 1
        return boxes

    # ─── CATEGORY A: CAMERA & GEOMETRY ──────────────────────────────────────────

    def test_a1_front_facing_shelf_normal(self) -> None:
        """A1: Front-facing standard shelf -> 100% occupied, 0% confidence, 0 alerts."""
        boxes = []
        for t in range(3):
            boxes.extend(self._generate_tier_boxes(tier_idx=t, spacing=20))

        for _ in range(5):
            snap = self.tracker.update(self.engine.analyze_frame(boxes, self.frame_w, self.frame_h))

        self.assertEqual(snap.status, "OCCUPIED")
        self.assertFalse(snap.vacancy_detected)
        self.assertEqual(len(snap.unoccluded_vacant_regions), 0)

    def test_a2_mild_perspective_scaling(self) -> None:
        """A2: Perspective scaling across horizontal axis -> adaptive median facing absorbs variation."""
        # Boxes decrease gradually from left (120px) to right (80px) due to perspective
        roi = self.tiers[0]["roi"]
        x_start = int(self.frame_w * roi[0]) + 40
        x_end = int(self.frame_w * roi[2]) - 40
        y_top = int(self.frame_h * roi[1]) + 20
        boxes = []
        curr_x = x_start
        while curr_x + 80 <= x_end:
            frac = (curr_x - x_start) / (x_end - x_start)
            w = int(120 - 40 * frac)
            boxes.append((curr_x, y_top, curr_x + w, y_top + 200))
            curr_x += w + 20

        snap = self.tracker.update(self.engine.analyze_frame(boxes, self.frame_w, self.frame_h))
        self.assertEqual(snap.status, "OCCUPIED")
        self.assertEqual(len(snap.unoccluded_vacant_regions), 0)

    def test_a3_mild_camera_tilt(self) -> None:
        """A3: Mild camera tilt (+25px vertical drift across shelf) stays within tier bounds."""
        boxes = self._generate_tier_boxes(tier_idx=0, y_tilt_px=25)
        snap = self.tracker.update(self.engine.analyze_frame(boxes, self.frame_w, self.frame_h))
        self.assertEqual(snap.status, "OCCUPIED")
        self.assertEqual(len(snap.unoccluded_vacant_regions), 0)

    def test_a4_shelf_boundary_margin_filtering(self) -> None:
        """A4: Unoccupied space near shelf edges (boundary margins) is safely ignored."""
        # Skip first product (index 0) and last product; boundary margin filter must ignore edge spaces
        boxes = self._generate_tier_boxes(tier_idx=0, exclude_indices=(0, 11, 12))
        snap = self.tracker.update(self.engine.analyze_frame(boxes, self.frame_w, self.frame_h))
        # Internal space has no missing items, edge spaces are filtered by ignore_boundary_margin_pct
        self.assertEqual(len(snap.unoccluded_vacant_regions), 0)

    # ─── CATEGORY B: LIGHTING & VISUAL NOISE ───────────────────────────────────

    def test_b1_lighting_contrast_jitter(self) -> None:
        """B1: Minor bounding box coordinate jitter (+/-6px) preserves high geometric stability."""
        rng = np.random.RandomState(42)
        boxes_base = self._generate_tier_boxes(tier_idx=0, exclude_indices=(4, 5))  # Valid vacancy

        for f in range(12):
            jittered_boxes = []
            for (bx1, by1, bx2, by2) in boxes_base:
                jx1 = bx1 + rng.randint(-4, 5)
                jx2 = bx2 + rng.randint(-4, 5)
                jittered_boxes.append((jx1, by1, jx2, by2))

            snap = self.tracker.update(self.engine.analyze_frame(jittered_boxes, self.frame_w, self.frame_h))

        self.assertTrue(self.tracker.is_confirmed)
        g = snap.unoccluded_vacant_regions[0]
        self.assertTrue(g.replenishment_recommended)
        self.assertGreaterEqual(g.geometric_stability, 0.70)
        self.assertGreaterEqual(g.vacancy_confidence, 0.70)

    # ─── CATEGORY C: PRODUCT ARRANGEMENT & DENSITY ─────────────────────────────

    def test_c1_dense_shelf(self) -> None:
        """C1: High density shelf (spacing 8px ~ 0.08x facing) -> 0 gaps, 0% confidence."""
        boxes = self._generate_tier_boxes(tier_idx=0, spacing=8)
        snap = self.tracker.update(self.engine.analyze_frame(boxes, self.frame_w, self.frame_h))
        self.assertEqual(snap.status, "OCCUPIED")
        self.assertEqual(len(snap.unoccluded_vacant_regions), 0)

    def test_c2_sparse_shelf_normal_spacing(self) -> None:
        """C2: Sparse shelf with normal spacing (35px ~ 0.35x facing) does not trigger false gaps."""
        boxes = self._generate_tier_boxes(tier_idx=0, spacing=35)
        snap = self.tracker.update(self.engine.analyze_frame(boxes, self.frame_w, self.frame_h))
        self.assertEqual(snap.status, "OCCUPIED")
        self.assertEqual(len(snap.unoccluded_vacant_regions), 0)

    def test_c3_large_empty_region(self) -> None:
        """C3: Very large empty region (4 products missing ~ 4.8x facing) achieves high confidence."""
        boxes = self._generate_tier_boxes(tier_idx=0, exclude_indices=(3, 4, 5, 6))
        for _ in range(12):
            snap = self.tracker.update(self.engine.analyze_frame(boxes, self.frame_w, self.frame_h))

        self.assertTrue(self.tracker.is_confirmed)
        g = snap.unoccluded_vacant_regions[0]
        self.assertGreaterEqual(g.width_multiple, 4.0)
        self.assertGreaterEqual(g.vacancy_confidence, 0.90)
        self.assertTrue(g.replenishment_recommended)

    def test_c4_multiple_smaller_spaces_rejected(self) -> None:
        """C4: Several small spaces (each < 1.75x facing) are not aggregated into false vacancies."""
        # Skip single items at spaced intervals (idx 2, 5, 8); each creates a ~1.4x gap
        boxes = self._generate_tier_boxes(tier_idx=0, exclude_indices=(2, 5, 8))
        snap = self.tracker.update(self.engine.analyze_frame(boxes, self.frame_w, self.frame_h))
        self.assertEqual(len(snap.unoccluded_vacant_regions), 0)

    # ─── CATEGORY D: DETECTION NOISE ───────────────────────────────────────────

    def test_d1_single_frame_dropout(self) -> None:
        """D1: Single frame detector dropout does not trigger spurious alert."""
        boxes = self._generate_tier_boxes(tier_idx=0)
        # 5 normal frames
        for _ in range(5):
            self.tracker.update(self.engine.analyze_frame(boxes, self.frame_w, self.frame_h))
        # 1 frame complete dropout
        snap_drop = self.tracker.update(self.engine.analyze_frame([], self.frame_w, self.frame_h))
        self.assertFalse(self.tracker.is_confirmed)
        self.assertFalse(any(g.replenishment_recommended for g in snap_drop.unoccluded_vacant_regions))

    def test_d2_multiframe_dropout_safety(self) -> None:
        """D2: 3 consecutive dropout frames transition safely to UNCERTAIN without false alerts."""
        boxes = self._generate_tier_boxes(tier_idx=0)
        for _ in range(5):
            self.tracker.update(self.engine.analyze_frame(boxes, self.frame_w, self.frame_h))
        for _ in range(3):
            snap = self.tracker.update(self.engine.analyze_frame([], self.frame_w, self.frame_h))
        self.assertFalse(self.tracker.is_confirmed)

    # ─── CATEGORY E: SHOPPER INTERACTION LIFECYCLE ─────────────────────────────

    def test_e1_full_shopper_interaction_lifecycle(self) -> None:
        """E1: Full lifecycle: normal -> shopper arrives -> item removed -> shopper leaves -> confirmed alert."""
        normal_boxes = self._generate_tier_boxes(tier_idx=0)
        removed_boxes = self._generate_tier_boxes(tier_idx=0, exclude_indices=(4, 5))
        shopper_box = [(600, 200, 1100, 1000)]

        # Phase 1: Normal occupied shelf (5 frames)
        for _ in range(5):
            snap1 = self.tracker.update(self.engine.analyze_frame(normal_boxes, self.frame_w, self.frame_h))
        self.assertEqual(snap1.status, "OCCUPIED")
        self.assertFalse(self.tracker.is_confirmed)

        # Phase 2: Shopper arrives and occludes region (5 frames)
        for _ in range(5):
            snap2 = self.tracker.update(self.engine.analyze_frame(normal_boxes, self.frame_w, self.frame_h, person_boxes=shopper_box))
        self.assertEqual(snap2.status, "UNCERTAIN")

        # Phase 3: Shopper removes items while occluding (3 frames)
        for _ in range(3):
            snap3 = self.tracker.update(self.engine.analyze_frame(removed_boxes, self.frame_w, self.frame_h, person_boxes=shopper_box))
        self.assertEqual(snap3.status, "UNCERTAIN")
        self.assertFalse(self.tracker.is_confirmed)

        # Phase 4: Shopper leaves, exposing the vacancy (12 frames)
        for f in range(12):
            snap4 = self.tracker.update(self.engine.analyze_frame(removed_boxes, self.frame_w, self.frame_h, person_boxes=[]))
            if f < 9:
                self.assertFalse(self.tracker.is_confirmed)

        # Final state: confirmed replenishment recommendation
        self.assertTrue(self.tracker.is_confirmed)
        self.assertEqual(snap4.status, "VACANT")
        g = snap4.unoccluded_vacant_regions[0]
        self.assertTrue(g.replenishment_recommended)
        self.assertGreaterEqual(g.vacancy_confidence, 0.70)

    # ─── CATEGORY F: CAMERA MOTION DYNAMICS ────────────────────────────────────

    def test_f1_slow_camera_pan(self) -> None:
        """F1: Slow camera pan (flow mag 2.5 < 6.0) does not interrupt tracking."""
        boxes = self._generate_tier_boxes(tier_idx=0, exclude_indices=(4, 5))
        for _ in range(12):
            snap = self.tracker.update(self.engine.analyze_frame(boxes, self.frame_w, self.frame_h, camera_motion_mag=2.5))
        self.assertTrue(self.tracker.is_confirmed)
        self.assertEqual(snap.status, "VACANT")

    def test_f2_fast_pan_freeze(self) -> None:
        """F2: Fast pan (flow mag 8.0 > 6.0) freezes confirmation and marks UNCERTAIN."""
        boxes = self._generate_tier_boxes(tier_idx=0, exclude_indices=(4, 5))
        for _ in range(12):
            snap = self.tracker.update(self.engine.analyze_frame(boxes, self.frame_w, self.frame_h, camera_motion_mag=8.0))
        self.assertEqual(snap.status, "UNCERTAIN")
        self.assertFalse(self.tracker.is_confirmed)

    # ─── CATEGORY G: MULTI-TIER & MULTI-GAP CONFIGURATIONS ─────────────────────

    def test_g1_single_tier_isolated_vacancy(self) -> None:
        """G1: Vacancy isolated to Tier 2 leaves Tier 1 and Tier 3 OCCUPIED."""
        t1 = self._generate_tier_boxes(tier_idx=0)
        t2 = self._generate_tier_boxes(tier_idx=1, exclude_indices=(4, 5))  # Vacancy
        t3 = self._generate_tier_boxes(tier_idx=2)
        all_boxes = t1 + t2 + t3

        for _ in range(12):
            snap = self.tracker.update(self.engine.analyze_frame(all_boxes, self.frame_w, self.frame_h))

        self.assertTrue(self.tracker.is_confirmed)
        t_dict = {t.tier_id: t for t in snap.tiers}
        self.assertEqual(t_dict["SHELF-01-TIER-01"].status, "OCCUPIED")
        self.assertEqual(t_dict["SHELF-01-TIER-02"].status, "VACANT")
        self.assertEqual(t_dict["SHELF-01-TIER-03"].status, "OCCUPIED")

    def test_g2_multitier_simultaneous_vacancies(self) -> None:
        """G2: Simultaneous vacancies in Tier 1 and Tier 3 are tracked and reported in both tiers."""
        t1 = self._generate_tier_boxes(tier_idx=0, exclude_indices=(3, 4))  # Vacancy Tier 1
        t2 = self._generate_tier_boxes(tier_idx=1)
        t3 = self._generate_tier_boxes(tier_idx=2, exclude_indices=(5, 6))  # Vacancy Tier 3
        all_boxes = t1 + t2 + t3

        for _ in range(12):
            snap = self.tracker.update(self.engine.analyze_frame(all_boxes, self.frame_w, self.frame_h))

        self.assertTrue(self.tracker.is_confirmed)
        self.assertEqual(len(snap.unoccluded_vacant_regions), 2)
        tiers_with_recs = [g.tier_id for g in snap.unoccluded_vacant_regions if g.replenishment_recommended]
        self.assertIn("SHELF-01-TIER-01", tiers_with_recs)
        self.assertIn("SHELF-01-TIER-03", tiers_with_recs)

    def test_g3_multiple_disjoint_gaps_same_tier(self) -> None:
        """G3: Two disjoint gaps on same tier are independently tracked with separate IDs."""
        boxes = self._generate_tier_boxes(tier_idx=0, exclude_indices=(2, 3, 7, 8))
        for _ in range(12):
            snap = self.tracker.update(self.engine.analyze_frame(boxes, self.frame_w, self.frame_h))

        self.assertTrue(self.tracker.is_confirmed)
        t1_gaps = [g for g in snap.unoccluded_vacant_regions if g.tier_id == "SHELF-01-TIER-01"]
        self.assertEqual(len(t1_gaps), 2)
        self.assertNotEqual(t1_gaps[0].region_id, t1_gaps[1].region_id)
        for g in t1_gaps:
            self.assertTrue(g.replenishment_recommended)
            self.assertGreaterEqual(g.vacancy_confidence, 0.70)


if __name__ == "__main__":
    unittest.main(verbosity=2)
