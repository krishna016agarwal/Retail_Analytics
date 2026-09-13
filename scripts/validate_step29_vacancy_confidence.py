"""Step 29: Vacancy Confidence & Replenishment Signal Validation Suite.

Validates the 10 core operational scenarios required by Step 29:
1. Normal Spacing Without Vacancy: Normal product spacing yields no vacancy candidate, 0% confidence, no alert.
2. Small Temporary Gap (1-3 frames): Candidate gap does not persist long enough; never reaches confirmation or replenishment.
3. Persistent Meaningful Gap (>= 10 frames): Confirmed empty space yields high Visual Vacancy Confidence (>= 0.70) and triggers replenishment recommendation.
4. Shopper Blocks Gap: Person occlusion freezes temporal confirmation, sets status to UNCERTAIN, and zeros confidence.
5. Shopper Leaves: Once person moves away, vacancy observation resumes and confirms cleanly.
6. Camera Motion: Optical flow motion suppresses spurious gap confirmation.
7. Detector Dropout: Single-frame dropouts do not generate spurious confirmed replenishment recommendations.
8. Partial Product Detection (2D projection): Step 28 2D projection prevents pseudo-gaps from generating artificial vacancy signals.
9. Gap Geometry Jumps: Candidate gaps that jump horizontally fail center-drift tolerance, resetting persistence and preventing replenishment triggers.
10. Multiple Valid Gaps: Distinct unoccupied regions in the same shelf tier are tracked and reported as independent regions.
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


class TestStep29VacancyConfidence(unittest.TestCase):
    """Step 29 Validation Suite for Visual Vacancy Confidence and Replenishment Signals."""

    def setUp(self) -> None:
        self.frame_w = 1762
        self.frame_h = 2350
        self.tier_roi = (0.02, 0.12, 0.98, 0.35)  # Tier 1 (y: 282..822)
        self.tiers = [
            {
                "tier_id": "SHELF-01-TIER-01",
                "name": "Tier 1 — Top (2L Bottles)",
                "roi": self.tier_roi,
            },
        ]
        self.engine = ShelfVacancyEngine(
            shelf_id="SHELF-01",
            roi=(0.02, 0.12, 0.98, 0.88),
            tiers=self.tiers,
            min_gap_multiplier=1.75,
            min_absolute_gap_px=45,
            max_normal_spacing_ratio=0.40,
        )
        self.tracker = ShelfVacancyTracker(
            min_consecutive_frames=10,
            recovery_frames=5,
            min_replenishment_confidence=0.70,
            max_center_drift_ratio=0.25,
            stability_history_window=5,
            confidence_persistence_weight=0.40,
            confidence_size_weight=0.30,
            confidence_stability_weight=0.30,
        )

    def _pack_shelf(
        self,
        item_w: int = 140,
        item_h: int = 350,
        y_top: int = 400,
        spacing: int = 30,
        exclude_indices: Tuple[int, ...] = (),
    ) -> List[Tuple[int, int, int, int]]:
        """Generate evenly spaced product bounding boxes across Tier 1."""
        x_start = int(self.frame_w * self.tier_roi[0]) + 50
        x_end = int(self.frame_w * self.tier_roi[2]) - 50
        boxes = []
        curr_x = x_start
        idx = 0
        while curr_x + item_w <= x_end:
            if idx not in exclude_indices:
                boxes.append((curr_x, y_top, curr_x + item_w, y_top + item_h))
            curr_x += item_w + spacing
            idx += 1
        return boxes

    def test_01_normal_spacing_no_alert(self) -> None:
        """Scenario 1: Normal spacing without vacancy -> 0% confidence, no alert."""
        boxes = self._pack_shelf(spacing=25)  # Normal spacing ~0.18x item width
        for _ in range(12):
            snap = self.engine.analyze_frame(
                product_boxes=boxes,
                frame_w=self.frame_w,
                frame_h=self.frame_h,
            )
            updated = self.tracker.update(snap)

        self.assertEqual(updated.status, "OCCUPIED")
        self.assertFalse(updated.vacancy_detected)
        self.assertEqual(len(updated.unoccluded_vacant_regions), 0)
        self.assertFalse(self.tracker.is_confirmed)

    def test_02_small_temporary_gap(self) -> None:
        """Scenario 2: Small temporary gap (1-3 frames) -> no confirmation, no replenishment."""
        # Remove items 4 and 5 for 3 frames, creating a temporary valid candidate gap (2.64x facing)
        gap_boxes = self._pack_shelf(exclude_indices=(4, 5))
        normal_boxes = self._pack_shelf()

        for _ in range(3):
            snap = self.engine.analyze_frame(
                product_boxes=gap_boxes,
                frame_w=self.frame_w,
                frame_h=self.frame_h,
            )
            updated = self.tracker.update(snap)
            self.assertEqual(updated.temporal_state, VacancyTemporalState.TEMPORARY_VACANCY.value)
            # Must NOT be confirmed or recommended for replenishment yet
            for g in updated.unoccluded_vacant_regions:
                self.assertFalse(g.replenishment_recommended)
                self.assertLess(g.persistence_frames, 10)

        # Gap filled
        for _ in range(2):
            snap = self.engine.analyze_frame(
                product_boxes=normal_boxes,
                frame_w=self.frame_w,
                frame_h=self.frame_h,
            )
            updated = self.tracker.update(snap)

        self.assertFalse(self.tracker.is_confirmed)
        self.assertEqual(updated.status, "OCCUPIED")

    def test_03_persistent_meaningful_gap(self) -> None:
        """Scenario 3: Persistent gap (>=10 frames) -> confirmed, confidence >= 0.70, replenishment recommended."""
        # Exclude items 3 and 4 creating a ~310px gap (~2.2x facing width)
        boxes = self._pack_shelf(exclude_indices=(3, 4))

        for frame_idx in range(12):
            snap = self.engine.analyze_frame(
                product_boxes=boxes,
                frame_w=self.frame_w,
                frame_h=self.frame_h,
            )
            updated = self.tracker.update(snap)

            if frame_idx < 9:
                self.assertFalse(self.tracker.is_confirmed)
            else:
                self.assertTrue(self.tracker.is_confirmed)
                self.assertEqual(updated.status, "VACANT")
                self.assertGreater(len(updated.unoccluded_vacant_regions), 0)
                g = updated.unoccluded_vacant_regions[0]
                self.assertGreaterEqual(g.persistence_frames, 10)
                # Visual Vacancy Confidence must be >= 0.70
                self.assertGreaterEqual(g.vacancy_confidence, 0.70)
                self.assertTrue(g.replenishment_recommended)

    def test_04_shopper_blocks_gap(self) -> None:
        """Scenario 4: Shopper blocks gap -> status UNCERTAIN, confidence 0.0, replenishment suppressed."""
        boxes = self._pack_shelf(exclude_indices=(3, 4))
        # Warm up 5 frames of vacancy
        for _ in range(5):
            snap = self.engine.analyze_frame(
                product_boxes=boxes,
                frame_w=self.frame_w,
                frame_h=self.frame_h,
            )
            self.tracker.update(snap)

        # Shopper arrives and covers the gap coordinates (x: 500..800, y: 300..1200)
        shopper_box = [(500, 300, 800, 1200)]
        for _ in range(6):
            snap = self.engine.analyze_frame(
                product_boxes=boxes,
                frame_w=self.frame_w,
                frame_h=self.frame_h,
                person_boxes=shopper_box,
            )
            updated = self.tracker.update(snap)
            self.assertEqual(updated.status, "UNCERTAIN")
            # All vacant regions must report 0 confidence while occluded
            for g in updated.vacant_regions:
                self.assertEqual(g.vacancy_confidence, 0.0)
                self.assertFalse(g.replenishment_recommended)

    def test_05_shopper_leaves(self) -> None:
        """Scenario 5: Shopper leaves -> confirmation resumes and reaches replenishment recommendation."""
        boxes = self._pack_shelf(exclude_indices=(3, 4))
        # 5 frames of unoccluded vacancy
        for _ in range(5):
            snap = self.engine.analyze_frame(
                product_boxes=boxes,
                frame_w=self.frame_w,
                frame_h=self.frame_h,
            )
            self.tracker.update(snap)

        # 4 frames occluded
        shopper_box = [(500, 300, 800, 1200)]
        for _ in range(4):
            snap = self.engine.analyze_frame(
                product_boxes=boxes,
                frame_w=self.frame_w,
                frame_h=self.frame_h,
                person_boxes=shopper_box,
            )
            self.tracker.update(snap)

        # Shopper leaves: 6 more frames of unoccluded vacancy
        for f in range(6):
            snap = self.engine.analyze_frame(
                product_boxes=boxes,
                frame_w=self.frame_w,
                frame_h=self.frame_h,
                person_boxes=[],
            )
            updated = self.tracker.update(snap)

        self.assertTrue(self.tracker.is_confirmed)
        self.assertEqual(updated.status, "VACANT")
        g = updated.unoccluded_vacant_regions[0]
        self.assertTrue(g.replenishment_recommended)
        self.assertGreaterEqual(g.vacancy_confidence, 0.70)

    def test_06_camera_panning_motion_freeze(self) -> None:
        """Scenario 6: Camera motion (optical flow mag > threshold) freezes confirmation."""
        boxes = self._pack_shelf(exclude_indices=(3, 4))
        # Camera is panning rapidly: optical flow mag = 7.0 (exceeds 6.0 motion threshold)
        for _ in range(12):
            snap = self.engine.analyze_frame(
                product_boxes=boxes,
                frame_w=self.frame_w,
                frame_h=self.frame_h,
                camera_motion_mag=7.0,
            )
            updated = self.tracker.update(snap)
            self.assertEqual(updated.status, "UNCERTAIN")
            self.assertFalse(self.tracker.is_confirmed)

    def test_07_detector_dropout_resilience(self) -> None:
        """Scenario 7: Transient detector dropouts do not create false replenishment confirmations."""
        normal_boxes = self._pack_shelf()
        # Normal frames
        for _ in range(5):
            snap = self.engine.analyze_frame(
                product_boxes=normal_boxes,
                frame_w=self.frame_w,
                frame_h=self.frame_h,
            )
            self.tracker.update(snap)

        # Single frame dropout (detector drops all boxes)
        snap_empty = self.engine.analyze_frame(
            product_boxes=[],
            frame_w=self.frame_w,
            frame_h=self.frame_h,
        )
        updated = self.tracker.update(snap_empty)
        # Transient frame cannot trigger confirmed replenishment
        self.assertFalse(self.tracker.is_confirmed)
        for g in updated.unoccluded_vacant_regions:
            self.assertFalse(g.replenishment_recommended)

    def test_08_partial_product_detection_2d_projection(self) -> None:
        """Scenario 8: Partial neck/cap detections do not trigger false vacancies (Step 28 2D occupancy)."""
        # Exact bottles setup from Step 28 failure case:
        boxes = [
            (292, 441, 454, 685),
            (482, 438, 647, 741),
            (665, 445, 823, 793),
            # Middle 3 bottles detected neck/cap only:
            (994, 308, 1128, 541),
            (1148, 315, 1278, 558),
            (1299, 318, 1438, 562),
            (1450, 350, 1610, 760),
        ]
        for _ in range(12):
            snap = self.engine.analyze_frame(
                product_boxes=boxes,
                frame_w=self.frame_w,
                frame_h=self.frame_h,
            )
            updated = self.tracker.update(snap)

        # 2D projection must reject pseudo-gap across neck region
        self.assertEqual(len(updated.unoccluded_vacant_regions), 0)
        self.assertEqual(updated.status, "OCCUPIED")
        self.assertFalse(self.tracker.is_confirmed)

    def test_09_gap_geometry_jumps_reset_persistence(self) -> None:
        """Scenario 9: Unstable/jumping gap candidates fail drift tolerance and reset persistence."""
        # Gap location jumps across frames:
        # Frame 0-2: at X=[300, 550]
        # Frame 3-5: at X=[800, 1050] (huge jump)
        # Frame 6-8: at X=[1300, 1550] (huge jump)
        for f in range(9):
            if f < 3:
                boxes = self._pack_shelf(exclude_indices=(1, 2))
            elif f < 6:
                boxes = self._pack_shelf(exclude_indices=(4, 5))
            else:
                boxes = self._pack_shelf(exclude_indices=(7, 8))

            snap = self.engine.analyze_frame(
                product_boxes=boxes,
                frame_w=self.frame_w,
                frame_h=self.frame_h,
            )
            updated = self.tracker.update(snap)
            # Due to spatial jumps, no track should ever reach persistence >= 5
            for g in updated.unoccluded_vacant_regions:
                self.assertLess(g.persistence_frames, 5)
                self.assertFalse(g.replenishment_recommended)

        self.assertFalse(self.tracker.is_confirmed)

    def test_10_multiple_valid_gaps_independent_tracking(self) -> None:
        """Scenario 10: Multiple distinct vacant regions on same tier are tracked independently."""
        # Exclude indices 2, 3 (Gap 1) and 6, 7 (Gap 2) so each gap is ~2.64x facing width (> 1.75x)
        boxes = self._pack_shelf(exclude_indices=(2, 3, 6, 7))

        for _ in range(12):
            snap = self.engine.analyze_frame(
                product_boxes=boxes,
                frame_w=self.frame_w,
                frame_h=self.frame_h,
            )
            updated = self.tracker.update(snap)

        self.assertTrue(self.tracker.is_confirmed)
        self.assertEqual(len(updated.unoccluded_vacant_regions), 2)

        gap_ids = [g.region_id for g in updated.unoccluded_vacant_regions]
        self.assertEqual(len(set(gap_ids)), 2, "Both gaps must have distinct track IDs")

        for g in updated.unoccluded_vacant_regions:
            self.assertGreaterEqual(g.persistence_frames, 10)
            self.assertGreaterEqual(g.vacancy_confidence, 0.70)
            self.assertTrue(g.replenishment_recommended)
            self.assertGreaterEqual(g.geometric_stability, 0.80)


if __name__ == "__main__":
    unittest.main(verbosity=2)
