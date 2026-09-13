"""Step 28: Physical Shelf Tier Partitioning & 2D Occupancy Validation Suite.

Validates that:
1. Exact Step 27 Failure Mode is Solved:
   Products with mixed full-body and neck/cap detections on the same physical tier
   no longer get fragmented into separate pseudo-rows or trigger the ~617px pseudo-gap.
   Expected: FALSE VACANCY = 0.
2. Genuine Vacancy Detection:
   A real empty space without products across the entire physical tier is correctly detected.
3. Shopper Occlusion Safety:
   Person bounding boxes overlapping candidate gaps freeze temporal confirmation.
4. Camera Panning / Boundary Immunity:
   Products near image boundaries entering/exiting FOV do not trigger false vacancies.
5. Coordinate Jitter Durability:
   Small detector bounding box coordinate fluctuations do not create artificial gaps.
"""

import os
import pathlib
import sys
import unittest
import numpy as np

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inventory.shelf_vacancy import (
    ShelfVacancyEngine,
    ShelfVacancyTracker,
    VacancyTemporalState,
    ShelfVacancySnapshot,
)


class TestStep28TierOccupancyValidation(unittest.TestCase):
    """Rigorous test suite for physical shelf tier partitioning and 2D occupancy."""

    def setUp(self):
        self.frame_w = 1762
        self.frame_h = 2350
        self.tiers = [
            {
                "tier_id": "SHELF-01-TIER-01",
                "name": "Tier 1 — Top (2L Bottles)",
                "roi": (0.02, 0.12, 0.98, 0.35),
            },
            {
                "tier_id": "SHELF-01-TIER-02",
                "name": "Tier 2 — Mid (20oz Bottles)",
                "roi": (0.02, 0.36, 0.98, 0.59),
            },
            {
                "tier_id": "SHELF-01-TIER-03",
                "name": "Tier 3 — Lower (12-Pack Cans)",
                "roi": (0.02, 0.60, 0.98, 0.88),
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
        self.tracker = ShelfVacancyTracker(min_consecutive_frames=10, recovery_frames=5)

    def test_01_reproduce_and_solve_step27_failure(self):
        """CRITICAL: Replicate the exact Step 27 failure on Tier 1.

        Physical shelf contains 7 bottles.
        Left 3 bottles: detected as full-body (y: 440..750, h=310px, cy=595px)
        Middle 3 bottles: detected as neck/cap only (y: 310..550, h=240px, cy=430px)
        Right 1 bottle: detected as full-body (y: 350..760, h=410px, cy=555px)

        Old 1D centroid clustering split them into ROW-01 (neck) and ROW-02 (full),
        creating a false 617px gap in ROW-02 between left bottles and right bottle.
        New 2D horizontal occupancy projection MUST project all products onto the
        shelf axis, merging contiguous intervals and producing ZERO false vacancies.
        """
        # Actual bounding boxes from shelf_pan_demo Frame 0:
        boxes = [
            # Left bottles (full body)
            (292, 441, 454, 685),
            (482, 438, 647, 741),
            (665, 445, 823, 793),
            # Middle 3 bottles (neck/cap partial detections)
            (994, 308, 1128, 541),
            (1148, 315, 1278, 558),
            (1300, 320, 1421, 495),
            # Right bottle (full body)
            (1440, 345, 1556, 763),
        ]

        snapshot = self.engine.analyze_frame(
            product_boxes=boxes,
            frame_w=self.frame_w,
            frame_h=self.frame_h,
        )

        tier1 = next(t for t in snapshot.tiers if t.tier_id == "SHELF-01-TIER-01")

        # 1. Verify all 7 bottles were assigned to Tier 1
        self.assertEqual(tier1.product_count, 7)

        # 2. Verify FALSE VACANCIES = 0
        self.assertEqual(
            len(tier1.gaps),
            0,
            f"Expected 0 vacancies on Tier 1, but found: {[g.to_dict() for g in tier1.gaps]}",
        )
        self.assertEqual(tier1.status, "OCCUPIED")

        # 3. Verify overall shelf status is OCCUPIED
        self.assertFalse(snapshot.vacancy_detected)
        self.assertEqual(len(snapshot.unoccluded_vacant_regions), 0)

        # 4. Verify temporal confirmation remains NORMAL across 15 frames
        tracker = ShelfVacancyTracker(min_consecutive_frames=10)
        for _ in range(15):
            snap = self.engine.analyze_frame(
                product_boxes=boxes,
                frame_w=self.frame_w,
                frame_h=self.frame_h,
            )
            snap = tracker.update(snap)
            self.assertEqual(snap.temporal_state, VacancyTemporalState.NORMAL.value)
            self.assertFalse(tracker.is_confirmed)

    def test_02_genuine_vacancy_in_tier(self):
        """Verify that a genuine empty space (e.g. missing 3 bottles) is confirmed."""
        # Shelf with bottle 1, 2, then a genuine 300px empty space, then bottle 4, 5
        boxes = [
            (200, 400, 350, 700),
            (370, 400, 520, 700),
            # Genuine empty void: x = 520 to 850 (width = 330px, approx 2.2x median)
            (850, 400, 1000, 700),
            (1020, 400, 1170, 700),
            (1190, 400, 1340, 700),
        ]

        tracker = ShelfVacancyTracker(min_consecutive_frames=10)
        for frame_idx in range(12):
            snapshot = self.engine.analyze_frame(
                product_boxes=boxes,
                frame_w=self.frame_w,
                frame_h=self.frame_h,
            )
            snapshot = tracker.update(snapshot)
            if frame_idx < 9:
                self.assertEqual(snapshot.temporal_state, VacancyTemporalState.TEMPORARY_VACANCY.value)
                self.assertFalse(tracker.is_confirmed)
            else:
                self.assertEqual(snapshot.temporal_state, VacancyTemporalState.VACANCY_CONFIRMED.value)
                self.assertTrue(tracker.is_confirmed)
                self.assertEqual(snapshot.status, "VACANT")

    def test_03_shopper_occlusion_freezes_tier_confirmation(self):
        """Shopper overlapping vacancy freezes confirmation counter."""
        boxes = [
            (200, 400, 350, 700),
            (370, 400, 520, 700),
            # Empty void: 520 to 850
            (850, 400, 1000, 700),
            (1020, 400, 1170, 700),
        ]
        person_box = (500, 300, 800, 1200)  # Blocks the gap

        tracker = ShelfVacancyTracker(min_consecutive_frames=10)
        # Run 15 frames with shopper present
        for _ in range(15):
            snapshot = self.engine.analyze_frame(
                product_boxes=boxes,
                frame_w=self.frame_w,
                frame_h=self.frame_h,
                person_boxes=[person_box],
            )
            snapshot = tracker.update(snapshot)
            self.assertEqual(snapshot.status, "UNCERTAIN")
            self.assertFalse(tracker.is_confirmed)
            self.assertEqual(len(snapshot.unoccluded_vacant_regions), 0)

    def test_04_camera_panning_boundary_suppression(self):
        """Products entering/leaving tier boundary margins do not trigger vacancies."""
        # Products shifted close to left edge (x1=10) and right edge (x2=1740)
        boxes = [
            (40, 400, 190, 700),
            (210, 400, 360, 700),
            (380, 400, 530, 700),
            (550, 400, 700, 700),
        ]
        snapshot = self.engine.analyze_frame(
            product_boxes=boxes,
            frame_w=self.frame_w,
            frame_h=self.frame_h,
        )
        tier1 = next(t for t in snapshot.tiers if t.tier_id == "SHELF-01-TIER-01")
        # Gaps beyond product 4 (towards right edge of image) must be ignored
        self.assertEqual(len(tier1.gaps), 0)

    def test_05_coordinate_jitter_durability(self):
        """Gaussian jitter on product coordinates does not create pseudo-gaps."""
        base_boxes = [(100 + i * 160, 400, 240 + i * 160, 700) for i in range(8)]
        np.random.seed(42)

        for _ in range(20):
            jittered = [
                (
                    b[0] + int(np.random.normal(0, 3)),
                    b[1] + int(np.random.normal(0, 3)),
                    b[2] + int(np.random.normal(0, 3)),
                    b[3] + int(np.random.normal(0, 3)),
                )
                for b in base_boxes
            ]
            snapshot = self.engine.analyze_frame(
                product_boxes=jittered,
                frame_w=self.frame_w,
                frame_h=self.frame_h,
            )
            tier1 = next(t for t in snapshot.tiers if t.tier_id == "SHELF-01-TIER-01")
            self.assertEqual(len(tier1.gaps), 0)


if __name__ == "__main__":
    unittest.main()
