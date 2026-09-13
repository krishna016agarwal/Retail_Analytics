"""Step 26: Vacancy Detection Accuracy & Robustness Validation Suite.

Validates:
1. TEST A — Normal Shelf Spacing: Normal inter-product packing gaps do NOT generate alerts.
2. TEST B — Persistent Empty Space: Confirms genuine large gap after temporal threshold.
3. TEST C — Temporary Detector Miss: 1-3 frames of dropped detection do NOT trigger alerts.
4. TEST D — Shopper Occlusion: Freezes confirmation during customer occlusion; recovers after.
5. TEST E — Camera Motion / Panning: Distinguishes 'Out of View' at frame edges from 'Empty Shelf'.
6. TEST F — Lighting / Detection Noise: Jitter & confidence drops do not cause false alerts.
7. TEST G — Video & Run Synchronization: Verifies run_id generation and clean switching.
"""

import json
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
    VacantRegion,
)
from inventory.inventory_api import _run_pipeline_sync


class TestStep26VacancyRobustness(unittest.TestCase):

    def setUp(self):
        self.engine = ShelfVacancyEngine(
            shelf_id="SHELF-01",
            roi=(0.0, 0.0, 1.0, 1.0),
            min_gap_multiplier=1.75,
            min_absolute_gap_px=45,
            max_normal_spacing_ratio=0.40,
        )
        self.tracker = ShelfVacancyTracker(
            min_consecutive_frames=10,
            recovery_frames=5,
        )

    # ----------------------------------------------------------------------
    # TEST A: Normal Shelf Spacing
    # ----------------------------------------------------------------------
    def test_A_normal_shelf_spacing(self):
        """Normal gaps between packed products must NOT generate vacancy alerts."""
        # Create a shelf row with 10 products, median width 60px, spacing 5-15px
        boxes = []
        x = 50
        for i in range(10):
            w = 60
            spacing = 10 if i % 2 == 0 else 14  # within 0.40 * 60 = 24px tolerance
            boxes.append((x, 100, x + w, 180))
            x += w + spacing

        for frame_idx in range(15):
            snap = self.engine.analyze_frame(boxes, frame_w=1200, frame_h=600)
            tracked = self.tracker.update(snap)

            self.assertFalse(snap.vacancy_detected, f"Frame {frame_idx}: Normal spacing triggered vacancy")
            self.assertEqual(len(snap.vacant_regions), 0)
            self.assertEqual(tracked.status, "OCCUPIED")
            self.assertEqual(self.tracker.current_state, VacancyTemporalState.NORMAL)
            self.assertFalse(self.tracker.is_confirmed)

    # ----------------------------------------------------------------------
    # TEST B: Persistent Empty Space
    # ----------------------------------------------------------------------
    def test_B_persistent_empty_space(self):
        """A genuinely large empty shelf region must confirm vacancy after N frames."""
        # Row with 3 products, a 220px gap (~3.6x median width), and 3 products
        boxes = [
            (50, 100, 110, 180),   # w=60
            (120, 100, 180, 180),  # w=60, gap=10
            (190, 100, 250, 180),  # w=60, gap=10
            # --- 220px EMPTY SPACE ---
            (470, 100, 530, 180),  # w=60, gap=220
            (540, 100, 600, 180),  # w=60, gap=10
            (610, 100, 670, 180),  # w=60, gap=10
        ]

        states_observed = []
        for frame_idx in range(15):
            snap = self.engine.analyze_frame(boxes, frame_w=1200, frame_h=600)
            tracked = self.tracker.update(snap)
            states_observed.append(tracked.temporal_state)

            self.assertTrue(snap.vacancy_detected)
            self.assertEqual(len(snap.vacant_regions), 1)

        # Verification of state progression
        self.assertEqual(states_observed[0], VacancyTemporalState.TEMPORARY_VACANCY.value)
        self.assertEqual(states_observed[8], VacancyTemporalState.TEMPORARY_VACANCY.value)
        # At frame 9 (10th frame), state must become VACANCY_CONFIRMED
        self.assertEqual(states_observed[9], VacancyTemporalState.VACANCY_CONFIRMED.value)
        self.assertEqual(states_observed[14], VacancyTemporalState.VACANCY_CONFIRMED.value)
        self.assertTrue(self.tracker.is_confirmed)

    # ----------------------------------------------------------------------
    # TEST C: Temporary Detector Miss
    # ----------------------------------------------------------------------
    def test_C_temporary_detector_miss(self):
        """Temporary detector misses (1-3 frames) must NOT confirm a vacancy."""
        normal_boxes = [
            (50, 100, 110, 180),
            (120, 100, 180, 180),
            (190, 100, 250, 180),
            (260, 100, 320, 180),
            (330, 100, 390, 180),
            (400, 100, 460, 180),
        ]

        # Miss frame: 2 products in the middle drop out for 3 frames
        miss_boxes = [
            (50, 100, 110, 180),
            (120, 100, 180, 180),
            # missing (190..250) and (260..320) -> gap from 180 to 330 = 150px
            (330, 100, 390, 180),
            (400, 100, 460, 180),
        ]

        # 5 normal frames
        for _ in range(5):
            snap = self.engine.analyze_frame(normal_boxes, 1000, 600)
            self.tracker.update(snap)
            self.assertEqual(self.tracker.current_state, VacancyTemporalState.NORMAL)

        # 3 missing frames (detector glitch / transient flicker)
        for f in range(3):
            snap = self.engine.analyze_frame(miss_boxes, 1000, 600)
            tracked = self.tracker.update(snap)
            # Must enter TEMPORARY_VACANCY, but MUST NOT be VACANCY_CONFIRMED
            self.assertEqual(tracked.temporal_state, VacancyTemporalState.TEMPORARY_VACANCY.value)
            self.assertFalse(self.tracker.is_confirmed, f"Flicker frame {f} falsely confirmed vacancy!")

        # Detections return to normal
        for _ in range(5):
            snap = self.engine.analyze_frame(normal_boxes, 1000, 600)
            tracked = self.tracker.update(snap)

        # Must revert back to NORMAL without ever firing an alert
        self.assertEqual(self.tracker.current_state, VacancyTemporalState.NORMAL)
        self.assertFalse(self.tracker.is_confirmed)

    # ----------------------------------------------------------------------
    # TEST D: Shopper Occlusion & Freeze
    # ----------------------------------------------------------------------
    def test_D_shopper_occlusion_freeze(self):
        """Shopper blocking shelf sets UNCERTAIN and freezes confirmation."""
        # Shelf with an apparent gap where a shopper is standing
        boxes = [
            (50, 100, 110, 180),
            (120, 100, 180, 180),
            # Gap between 180 and 420 (width=240px)
            (420, 100, 480, 180),
            (490, 100, 550, 180),
        ]
        # Shopper bbox directly covering the gap [180, 420]
        shopper_box = [(170, 50, 430, 550)]

        # Run 15 frames with shopper present
        for f in range(15):
            snap = self.engine.analyze_frame(boxes, 1000, 600, person_boxes=shopper_box)
            tracked = self.tracker.update(snap)

            self.assertTrue(snap.is_occluded)
            self.assertEqual(snap.status, "UNCERTAIN")
            self.assertTrue(snap.vacant_regions[0].is_occluded_by_person)
            # Confirmation must be FROZEN
            self.assertNotEqual(
                self.tracker.current_state,
                VacancyTemporalState.VACANCY_CONFIRMED,
                f"Frame {f}: Shopper occlusion falsely confirmed as vacant!",
            )

        # Now shopper walks away (gap remains genuine empty space)
        for f in range(12):
            snap = self.engine.analyze_frame(boxes, 1000, 600, person_boxes=[])
            tracked = self.tracker.update(snap)

        # After shopper leaves and vacancy persists for 10 frames -> VACANCY_CONFIRMED
        self.assertEqual(self.tracker.current_state, VacancyTemporalState.VACANCY_CONFIRMED)

    # ----------------------------------------------------------------------
    # TEST E: Camera Motion / Panning & 'Out of View' vs 'Empty Shelf'
    # ----------------------------------------------------------------------
    def test_E_out_of_view_vs_empty_shelf(self):
        """Products entering or leaving camera FOV must NOT be counted as vacancies."""
        # Simulating camera panning to the right across 10 frames
        # Products shift to the left by 30px each frame
        # Some products exit the frame at X < 0, new ones enter at X > 800
        # The shelf itself has NO internal gaps (products are continuously spaced)

        for pan_step in range(10):
            shift = pan_step * 30
            # Continuous row from X=50 to X=750 in world coords
            # Shifted coords: x_screen = x_world - shift
            row_boxes = []
            for world_x in range(50, 950, 70):
                screen_x1 = world_x - shift
                screen_x2 = screen_x1 + 60
                # Only include products that are inside the visible frame [0, 800]
                if screen_x2 > 20 and screen_x1 < 780:
                    bx1 = max(0, screen_x1)
                    bx2 = min(800, screen_x2)
                    if bx2 - bx1 >= 30:  # partially or fully visible
                        row_boxes.append((bx1, 100, bx2, 180))

            snap = self.engine.analyze_frame(
                row_boxes,
                frame_w=800,
                frame_h=600,
                camera_motion_mag=3.0,  # 3.0 px/frame optical flow
            )
            tracked = self.tracker.update(snap)

            # Even though products exit on left and enter on right, internal gaps are 10px
            self.assertEqual(
                len(snap.vacant_regions),
                0,
                f"Pan step {pan_step}: Boundary exiting/entering created false gap! Gaps: {snap.vacant_regions}",
            )
            self.assertFalse(self.tracker.is_confirmed)

    # ----------------------------------------------------------------------
    # TEST F: Detection Noise & Bounding-Box Jitter
    # ----------------------------------------------------------------------
    def test_F_detection_noise_jitter(self):
        """Random coordinate jitter and minor detector noise do not cause alerts."""
        base_boxes = [
            (50, 100, 110, 180),
            (120, 100, 180, 180),
            (190, 100, 250, 180),
            (260, 100, 320, 180),
            (330, 100, 390, 180),
        ]
        rng = np.random.default_rng(42)

        for f in range(20):
            jittered = []
            for b in base_boxes:
                jx1 = b[0] + int(rng.integers(-4, 5))
                jy1 = b[1] + int(rng.integers(-3, 4))
                jx2 = b[2] + int(rng.integers(-4, 5))
                jy2 = b[3] + int(rng.integers(-3, 4))
                jittered.append((jx1, jy1, jx2, jy2))

            snap = self.engine.analyze_frame(jittered, 1000, 600)
            tracked = self.tracker.update(snap)

            self.assertFalse(snap.vacancy_detected, f"Noise frame {f} triggered false vacancy")
            self.assertFalse(self.tracker.is_confirmed)

    # ----------------------------------------------------------------------
    # TEST G: Video Selection & Dashboard Run Synchronization
    # ----------------------------------------------------------------------
    def test_G_video_and_run_synchronization(self):
        """Switching videos generates distinct run_ids with matching video metadata."""
        # 1. Run on shelf_pan_demo.mp4
        rep1 = _run_pipeline_sync(max_frames=12, video_source_input="shelf_pan_demo.mp4")
        run_id1 = rep1.get("run_id")
        self.assertIsNotNone(run_id1)
        self.assertIn("shelf_pan_demo", rep1.get("video_source", ""))
        self.assertEqual(rep1.get("camera_id"), "CAM-01")
        self.assertEqual(len(rep1.get("shelves", [])), 1)
        self.assertEqual(rep1["shelves"][0]["shelf_id"], "SHELF-01")

        # 2. Switch to inventory2.mp4
        rep2 = _run_pipeline_sync(max_frames=12, video_source_input="inventory2.mp4")
        run_id2 = rep2.get("run_id")
        self.assertIsNotNone(run_id2)
        self.assertNotEqual(run_id1, run_id2, "Switching video must generate a new distinct run_id")
        self.assertIn("inventory2", rep2.get("video_source", ""))
        self.assertEqual(len(rep2.get("shelves", [])), 1)

        # 3. Switch back to shelf_pan_demo.mp4
        rep3 = _run_pipeline_sync(max_frames=12, video_source_input="shelf_pan_demo.mp4")
        run_id3 = rep3.get("run_id")
        self.assertIsNotNone(run_id3)
        self.assertNotEqual(run_id2, run_id3)
        self.assertIn("shelf_pan_demo", rep3.get("video_source", ""))

        # Check zero leakage of legacy SKU terms across all three runs
        for rep in [rep1, rep2, rep3]:
            rep_str = json.dumps(rep).lower()
            for term in ["7-up", "sudafed", "coke", "pepsi", "planogram", "tier-08"]:
                self.assertNotIn(term, rep_str, f"Found legacy term '{term}' in report!")


if __name__ == "__main__":
    print("=" * 80)
    print("STEP 26: VACANCY DETECTION ACCURACY & ROBUSTNESS VALIDATION SUITE")
    print("=" * 80)
    unittest.main()
