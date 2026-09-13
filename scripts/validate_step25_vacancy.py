"""Automated Validation Suite for Generic Shelf Vacancy Detection (Step 25).

Verifies:
1. Config: configs/shelf_vacancy_config.json loads valid SHELF-01 ROI and parameters.
2. Engine: ShelfVacancyEngine correctly groups product rows and detects internal gaps.
3. Adaptive Threshold: Scales with median product width, ignores normal packing spacing.
4. Temporal Confirmation: Transitions NORMAL -> TEMPORARY_VACANCY -> VACANCY_CONFIRMED.
5. Shopper Occlusion: Freezes confirmation when shopper is present.
6. Schema Compliance: Output JSON matches Section 9 V1 schema.
7. Zero Leakage: Zero occurrences of legacy SKU/brand names or fictitious tiers.
"""

import json
import os
import pathlib
import sys
import unittest

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


class TestShelfVacancyArchitecture(unittest.TestCase):

    def test_01_config_loads(self):
        cfg_file = ROOT_DIR / "configs" / "shelf_vacancy_config.json"
        self.assertTrue(cfg_file.is_file(), "Missing configs/shelf_vacancy_config.json")
        with open(cfg_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data.get("camera_id"), "CAM-01")
        self.assertIn("shelves", data)
        self.assertEqual(data["shelves"][0]["shelf_id"], "SHELF-01")
        self.assertEqual(len(data["shelves"][0]["roi"]), 4)

    def test_02_row_grouping_and_gap_detection(self):
        engine = ShelfVacancyEngine(
            shelf_id="SHELF-01",
            roi=(0.0, 0.0, 1.0, 1.0),
            min_gap_multiplier=1.75,
            min_absolute_gap_px=40,
        )

        # Row 1: 4 products with normal 10px spacing (no gap)
        # Row 2: 2 products, a 200px gap, then 2 products
        mock_boxes = [
            # Row 1 (Y ~ 100)
            (50, 80, 100, 120),    # w=50
            (110, 82, 160, 118),   # w=50, gap=10 (normal)
            (170, 79, 220, 121),   # w=50, gap=10 (normal)
            (230, 81, 280, 119),   # w=50, gap=10 (normal)
            # Row 2 (Y ~ 300)
            (50, 280, 100, 320),   # w=50
            (110, 282, 160, 318),  # w=50, gap=10 (normal)
            (360, 279, 410, 321),  # w=50, gap=200 (LARGE GAP: 4.0x median)
            (420, 281, 470, 319),  # w=50, gap=10 (normal)
        ]

        snapshot = engine.analyze_frame(mock_boxes, frame_w=1000, frame_h=600)
        self.assertEqual(len(snapshot.rows), 2, "Must identify exactly 2 distinct rows")
        self.assertTrue(snapshot.vacancy_detected, "Must detect vacancy on Row 2")
        self.assertEqual(len(snapshot.vacant_regions), 1, "Must detect exactly 1 internal gap")
        gap = snapshot.vacant_regions[0]
        self.assertEqual(gap.width_px, 200.0)
        self.assertAlmostEqual(gap.width_multiple, 4.0, delta=0.1)

    def test_03_temporal_confirmation_state_machine(self):
        tracker = ShelfVacancyTracker(min_consecutive_frames=5, recovery_frames=3)
        self.assertEqual(tracker.current_state, VacancyTemporalState.NORMAL)

        vacant_snap = ShelfVacancySnapshot(
            shelf_id="SHELF-01",
            status="VACANT",
            vacancy_detected=True,
            vacancy_score=0.25,
            occupancy_pct=60.0,
            detected_facings_count=20,
        )
        occupied_snap = ShelfVacancySnapshot(
            shelf_id="SHELF-01",
            status="OCCUPIED",
            vacancy_detected=False,
            vacancy_score=0.0,
            occupancy_pct=90.0,
            detected_facings_count=25,
        )

        # 1 frame vacant -> TEMPORARY_VACANCY
        s = tracker.update(vacant_snap)
        self.assertEqual(tracker.current_state, VacancyTemporalState.TEMPORARY_VACANCY)

        # 3 more frames vacant -> still TEMPORARY_VACANCY (4/5)
        for _ in range(3):
            tracker.update(vacant_snap)
        self.assertEqual(tracker.current_state, VacancyTemporalState.TEMPORARY_VACANCY)

        # 5th frame vacant -> VACANCY_CONFIRMED
        s = tracker.update(vacant_snap)
        self.assertEqual(tracker.current_state, VacancyTemporalState.VACANCY_CONFIRMED)
        self.assertTrue(tracker.is_confirmed)

        # 3 frames occupied -> RESTORED
        for _ in range(3):
            tracker.update(occupied_snap)
        self.assertEqual(tracker.current_state, VacancyTemporalState.RESTORED)

    def test_04_shopper_occlusion_freeze(self):
        engine = ShelfVacancyEngine(shelf_id="SHELF-01", roi=(0.0, 0.0, 1.0, 1.0))
        tracker = ShelfVacancyTracker(min_consecutive_frames=3)

        mock_boxes = [
            (50, 100, 100, 150),
            (400, 100, 450, 150),
            (460, 100, 510, 150),
        ]
        # Shopper standing in the gap between 100 and 400
        shopper_boxes = [(150, 50, 350, 500)]

        snap = engine.analyze_frame(mock_boxes, 1000, 600, person_boxes=shopper_boxes)
        self.assertTrue(snap.is_occluded)
        self.assertEqual(snap.status, "UNCERTAIN")
        # Gap should be flagged as occluded
        self.assertTrue(snap.vacant_regions[0].is_occluded_by_person)

        # Tracker must freeze and NOT confirm vacancy
        for _ in range(5):
            tracker.update(snap)
        self.assertNotEqual(tracker.current_state, VacancyTemporalState.VACANCY_CONFIRMED)

    def test_05_end_to_end_v1_pipeline_run(self):
        report = _run_pipeline_sync(max_frames=15, video_source_input="shelf_pan_demo.mp4")

        # Verify V1 schema
        self.assertEqual(report.get("camera_id"), "CAM-01")
        self.assertIn("shelves", report)
        self.assertEqual(len(report["shelves"]), 1)
        shelf = report["shelves"][0]
        self.assertEqual(shelf["shelf_id"], "SHELF-01")
        self.assertIn(shelf["status"], ["VACANT", "OCCUPIED", "UNCERTAIN"])
        self.assertIn("occupancy_pct", shelf)
        self.assertIn("vacancy_score", shelf)
        self.assertIn("temporal_state", shelf)

        # Verify alerts
        alerts = report.get("active_alerts", [])
        if shelf["status"] == "VACANT":
            self.assertTrue(len(alerts) > 0)
            self.assertEqual(alerts[0]["type"], "SHELF_VACANCY")
            self.assertEqual(alerts[0]["title"], "SHELF 1 — EMPTY SPACE DETECTED")

        # Verify zero occurrences of fabricated SKUs or planogram tiers
        rep_json_str = json.dumps(report).lower()
        forbidden = [
            "7-up", "sudafed", "coke", "pepsi", "bisleri",
            "tier-01", "tier-02", "prime eye-level", "floor staging"
        ]
        for term in forbidden:
            self.assertNotIn(term, rep_json_str, f"Forbidden legacy term leaked into report: {term}")


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING STEP 25 AUTOMATED VALIDATION SUITE")
    print("=" * 80)
    unittest.main()
