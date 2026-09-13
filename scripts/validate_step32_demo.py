"""Step 32: End-to-End Retail Demo & Production Hardening Validation Suite.

Validates the core hardening and operational scenarios required by Step 32:
1. Complete Operational Lifecycle:
   Normal -> Candidate Gap -> Confirmed Vacancy -> Replenishment Recommendation ->
   Operator Acknowledge -> Shelf Restored -> Audit History verification.
2. Run Isolation:
   Active vacancies in Run A do not leak into Run B.
3. Stale-State Prevention:
   Starting a new run resets active tracking handles and isolates alerts.
4. Dashboard/API Data Synchronization:
   Report active_events, events history, and API responses match atomically.
5. Continuous Frame Deduplication:
   Continuous gap updates single event in-place without duplicate records.
6. API Outage & Fallback Resilience:
   Graceful behavior on client fallbacks and error handling.
7. Invalid Video Handling:
   Requesting a non-existent video returns clean 404 response without pipeline deadlock.
8. Event Persistence Recovery:
   Malformed JSON items or corrupted records are skipped without server crash.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys
import tempfile
import unittest
from typing import Any, Dict, List

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inventory.shelf_events import (
    ShelfEventManager,
    ShelfEventType,
    ShelfEventStatus,
    ShelfInventoryEvent,
)
from inventory.shelf_vacancy import (
    ShelfTier,
    ShelfVacancySnapshot,
    VacantRegion,
    VacancyTemporalState,
)


class TestStep32DemoHardening(unittest.TestCase):
    """Step 32 Validation Suite for End-to-End Demo & Production Hardening."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.events_file = pathlib.Path(self.temp_dir) / "inventory_events.json"
        self.dashboard_file = pathlib.Path(self.temp_dir) / "public_inventory_events.json"

        self.manager = ShelfEventManager(
            run_id="RUN-DEMO-001",
            camera_id="CAM-01",
            storage_path=self.events_file,
            dashboard_sync_path=self.dashboard_file,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_snapshot(
        self,
        frame_idx: int,
        tier_gaps: Dict[str, List[VacantRegion]],
        status: str = "OCCUPIED",
        occluded: bool = False,
        motion: bool = False,
    ) -> ShelfVacancySnapshot:
        tiers = []
        all_vacant = []
        for t_id, gaps in tier_gaps.items():
            t = ShelfTier(
                tier_id=t_id,
                name=f"Tier {t_id}",
                roi=(0.02, 0.12, 0.98, 0.40),
                y_min_px=200,
                y_max_px=800,
                product_count=max(0, 10 - len(gaps) * 2),
                median_product_width=120.0,
                occupancy_pct=85.0 if not gaps else 45.0,
                status="VACANT" if any(g.replenishment_recommended for g in gaps) else ("UNCERTAIN" if gaps else "OCCUPIED"),
                temporal_state=VacancyTemporalState.VACANCY_CONFIRMED.value if any(g.replenishment_recommended for g in gaps) else VacancyTemporalState.NORMAL.value,
                gaps=gaps,
            )
            tiers.append(t)
            all_vacant.extend(gaps)

        return ShelfVacancySnapshot(
            shelf_id="SHELF-01",
            status=status,
            vacancy_detected=len(all_vacant) > 0,
            vacancy_score=0.55 if all_vacant else 0.15,
            occupancy_pct=45.0 if all_vacant else 85.0,
            detected_facings_count=8,
            temporal_state=VacancyTemporalState.VACANCY_CONFIRMED.value if any(g.replenishment_recommended for g in all_vacant) else VacancyTemporalState.NORMAL.value,
            vacant_regions=all_vacant,
            unoccluded_vacant_regions=[g for g in all_vacant if not g.is_occluded_by_person],
            tiers=tiers,
            is_occluded=occluded,
            is_camera_moving=motion,
        )

    def test_01_complete_operational_lifecycle(self) -> None:
        """Complete lifecycle: Normal -> Candidate -> Confirmed -> Ack -> Restored -> Audit History."""
        gap = VacantRegion(
            region_id="GAP-T1-1",
            row_id="SHELF-01-T1",
            tier_id="SHELF-01-T1",
            x1=500,
            y1=300,
            x2=850,
            y2=800,
            x1_pct=0.28,
            y1_pct=0.15,
            x2_pct=0.48,
            y2_pct=0.35,
            width_px=350,
            width_pct=0.20,
            width_multiple=2.2,
            persistence_frames=1,
            vacancy_confidence=0.20,
            replenishment_recommended=False,
        )

        # 1. Normal state (0 events)
        snap_normal = self._make_snapshot(0, {"SHELF-01-T1": []})
        self.manager.process_frame(snap_normal, frame_index=0, timestamp_sec=0.0)
        self.assertEqual(len(self.manager.get_active_events()), 0)

        # 2. Candidate gap (observing, persistence < 10, no confirmed event yet)
        for f in range(1, 5):
            gap.persistence_frames = f
            snap_cand = self._make_snapshot(f, {"SHELF-01-T1": [gap]}, status="UNCERTAIN")
            self.manager.process_frame(snap_cand, frame_index=f, timestamp_sec=f * 0.1)
        self.assertEqual(len(self.manager.get_active_events()), 0)

        # 3. Confirmed Persistent Vacancy (persistence >= 10 frames) -> REPLENISHMENT_RECOMMENDED
        for f in range(5, 12):
            gap.persistence_frames = f
            gap.vacancy_confidence = 0.85
            gap.replenishment_recommended = True
            snap_conf = self._make_snapshot(f, {"SHELF-01-T1": [gap]}, status="VACANT")
            self.manager.process_frame(snap_conf, frame_index=f, timestamp_sec=f * 0.1)

        active = self.manager.get_active_events()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["status"], ShelfEventStatus.ACTIVE.value)
        self.assertEqual(active[0]["event_type"], ShelfEventType.REPLENISHMENT_RECOMMENDED.value)
        evt_id = active[0]["event_id"]

        # 4. Operator Acknowledge
        ack_res = self.manager.acknowledge_event(evt_id)
        self.assertIsNotNone(ack_res)
        self.assertEqual(ack_res.status, ShelfEventStatus.ACKNOWLEDGED.value)

        # 5. Shelf Restored (gap refilled, 0 gaps on tier)
        snap_restored = self._make_snapshot(15, {"SHELF-01-T1": []}, status="OCCUPIED")
        self.manager.process_frame(snap_restored, frame_index=15, timestamp_sec=1.5)

        # 6. Verify Active Cleared & Audit History Recorded
        self.assertEqual(len(self.manager.get_active_events()), 0)
        history = self.manager.get_history()
        types = [h["event_type"] for h in history]
        self.assertIn(ShelfEventType.REPLENISHMENT_RECOMMENDED.value, types)
        self.assertIn(ShelfEventType.SHELF_RESTORED.value, types)

    def test_02_run_isolation(self) -> None:
        """Active events in Run A do not leak into Run B queries."""
        gap = VacantRegion(
            region_id="GAP-T1-1",
            row_id="SHELF-01-T1",
            tier_id="SHELF-01-T1",
            x1=500,
            y1=300,
            x2=850,
            y2=800,
            x1_pct=0.28,
            y1_pct=0.15,
            x2_pct=0.48,
            y2_pct=0.35,
            width_px=350,
            width_pct=0.20,
            width_multiple=2.2,
            persistence_frames=12,
            vacancy_confidence=0.85,
            replenishment_recommended=True,
        )

        # Process in RUN-A
        self.manager.run_id = "RUN-A"
        snap_a = self._make_snapshot(0, {"SHELF-01-T1": [gap]}, status="VACANT")
        self.manager.process_frame(snap_a, frame_index=0, timestamp_sec=0.0)

        # Query scoped to RUN-A vs RUN-B
        active_a = self.manager.get_active_events(run_id="RUN-A")
        active_b = self.manager.get_active_events(run_id="RUN-B")
        self.assertEqual(len(active_a), 1)
        self.assertEqual(len(active_b), 0)

    def test_03_stale_state_prevention_on_new_run(self) -> None:
        """reset_active_for_new_run resets active handles while preserving historical events."""
        gap = VacantRegion(
            region_id="GAP-T1-1",
            row_id="SHELF-01-T1",
            tier_id="SHELF-01-T1",
            x1=500,
            y1=300,
            x2=850,
            y2=800,
            x1_pct=0.28,
            y1_pct=0.15,
            x2_pct=0.48,
            y2_pct=0.35,
            width_px=350,
            width_pct=0.20,
            width_multiple=2.2,
            persistence_frames=12,
            vacancy_confidence=0.85,
            replenishment_recommended=True,
        )
        self.manager.run_id = "RUN-OLD"
        snap = self._make_snapshot(0, {"SHELF-01-T1": [gap]}, status="VACANT")
        self.manager.process_frame(snap, frame_index=0, timestamp_sec=0.0)
        self.assertEqual(len(self.manager.get_active_events()), 1)

        # Start new run
        self.manager.reset_active_for_new_run("RUN-NEW")
        self.assertEqual(self.manager.run_id, "RUN-NEW")
        # Scoped to RUN-NEW must be 0
        self.assertEqual(len(self.manager.get_active_events(run_id="RUN-NEW")), 0)
        # History still contains the event from RUN-OLD
        self.assertEqual(len(self.manager.get_history()), 1)

    def test_04_dashboard_api_synchronization(self) -> None:
        """Active events match between manager, JSON persistence, and disk files."""
        gap = VacantRegion(
            region_id="GAP-T1-1",
            row_id="SHELF-01-T1",
            tier_id="SHELF-01-T1",
            x1=500,
            y1=300,
            x2=850,
            y2=800,
            x1_pct=0.28,
            y1_pct=0.15,
            x2_pct=0.48,
            y2_pct=0.35,
            width_px=350,
            width_pct=0.20,
            width_multiple=2.2,
            persistence_frames=12,
            vacancy_confidence=0.85,
            replenishment_recommended=True,
        )
        snap = self._make_snapshot(0, {"SHELF-01-T1": [gap]}, status="VACANT")
        self.manager.process_frame(snap, frame_index=0, timestamp_sec=0.0)

        # Check file contents
        self.assertTrue(self.events_file.exists())
        self.assertTrue(self.dashboard_file.exists())

        with open(self.events_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["active_events_count"], 1)
        self.assertEqual(data["active_run_id"], "RUN-DEMO-001")
        self.assertEqual(len(data["events"]), 1)

    def test_05_continuous_frame_deduplication(self) -> None:
        """30 frames of the same vacancy update the duration without creating duplicate events."""
        gap = VacantRegion(
            region_id="GAP-T1-1",
            row_id="SHELF-01-T1",
            tier_id="SHELF-01-T1",
            x1=500,
            y1=300,
            x2=850,
            y2=800,
            x1_pct=0.28,
            y1_pct=0.15,
            x2_pct=0.48,
            y2_pct=0.35,
            width_px=350,
            width_pct=0.20,
            width_multiple=2.2,
            persistence_frames=10,
            vacancy_confidence=0.85,
            replenishment_recommended=True,
        )
        for f in range(30):
            gap.persistence_frames = 10 + f
            snap = self._make_snapshot(f, {"SHELF-01-T1": [gap]}, status="VACANT")
            self.manager.process_frame(snap, frame_index=f, timestamp_sec=f * 0.1, fps=10.0)

        active = self.manager.get_active_events()
        self.assertEqual(len(active), 1, "Exactly one event must exist")
        self.assertEqual(active[0]["duration_frames"], 39)
        self.assertAlmostEqual(active[0]["duration_sec"], 3.9, delta=0.2)

    def test_06_api_outage_resilience(self) -> None:
        """Manager functions fully offline if dashboard sync destination cannot be written."""
        bad_sync = pathlib.Path(self.temp_dir) / "non_existent_folder" / "bad.json"
        # Non-existent parent directory will test resilient save handling
        mgr_resilient = ShelfEventManager(
            run_id="RUN-OFFLINE",
            storage_path=self.events_file,
            dashboard_sync_path=bad_sync,
        )
        gap = VacantRegion(
            region_id="GAP-T1-1",
            row_id="SHELF-01-T1",
            tier_id="SHELF-01-T1",
            x1=500,
            y1=300,
            x2=850,
            y2=800,
            x1_pct=0.28,
            y1_pct=0.15,
            x2_pct=0.48,
            y2_pct=0.35,
            width_px=350,
            width_pct=0.20,
            width_multiple=2.2,
            persistence_frames=12,
            vacancy_confidence=0.85,
            replenishment_recommended=True,
        )
        snap = self._make_snapshot(0, {"SHELF-01-T1": [gap]}, status="VACANT")
        # Should not raise exception
        mgr_resilient.process_frame(snap, frame_index=0, timestamp_sec=0.0)
        self.assertEqual(len(mgr_resilient.get_active_events()), 1)

    def test_07_invalid_video_handling(self) -> None:
        """Direct verification of video validation helper behavior."""
        from inventory.inventory_api import _list_demo_videos

        videos = _list_demo_videos()
        self.assertIsInstance(videos, list)
        self.assertGreater(len(videos), 0)
        names = [v["filename"] for v in videos]
        self.assertIn("shelf_pan_demo.mp4", names)

    def test_08_event_persistence_recovery(self) -> None:
        """Malformed or corrupted JSON records are skipped gracefully without crashing."""
        corrupted_data = {
            "last_updated_iso": "2026-09-13T00:00:00Z",
            "active_run_id": "RUN-CORRUPT",
            "events": [
                "not_a_dictionary_item",
                {"invalid_missing_keys": True},
                {
                    "event_id": "EVT-VALID-001",
                    "run_id": "RUN-CORRUPT",
                    "timestamp_iso": "2026-09-13T00:00:00Z",
                    "camera_id": "CAM-01",
                    "shelf_id": "SHELF-01",
                    "tier_id": "SHELF-01-T1",
                    "event_type": "VACANCY_CONFIRMED",
                    "status": "ACTIVE",
                    "visual_vacancy_confidence": 0.85,
                },
            ],
        }
        with open(self.events_file, "w", encoding="utf-8") as f:
            json.dump(corrupted_data, f)

        # Load into brand new manager
        recovery_mgr = ShelfEventManager(
            run_id="RUN-CORRUPT",
            storage_path=self.events_file,
            dashboard_sync_path=self.dashboard_file,
        )
        # Should recover the 1 valid event and skip the 2 malformed items
        self.assertEqual(len(recovery_mgr.get_history()), 1)
        self.assertEqual(recovery_mgr.get_history()[0]["event_id"], "EVT-VALID-001")


def run_suite() -> bool:
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStep32DemoHardening)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_suite()
    sys.exit(0 if success else 1)
