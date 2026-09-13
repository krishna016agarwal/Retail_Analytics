"""Step 31: Inventory Event & Replenishment Action Layer Validation Suite.

Validates the 11 core operational scenarios required by Step 31:
1. Normal shelf creates 0 events (no false alarms on normal stocked shelves).
2. Temporary gap does not create replenishment event (candidate gap disappears before confirmation).
3. Persistent vacancy creates 1 confirmed event (candidate reaches threshold, logs confirmed event).
4. Duplicate frames update duration without duplicate events (active event duration grows, no duplicate records).
5. Shopper occlusion suppresses event creation and freezes confirmation.
6. Camera motion suppresses event creation.
7. Restored shelf transitions active event to RESOLVED and logs SHELF_RESTORED.
8. Multiple disjoint gaps create independent events on separate tiers or regions.
9. run_id separation isolates events across different runs.
10. Event persistence survives disk reload (output/ and dashboard/public/ sync).
11. Acknowledgment and manual resolution mutations function properly (ACTIVE -> ACKNOWLEDGED -> RESOLVED).
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


class TestStep31InventoryEvents(unittest.TestCase):
    """Step 31 Validation Suite for Event & Replenishment Action Layer."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.events_file = pathlib.Path(self.temp_dir) / "inventory_events.json"
        self.dashboard_file = pathlib.Path(self.temp_dir) / "public_inventory_events.json"

        self.manager = ShelfEventManager(
            run_id="run_step31_test",
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

    def test_01_normal_shelf_creates_zero_events(self) -> None:
        """Normal stocked shelf creates 0 events across 30 frames."""
        for f in range(30):
            snap = self._make_snapshot(frame_idx=f, tier_gaps={"SHELF-01-T1": []})
            events = self.manager.process_frame(snap, frame_index=f, timestamp_sec=f * 0.1, fps=10.0)
            self.assertEqual(len(events), 0)

        self.assertEqual(len(self.manager.get_active_events()), 0)
        self.assertEqual(len(self.manager.get_history()), 0)

    def test_02_temporary_gap_does_not_create_replenishment_event(self) -> None:
        """A transient gap of 3 frames disappears before reaching persistence threshold (10 frames)."""
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
            persistence_frames=2,
            vacancy_confidence=0.35,
            replenishment_recommended=False,
        )

        for f in range(3):
            gap.persistence_frames = f + 1
            snap = self._make_snapshot(
                frame_idx=f,
                tier_gaps={"SHELF-01-T1": [gap]},
                status="UNCERTAIN",
            )
            self.manager.process_frame(snap, frame_index=f, timestamp_sec=f * 0.1, fps=10.0)

        # Gap disappears at frame 4
        snap_clear = self._make_snapshot(frame_idx=4, tier_gaps={"SHELF-01-T1": []})
        self.manager.process_frame(snap_clear, frame_index=4, timestamp_sec=0.4, fps=10.0)

        replenish_events = [
            e for e in self.manager.get_history()
            if e["event_type"] == ShelfEventType.REPLENISHMENT_RECOMMENDED.value
        ]
        self.assertEqual(len(replenish_events), 0, "Transient gap must not trigger replenishment recommendation")

    def test_03_persistent_vacancy_creates_one_confirmed_event(self) -> None:
        """A persistent gap that reaches 10 frames triggers a confirmed replenishment alert."""
        for f in range(15):
            persist = f + 1
            conf = 0.40 if persist < 10 else 0.85
            rec = persist >= 10
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
                persistence_frames=persist,
                vacancy_confidence=conf,
                replenishment_recommended=rec,
            )
            snap = self._make_snapshot(
                frame_idx=f,
                tier_gaps={"SHELF-01-T1": [gap]},
                status="VACANT" if rec else "UNCERTAIN",
            )
            self.manager.process_frame(snap, frame_index=f, timestamp_sec=f * 0.1, fps=10.0)

        active = self.manager.get_active_events()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["status"], ShelfEventStatus.ACTIVE.value)
        self.assertGreaterEqual(active[0]["visual_vacancy_confidence"], 0.70)
        self.assertEqual(active[0]["tier_id"], "SHELF-01-T1")
        self.assertEqual(active[0]["event_type"], ShelfEventType.REPLENISHMENT_RECOMMENDED.value)

    def test_04_duplicate_frames_update_duration_without_duplicate_events(self) -> None:
        """Subsequent frames with the same active gap update duration, not creating duplicate events."""
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

        for f in range(20):
            gap.persistence_frames = 10 + f
            snap = self._make_snapshot(
                frame_idx=f,
                tier_gaps={"SHELF-01-T1": [gap]},
                status="VACANT",
            )
            self.manager.process_frame(snap, frame_index=f, timestamp_sec=f * 0.1, fps=10.0)

        active = self.manager.get_active_events()
        self.assertEqual(len(active), 1, "Exactly one active event should exist despite 20 frames")
        self.assertGreaterEqual(active[0]["duration_frames"], 25)
        self.assertAlmostEqual(active[0]["duration_sec"], 2.9, delta=0.5)

    def test_05_shopper_occlusion_freezes_and_suppresses(self) -> None:
        """When a person occludes the shelf, no new events are created."""
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
            vacancy_confidence=0.80,
            replenishment_recommended=True,
        )

        # 1. Establish active event
        snap = self._make_snapshot(frame_idx=0, tier_gaps={"SHELF-01-T1": [gap]}, status="VACANT")
        self.manager.process_frame(snap, frame_index=0, timestamp_sec=0.0, fps=10.0)
        self.assertEqual(len(self.manager.get_active_events()), 1)

        # 2. Shopper occludes shelf
        snap_occ = self._make_snapshot(
            frame_idx=1,
            tier_gaps={"SHELF-01-T1": [gap]},
            status="UNCERTAIN",
            occluded=True,
        )
        new_events = self.manager.process_frame(snap_occ, frame_index=1, timestamp_sec=0.1, fps=10.0)
        self.assertEqual(len(new_events), 0, "No new events during shopper occlusion")

    def test_06_camera_motion_suppresses_event_creation(self) -> None:
        """When camera is moving, event creation is suppressed."""
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
        snap_motion = self._make_snapshot(
            frame_idx=0,
            tier_gaps={"SHELF-01-T1": [gap]},
            status="UNCERTAIN",
            motion=True,
        )
        events = self.manager.process_frame(snap_motion, frame_index=0, timestamp_sec=0.0, fps=10.0)
        self.assertEqual(len(events), 0)
        self.assertEqual(len(self.manager.get_active_events()), 0)

    def test_07_restored_shelf_resolves_active_event_and_logs_restored(self) -> None:
        """When shelf is restocked, the active event resolves and a SHELF_RESTORED event is logged."""
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
        # 1. Trigger active vacancy
        snap_vacant = self._make_snapshot(frame_idx=0, tier_gaps={"SHELF-01-T1": [gap]}, status="VACANT")
        self.manager.process_frame(snap_vacant, frame_index=0, timestamp_sec=0.0, fps=10.0)
        self.assertEqual(len(self.manager.get_active_events()), 1)

        # 2. Shelf restored (0 gaps)
        snap_restored = self._make_snapshot(frame_idx=1, tier_gaps={"SHELF-01-T1": []}, status="OCCUPIED")
        self.manager.process_frame(snap_restored, frame_index=1, timestamp_sec=0.1, fps=10.0)

        # Active events must now be 0
        self.assertEqual(len(self.manager.get_active_events()), 0)

        # History must show resolved event and SHELF_RESTORED event
        restored_events = [
            e for e in self.manager.get_history()
            if e["event_type"] == ShelfEventType.SHELF_RESTORED.value
        ]
        self.assertEqual(len(restored_events), 1)
        self.assertEqual(restored_events[0]["status"], ShelfEventStatus.RESOLVED.value)

    def test_08_multiple_disjoint_gaps_create_independent_events(self) -> None:
        """Two distinct gaps on different tiers create two separate tracked events."""
        gap1 = VacantRegion(
            region_id="GAP-T1-1",
            row_id="SHELF-01-T1",
            tier_id="SHELF-01-T1",
            x1=100,
            y1=300,
            x2=400,
            y2=800,
            x1_pct=0.05,
            y1_pct=0.15,
            x2_pct=0.25,
            y2_pct=0.35,
            width_px=300,
            width_pct=0.20,
            width_multiple=2.0,
            persistence_frames=12,
            vacancy_confidence=0.82,
            replenishment_recommended=True,
        )
        gap2 = VacantRegion(
            region_id="GAP-T2-1",
            row_id="SHELF-01-T2",
            tier_id="SHELF-01-T2",
            x1=800,
            y1=850,
            x2=1100,
            y2=1300,
            x1_pct=0.45,
            y1_pct=0.40,
            x2_pct=0.65,
            y2_pct=0.60,
            width_px=300,
            width_pct=0.20,
            width_multiple=2.0,
            persistence_frames=12,
            vacancy_confidence=0.85,
            replenishment_recommended=True,
        )
        snap = self._make_snapshot(
            frame_idx=0,
            tier_gaps={"SHELF-01-T1": [gap1], "SHELF-01-T2": [gap2]},
            status="VACANT",
        )
        self.manager.process_frame(snap, frame_index=0, timestamp_sec=0.0, fps=10.0)
        active = self.manager.get_active_events()
        self.assertEqual(len(active), 2)
        tier_ids = {e["tier_id"] for e in active}
        self.assertIn("SHELF-01-T1", tier_ids)
        self.assertIn("SHELF-01-T2", tier_ids)

    def test_09_run_id_separation_isolates_events(self) -> None:
        """Switching run_id clear/filters history properly."""
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
        snap = self._make_snapshot(frame_idx=0, tier_gaps={"SHELF-01-T1": [gap]}, status="VACANT")

        self.manager.process_frame(snap, frame_index=0, timestamp_sec=0.0, fps=10.0)
        self.assertEqual(len(self.manager.get_history(run_id="run_step31_test")), 1)
        self.assertEqual(len(self.manager.get_history(run_id="run_different")), 0)

    def test_10_event_persistence_survives_disk_reload(self) -> None:
        """Events written to disk are cleanly loaded by a brand-new manager instance."""
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
        snap = self._make_snapshot(frame_idx=0, tier_gaps={"SHELF-01-T1": [gap]}, status="VACANT")
        self.manager.process_frame(snap, frame_index=0, timestamp_sec=0.0, fps=10.0)

        self.assertTrue(self.events_file.exists())
        self.assertTrue(self.dashboard_file.exists())

        # Create new manager pointing to same files
        reloaded_mgr = ShelfEventManager(
            run_id="run_step31_test",
            storage_path=self.events_file,
            dashboard_sync_path=self.dashboard_file,
        )
        self.assertEqual(len(reloaded_mgr.get_active_events()), 1)
        self.assertEqual(len(reloaded_mgr.get_history()), 1)
        self.assertEqual(reloaded_mgr.get_active_events()[0]["tier_id"], "SHELF-01-T1")

    def test_11_acknowledge_and_resolve_mutations(self) -> None:
        """Interactive ACKNOWLEDGE and RESOLVE operations update status and write through."""
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
        snap = self._make_snapshot(frame_idx=0, tier_gaps={"SHELF-01-T1": [gap]}, status="VACANT")
        self.manager.process_frame(snap, frame_index=0, timestamp_sec=0.0, fps=10.0)

        active = self.manager.get_active_events()
        self.assertEqual(len(active), 1)
        evt_id = active[0]["event_id"]

        # 1. Acknowledge
        ack_res = self.manager.acknowledge_event(evt_id)
        self.assertIsNotNone(ack_res)
        self.assertEqual(ack_res.status, ShelfEventStatus.ACKNOWLEDGED.value)
        self.assertIsNotNone(ack_res.acknowledged_at)

        # 2. Resolve
        res_res = self.manager.resolve_event(evt_id)
        self.assertIsNotNone(res_res)
        self.assertEqual(res_res.status, ShelfEventStatus.RESOLVED.value)
        self.assertIsNotNone(res_res.resolved_at)
        self.assertEqual(len(self.manager.get_active_events()), 0)


def run_suite() -> bool:
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStep31InventoryEvents)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_suite()
    sys.exit(0 if success else 1)
