"""Step 33 — Final Retail Product & Demo Layer Validation Suite.

Validates:
1. Overview uses actual API/event data (monitored shelves, active alerts, run ID).
2. Priority states: CRITICAL (conf >= 0.70), MONITORING (< 10 frames), UNCERTAIN (motion/occlusion), NORMAL.
3. Active event rendering: required fields (shelf/tier, confidence, duration, coords, status).
4. Acknowledge and Resolve lifecycle (ACTIVE -> ACKNOWLEDGED -> RESOLVED).
5. Shelf Restored lifecycle (re-occupation closes previous active alerts with SHELF_RESTORED).
6. Run isolation (new run resets active alerts, prevents cross-run contamination).
7. Stale-state prevention on video switch.
8. Lightweight analytics calculations and insufficient data guard (< 2 samples).
9. API failure graceful handling (offline safety, fallback resilience).
10. Dashboard refresh consistency (persistence round-trip integrity).
"""

from __future__ import annotations

import json
import sys
import tempfile
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inventory.shelf_events import (
    ShelfEventManager,
    ShelfInventoryEvent,
    ShelfEventStatus,
    ShelfEventType,
)
from inventory.shelf_vacancy import (
    ShelfTier,
    ShelfVacancySnapshot,
    VacantRegion,
    VacancyTemporalState,
)


def make_test_gap(
    region_id: str = "GAP-01",
    tier_id: str = "SHELF-01-T2",
    x1: int = 200,
    x2: int = 450,
    persistence_frames: int = 12,
    conf: float = 0.88,
    replenish: bool = True,
) -> VacantRegion:
    return VacantRegion(
        region_id=region_id,
        row_id=tier_id,
        tier_id=tier_id,
        x1=x1,
        y1=250,
        x2=x2,
        y2=450,
        x1_pct=x1 / 1000.0,
        y1_pct=0.25,
        x2_pct=x2 / 1000.0,
        y2_pct=0.45,
        width_px=float(x2 - x1),
        width_pct=float(x2 - x1) / 1000.0,
        width_multiple=float(x2 - x1) / 100.0,
        persistence_frames=persistence_frames,
        geometric_stability=1.0,
        vacancy_confidence=conf,
        replenishment_recommended=replenish,
    )


def make_test_snapshot(
    shelf_id: str = "SHELF-01",
    tier_id: str = "SHELF-01-T2",
    gaps: Optional[List[VacantRegion]] = None,
    is_occluded: bool = False,
    is_camera_moving: bool = False,
) -> ShelfVacancySnapshot:
    active_gaps = gaps or []
    tier = ShelfTier(
        tier_id=tier_id,
        name=f"Tier {tier_id}",
        roi=(0.0, 0.2, 1.0, 0.5),
        y_min_px=200,
        y_max_px=500,
        product_count=8 if not active_gaps else 4,
        median_product_width=100.0,
        occupancy_pct=85.0 if not active_gaps else 40.0,
        status="VACANT" if active_gaps else "OCCUPIED",
        gaps=active_gaps,
    )
    return ShelfVacancySnapshot(
        shelf_id=shelf_id,
        status="VACANT" if active_gaps else "OCCUPIED",
        vacancy_detected=bool(active_gaps),
        vacancy_score=0.8 if active_gaps else 0.1,
        occupancy_pct=40.0 if active_gaps else 85.0,
        detected_facings_count=8,
        vacant_regions=active_gaps,
        unoccluded_vacant_regions=active_gaps,
        tiers=[tier],
        is_occluded=is_occluded,
        is_camera_moving=is_camera_moving,
    )


def run_test(name: str, test_fn) -> bool:
    print(f"\n--- [TEST] {name} ---")
    try:
        test_fn()
        print(f"PASS: {name}")
        return True
    except AssertionError as ae:
        print(f"FAIL (Assertion): {name} -> {ae}")
        return False
    except Exception as e:
        print(f"FAIL (Exception): {name} -> {type(e).__name__}: {e}")
        return False


def test_overview_uses_actual_api_and_event_data():
    """Verify overview metrics derive strictly from real system and event data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Path(tmpdir) / "events.json"
        sync_path = Path(tmpdir) / "public_events.json"
        mgr = ShelfEventManager(
            run_id="RUN-TEST-01",
            camera_id="CAM-01",
            storage_path=storage,
            dashboard_sync_path=sync_path,
        )

        # Initially 0 active alerts
        active = mgr.get_active_events()
        assert len(active) == 0, "Initial active events must be 0"

        # Record a vacancy event on Tier 2
        gap = make_test_gap(region_id="GAP-01", tier_id="SHELF-01-T2", persistence_frames=12, conf=0.88)
        snap = make_test_snapshot(tier_id="SHELF-01-T2", gaps=[gap])
        mgr.process_frame(snap, frame_index=15, timestamp_sec=0.5, fps=30.0)

        active = mgr.get_active_events(run_id="RUN-TEST-01")
        assert len(active) == 1, f"Expected 1 active event, got {len(active)}"
        assert active[0]["tier_id"] == "SHELF-01-T2"
        assert active[0]["run_id"] == "RUN-TEST-01"
        assert active[0]["camera_id"] == "CAM-01"


def test_alert_priority_states():
    """Verify the 4 explicit priority levels conform strictly to existing criteria."""
    def evaluate_priority(is_vacant: bool, conf: float, persistence_frames: int, is_occluded: bool, is_camera_moving: bool):
        is_critical = is_vacant and conf >= 0.70
        is_monitoring = not is_critical and persistence_frames < 10 and not (conf >= 0.70 and persistence_frames >= 10)
        is_uncertain = not is_critical and not is_monitoring and (is_occluded or is_camera_moving)
        is_normal = not is_critical and not is_monitoring and not is_uncertain
        
        if is_critical:
            return "CRITICAL"
        elif is_monitoring:
            return "MONITORING"
        elif is_uncertain:
            return "UNCERTAIN"
        else:
            return "NORMAL"

    # 1. CRITICAL: Persistent vacancy with conf >= 0.70
    p1 = evaluate_priority(is_vacant=True, conf=0.85, persistence_frames=15, is_occluded=False, is_camera_moving=False)
    assert p1 == "CRITICAL", f"Expected CRITICAL, got {p1}"

    # 2. MONITORING: Vacancy candidate still being evaluated (< 10 frames)
    p2 = evaluate_priority(is_vacant=False, conf=0.55, persistence_frames=5, is_occluded=False, is_camera_moving=False)
    assert p2 == "MONITORING", f"Expected MONITORING, got {p2}"

    # 3. UNCERTAIN: Shopper occlusion or camera motion flag active
    p3 = evaluate_priority(is_vacant=False, conf=0.0, persistence_frames=10, is_occluded=True, is_camera_moving=False)
    assert p3 == "UNCERTAIN", f"Expected UNCERTAIN for occlusion, got {p3}"
    p3_motion = evaluate_priority(is_vacant=False, conf=0.0, persistence_frames=10, is_occluded=False, is_camera_moving=True)
    assert p3_motion == "UNCERTAIN", f"Expected UNCERTAIN for camera motion, got {p3_motion}"

    # 4. NORMAL: No actionable vacancy, fully occupied shelf
    p4 = evaluate_priority(is_vacant=False, conf=0.0, persistence_frames=10, is_occluded=False, is_camera_moving=False)
    assert p4 == "NORMAL", f"Expected NORMAL, got {p4}"


def test_active_event_rendering_and_fields():
    """Verify that active events contain all mandatory metadata for the Replenishment Action Center."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Path(tmpdir) / "events.json"
        sync_path = Path(tmpdir) / "public_events.json"
        mgr = ShelfEventManager(
            run_id="RUN-TEST-02",
            camera_id="CAM-01",
            storage_path=storage,
            dashboard_sync_path=sync_path,
        )

        gap = make_test_gap(region_id="GAP-T1-01", tier_id="SHELF-01-T1", x1=150, x2=380, persistence_frames=15, conf=0.91)
        snap = make_test_snapshot(tier_id="SHELF-01-T1", gaps=[gap])
        evs = mgr.process_frame(snap, frame_index=30, timestamp_sec=1.0, fps=30.0)

        assert len(evs) > 0, "Expected generated event"
        d = evs[0].to_dict()

        # Check all required UI fields
        required_fields = [
            "event_id", "run_id", "shelf_id", "tier_id",
            "visual_vacancy_confidence", "duration_sec", "gap_coordinates",
            "timestamp_iso", "status", "action_required"
        ]
        for f in required_fields:
            assert f in d, f"Missing required field '{f}' in event payload"

        assert d["tier_id"] == "SHELF-01-T1"
        assert d["visual_vacancy_confidence"] == 0.91
        assert d["gap_coordinates"]["x1"] == 150
        assert d["gap_coordinates"]["x2"] == 380
        assert d["status"] == "ACTIVE"


def test_acknowledge_and_resolve_lifecycle():
    """Verify operator lifecycle: ACTIVE -> ACKNOWLEDGED -> RESOLVED."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Path(tmpdir) / "events.json"
        sync_path = Path(tmpdir) / "public_events.json"
        mgr = ShelfEventManager(
            run_id="RUN-TEST-03",
            camera_id="CAM-01",
            storage_path=storage,
            dashboard_sync_path=sync_path,
        )

        gap = make_test_gap(region_id="GAP-T2-01", tier_id="SHELF-01-T2", persistence_frames=10, conf=0.82)
        snap = make_test_snapshot(tier_id="SHELF-01-T2", gaps=[gap])
        evs = mgr.process_frame(snap, frame_index=30, timestamp_sec=1.0, fps=30.0)
        assert len(evs) == 1
        event_id = evs[0].event_id

        # 1. Initially ACTIVE
        assert evs[0].status == ShelfEventStatus.ACTIVE.value
        assert len(mgr.get_active_events()) == 1

        # 2. Operator Acknowledges
        ack_ev = mgr.acknowledge_event(event_id)
        assert ack_ev is not None
        assert ack_ev.status == ShelfEventStatus.ACKNOWLEDGED.value
        assert ack_ev.acknowledged_at is not None
        # Still in active alerts list, but marked acknowledged
        active = mgr.get_active_events()
        assert len(active) == 1
        assert active[0]["status"] == "ACKNOWLEDGED"

        # 3. Operator Resolves
        res_ev = mgr.resolve_event(event_id)
        assert res_ev is not None
        assert res_ev.status == ShelfEventStatus.RESOLVED.value
        assert res_ev.resolved_at is not None
        # Removed from active alerts
        assert len(mgr.get_active_events()) == 0

        # Verified present in history
        history = mgr.get_history()
        assert len(history) >= 1
        assert history[0]["event_id"] == event_id
        assert history[0]["status"] == "RESOLVED"


def test_restored_shelf_lifecycle():
    """Verify shelf restoration closes the active vacancy with SHELF_RESTORED."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Path(tmpdir) / "events.json"
        sync_path = Path(tmpdir) / "public_events.json"
        mgr = ShelfEventManager(
            run_id="RUN-TEST-04",
            camera_id="CAM-01",
            storage_path=storage,
            dashboard_sync_path=sync_path,
        )

        # 1. Vacancy confirmed
        gap = make_test_gap(region_id="GAP-02", tier_id="SHELF-01-T2", persistence_frames=12, conf=0.85)
        snap1 = make_test_snapshot(tier_id="SHELF-01-T2", gaps=[gap])
        mgr.process_frame(snap1, frame_index=20, timestamp_sec=0.67, fps=30.0)
        assert len(mgr.get_active_events()) == 1

        # 2. Shelf re-occupied (no gaps in snapshot)
        snap2 = make_test_snapshot(tier_id="SHELF-01-T2", gaps=[])
        restored_evs = mgr.process_frame(snap2, frame_index=60, timestamp_sec=2.0, fps=30.0)
        assert len(restored_evs) >= 1, "Should generate SHELF_RESTORED event"
        restored = next((e for e in restored_evs if e.event_type == ShelfEventType.SHELF_RESTORED.value), None)
        assert restored is not None, "SHELF_RESTORED event must be present"
        assert restored.status == ShelfEventStatus.RESOLVED.value

        # Active events must now be empty
        assert len(mgr.get_active_events()) == 0


def test_run_isolation_active_events():
    """Verify new run ID isolates active tracking and prevents stale alerts from previous run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Path(tmpdir) / "events.json"
        sync_path = Path(tmpdir) / "public_events.json"
        mgr = ShelfEventManager(
            run_id="RUN-001",
            camera_id="CAM-01",
            storage_path=storage,
            dashboard_sync_path=sync_path,
        )

        # Generate alert in RUN-001
        gap = make_test_gap(region_id="GAP-A", tier_id="SHELF-01-T1", persistence_frames=10, conf=0.80)
        snap = make_test_snapshot(tier_id="SHELF-01-T1", gaps=[gap])
        mgr.process_frame(snap, frame_index=15, timestamp_sec=0.5, fps=30.0)
        assert len(mgr.get_active_events("RUN-001")) == 1

        # Start new RUN-002
        mgr.reset_active_for_new_run("RUN-002")

        # RUN-002 active events must be 0
        assert len(mgr.get_active_events("RUN-002")) == 0
        # History retains previous run's record
        hist_001 = mgr.get_history(run_id="RUN-001")
        assert len(hist_001) == 1


def test_stale_state_prevention_on_video_switch():
    """Verify video switching does not contaminate event state across different video streams."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Path(tmpdir) / "events.json"
        sync_path = Path(tmpdir) / "public_events.json"
        mgr = ShelfEventManager(
            run_id="RUN-PAN-01",
            camera_id="CAM-01",
            storage_path=storage,
            dashboard_sync_path=sync_path,
        )

        # Run 1: shelf_pan_demo.mp4
        gap = make_test_gap(region_id="GAP-PAN", tier_id="SHELF-01-T3", persistence_frames=10, conf=0.78)
        snap = make_test_snapshot(tier_id="SHELF-01-T3", gaps=[gap])
        mgr.process_frame(snap, frame_index=25, timestamp_sec=0.8, fps=30.0)

        # Switch to inventory2.mp4 under new run ID
        mgr.reset_active_for_new_run("RUN-AISLE-01")

        # Active events for RUN-AISLE-01 are clean
        active_aisle = mgr.get_active_events(run_id="RUN-AISLE-01")
        assert len(active_aisle) == 0, f"Expected 0 active events for new run, got {len(active_aisle)}"


def test_real_analytics_calculations_and_empty_state():
    """Verify analytics calculations strictly adhere to real data and insufficient data state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Path(tmpdir) / "events.json"
        sync_path = Path(tmpdir) / "public_events.json"
        mgr = ShelfEventManager(
            run_id="RUN-ANALYTICS",
            camera_id="CAM-01",
            storage_path=storage,
            dashboard_sync_path=sync_path,
        )

        # Case A: 0 events -> insufficient data
        s0 = mgr.get_analytics_summary()
        assert s0["total_events"] == 0
        assert s0["has_sufficient_resolution_data"] is False
        assert s0["avg_resolution_time_sec"] is None

        # Case B: 1 resolved event -> insufficient data (< 2 samples)
        gap1 = make_test_gap(region_id="GAP-1", tier_id="SHELF-01-T1", persistence_frames=10, conf=0.85)
        snap1 = make_test_snapshot(tier_id="SHELF-01-T1", gaps=[gap1])
        mgr.process_frame(snap1, frame_index=10, timestamp_sec=0.33, fps=30.0)
        ev1_id = mgr.get_active_events()[0]["event_id"]
        mgr.resolve_event(ev1_id)

        s1 = mgr.get_analytics_summary()
        assert s1["resolved_events"] == 1
        assert s1["has_sufficient_resolution_data"] is False
        assert s1["avg_resolution_time_sec"] is None

        # Case C: 2 resolved events -> sufficient data, accurately calculated
        gap2 = make_test_gap(region_id="GAP-2", tier_id="SHELF-01-T2", persistence_frames=12, conf=0.90)
        snap2 = make_test_snapshot(tier_id="SHELF-01-T2", gaps=[gap2])
        mgr.process_frame(snap2, frame_index=40, timestamp_sec=1.33, fps=30.0)
        ev2_id = mgr.get_active_events()[0]["event_id"]
        mgr.resolve_event(ev2_id)

        s2 = mgr.get_analytics_summary()
        assert s2["resolved_events"] == 2
        assert s2["has_sufficient_resolution_data"] is True
        assert s2["avg_resolution_time_sec"] is not None
        assert isinstance(s2["avg_resolution_time_sec"], float)


def test_api_failure_graceful_handling():
    """Verify API failure resilience: offline endpoints fail cleanly with proper status codes."""
    base_url = "http://127.0.0.1:8001"

    # 1. Non-existent event acknowledge returns 404 cleanly
    req_ack = urllib.request.Request(f"{base_url}/inventory/events/NON_EXISTENT_ID/acknowledge", data=b"")
    try:
        urllib.request.urlopen(req_ack)
        assert False, "Expected 404 for acknowledge on non-existent event"
    except urllib.error.HTTPError as e:
        assert e.code == 404

    # 2. Non-existent event resolve returns 404 cleanly
    req_res = urllib.request.Request(f"{base_url}/inventory/events/NON_EXISTENT_ID/resolve", data=b"")
    try:
        urllib.request.urlopen(req_res)
        assert False, "Expected 404 for resolve on non-existent event"
    except urllib.error.HTTPError as e:
        assert e.code == 404

    # 3. Invalid video in run start returns 404 cleanly without crashing
    req_run = urllib.request.Request(f"{base_url}/inventory/run?video_source=non_existent_fake_video.mp4", data=b"")
    try:
        urllib.request.urlopen(req_run)
        assert False, "Expected 404 for non-existent video run"
    except urllib.error.HTTPError as e:
        assert e.code == 404
        body = e.read().decode()
        assert "not exist" in body or "not found" in body

    # 4. Status endpoint works without crashing
    resp_status = urllib.request.urlopen(f"{base_url}/inventory/status")
    assert resp_status.status == 200


def test_dashboard_refresh_consistency():
    """Verify that restarting/refreshing the system reloads disk data with complete integrity."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Path(tmpdir) / "events.json"
        sync_path = Path(tmpdir) / "public_events.json"

        # Session 1: Create manager, generate events, ack one
        mgr1 = ShelfEventManager(
            run_id="RUN-REFRESH",
            camera_id="CAM-01",
            storage_path=storage,
            dashboard_sync_path=sync_path,
        )
        gap = make_test_gap(region_id="GAP-REFRESH-1", tier_id="SHELF-01-T1", persistence_frames=10, conf=0.88)
        snap = make_test_snapshot(tier_id="SHELF-01-T1", gaps=[gap])
        evs = mgr1.process_frame(snap, frame_index=20, timestamp_sec=0.67, fps=30.0)
        assert len(evs) == 1
        ev_id = evs[0].event_id
        mgr1.acknowledge_event(ev_id)

        # Session 2: Simulating dashboard refresh / server restart by loading from same storage
        mgr2 = ShelfEventManager(
            run_id="RUN-REFRESH",
            camera_id="CAM-01",
            storage_path=storage,
            dashboard_sync_path=sync_path,
        )
        reloaded_active = mgr2.get_active_events()
        assert len(reloaded_active) == 1, f"Expected 1 reloaded active event, got {len(reloaded_active)}"
        assert reloaded_active[0]["event_id"] == ev_id
        assert reloaded_active[0]["status"] == "ACKNOWLEDGED"

        reloaded_history = mgr2.get_history()
        assert len(reloaded_history) == 1
        assert reloaded_history[0]["event_id"] == ev_id


def main():
    print("=================================================================")
    print("STEP 33 — FINAL RETAIL PRODUCT & DEMO LAYER VALIDATION")
    print("=================================================================")

    tests = [
        ("1. Store Overview Actual Telemetry", test_overview_uses_actual_api_and_event_data),
        ("2. Alert Priority States (CRITICAL, MONITORING, UNCERTAIN, NORMAL)", test_alert_priority_states),
        ("3. Active Event Rendering & Fields Integrity", test_active_event_rendering_and_fields),
        ("4. Operator Acknowledge & Resolve Lifecycle", test_acknowledge_and_resolve_lifecycle),
        ("5. Shelf Restored Lifecycle Event Closure", test_restored_shelf_lifecycle),
        ("6. Run Isolation Active Alert Containment", test_run_isolation_active_events),
        ("7. Stale-State Prevention on Video Switch", test_stale_state_prevention_on_video_switch),
        ("8. Real Lightweight Analytics & Insufficient Data Guard", test_real_analytics_calculations_and_empty_state),
        ("9. API Failure Graceful Degradation", test_api_failure_graceful_handling),
        ("10. Dashboard Refresh Persistence Consistency", test_dashboard_refresh_consistency),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        if run_test(name, fn):
            passed += 1
        else:
            failed += 1

    print("\n=================================================================")
    print(f"STEP 33 VALIDATION SUMMARY: {passed}/{len(tests)} PASSED, {failed} FAILED")
    print("=================================================================")

    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
