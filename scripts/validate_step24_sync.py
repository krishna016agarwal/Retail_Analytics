#!/usr/bin/env python3
"""Step 24 Validation Suite — Inventory Video, Pipeline, API, and Dashboard Synchronization.

Tests:
  Test A: shelf_pan_demo.mp4 -> 30 frames
  Test B: shelf_pan_demo.mp4 -> 75 frames
  Test C: inventory2.mp4 -> separate run
  Test D: Stale data protection / immediate RUNNING status
  Test E: Consecutive runs with alternating videos
  Test F: Top-shelf count stability (no background ShelfSnapshotWorker interference)
"""

import json
import sys
import time
import urllib.parse
import urllib.request

BASE_URL = "http://127.0.0.1:8001"

def api_get(endpoint):
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def api_post(endpoint):
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def wait_for_run_completion(max_wait_sec=120):
    start = time.time()
    while time.time() - start < max_wait_sec:
        st = api_get("/inventory/run/status")
        state = st.get("state")
        if state == "COMPLETED":
            return st
        elif state == "FAILED":
            raise RuntimeError(f"Run failed: {st.get('error_message')}")
        time.sleep(1.0)
    raise TimeoutError("Pipeline run timed out")

def run_test_a():
    print("\n" + "="*60)
    print("TEST A: shelf_pan_demo.mp4 -> 30 frames")
    print("="*60)
    res = api_post("/inventory/run?max_frames=30&video_source=shelf_pan_demo.mp4")
    print(f"Trigger response: {res}")
    run_id = res.get("run_id")
    assert res.get("status") == "started", "Failed to start run"
    assert res.get("video_source") == "shelf_pan_demo.mp4"
    assert res.get("max_frames") == 30

    final_st = wait_for_run_completion()
    print(f"Completed run: {final_st.get('run_id')} in {final_st.get('elapsed_sec'):.2f}s ({final_st.get('fps'):.2f} FPS)")
    assert final_st.get("frames_processed") == 30, f"Expected 30 frames, got {final_st.get('frames_processed')}"
    assert final_st.get("video_source") == "shelf_pan_demo.mp4"

    rep = api_get("/inventory/report")
    assert "shelf_pan_demo.mp4" in rep.get("video_source", "")
    assert rep.get("total_frames") == 30
    assert rep.get("catalog_skus_registered") == 6
    skus = {s["sku_id"]: s for s in rep.get("sku_inventory_summary", [])}
    print(f"SKUs reported: {list(skus.keys())}")
    print(f"Stable facings total: {rep.get('stable_facings')}, Active visible: {rep.get('total_active_visible_facings')}")
    print(">>> TEST A PASSED")
    return rep

def run_test_b():
    print("\n" + "="*60)
    print("TEST B: shelf_pan_demo.mp4 -> 75 frames")
    print("="*60)
    res = api_post("/inventory/run?max_frames=75&video_source=shelf_pan_demo.mp4")
    print(f"Trigger response: {res}")
    assert res.get("status") == "started"
    assert res.get("video_source") == "shelf_pan_demo.mp4"
    assert res.get("max_frames") == 75

    final_st = wait_for_run_completion()
    print(f"Completed run: {final_st.get('run_id')} in {final_st.get('elapsed_sec'):.2f}s ({final_st.get('fps'):.2f} FPS)")
    assert final_st.get("frames_processed") == 75
    assert final_st.get("video_source") == "shelf_pan_demo.mp4"

    rep = api_get("/inventory/report")
    assert "shelf_pan_demo.mp4" in rep.get("video_source", "")
    assert rep.get("total_frames") == 75
    print(f"Stable facings total: {rep.get('stable_facings')}, Active visible: {rep.get('total_active_visible_facings')}")
    print(">>> TEST B PASSED")
    return rep

def run_test_c():
    print("\n" + "="*60)
    print("TEST C: inventory2.mp4 -> separate run (50 frames)")
    print("="*60)
    res = api_post("/inventory/run?max_frames=50&video_source=inventory2.mp4")
    print(f"Trigger response: {res}")
    assert res.get("status") == "started"
    assert res.get("video_source") == "inventory2.mp4"
    assert res.get("max_frames") == 50

    final_st = wait_for_run_completion()
    print(f"Completed run: {final_st.get('run_id')} in {final_st.get('elapsed_sec'):.2f}s ({final_st.get('fps'):.2f} FPS)")
    assert final_st.get("frames_processed") == 50
    assert final_st.get("video_source") == "inventory2.mp4"

    rep = api_get("/inventory/report")
    assert "inventory2.mp4" in rep.get("video_source", "")
    assert rep.get("total_frames") == 50
    print(f"Report correctly switched to inventory2.mp4: {rep.get('video_source')}")
    print(f"Stable facings: {rep.get('stable_facings')}, Alerts: {rep.get('active_alerts_count')}")
    print(">>> TEST C PASSED")
    return rep

def run_test_d_and_e():
    print("\n" + "="*60)
    print("TEST D & E: Stale Data Protection & Consecutive Runs")
    print("="*60)
    # Consecutive run switching back to shelf_pan_demo.mp4
    res = api_post("/inventory/run?max_frames=30&video_source=shelf_pan_demo.mp4")
    # Immediate status check
    imm_st = api_get("/inventory/run/status")
    print(f"Immediate status check: state={imm_st.get('state')}, video={imm_st.get('video_source')}, frames={imm_st.get('frames_processed')}")
    assert imm_st.get("state") == "RUNNING"
    assert imm_st.get("video_source") == "shelf_pan_demo.mp4"

    final_st = wait_for_run_completion()
    rep = api_get("/inventory/report")
    assert "shelf_pan_demo.mp4" in rep.get("video_source", "")
    assert rep.get("total_frames") == 30
    print(f"Consecutive run completed cleanly: {rep.get('run_id')} on {rep.get('video_source')}")
    print(">>> TEST D & E PASSED")

def run_test_f():
    print("\n" + "="*60)
    print("TEST F: Top-Shelf Count Stability (No Background Jumping)")
    print("="*60)
    # Read report 3 times across 8 seconds
    counts = []
    for i in range(3):
        rep = api_get("/inventory/report")
        stable = rep.get("stable_facings")
        active = rep.get("total_active_visible_facings")
        counts.append((stable, active))
        print(f"Sample {i+1}: stable_facings={stable}, active_facings={active}")
        time.sleep(2.5)

    assert len(set(counts)) == 1, f"Counts changed without running a pipeline! Got: {counts}"
    print(f"Verified 100% count stability across samples: {counts[0]}")
    print(">>> TEST F PASSED")

if __name__ == "__main__":
    t_start = time.time()
    rep_a = run_test_a()
    rep_b = run_test_b()
    rep_c = run_test_c()
    run_test_d_and_e()
    run_test_f()
    print("\n" + "="*60)
    print(f"ALL STEP 24 VALIDATION TESTS PASSED! Total duration: {time.time() - t_start:.1f}s")
    print("="*60)
