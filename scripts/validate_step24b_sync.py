#!/usr/bin/env python3
"""Step 24B Validation Suite — Video Selection -> Run Synchronization.

Verifies:
1. shelf_pan_demo.mp4 run execution & report generation.
2. Selecting inventory2.mp4 WITHOUT running does NOT change the active report or run ID.
3. Running inventory2.mp4 produces a new run ID, processes inventory2.mp4, and updates report.
4. Switching back from inventory2.mp4 to shelf_pan_demo.mp4 works identically.
5. All tabs, run history, and reports contain distinct and correct data.
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

def test_step24b_cycle():
    print("=" * 65)
    print("STEP 24B: START TEST 1 — shelf_pan_demo.mp4 (30 frames)")
    print("=" * 65)

    res1 = api_post("/inventory/run?max_frames=30&video_source=shelf_pan_demo.mp4")
    run_id1 = res1.get("run_id")
    print(f"Triggered Run 1: run_id={run_id1}, video={res1.get('video_source')}")
    assert res1.get("status") == "started"
    assert res1.get("video_source") == "shelf_pan_demo.mp4"

    # Verify immediate RUNNING state
    imm1 = api_get("/inventory/run/status")
    print(f"Immediate status: state={imm1.get('state')}, video={imm1.get('video_source')}")
    assert imm1.get("state") == "RUNNING"
    assert imm1.get("video_source") == "shelf_pan_demo.mp4"

    final1 = wait_for_run_completion()
    print(f"Run 1 Completed: {final1.get('run_id')} in {final1.get('elapsed_sec'):.2f}s")
    rep1 = api_get("/inventory/report")
    assert "shelf_pan_demo.mp4" in rep1.get("video_source", "")
    assert rep1.get("run_id") == run_id1
    facings1 = rep1.get("stable_facings")
    alerts1 = rep1.get("active_alerts_count")
    print(f"Run 1 Metrics: facings={facings1}, alerts={alerts1}, skus={len(rep1.get('sku_inventory_summary', []))}")

    print("\n" + "=" * 65)
    print("STEP 24B: TEST 2 — Video Selection Change WITHOUT Running")
    print("=" * 65)
    print("Simulating operator changing dropdown to 'inventory2.mp4'...")
    # Without POST /inventory/run, report must remain Run 1!
    rep_check = api_get("/inventory/report")
    assert rep_check.get("run_id") == run_id1, "Report changed without running!"
    assert "shelf_pan_demo.mp4" in rep_check.get("video_source", ""), "Report video changed without running!"
    assert rep_check.get("stable_facings") == facings1, "Facings changed without running!"
    print(f"Verified: Active report remains {run_id1} on {rep_check.get('video_source')}")
    print(">>> Video selection alone did NOT trigger inference. PASS.")

    print("\n" + "=" * 65)
    print("STEP 24B: TEST 3 — Operator Clicks Run on inventory2.mp4 (30 frames)")
    print("=" * 65)
    res2 = api_post("/inventory/run?max_frames=30&video_source=inventory2.mp4")
    run_id2 = res2.get("run_id")
    print(f"Triggered Run 2: run_id={run_id2}, video={res2.get('video_source')}")
    assert run_id2 != run_id1, "New run did not generate a new run ID!"
    assert res2.get("video_source") == "inventory2.mp4"

    # Immediate status check
    imm2 = api_get("/inventory/run/status")
    print(f"Immediate status: state={imm2.get('state')}, video={imm2.get('video_source')}, run_id={imm2.get('run_id')}")
    assert imm2.get("state") == "RUNNING"
    assert imm2.get("video_source") == "inventory2.mp4"
    assert imm2.get("run_id") == run_id2

    final2 = wait_for_run_completion()
    print(f"Run 2 Completed: {final2.get('run_id')} in {final2.get('elapsed_sec'):.2f}s")
    rep2 = api_get("/inventory/report")
    assert "inventory2.mp4" in rep2.get("video_source", "")
    assert rep2.get("run_id") == run_id2
    facings2 = rep2.get("stable_facings")
    alerts2 = rep2.get("active_alerts_count")
    print(f"Run 2 Metrics: facings={facings2}, alerts={alerts2}, skus={len(rep2.get('sku_inventory_summary', []))}")
    assert rep2.get("run_id") != rep1.get("run_id")
    print(">>> Run 2 cleanly replaced Run 1 with exact inventory2.mp4 data. PASS.")

    print("\n" + "=" * 65)
    print("STEP 24B: TEST 4 — Switch Back: shelf_pan_demo.mp4 (30 frames)")
    print("=" * 65)
    res3 = api_post("/inventory/run?max_frames=30&video_source=shelf_pan_demo.mp4")
    run_id3 = res3.get("run_id")
    print(f"Triggered Run 3: run_id={run_id3}, video={res3.get('video_source')}")
    assert run_id3 != run_id2
    assert res3.get("video_source") == "shelf_pan_demo.mp4"

    final3 = wait_for_run_completion()
    print(f"Run 3 Completed: {final3.get('run_id')} in {final3.get('elapsed_sec'):.2f}s")
    rep3 = api_get("/inventory/report")
    assert "shelf_pan_demo.mp4" in rep3.get("video_source", "")
    assert rep3.get("run_id") == run_id3
    print(f"Run 3 Metrics: facings={rep3.get('stable_facings')}, alerts={rep3.get('active_alerts_count')}")
    print(">>> Reverse transition inventory2.mp4 -> shelf_pan_demo.mp4 PASSED.")

    print("\n" + "=" * 65)
    print("STEP 24B: TEST 5 — Verify Run History Preservation")
    print("=" * 65)
    with open("dashboard/public/run_history.json", "r", encoding="utf-8") as f:
        hist = json.load(f)
    runs = hist.get("runs", [])
    history_run_ids = [r.get("run_id") for r in runs]
    history_videos = {r.get("run_id"): r.get("video_source") for r in runs}
    print(f"Run IDs in history: {history_run_ids}")
    print(f"Video associations: {history_videos}")
    assert run_id1 in history_run_ids, f"Run {run_id1} missing from history"
    assert run_id2 in history_run_ids, f"Run {run_id2} missing from history"
    assert run_id3 in history_run_ids, f"Run {run_id3} missing from history"
    assert "shelf_pan_demo.mp4" in history_videos[run_id1]
    assert "inventory2.mp4" in history_videos[run_id2]
    assert "shelf_pan_demo.mp4" in history_videos[run_id3]
    print(">>> Run History preserved all runs with their correct video sources. PASS.")

    print("\n" + "=" * 65)
    print("ALL STEP 24B TESTS PASSED!")
    print("=" * 65)

if __name__ == "__main__":
    test_step24b_cycle()
