"""
scripts/diagnose_step24c.py
Diagnostic script for Step 24C Phase 1 and Phase 2.
Executes Run A (shelf_pan_demo.mp4, 30 frames) and Run B (inventory2.mp4, 75 frames)
against http://127.0.0.1:8001/inventory/run, polls status, inspects JSON files,
calls GET /inventory/report, and compares them side-by-side.
"""

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

BASE_URL = "http://127.0.0.1:8001"
ROOT_DIR = Path(__file__).resolve().parent.parent

def api_post(endpoint: str, params: dict = None) -> dict:
    url = f"{BASE_URL}{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def api_get(endpoint: str) -> dict:
    url = f"{BASE_URL}{endpoint}?_t={int(time.time()*1000)}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def poll_until_completed(timeout_sec: int = 120):
    start = time.time()
    while time.time() - start < timeout_sec:
        st = api_get("/inventory/run/status")
        state = st.get("state")
        frames = st.get("frames_processed", 0)
        total = st.get("total_frames", 0)
        fps = st.get("fps", 0)
        print(f"  [Poll] state={state} frames={frames}/{total} fps={fps:.1f}", flush=True)
        if state == "COMPLETED":
            return st
        if state == "FAILED":
            raise RuntimeError(f"Run failed: {st.get('error_message')}")
        time.sleep(1.0)
    raise TimeoutError("Pipeline run timed out")

def extract_metrics(report_data: dict) -> dict:
    return {
        "run_id": report_data.get("run_id"),
        "video_source": report_data.get("video_source"),
        "total_frames": report_data.get("total_frames"),
        "timestamp_iso": report_data.get("timestamp_iso"),
        "stable_facings": report_data.get("stable_facings"),
        "active_visible_facings": report_data.get("total_active_visible_facings"),
        "total_alerts": report_data.get("total_alerts_fired"),
        "recent_events_count": len(report_data.get("recent_events", [])),
        "skus": {
            s["sku_id"]: {
                "stable": s.get("stable_facings", 0),
                "active": s.get("current_active_facings", 0),
                "status": s.get("shelf_status"),
            }
            for s in report_data.get("sku_inventory_summary", [])
        }
    }

def read_json_file(rel_path: str) -> dict:
    p = ROOT_DIR / rel_path
    if not p.is_file():
        return {"error": f"File {rel_path} not found"}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def run_test(label: str, video: str, frames: int) -> dict:
    print(f"\n==========================================")
    print(f"STARTING {label}: {video} ({frames} frames)")
    print(f"==========================================")

    # 1. Trigger run
    start_resp = api_post("/inventory/run", {"video_source": video, "max_frames": frames})
    assigned_run_id = start_resp.get("run_id")
    print(f"Triggered POST /inventory/run -> assigned_run_id: {assigned_run_id}")

    # 2. Poll until completed
    status = poll_until_completed()
    print(f"Pipeline finished. Status run_id={status.get('run_id')} elapsed={status.get('elapsed_sec')}s")

    # 3. Read output/inventory_report/inventory_report.json
    out_json = read_json_file("output/inventory_report/inventory_report.json")
    out_metrics = extract_metrics(out_json)

    # 4. Read dashboard/public/inventory_report.json
    pub_json = read_json_file("dashboard/public/inventory_report.json")
    pub_metrics = extract_metrics(pub_json)

    # 5. Call GET /inventory/report
    api_rep = api_get("/inventory/report")
    api_metrics = extract_metrics(api_rep)

    return {
        "label": label,
        "assigned_run_id": assigned_run_id,
        "out_file": out_metrics,
        "pub_file": pub_metrics,
        "api_endpoint": api_metrics,
    }

if __name__ == "__main__":
    # Test A: shelf_pan_demo.mp4 30 frames
    res_a = run_test("RUN A", "shelf_pan_demo.mp4", 30)

    time.sleep(1.0)

    # Test B: inventory2.mp4 75 frames
    res_b = run_test("RUN B", "inventory2.mp4", 75)

    print("\n\n==========================================")
    print("PHASE 2 DIAGNOSTIC COMPARISON RESULTS")
    print("==========================================")

    for res in [res_a, res_b]:
        print(f"\n--- {res['label']} (Assigned: {res['assigned_run_id']}) ---")
        api = res["api_endpoint"]
        print(f"  API run_id:           {api['run_id']}")
        print(f"  API video_source:     {api['video_source']}")
        print(f"  API total_frames:     {api['total_frames']}")
        print(f"  API stable_facings:   {api['stable_facings']}")
        print(f"  API active_facings:   {api['active_visible_facings']}")
        print(f"  API total_alerts:     {api['total_alerts']}")
        print(f"  API events_count:     {api['recent_events_count']}")
        print(f"  API timestamp:        {api['timestamp_iso']}")
        print("  SKUs:")
        for sku_id, info in api["skus"].items():
            print(f"    {sku_id:22s} stable={info['stable']:2d} active={info['active']:2d} status={info['status']}")

        # Compare API vs out_file
        match_out = (api["run_id"] == res["out_file"]["run_id"])
        match_pub = (api["run_id"] == res["pub_file"]["run_id"])
        print(f"  Sync Check: API matches output.json: {match_out} | API matches public.json: {match_pub}")

    print("\n--- DIFFERENCE PROOF (RUN A vs RUN B) ---")
    a_api = res_a["api_endpoint"]
    b_api = res_b["api_endpoint"]
    print(f"Run A run_id: {a_api['run_id']}  vs  Run B run_id: {b_api['run_id']}")
    print(f"Run A video:  {a_api['video_source'].split('/')[-1].split(chr(92))[-1]}  vs  Run B video:  {b_api['video_source'].split('/')[-1].split(chr(92))[-1]}")
    print(f"Run A frames: {a_api['total_frames']}  vs  Run B frames: {b_api['total_frames']}")
    print(f"Run A stable: {a_api['stable_facings']}  vs  Run B stable: {b_api['stable_facings']}")
    print(f"Run A alerts: {a_api['total_alerts']}  vs  Run B alerts: {b_api['total_alerts']}")
