"""Inventory Reporting REST API (Steps 10 & 14).

Standalone FastAPI application that serves the StoreInventoryReport
produced by the existing retail inventory pipeline, and orchestrates
repeatable, end-to-end inventory runs with live progress tracking.

Completely isolated from src/, main.py, and the crowd/queue pipeline.
Does NOT retrain or modify any model weights or algorithms.

Endpoints
---------
GET  /inventory/health          -- liveness check and pipeline state
GET  /inventory/videos          -- available demo video sources
GET  /inventory/run/status      -- live pipeline execution progress & state
POST /inventory/run             -- trigger pipeline execution with progress
GET  /inventory/report          -- latest report (instant, cached or disk)
GET  /inventory/report/text     -- manager-readable text summary
"""

from __future__ import annotations

import json
import pathlib
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path when imported as a module
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

# ─── Default paths (relative to project root) ─────────────────────────────────
_DEFAULT_MODEL   = "inventory_data/custom_model/retail_detector_exp2.pt"
_DEFAULT_VIDEO   = "inventory_data/demo_videos/shelf_pan_demo.mp4"
_DEFAULT_CATALOG = "inventory_data/catalogs/demo_store_catalog.json"
_DEFAULT_JSON    = "output/inventory_report/inventory_report.json"

# ─── FastAPI application ───────────────────────────────────────────────────────
app = FastAPI(
    title="Retail Inventory API",
    description=(
        "Serves the StoreInventoryReport and orchestrates repeatable "
        "retail inventory pipeline runs (Steps 1-14). Completely isolated "
        "from the crowd/queue analytics pipeline."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory report cache & run progress state ──────────────────────────────
_cache_lock      = threading.Lock()
_cached_report: Optional[Dict[str, Any]] = None
_cache_source    = "NONE"          # "DISK" | "PIPELINE" | "NONE"
_cache_timestamp: Optional[str]  = None
_pipeline_running = False

_progress_lock   = threading.Lock()
_run_progress: Dict[str, Any] = {
    "state": "READY",              # "READY" | "RUNNING" | "COMPLETED" | "FAILED"
    "run_id": None,
    "video_source": "shelf_pan_demo.mp4",
    "frames_processed": 0,
    "total_frames": 75,
    "progress_percent": 0.0,
    "elapsed_sec": 0.0,
    "fps": 0.0,
    "error_message": None,
    "latest_run": None,
}


def _list_demo_videos() -> List[Dict[str, Any]]:
    """Return available demo videos from inventory_data/demo_videos."""
    demo_dir = ROOT_DIR / "inventory_data" / "demo_videos"
    videos = []
    if demo_dir.is_dir():
        for p in sorted(demo_dir.glob("*.mp4")):
            videos.append({
                "filename": p.name,
                "relative_path": f"inventory_data/demo_videos/{p.name}",
                "size_bytes": p.stat().st_size,
                "label": f"{p.name} (Shelf Panoramic Demo)",
            })
    if not videos:
        videos.append({
            "filename": "shelf_pan_demo.mp4",
            "relative_path": "inventory_data/demo_videos/shelf_pan_demo.mp4",
            "size_bytes": 6697151,
            "label": "shelf_pan_demo.mp4 (Shelf Panoramic Demo)",
        })
    return videos


def _load_report_from_disk() -> Optional[Dict[str, Any]]:
    """Load the latest JSON report from disk if it exists."""
    p = ROOT_DIR / _DEFAULT_JSON
    if p.is_file():
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _get_next_run_id() -> str:
    """Determine the next sequential run ID (e.g. RUN-004) from run_history.json."""
    history_file = ROOT_DIR / "dashboard" / "public" / "run_history.json"
    max_n = 3
    if history_file.is_file():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for r in data.get("runs", []):
                    rid = r.get("run_id", "")
                    if rid.startswith("RUN-"):
                        try:
                            n = int(rid.split("-")[1])
                            if n > max_n:
                                max_n = n
                        except (IndexError, ValueError):
                            pass
        except Exception:
            pass
    return f"RUN-{max_n + 1:03d}"


def _update_run_history(
    report_dict: Dict[str, Any],
    run_id: str,
    video_name: str,
    processing_time_sec: float
) -> Dict[str, Any]:
    """Append the completed live run to run_history.json, preserving demo snapshot labels."""
    history_files = [
        ROOT_DIR / "dashboard" / "public" / "run_history.json",
        ROOT_DIR / "dashboard" / "public" / "inventory_run_history.json",
    ]
    runs = []

    # Read existing runs from primary file
    for hf in history_files:
        if hf.is_file():
            try:
                with open(hf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "runs" in data:
                        runs = data["runs"]
                        break
            except Exception:
                pass

    # Mark all previous runs as not latest
    for r in runs:
        r["is_latest"] = False

    # Construct the new live run record
    new_run_entry = {
        "run_id": run_id,
        "run_label": f"CURRENT LIVE PIPELINE RUN ({run_id})",
        "is_demo_snapshot": False,
        "is_latest": True,
        "disclaimer": None,
        "timestamp": report_dict.get("timestamp_iso") or datetime.now(timezone.utc).isoformat(),
        "video_source": video_name,
        "total_frames": report_dict.get("total_frames", 75),
        "processing_time_sec": processing_time_sec,
        "active_visible_facings": report_dict.get("total_active_visible_facings", 0),
        "stable_facings": report_dict.get("stable_facings", 0),
        "uncertain_facings": report_dict.get("uncertain_facings", 0),
        "unknown_facings": report_dict.get("total_unknown_facings", 0),
        "catalog_skus_visible": report_dict.get("catalog_skus_visible", 0),
        "in_stock_skus": report_dict.get("in_stock_skus_count", 0),
        "low_stock_skus": report_dict.get("low_stock_skus_count", 0),
        "out_of_view_skus": report_dict.get("out_of_view_skus_count", 0),
        "total_alerts": report_dict.get("total_alerts_fired", 0),
        "high_alerts": report_dict.get("alerts_by_severity", {}).get("HIGH", 0),
        "medium_alerts": report_dict.get("alerts_by_severity", {}).get("MEDIUM", 0),
        "verification_required": report_dict.get("verification_required_count", 0),
        "shelf_health": report_dict.get("shelf_health", "GOOD"),
        "health_reason": report_dict.get("health_reason", ""),
        "evidence_video": "/evidence/inventory_alerts_video.mp4",
        "skus": [
            {
                "sku_id": s["sku_id"],
                "product_name": s["product_name"],
                "category": s.get("category", "General"),
                "stable_facings": s.get("stable_facings", 0),
                "shelf_status": s.get("shelf_status", "UNKNOWN"),
            }
            for s in report_dict.get("sku_inventory_summary", [])
        ],
    }

    # If run_id already exists, replace it; otherwise append
    existing_idx = next((i for i, r in enumerate(runs) if r.get("run_id") == run_id), None)
    if existing_idx is not None:
        runs[existing_idx] = new_run_entry
    else:
        runs.append(new_run_entry)

    # Save to both history locations
    for hf in history_files:
        try:
            hf.parent.mkdir(parents=True, exist_ok=True)
            with open(hf, "w", encoding="utf-8") as f:
                json.dump({"runs": runs}, f, indent=2)
        except Exception as e:
            print(f"[InventoryAPI] Failed saving run history to {hf}: {e}", flush=True)

    # Sync inventory_report.json to dashboard/public/inventory_report.json
    pub_rep = ROOT_DIR / "dashboard" / "public" / "inventory_report.json"
    try:
        pub_rep.parent.mkdir(parents=True, exist_ok=True)
        with open(pub_rep, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)
    except Exception as e:
        print(f"[InventoryAPI] Failed syncing inventory_report.json to public: {e}", flush=True)

    return new_run_entry


def _run_pipeline_sync(
    max_frames: int = 75,
    video_source_input: Optional[str] = None,
    run_id: Optional[str] = None
) -> Dict[str, Any]:
    """Run the full inventory pipeline synchronously with real-time progress updates.

    Reuses all existing pipeline modules exactly as implemented in Steps 1-8.
    Does NOT modify any model weights, tracker, or crowd/queue code.
    """
    import cv2
    from inventory.catalog import SKUCatalog
    from inventory.config import InventoryModelConfig
    from inventory.inventory_aggregator import InventoryAggregatorConfig, SKUInventoryAggregator
    from inventory.inventory_alerts import InventoryAlertConfig, InventoryAlertDetector
    from inventory.inventory_events import InventoryEventConfig, InventoryEventDetector
    from inventory.inventory_report import StoreInventoryReportBuilder
    from inventory.shelf_detector import ShelfProductDetector
    from inventory.shelf_state import ProductTrackStateTracker

    model_path   = ROOT_DIR / _DEFAULT_MODEL
    catalog_path = ROOT_DIR / _DEFAULT_CATALOG
    out_json     = ROOT_DIR / _DEFAULT_JSON

    if video_source_input:
        chosen_video = ROOT_DIR / video_source_input
        if not chosen_video.is_file():
            # Try relative to demo_videos
            chosen_video = ROOT_DIR / "inventory_data" / "demo_videos" / video_source_input
    else:
        chosen_video = ROOT_DIR / _DEFAULT_VIDEO

    if not model_path.is_file():
        raise FileNotFoundError(f"Detector model not found: {model_path}")
    if not chosen_video.is_file():
        raise FileNotFoundError(f"Video file not found: {chosen_video}")

    assigned_run_id = run_id or _get_next_run_id()
    video_name = chosen_video.name

    # Initialise pipeline components (identical config to demo_inventory_report.py)
    det_cfg = InventoryModelConfig(
        model_path=str(model_path),
        model_tier="retail_specific",
        device="cpu",
        confidence_threshold=0.30,
        target_classes=None,
    )
    detector = ShelfProductDetector(det_cfg)

    catalog    = None
    recognizer = None
    if catalog_path.is_file():
        from inventory.sku_recognizer import SpatialColorTextureRecognizer
        catalog    = SKUCatalog.load_json(str(catalog_path))
        recognizer = SpatialColorTextureRecognizer(catalog=catalog, match_threshold=0.65)
        recognizer.build_index()

    tracker = ProductTrackStateTracker(
        min_hits_for_stable=3,
        max_misses_for_removal=5,
        displacement_change_ratio=0.25,
        iou_change_threshold=0.60,
    )
    event_detector = InventoryEventDetector(
        config=InventoryEventConfig(
            min_hits_for_stable=3,
            max_misses_for_removal=5,
            min_move_displacement_ratio=0.22,
            sustained_move_frames=2,
            sku_change_min_confidence=0.70,
            sku_recheck_interval=5,
        ),
        recognizer=recognizer,
    )
    aggregator = SKUInventoryAggregator(
        catalog=catalog,
        config=InventoryAggregatorConfig(
            in_stock_threshold=3,
            low_stock_threshold=2,
            snapshot_interval=15,
        ),
    )
    alert_detector = InventoryAlertDetector(
        config=InventoryAlertConfig(
            low_stock_threshold=2,
            stockout_consecutive_frames=10,
            rapid_removal_count=3,
            rapid_removal_window_frames=20,
            movement_alert_count=3,
            movement_window_frames=20,
            uncertain_sku_ratio_threshold=0.30,
            uncertain_sku_min_count=15,
            alert_cooldown_frames=15,
        ),
        catalog=catalog,
    )

    cap         = cv2.VideoCapture(str(chosen_video))
    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap_count   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total       = min(max_frames, cap_count if cap_count > 0 else max_frames)
    frame_idx   = 0
    all_events  = []
    last_ts_sec = 0.0

    start_time = time.time()

    # Initialise progress state
    with _progress_lock:
        _run_progress.update({
            "state": "RUNNING",
            "run_id": assigned_run_id,
            "video_source": video_name,
            "frames_processed": 0,
            "total_frames": total,
            "progress_percent": 0.0,
            "elapsed_sec": 0.0,
            "fps": 0.0,
            "error_message": None,
        })

    while cap.isOpened() and frame_idx < total:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        ts = frame_idx / fps
        last_ts_sec = ts

        batch     = detector.detect(frame=frame, frame_index=frame_idx, track=True, persist=True, tracker="bytetrack.yaml")
        snapshot  = tracker.update(batch=batch, frame_index=frame_idx)
        evts      = event_detector.process_frame(batch=batch, snapshot=snapshot, frame_index=frame_idx, timestamp_sec=ts, frame_bgr=frame)
        all_events.extend(evts)
        sku_stats = aggregator.update(snapshot=snapshot, events=evts, frame_index=frame_idx, timestamp_sec=ts)
        alert_detector.process_frame(sku_stats=sku_stats, frame_events=evts, frame_index=frame_idx, timestamp_sec=ts)
        frame_idx += 1

        # Emit real-time progress
        elapsed = time.time() - start_time
        curr_rate = round(frame_idx / max(elapsed, 0.001), 1)
        with _progress_lock:
            _run_progress["frames_processed"] = frame_idx
            _run_progress["total_frames"] = total
            _run_progress["progress_percent"] = round((frame_idx / max(total, 1)) * 100, 1)
            _run_progress["elapsed_sec"] = round(elapsed, 2)
            _run_progress["fps"] = curr_rate

    cap.release()
    total_elapsed = round(time.time() - start_time, 2)

    report = StoreInventoryReportBuilder.build_report(
        sku_stats=aggregator.current_stats,
        all_alerts=alert_detector.all_alerts,
        recent_events=all_events,
        frame_index=max(0, frame_idx - 1),
        timestamp_sec=last_ts_sec,
        total_frames=frame_idx,
        video_source=str(chosen_video),
        catalog_skus_registered=len(catalog) if catalog else 0,
    )

    report_dict = report.to_dict()

    # Persist report to output/
    report.save_json(out_json)

    # Update run history dataset
    run_entry = _update_run_history(
        report_dict=report_dict,
        run_id=assigned_run_id,
        video_name=video_name,
        processing_time_sec=total_elapsed,
    )

    # Set progress state to COMPLETED
    with _progress_lock:
        _run_progress["state"] = "COMPLETED"
        _run_progress["frames_processed"] = frame_idx
        _run_progress["progress_percent"] = 100.0
        _run_progress["elapsed_sec"] = total_elapsed
        _run_progress["latest_run"] = run_entry

    return report_dict


# ─── Startup: pre-warm cache from disk ────────────────────────────────────────
@app.on_event("startup")
def _startup_preload():
    global _cached_report, _cache_source, _cache_timestamp
    data = _load_report_from_disk()
    if data:
        with _cache_lock:
            _cached_report   = data
            _cache_source    = "DISK"
            _cache_timestamp = data.get("timestamp_iso") or datetime.now(timezone.utc).isoformat()


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/inventory/health", summary="Inventory API Health Check")
def inventory_health():
    """Verify the inventory API is alive and report cache status."""
    with _cache_lock:
        has_cache = _cached_report is not None
        source    = _cache_source
        ts        = _cache_timestamp
    with _progress_lock:
        run_st = _run_progress.get("state", "READY")
    return {
        "status":           "healthy",
        "report_cached":    has_cache,
        "cache_source":     source,
        "cache_timestamp":  ts,
        "pipeline_running": _pipeline_running or (run_st == "RUNNING"),
        "run_state":        run_st,
    }


@app.get("/inventory/videos", summary="Available Inventory Demo Videos")
def get_inventory_videos():
    """Return available demo video sources for running inventory analysis."""
    return {"videos": _list_demo_videos()}


@app.get("/inventory/run/status", summary="Current Pipeline Run Status & Progress")
def get_inventory_run_status():
    """Return live execution state, frame progress, elapsed time, and rate."""
    with _progress_lock:
        return dict(_run_progress)


@app.get("/inventory/report", summary="Get Latest Inventory Report")
def get_inventory_report():
    """Return the latest cached StoreInventoryReport JSON."""
    global _cached_report, _cache_source, _cache_timestamp

    with _cache_lock:
        data   = _cached_report
        source = _cache_source
        ts     = _cache_timestamp

    if data is None:
        data = _load_report_from_disk()
        if data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No inventory report available. Run POST /inventory/run first.",
            )
        with _cache_lock:
            _cached_report   = data
            _cache_source    = "DISK"
            _cache_timestamp = data.get("timestamp_iso")
        source = "DISK"
        ts     = data.get("timestamp_iso")

    enriched = dict(data)
    enriched["_api_source"]       = source
    enriched["_api_served_at"]    = datetime.now(timezone.utc).isoformat()
    enriched["_pipeline_running"] = _pipeline_running
    return JSONResponse(content=enriched)


@app.post("/inventory/run", summary="Launch Inventory Analysis Run")
def launch_inventory_run(
    background_tasks: BackgroundTasks,
    max_frames: int = Query(75, ge=5, le=500, description="Max frames to process"),
    video_source: Optional[str] = Query(None, description="Video filename or path relative to demo_videos"),
):
    """Trigger a fresh inventory pipeline run with real-time progress updates.

    Prevents duplicate runs while an existing run is in progress.
    """
    global _pipeline_running

    with _progress_lock:
        if _run_progress.get("state") == "RUNNING" or _pipeline_running:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "status": "already_running",
                    "message": "A pipeline run is currently active. Poll GET /inventory/run/status.",
                    "progress": dict(_run_progress),
                },
            )

    assigned_run_id = _get_next_run_id()

    def _background_worker():
        global _cached_report, _cache_source, _cache_timestamp, _pipeline_running
        _pipeline_running = True
        try:
            result = _run_pipeline_sync(
                max_frames=max_frames,
                video_source_input=video_source,
                run_id=assigned_run_id,
            )
            with _cache_lock:
                _cached_report   = result
                _cache_source    = "PIPELINE"
                _cache_timestamp = result.get("timestamp_iso") or datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            print(f"[InventoryAPI] Pipeline execution error: {exc}", flush=True)
            with _progress_lock:
                _run_progress["state"] = "FAILED"
                _run_progress["error_message"] = str(exc)
        finally:
            _pipeline_running = False

    background_tasks.add_task(_background_worker)
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "status": "started",
            "run_id": assigned_run_id,
            "video_source": video_source or "shelf_pan_demo.mp4",
            "max_frames": max_frames,
            "message": f"Run {assigned_run_id} started. Poll GET /inventory/run/status for progress.",
        },
    )


@app.get("/inventory/report/text", response_class=PlainTextResponse, summary="Manager Text Report")
def get_inventory_text():
    """Return the manager-readable text summary of the latest inventory report."""
    p = ROOT_DIR / "output" / "inventory_report" / "inventory_report.txt"
    if p.is_file():
        return p.read_text(encoding="utf-8")
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Text report not found. Run POST /inventory/run first.",
    )
