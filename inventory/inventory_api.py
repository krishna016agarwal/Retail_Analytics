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

import cv2
import numpy as np


# Ensure project root is on sys.path when imported as a module
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

# ─── Default paths (relative to project root) ─────────────────────────────────
_DEFAULT_MODEL   = "inventory_data/custom_model/retail_detector_exp2.pt"
_DEFAULT_VIDEO   = "inventory_data/demo_videos/shelf_pan_demo.mp4" if (ROOT_DIR / "inventory_data" / "demo_videos" / "shelf_pan_demo.mp4").is_file() else "videos/inventory2.mp4"
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

from inventory.shelf_events import ShelfEventManager, ShelfEventType, ShelfEventStatus
_event_manager = ShelfEventManager()


def _list_demo_videos() -> List[Dict[str, Any]]:
    """Return available demo videos including shelf_pan_demo.mp4 and inventory2.mp4."""
    videos = []
    seen = set()

    candidates = [
        (ROOT_DIR / "inventory_data" / "demo_videos" / "shelf_pan_demo.mp4", "shelf_pan_demo.mp4 (Shelf Panoramic Demo)"),
        (ROOT_DIR / "videos" / "inventory2.mp4", "inventory2.mp4 (Store Aisle Shelves Demo)"),
        (ROOT_DIR / "videos" / "inventory.mp4", "inventory.mp4 (Store Overview Demo)"),
    ]

    for p, label in candidates:
        if p.is_file() and p.name not in seen:
            seen.add(p.name)
            videos.append({
                "filename": p.name,
                "relative_path": str(p.relative_to(ROOT_DIR)).replace("\\", "/"),
                "size_bytes": p.stat().st_size,
                "label": label,
            })

    demo_dir = ROOT_DIR / "inventory_data" / "demo_videos"
    if demo_dir.is_dir():
        for p in sorted(demo_dir.glob("*.mp4")):
            if p.name not in seen:
                seen.add(p.name)
                videos.append({
                    "filename": p.name,
                    "relative_path": str(p.relative_to(ROOT_DIR)).replace("\\", "/"),
                    "size_bytes": p.stat().st_size,
                    "label": f"{p.name} (Demo Video)",
                })
    return videos


def _list_models() -> List[Dict[str, Any]]:
    """Return available detection models."""
    models = []
    seen = set()
    candidates = [
        (ROOT_DIR / "detect_product_empty_space.pt", "detect_product_empty_space.pt (Product & Empty Space Model)"),
        (ROOT_DIR / "inventory_data" / "custom_model" / "retail_detector_exp2.pt", "retail_detector_exp2.pt (Retail Detector Exp 2)"),
        (ROOT_DIR / "yolo11n.pt", "yolo11n.pt (COCO Baseline)"),
    ]
    for p, label in candidates:
        if p.is_file() and p.name not in seen:
            seen.add(p.name)
            models.append({
                "filename": p.name,
                "relative_path": str(p.relative_to(ROOT_DIR)).replace("\\", "/"),
                "size_bytes": p.stat().st_size,
                "label": label,
            })
    return models


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
        ROOT_DIR / "output" / "inventory_report" / "run_history.json",
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
    shelves_data = report_dict.get("shelves", [])
    primary_shelf = shelves_data[0] if shelves_data else {}
    s_status = primary_shelf.get("status", "OCCUPIED")
    is_vacant = primary_shelf.get("vacancy_detected", False)
    v_score = primary_shelf.get("vacancy_score", 0.0)
    occ_pct = primary_shelf.get("occupancy_pct", 0.0)

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
        "camera_id": report_dict.get("camera_id", "CAM-01"),
        "shelf_id": primary_shelf.get("shelf_id", "SHELF-01"),
        "shelf_status": s_status,
        "vacancy_detected": is_vacant,
        "vacancy_score": v_score,
        "occupancy_pct": occ_pct,
        "active_visible_facings": report_dict.get("total_active_visible_facings", 0),
        "stable_facings": report_dict.get("stable_facings", 0),
        "uncertain_facings": report_dict.get("uncertain_facings", 0),
        "total_alerts": report_dict.get("total_alerts_fired", 0),
        "high_alerts": report_dict.get("alerts_by_severity", {}).get("HIGH", 0),
        "shelf_health": report_dict.get("shelf_health", "HEALTHY"),
        "health_reason": report_dict.get("health_reason", ""),
        "shelves": shelves_data,
        "alerts": report_dict.get("active_alerts", []),
        "evidence_video": f"/inventory/video/{video_name}",
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
    video_source_or_max_frames: Optional[Any] = None,
    video_source_input: Optional[str] = None,
    run_id: Optional[str] = None,
    max_frames: Optional[int] = None,
    model_name_or_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full inventory pipeline synchronously with real-time progress updates.

    Reuses all existing pipeline modules exactly as implemented in Steps 1-8.
    Does NOT modify any model weights, tracker, or crowd/queue code.
    Processes full source video to EOF by default unless max_frames is explicitly specified.
    """
    if isinstance(video_source_or_max_frames, str):
        video_source_input = video_source_or_max_frames
        actual_max_frames = max_frames
    elif isinstance(video_source_or_max_frames, int):
        actual_max_frames = max_frames if max_frames is not None else video_source_or_max_frames
    else:
        actual_max_frames = max_frames

    if actual_max_frames is not None and actual_max_frames <= 0:
        actual_max_frames = None

    max_frames = actual_max_frames
    import cv2
    from inventory.catalog import SKUCatalog
    from inventory.config import InventoryModelConfig
    from inventory.inventory_aggregator import InventoryAggregatorConfig, SKUInventoryAggregator
    from inventory.inventory_alerts import InventoryAlertConfig, InventoryAlertDetector
    from inventory.inventory_events import InventoryEventConfig, InventoryEventDetector
    from inventory.inventory_report import StoreInventoryReportBuilder
    from inventory.shelf_detector import ShelfProductDetector
    from inventory.shelf_state import ProductTrackStateTracker

    catalog_path = ROOT_DIR / _DEFAULT_CATALOG
    out_json     = ROOT_DIR / _DEFAULT_JSON

    if model_name_or_path:
        cand_m = Path(model_name_or_path)
        if (ROOT_DIR / cand_m.name).is_file():
            model_path = (ROOT_DIR / cand_m.name).resolve()
        elif (ROOT_DIR / "inventory_data" / "custom_model" / cand_m.name).is_file():
            model_path = (ROOT_DIR / "inventory_data" / "custom_model" / cand_m.name).resolve()
        elif cand_m.is_file():
            model_path = cand_m.resolve()
        elif (ROOT_DIR / model_name_or_path).is_file():
            model_path = (ROOT_DIR / model_name_or_path).resolve()
        else:
            model_path = (ROOT_DIR / _DEFAULT_MODEL).resolve()
    else:
        model_path = (ROOT_DIR / _DEFAULT_MODEL).resolve()

    if video_source_input:
        cand = Path(video_source_input)
        if cand.is_file():
            chosen_video = cand
        elif (ROOT_DIR / video_source_input).is_file():
            chosen_video = ROOT_DIR / video_source_input
        elif (ROOT_DIR / "inventory_data" / "demo_videos" / cand.name).is_file():
            chosen_video = ROOT_DIR / "inventory_data" / "demo_videos" / cand.name
        elif (ROOT_DIR / "videos" / cand.name).is_file():
            chosen_video = ROOT_DIR / "videos" / cand.name
        else:
            chosen_video = ROOT_DIR / _DEFAULT_VIDEO
    else:
        chosen_video = ROOT_DIR / _DEFAULT_VIDEO

    if not model_path.is_file():
        raise FileNotFoundError(f"Detector model not found: {model_path}")
    if not chosen_video.is_file():
        raise FileNotFoundError(f"Video file not found: {chosen_video}")

    assigned_run_id = run_id or _get_next_run_id()
    video_name = chosen_video.name

    tracker_cfg = str(ROOT_DIR / "inventory" / "bytetrack_shelf.yaml") if (ROOT_DIR / "inventory" / "bytetrack_shelf.yaml").is_file() else None

    # Calibrate confidence threshold: detect_product_empty_space uses 0.15 for high recall on empty spaces and products
    conf_threshold = 0.15 if "detect_product_empty_space" in model_path.name else 0.30

    # Initialise pipeline components
    det_cfg = InventoryModelConfig(
        model_path=str(model_path),
        model_tier="retail_specific",
        device="cpu",
        confidence_threshold=conf_threshold,
        target_classes=None,
        tracker_config_path=tracker_cfg,
    )
    detector = ShelfProductDetector(det_cfg)

    # Low-level product track tracker (validated from Steps 1-24)
    tracker = ProductTrackStateTracker(
        min_hits_for_stable=3,
        removal_grace_seconds=1.5,
        occlusion_freeze_enabled=True,
        displacement_change_ratio=0.25,
        iou_change_threshold=0.60,
    )

    # Load fallback configuration from configs/shelf_vacancy_config.json
    vacancy_cfg_file = ROOT_DIR / "configs" / "shelf_vacancy_config.json"
    shelf_roi = (0.02, 0.12, 0.98, 0.88)
    shelf_tiers = None
    v_cfg: Dict[str, Any] = {}
    if vacancy_cfg_file.is_file():
        try:
            with open(vacancy_cfg_file, "r", encoding="utf-8") as f:
                v_cfg = json.load(f)
                video_rois = v_cfg.get("video_rois", {})
                if video_name in video_rois:
                    v_entry = video_rois[video_name]
                    if isinstance(v_entry, dict):
                        shelf_roi = tuple(v_entry.get("shelf_roi", shelf_roi))
                        shelf_tiers = v_entry.get("tiers")
                    elif isinstance(v_entry, (list, tuple)):
                        shelf_roi = tuple(v_entry)
                elif v_cfg.get("shelves"):
                    s0 = v_cfg["shelves"][0]
                    if "roi" in s0:
                        shelf_roi = tuple(s0["roi"])
                    shelf_tiers = s0.get("tiers")
        except Exception:
            pass

    # Step 33.2: Automatic Shelf Geometry Discovery from visual video structure
    from inventory.shelf_geometry import AutomaticShelfGeometryDetector
    geo_detector = AutomaticShelfGeometryDetector(min_tier_height_ratio=0.08)
    fallback_cfg = {"shelf_roi": shelf_roi, "tiers": shelf_tiers}
    try:
        geo_res = geo_detector.detect_from_video(
            video_path=chosen_video,
            shelf_id="SHELF-01",
            configured_fallback=fallback_cfg,
        )
    except Exception as exc:
        print(f"[InventoryAPI] Automatic shelf discovery failed: {exc}", flush=True)
        geo_res = geo_detector._fallback("SHELF-01", fallback_cfg, str(exc))

    from inventory.shelf_vacancy import (
        ShelfVacancyEngine,
        ShelfVacancyTracker,
        VacancyTemporalState,
        annotate_vacancy_frame,
    )

    vacancy_engine = ShelfVacancyEngine(
        shelf_id="SHELF-01",
        roi=geo_res.shelf_roi_norm,
        geometry_result=geo_res,
        min_gap_multiplier=float(v_cfg.get("min_gap_multiplier", 1.75)),
        min_absolute_gap_px=int(v_cfg.get("min_absolute_gap_px", 45)),
    )
    vacancy_tracker = ShelfVacancyTracker(
        min_consecutive_frames=int(v_cfg.get("min_consecutive_frames", 10)),
        recovery_frames=int(v_cfg.get("recovery_frames", 5)),
        min_replenishment_confidence=float(v_cfg.get("min_replenishment_confidence", 0.70)),
        max_center_drift_ratio=float(v_cfg.get("max_center_drift_ratio", 0.25)),
        stability_history_window=int(v_cfg.get("stability_history_window", 5)),
        confidence_persistence_weight=float(v_cfg.get("confidence_persistence_weight", 0.40)),
        confidence_size_weight=float(v_cfg.get("confidence_size_weight", 0.30)),
        confidence_stability_weight=float(v_cfg.get("confidence_stability_weight", 0.30)),
    )

    global _event_manager
    _event_manager.reset_active_for_new_run(assigned_run_id)

    cap         = cv2.VideoCapture(str(chosen_video))
    fps         = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    cap_count   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    src_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    source_duration = round(cap_count / fps, 2) if (cap_count > 0 and fps > 0) else 0.0

    # Total frames to process: entire video if max_frames is None
    total       = cap_count if (max_frames is None or max_frames <= 0) else min(max_frames, cap_count if cap_count > 0 else max_frames)
    frame_idx   = 0
    last_ts_sec = 0.0
    prev_gray   = None
    last_v_snap = None
    last_frame  = None
    active_alerts: List[Dict[str, Any]] = []

    # Canonical Step 33.1 / 33.2: Initialize OpenCV VideoWriter with exact source parameters
    annotated_dir = ROOT_DIR / "output" / "inventory_report" / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    video_stem = chosen_video.stem
    raw_video_filename = f"{video_stem}_{assigned_run_id}_annotated.mp4"
    raw_video_path = annotated_dir / raw_video_filename

    video_writer = cv2.VideoWriter(
        str(raw_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (src_w, src_h),
    )

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

    annotated_frame = None

    while cap.isOpened() and (max_frames is None or frame_idx < max_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        ts = frame_idx / fps
        last_ts_sec = ts
        last_frame = frame
        h, w = frame.shape[:2]

        # 1. Optical flow camera motion
        small_gray = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY)
        flow_mag = 0.0
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, small_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_mag = float(np.mean(mag))
        prev_gray = small_gray

        # 2. Product and person detections
        batch = detector.detect(frame=frame, frame_index=frame_idx, track=True, persist=True, tracker=tracker_cfg or "bytetrack.yaml")
        tracker.update(batch=batch, frame_index=frame_idx, fps=fps)

        prod_boxes = [d.bbox for d in batch.product_detections]
        person_boxes = [d.bbox for d in batch.person_detections]
        empty_boxes = [(d.bbox, d.confidence) for d in getattr(batch, "empty_detections", [])]

        # 3. Shelf vacancy evaluation & temporal confirmation
        v_snap = vacancy_engine.analyze_frame(
            product_boxes=prod_boxes,
            frame_w=w,
            frame_h=h,
            person_boxes=person_boxes,
            camera_motion_mag=flow_mag,
            model_empty_boxes=empty_boxes,
            model_name=model_path.name,
        )
        last_v_snap = vacancy_tracker.update(v_snap)
        if not getattr(last_v_snap, "model_empty_boxes", None):
            last_v_snap.model_empty_boxes = empty_boxes
        last_v_snap.model_name = model_path.name

        # 4. Inventory Event Processing & Deduplication
        _event_manager.process_frame(
            snapshot=last_v_snap,
            frame_index=frame_idx,
            timestamp_sec=last_ts_sec,
            fps=fps,
        )

        active_events = _event_manager.get_active_events(run_id=assigned_run_id)
        active_alerts = [
            {
                "alert_id": ev["event_id"],
                "shelf_id": ev["shelf_id"],
                "tier_id": ev["tier_id"],
                "type": "SHELF_VACANCY",
                "severity": "HIGH",
                "title": f"{ev['tier_id']} — REPLENISHMENT RECOMMENDED",
                "message": ev["message"],
                "visual_vacancy_confidence": ev["visual_vacancy_confidence"],
                "duration_frames": ev["duration_frames"],
                "duration_sec": ev["duration_sec"],
                "timestamp": ev["timestamp_iso"],
                "status": ev["status"],
            }
            for ev in active_events
            if ev.get("action_required") or ev.get("event_type") == ShelfEventType.REPLENISHMENT_RECOMMENDED.value
        ]

        if not active_alerts and vacancy_tracker.is_confirmed:
            active_alerts.append({
                "alert_id": f"ALT-SHELF-01-{int(time.time())}",
                "shelf_id": "SHELF-01",
                "type": "SHELF_VACANCY",
                "severity": "HIGH",
                "title": "SHELF-01 — EMPTY SPACE DETECTED",
                "message": "Persistent empty/vacant shelf space detected on Shelf 1. Replenishment verification recommended.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "ACTIVE",
            })

        # 5. Canonical Step 33.1: Write real annotated frame to VideoWriter
        annotated_frame = annotate_vacancy_frame(
            frame=frame,
            snapshot=last_v_snap,
            roi_pct=shelf_roi,
            active_alert=bool(active_alerts) or vacancy_tracker.is_confirmed,
            model_name=model_path.name,
        )
        if video_writer is not None and video_writer.isOpened():
            video_writer.write(annotated_frame)

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
    if video_writer is not None:
        video_writer.release()
    total_elapsed = round(time.time() - start_time, 2)

    # Save visual evidence snapshot for dashboard
    evidence_dir = ROOT_DIR / "dashboard" / "public" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    if annotated_frame is not None:
        try:
            cv2.imwrite(str(evidence_dir / "latest_shelf_snapshot.jpg"), annotated_frame)
            cv2.imwrite(str(evidence_dir / f"{video_name}_vacancy.jpg"), annotated_frame)
        except Exception as e:
            print(f"[InventoryAPI] Failed saving visual snapshot: {e}", flush=True)

    # Produce web-optimized H.264 video via ffmpeg if available, otherwise retain mp4v fallback
    web_video_filename = f"{video_stem}_{assigned_run_id}_web.mp4"
    web_video_path = annotated_dir / web_video_filename
    h264_produced = False
    codec_used = "mp4v"

    try:
        import subprocess
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(raw_video_path),
            "-c:v", "libx264",
            "-r", str(fps),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(web_video_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode == 0 and web_video_path.is_file() and web_video_path.stat().st_size > 100:
            h264_produced = True
            codec_used = "H.264 (avc1)"
    except Exception as e:
        h264_produced = False

    playable_video_path = web_video_path if h264_produced else raw_video_path
    playable_filename = web_video_filename if h264_produced else raw_video_filename

    # Copy to dashboard public evidence directory for instant web streaming
    import shutil
    run_ev_path = evidence_dir / f"{assigned_run_id}_annotated.mp4"
    latest_ev_path = evidence_dir / "latest_annotated.mp4"
    try:
        shutil.copy2(str(playable_video_path), str(run_ev_path))
        shutil.copy2(str(playable_video_path), str(latest_ev_path))
    except Exception as e:
        print(f"[InventoryAPI] Failed copying annotated video to evidence dir: {e}", flush=True)

    annotated_frames = frame_idx
    annotated_fps = fps
    annotated_duration = round((annotated_frames / annotated_fps), 2) if annotated_fps > 0 else 0.0
    duration_ratio = round((annotated_duration / source_duration), 3) if source_duration > 0 else 1.0

    print(
        f"[InventoryAPI] Validation: Source={cap_count}f @ {fps:.1f}fps ({source_duration:.2f}s) | "
        f"Annotated={annotated_frames}f @ {annotated_fps:.1f}fps ({annotated_duration:.2f}s) | "
        f"Ratio={duration_ratio:.3f} | Geometry={geo_res.geometry_source} ({len(geo_res.tiers)} tiers, conf={geo_res.geometry_confidence:.2f})",
        flush=True,
    )

    report = StoreInventoryReportBuilder.build_shelf_vacancy_report(
        shelf_snapshot=last_v_snap or vacancy_engine.analyze_frame([], 1920, 1080),
        all_alerts=active_alerts,
        frame_index=max(0, frame_idx - 1),
        timestamp_sec=last_ts_sec,
        total_frames=frame_idx,
        video_source=video_name,
        camera_id="CAM-01",
        run_id=assigned_run_id,
    )

    report_dict = report.to_dict()
    report_dict["run_id"] = assigned_run_id
    report_dict["active_events"] = _event_manager.get_active_events(run_id=assigned_run_id)
    report_dict["events"] = _event_manager.get_history(run_id=assigned_run_id)
    report_dict["video_metadata"] = {
        "source_frames": cap_count,
        "source_fps": round(fps, 2),
        "source_width": src_w,
        "source_height": src_h,
        "source_duration": source_duration,
        "annotated_frames": annotated_frames,
        "annotated_fps": round(annotated_fps, 2),
        "annotated_duration": annotated_duration,
        "duration_ratio": duration_ratio,
    }
    report_dict["shelf_geometry"] = geo_res.to_dict()
    try:
        rel_model_str = str(model_path.resolve().relative_to(ROOT_DIR.resolve())).replace("\\", "/")
    except Exception:
        rel_model_str = model_path.name

    report_dict["model_info"] = {
        "model_name": model_path.name,
        "model_path": rel_model_str,
        "confidence_threshold": det_cfg.confidence_threshold,
    }
    report_dict["annotated_video"] = {
        "raw_path": str(raw_video_path.relative_to(ROOT_DIR)).replace("\\", "/"),
        "playable_path": str(playable_video_path.relative_to(ROOT_DIR)).replace("\\", "/"),
        "filename": playable_filename,
        "codec": codec_used,
        "h264_encoded": h264_produced,
        "stream_url": f"/inventory/video/annotated/{assigned_run_id}",
        "public_url": f"/evidence/{assigned_run_id}_annotated.mp4",
        "fps": round(fps, 1),
        "width": src_w,
        "height": src_h,
        "frames_written": frame_idx,
        "duration": annotated_duration,
        "source_duration": source_duration,
        "duration_ratio": duration_ratio,
    }


    # Persist report to output/
    try:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)
    except Exception as e:
        print(f"[InventoryAPI] Failed saving report to {out_json}: {e}", flush=True)

    # Update run history dataset
    run_entry = _update_run_history(
        report_dict=report_dict,
        run_id=assigned_run_id,
        video_name=video_name,
        processing_time_sec=total_elapsed,
    )

    # Atomically update backend in-memory cache BEFORE declaring run COMPLETED
    global _cached_report, _cache_source, _cache_timestamp
    with _cache_lock:
        _cached_report   = report_dict
        _cache_source    = "PIPELINE"
        _cache_timestamp = report_dict.get("timestamp_iso") or datetime.now(timezone.utc).isoformat()

    # Set progress state to COMPLETED (status poll will now find new report guaranteed)
    with _progress_lock:
        _run_progress["state"] = "COMPLETED"
        _run_progress["run_id"] = assigned_run_id
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


@app.get("/inventory/models", summary="Available Inventory Detection Models")
def get_inventory_models():
    """Return available detector models."""
    return {"models": _list_models()}


@app.api_route("/inventory/video/{filename}", methods=["GET", "HEAD"], summary="Stream Inventory Demo Video")
def stream_inventory_video(filename: str):
    """Serve demo video MP4 files with range-request support for HTML5 video playback."""
    safe_name = Path(filename).name
    candidates = [
        ROOT_DIR / "inventory_data" / "demo_videos" / safe_name,
        ROOT_DIR / "videos" / safe_name,
        ROOT_DIR / "output" / "inventory_report" / "annotated" / safe_name,
        ROOT_DIR / "dashboard" / "public" / "evidence" / safe_name,
    ]
    for p in candidates:
        if p.is_file():
            return FileResponse(
                path=str(p),
                media_type="video/mp4",
                filename=safe_name,
            )
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Video file '{safe_name}' not found.",
    )


@app.api_route("/inventory/video/annotated/{run_id}", methods=["GET", "HEAD"], summary="Stream Annotated Video by Run ID")
def stream_annotated_video_by_run(run_id: str):
    """Serve annotated video for specific run ID with range-request support."""
    safe_run = Path(run_id).name
    evidence_file = ROOT_DIR / "dashboard" / "public" / "evidence" / f"{safe_run}_annotated.mp4"
    if evidence_file.is_file():
        return FileResponse(
            path=str(evidence_file),
            media_type="video/mp4",
            filename=f"{safe_run}_annotated.mp4",
        )
    ann_dir = ROOT_DIR / "output" / "inventory_report" / "annotated"
    if ann_dir.is_dir():
        matches = list(ann_dir.glob(f"*_{safe_run}_*.mp4"))
        if matches:
            web_m = [m for m in matches if "_web.mp4" in m.name]
            target = web_m[0] if web_m else matches[0]
            return FileResponse(
                path=str(target),
                media_type="video/mp4",
                filename=target.name,
            )
    latest_file = ROOT_DIR / "dashboard" / "public" / "evidence" / "latest_annotated.mp4"
    if latest_file.is_file():
        return FileResponse(
            path=str(latest_file),
            media_type="video/mp4",
            filename="latest_annotated.mp4",
        )
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Annotated video for run '{safe_run}' not found.",
    )


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
    return JSONResponse(
        content=enriched,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    )


@app.post("/inventory/run", summary="Launch Inventory Analysis Run")
def launch_inventory_run(
    background_tasks: BackgroundTasks,
    max_frames: Optional[int] = Query(None, ge=1, le=5000, description="Max frames to process (None for complete video)"),
    video_source: Optional[str] = Query(None, description="Video filename or path relative to demo_videos"),
    model_name: Optional[str] = Query(None, description="Model filename or path"),
):
    """Trigger a fresh inventory pipeline run with real-time progress updates.

    Prevents duplicate runs while an existing run is in progress.
    Processes the entire source video to EOF by default unless max_frames is specified.
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
    resolved_video_name = Path(video_source).name if video_source else "shelf_pan_demo.mp4"
    actual_max_frames = max_frames if (max_frames is not None and max_frames > 0) else None

    # Validate video existence before launching background execution
    if video_source:
        cand = Path(video_source)
        possible = [
            cand,
            ROOT_DIR / video_source,
            ROOT_DIR / "inventory_data" / "demo_videos" / cand.name,
            ROOT_DIR / "videos" / cand.name,
        ]
        if not any(p.is_file() for p in possible):
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "status": "error",
                    "message": f"Requested video source '{video_source}' does not exist.",
                },
            )

    _pipeline_running = True
    with _progress_lock:
        _run_progress["state"] = "RUNNING"
        _run_progress["run_id"] = assigned_run_id
        _run_progress["video_source"] = resolved_video_name
        _run_progress["frames_processed"] = 0
        _run_progress["total_frames"] = actual_max_frames or 0
        _run_progress["progress_percent"] = 0.0
        _run_progress["elapsed_sec"] = 0.0
        _run_progress["error_message"] = None

    def _background_worker():
        global _cached_report, _cache_source, _cache_timestamp, _pipeline_running
        _pipeline_running = True
        try:
            _run_pipeline_sync(
                max_frames=actual_max_frames,
                video_source_input=video_source,
                run_id=assigned_run_id,
                model_name_or_path=model_name,
            )
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
            "video_source": resolved_video_name,
            "max_frames": actual_max_frames,
            "message": f"Run {assigned_run_id} started. Processing full source video unless debug frames set. Poll GET /inventory/run/status.",
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


# ─── Inventory Events & Replenishment Actions (Step 31) ──────────────────────

@app.get("/inventory/events/active", summary="Get Active Replenishment Alerts & Vacancy Events")
def get_active_inventory_events(
    run_id: Optional[str] = Query(None, description="Optional run_id to filter active events"),
):
    """Return all currently active replenishment recommendations and confirmed vacancies."""
    active_list = _event_manager.get_active_events(run_id=run_id)
    return {
        "status": "success",
        "active_events": active_list,
        "total_active": len(active_list),
    }


@app.get("/inventory/events/history", summary="Get Inventory Event History")
def get_inventory_event_history(
    run_id: Optional[str] = Query(None, description="Optional run_id to filter events"),
    limit: int = Query(50, ge=1, le=500, description="Max number of events to return"),
):
    """Return chronological event history (newest first)."""
    return {
        "status": "success",
        "events": _event_manager.get_history(run_id=run_id, limit=limit),
    }


@app.post("/inventory/events/{event_id}/acknowledge", summary="Acknowledge Replenishment Alert")
def acknowledge_inventory_event(event_id: str):
    """Mark an active replenishment alert as acknowledged by store personnel."""
    ev = _event_manager.acknowledge_event(event_id)
    if not ev:
        raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found.")
    return {
        "status": "success",
        "message": f"Event '{event_id}' acknowledged.",
        "event": ev.to_dict(),
    }


@app.post("/inventory/events/{event_id}/resolve", summary="Resolve / Close Vacancy Event")
def resolve_inventory_event(event_id: str):
    """Manually mark an active replenishment alert as resolved (shelf refilled)."""
    ev = _event_manager.resolve_event(event_id)
    if not ev:
        raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found.")
    return {
        "status": "success",
        "message": f"Event '{event_id}' resolved.",
        "event": ev.to_dict(),
    }


@app.get("/inventory/events/analytics", summary="Get Real Inventory Event Analytics")
def get_inventory_event_analytics(
    run_id: Optional[str] = Query(None, description="Optional run_id filter"),
):
    """Return lightweight summary analytics derived strictly from real persisted events."""
    return {
        "status": "success",
        "analytics": _event_manager.get_analytics_summary(run_id=run_id),
    }


@app.get("/inventory/status", summary="Current Store Shelf & Event Status Summary")
def get_current_inventory_status():
    """Return an instant summary of current shelf occupancy, vacancy confidence, and active alerts."""
    with _cache_lock:
        data = _cached_report
    if not data:
        data = _load_report_from_disk() or {}

    shelves = data.get("shelves", [])
    primary = shelves[0] if shelves else {}
    active_evs = _event_manager.get_active_events()

    return {
        "shelf_id": primary.get("shelf_id", "SHELF-01"),
        "status": primary.get("status", "OCCUPIED"),
        "visible_shelf_occupancy_pct": primary.get("occupancy_pct", 0.0),
        "visual_vacancy_confidence": primary.get("visual_vacancy_confidence", 0.0),
        "replenishment_recommended": primary.get("replenishment_recommended", False) or len(active_evs) > 0,
        "active_events_count": len(active_evs),
        "tiers": primary.get("tiers", []),
        "last_updated_iso": data.get("timestamp_iso"),
    }


# ─── Periodic Snapshot & Planogram Monitoring Endpoints (SIH PS 179) ─────────

@app.get("/inventory/snapshot/latest", summary="Get Latest Shelf Snapshot & Countdown")
def get_latest_shelf_snapshot():
    """Return the latest edge shelf row scan, countdown, and planogram status."""
    from inventory.shelf_snapshot_worker import ShelfSnapshotWorker
    worker = ShelfSnapshotWorker.get_instance()
    return worker.get_telemetry()


@app.post("/inventory/snapshot/stop", summary="Stop Periodic Shelf Video Monitoring")
def stop_shelf_snapshot():
    """Stop/pause periodic background video scanning."""
    from inventory.shelf_snapshot_worker import ShelfSnapshotWorker
    worker = ShelfSnapshotWorker.get_instance()
    worker.stop()
    return {
        "status": "stopped",
        "message": "Periodic shelf video monitoring stopped.",
        "is_running": False,
    }


@app.post("/inventory/snapshot/start", summary="Start Periodic Shelf Video Monitoring")
def start_shelf_snapshot():
    """Resume periodic background video scanning."""
    from inventory.shelf_snapshot_worker import ShelfSnapshotWorker
    worker = ShelfSnapshotWorker.get_instance()
    worker.start()
    return {
        "status": "started",
        "message": "Periodic shelf video monitoring started.",
        "is_running": True,
    }


@app.post("/inventory/snapshot/scan-now", summary="Trigger Immediate Shelf Snapshot")
def trigger_immediate_shelf_snapshot():
    """Force an immediate shelf frame analysis and reset countdown."""
    from inventory.shelf_snapshot_worker import ShelfSnapshotWorker
    worker = ShelfSnapshotWorker.get_instance()
    result = worker.trigger_scan_now()
    return {
        "status": "success",
        "message": "Immediate shelf scan completed successfully.",
        "result": result,
    }


@app.post("/inventory/snapshot/interval", summary="Set Periodic Scan Interval")
def set_snapshot_interval(interval_sec: int = Query(10, ge=3, le=3600)):
    """Set periodic scan interval in seconds (default: 10s for demo, 300s for prod)."""
    from inventory.shelf_snapshot_worker import ShelfSnapshotWorker
    worker = ShelfSnapshotWorker.get_instance()
    new_sec = worker.set_interval(interval_sec)
    return {
        "status": "success",
        "interval_seconds": new_sec,
        "message": f"Scan interval updated to {new_sec} seconds.",
    }
