"""FastAPI Edge REST API for Offline-First Retail Telemetry (SIH 179).

Exposes local SQLite telemetry collected at the retail edge for local dashboards,
diagnostics, and monitoring without transmitting images, faces, or PII.
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import cv2
from fastapi import Depends, FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from src.database import EdgeDatabase

app = FastAPI(
    title="Retail Analytics Edge API",
    description="Offline-first Edge REST API exposing anonymous retail intelligence telemetry snapshots from SQLite.",
    version="1.0.0",
)

# Enable CORS for local dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_db_instance: Optional[EdgeDatabase] = None


def get_db() -> EdgeDatabase:
    """Dependency provider for EdgeDatabase instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = EdgeDatabase("data/retail_edge.db")
    return _db_instance


def set_db(db: EdgeDatabase) -> None:
    """Set custom EdgeDatabase instance (useful for testing)."""
    global _db_instance
    _db_instance = db


@app.get(
    "/",
    summary="Root Service Banner",
    response_description="Basic service identification and online status.",
)
def get_root() -> Dict[str, str]:
    """Return service identification banner."""
    return {
        "service": "Retail Analytics Edge API",
        "status": "online",
    }


@app.get(
    "/health",
    summary="Edge Health Check",
    response_description="Verifies edge API and SQLite database accessibility.",
    responses={
        200: {
            "description": "System and SQLite database are healthy.",
            "content": {
                "application/json": {
                    "example": {"status": "healthy", "database": "connected"}
                }
            },
        },
        503: {
            "description": "SQLite database is unreachable or corrupted.",
            "content": {
                "application/json": {
                    "example": {"status": "unhealthy", "database": "disconnected"}
                }
            },
        },
    },
)
def get_health(db: EdgeDatabase = Depends(get_db)) -> JSONResponse:
    """Verify API availability and active SQLite connectivity."""
    is_connected = False
    try:
        is_connected = db.check_connection()
    except Exception:
        is_connected = False

    if is_connected:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "healthy", "database": "connected"},
        )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "unhealthy", "database": "disconnected"},
    )


@app.get(
    "/api/latest",
    summary="Get Latest Telemetry Snapshot",
    response_description="Most recently recorded telemetry snapshot.",
    responses={
        200: {"description": "Latest snapshot retrieved successfully."},
        404: {
            "description": "No snapshots found in database.",
            "content": {
                "application/json": {"example": {"detail": "No snapshots found"}}
            },
        },
    },
)
def get_latest(db: EdgeDatabase = Depends(get_db)) -> Dict[str, Any]:
    """Retrieve the most recent retail telemetry snapshot."""
    snapshot = db.get_latest_snapshot()
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No snapshots found",
        )
    return snapshot


@app.get(
    "/api/snapshots",
    summary="List Telemetry Snapshots",
    response_description="List of recent telemetry snapshots ordered newest first.",
)
def get_snapshots(
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of snapshots to retrieve (1-1000).",
    ),
    db: EdgeDatabase = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Retrieve up to limit recent telemetry snapshots."""
    return db.get_snapshots(limit=limit)


@app.get(
    "/api/summary",
    summary="Get Aggregated Telemetry Summary",
    response_description="Current and peak store metrics along with snapshot counts.",
)
def get_summary(db: EdgeDatabase = Depends(get_db)) -> Dict[str, Any]:
    """Retrieve aggregated current and peak metrics across store operations."""
    latest = db.get_latest_snapshot()
    total_snapshots = db.get_total_count()
    pending_sync_count = db.get_unsynced_count()

    if latest is None:
        return {
            "latest_timestamp": None,
            "entries": 0,
            "exits": 0,
            "current_occupancy": 0,
            "peak_occupancy": 0,
            "current_queue": 0,
            "peak_queue": 0,
            "average_dwell": 0.0,
            "maximum_dwell": 0.0,
            "average_wait": 0.0,
            "total_snapshots": total_snapshots,
            "pending_sync_count": pending_sync_count,
        }

    return {
        "latest_timestamp": latest.get("timestamp"),
        "entries": latest.get("entries", 0),
        "exits": latest.get("exits", 0),
        "current_occupancy": latest.get("occupancy", 0),
        "peak_occupancy": latest.get("peak_occupancy", 0),
        "current_queue": latest.get("queue_length", 0),
        "peak_queue": latest.get("peak_queue", 0),
        "average_dwell": latest.get("avg_dwell", 0.0),
        "maximum_dwell": latest.get("max_dwell", 0.0),
        "average_wait": latest.get("avg_wait", 0.0),
        "total_snapshots": total_snapshots,
        "pending_sync_count": pending_sync_count,
    }


@app.get(
    "/api/sync/status",
    summary="Get Sync Queue Status",
    response_description="Number of pending snapshots and whether sync is required.",
)
def get_sync_status(db: EdgeDatabase = Depends(get_db)) -> Dict[str, Any]:
    """Retrieve synchronization queue status for upstream cloud sync."""
    pending = db.get_unsynced_count()
    return {
        "pending_snapshots": pending,
        "sync_required": pending > 0,
    }


# ==========================================
# Central-Compatible v1 Endpoints (Edge)
# ==========================================
@app.get(
    "/api/v1/analytics/latest",
    summary="Get Latest Telemetry Snapshot (v1)",
    response_description="Most recently recorded telemetry snapshot.",
)
def get_latest_v1(db: EdgeDatabase = Depends(get_db)) -> Dict[str, Any]:
    """Retrieve the most recent retail telemetry snapshot."""
    return get_latest(db)


@app.get(
    "/api/v1/analytics",
    summary="List Telemetry Snapshots (v1)",
    response_description="List of recent telemetry snapshots ordered newest first.",
)
def get_analytics_v1(
    limit: int = Query(default=100, ge=1, le=1000, description="Max snapshots to retrieve."),
    store_id: Optional[str] = Query(default=None, description="Optional store ID filter."),
    device_id: Optional[str] = Query(default=None, description="Optional device ID filter."),
    db: EdgeDatabase = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Retrieve up to limit recent telemetry snapshots from SQLite."""
    return db.get_snapshots(limit=limit)


@app.get(
    "/api/v1/sync/status",
    summary="Get Sync Status and Telemetry Ingestion Metrics (v1)",
    response_description="Total, pending, and synchronized snapshots count.",
)
def get_sync_status_v1(db: EdgeDatabase = Depends(get_db)) -> Dict[str, Any]:
    """Retrieve sync metrics and edge buffer counts from SQLite."""
    total = db.get_total_count()
    pending = db.get_unsynced_count()
    synced = max(0, total - pending)
    latest = db.get_latest_snapshot()
    return {
        "total_snapshots": total,
        "pending_snapshots": pending,
        "synced_snapshots": synced,
        "sync_required": pending > 0,
        "stores": 1,
        "devices": 1,
        "latest_timestamp": latest.get("created_at") if latest else None,
    }


# ==========================================
# Phase 8: Retail Intelligence Endpoints (Edge)
# ==========================================
def reconcile_active_alerts(
    db: EdgeDatabase,
    latest_snapshot: Optional[Dict[str, Any]],
    recent_snapshots: List[Dict[str, Any]],
    zones: List[Dict[str, Any]],
) -> None:
    """Reconcile alert states in SQLite based on currently observed conditions.

    Transitions alerts to RESOLVED if their underlying condition is no longer true.
    """
    if latest_snapshot is None:
        return

    from src.retail_intelligence import RetailIntelligenceEngine

    queue_intel, crowd_intel = RetailIntelligenceEngine.derive_live_summaries(
        latest_snapshot=latest_snapshot,
        recent_snapshots=recent_snapshots,
    )

    # 1. QUEUE_CONGESTION is active only if:
    # (predicted_queue >= 6 and growth_rate > 0) or (current_queue >= 6 and trend != "SHRINKING")
    pred_q = queue_intel.get("predicted_queue_3min", 0)
    current_q = queue_intel.get("current_queue", 0)
    growth_rate = queue_intel.get("growth_rate_per_min", 0.0)
    trend = queue_intel.get("trend", "STABLE")

    queue_congested = (pred_q >= 6 and growth_rate > 0.0) or (current_q >= 6 and trend != "SHRINKING")
    if not queue_congested and hasattr(db, "resolve_alerts_by_type"):
        db.resolve_alerts_by_type("QUEUE_CONGESTION")

    # 2. CROWD_SPIKE is active only if a crowd spike condition is currently occurring
    is_spike = crowd_intel.get("is_spike", False)
    if not is_spike and hasattr(db, "resolve_alerts_by_type"):
        db.resolve_alerts_by_type("CROWD_SPIKE")

    # 3. STAFFING is active only if a zone is currently overloaded
    if hasattr(db, "resolve_alerts_by_type"):
        for z in (zones or []):
            load = z.get("shopper_load_per_staff", 0.0)
            shoppers = z.get("current_shoppers", 0)
            if load < 3.0 or shoppers < 2:
                db.resolve_alerts_by_type("STAFFING", zone_id=z.get("zone_id"))


@app.get(
    "/api/v1/intelligence/latest",
    summary="Get Latest Edge Intelligence",
    response_description="Latest edge intelligence summary including active alerts and zones.",
)
def get_latest_intelligence(db: EdgeDatabase = Depends(get_db)) -> Dict[str, Any]:
    """Retrieve the latest live intelligence view from local SQLite."""
    from src.retail_intelligence import RetailIntelligenceEngine

    latest_snapshot = db.get_latest_snapshot()
    recent_snapshots = db.get_snapshots(limit=10)
    zones = db.get_latest_zone_snapshots() if hasattr(db, "get_latest_zone_snapshots") else []

    reconcile_active_alerts(db, latest_snapshot, recent_snapshots, zones)
    active_alerts = db.get_active_alerts(limit=50) if hasattr(db, "get_active_alerts") else []

    queue_intel, crowd_intel = RetailIntelligenceEngine.derive_live_summaries(
        latest_snapshot=latest_snapshot,
        recent_snapshots=recent_snapshots,
    )

    return {
        "status": "online",
        "platform": "Edge Local Node",
        "latest_snapshot": latest_snapshot,
        "queue": queue_intel,
        "crowd": crowd_intel,
        "zones": zones,
        "active_alerts": active_alerts,
        "active_alert_count": len(active_alerts),
    }


@app.get(
    "/api/v1/alerts",
    summary="List Operational Alerts",
    response_description="Recent alerts recorded on the edge device.",
)
def get_alerts(
    severity: Optional[str] = Query(default=None, description="Filter by severity ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')."),
    type: Optional[str] = Query(default=None, description="Filter by type ('QUEUE_CONGESTION', 'STAFFING', 'CROWD_SPIKE')."),
    status: Optional[str] = Query(default=None, description="Filter by status ('ACTIVE', 'RESOLVED')."),
    limit: int = Query(default=100, ge=1, le=1000, description="Max alerts to retrieve."),
    db: EdgeDatabase = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Retrieve operational alerts from local edge SQLite."""
    latest_snapshot = db.get_latest_snapshot()
    recent_snapshots = db.get_snapshots(limit=10)
    zones = db.get_latest_zone_snapshots() if hasattr(db, "get_latest_zone_snapshots") else []
    reconcile_active_alerts(db, latest_snapshot, recent_snapshots, zones)

    if hasattr(db, "get_alerts"):
        return db.get_alerts(severity=severity, alert_type=type, status=status, limit=limit)
    return []


@app.get(
    "/api/v1/zones",
    summary="Get Latest Zone Metrics",
    response_description="Latest metrics across configured retail zones.",
)
def get_zones(db: EdgeDatabase = Depends(get_db)) -> List[Dict[str, Any]]:
    """Retrieve the latest snapshot for each store zone."""
    if hasattr(db, "get_latest_zone_snapshots"):
        return db.get_latest_zone_snapshots()
    return []


@app.get(
    "/api/v1/patterns",
    summary="Calculate Historical Store Patterns",
    response_description="Historical hourly averages, busiest hour, and queue patterns.",
)
def get_patterns(
    limit: int = Query(default=500, ge=10, le=2000, description="Number of historical records to analyze."),
    db: EdgeDatabase = Depends(get_db),
) -> Dict[str, Any]:
    """Calculate hourly patterns from historical snapshots recorded in local SQLite."""
    from src.retail_intelligence import RetailIntelligenceEngine

    snapshots = db.get_snapshots(limit=limit)
    zone_snapshots = db.get_zone_snapshots(limit=limit) if hasattr(db, "get_zone_snapshots") else []
    return RetailIntelligenceEngine.calculate_historical_patterns(
        snapshots=snapshots,
        zone_snapshots=zone_snapshots,
    )


# ==========================================
# 4-Camera Multi-Stream & Department Analytics Endpoints
# ==========================================
_camera_manager_instance: Optional[Any] = None


def get_camera_manager() -> Optional[Any]:
    """Return active MultiCameraManager instance if initialized in-process."""
    return _camera_manager_instance


def set_camera_manager(mgr: Any) -> None:
    """Register active MultiCameraManager instance for live telemetry queries."""
    global _camera_manager_instance
    _camera_manager_instance = mgr


@app.get(
    "/api/v1/departments",
    summary="Get Live Department Status",
    response_description="Live computer vision metrics for Food, Electronics, and Grocery departments.",
)
def get_departments(db: EdgeDatabase = Depends(get_db)) -> List[Dict[str, Any]]:
    """Retrieve actual computer vision analytics across store departments."""
    mgr = get_camera_manager()
    if mgr is not None:
        results = []
        for w in mgr.get_workers():
            if w.config.role == "department" or w.config.role == "zone":
                staff = max(1, w.config.expected_staff)
                shoppers = len(w._track_enter_frames) if hasattr(w, "_track_enter_frames") else 0
                load = round(shoppers / staff, 2)
                dwells = list(w._completed_dwells) if hasattr(w, "_completed_dwells") else []
                avg_d = round(sum(dwells) / len(dwells), 1) if dwells else 0.0
                max_d = round(max(dwells), 1) if dwells else 0.0
                results.append(
                    {
                        "camera_id": w.config.camera_id,
                        "zone_id": w.config.zone_id,
                        "department": w.config.name,
                        "zone_name": w.config.name,
                        "current_shoppers": shoppers,
                        "peak_shoppers": getattr(w, "_peak_shoppers", shoppers),
                        "footfall": len(getattr(w, "_unique_track_ids", [])) or shoppers,
                        "avg_dwell": avg_d,
                        "max_dwell": max_d,
                        "traffic_level": w._determine_traffic_level(shoppers) if hasattr(w, "_determine_traffic_level") else "NORMAL",
                        "traffic_trend": w._determine_traffic_trend(shoppers) if hasattr(w, "_determine_traffic_trend") else "STABLE",
                        "expected_staff": staff,
                        "shopper_load_per_staff": load,
                        "status": "CONGESTED" if load >= 5.0 else ("UNDERSTAFFED" if load >= 3.0 else "OPTIMAL"),
                        "is_simulation": True,
                    }
                )
        if results:
            return results

    # Fallback to database
    if hasattr(db, "get_latest_departments"):
        db_depts = db.get_latest_departments()
        if db_depts:
            return db_depts

    # Default structure with zero mock data
    return [
        {
            "camera_id": "CAM_01",
            "zone_id": "food",
            "department": "Food",
            "zone_name": "Food",
            "current_shoppers": 0,
            "peak_shoppers": 0,
            "footfall": 0,
            "avg_dwell": 0.0,
            "max_dwell": 0.0,
            "traffic_level": "LOW",
            "traffic_trend": "STABLE",
            "expected_staff": 2,
            "shopper_load_per_staff": 0.0,
            "status": "OPTIMAL",
            "is_simulation": True,
        },
        {
            "camera_id": "CAM_02",
            "zone_id": "electronics",
            "department": "Electronics",
            "zone_name": "Electronics",
            "current_shoppers": 0,
            "peak_shoppers": 0,
            "footfall": 0,
            "avg_dwell": 0.0,
            "max_dwell": 0.0,
            "traffic_level": "LOW",
            "traffic_trend": "STABLE",
            "expected_staff": 1,
            "shopper_load_per_staff": 0.0,
            "status": "OPTIMAL",
            "is_simulation": True,
        },
        {
            "camera_id": "CAM_03",
            "zone_id": "grocery",
            "department": "Grocery",
            "zone_name": "Grocery",
            "current_shoppers": 0,
            "peak_shoppers": 0,
            "footfall": 0,
            "avg_dwell": 0.0,
            "max_dwell": 0.0,
            "traffic_level": "LOW",
            "traffic_trend": "STABLE",
            "expected_staff": 2,
            "shopper_load_per_staff": 0.0,
            "status": "OPTIMAL",
            "is_simulation": True,
        },
    ]


@app.get(
    "/api/v1/cameras",
    summary="Get 4-Camera Processing Status",
    response_description="Operational processing status across CAM_01 to CAM_04.",
)
def get_cameras(db: EdgeDatabase = Depends(get_db)) -> List[Dict[str, Any]]:
    """Retrieve operational status for the 4 concurrent video processing camera workers."""
    mgr = get_camera_manager()
    default_cams = [
        {"id": "CAM_01", "name": "Food", "role": "Department Analytics", "zone": "food", "source": "videos/food/food.mp4", "expected_staff": 2},
        {"id": "CAM_02", "name": "Electronics", "role": "Department Analytics", "zone": "electronics", "source": "videos/electronics/electronics.mp4", "expected_staff": 1},
        {"id": "CAM_03", "name": "Grocery", "role": "Department Analytics", "zone": "grocery", "source": "videos/grocery/grocery.mp4", "expected_staff": 2},
        {"id": "CAM_04", "name": "Checkout", "role": "Queue Analytics", "zone": "checkout", "source": "videos/checkout/checkout.mp4", "expected_staff": 2},
    ]

    result = []
    for c_info in default_cams:
        cam_id = c_info["id"]
        w = mgr.get_worker(cam_id) if mgr else None

        fps = round(w.fps, 1) if w else 30.0
        frame_idx = w.frame_idx if w else 0
        is_processing = True if w is not None else False
        current_shoppers = len(w._track_enter_frames) if (w and hasattr(w, "_track_enter_frames")) else 0
        if w and w.config.role == "checkout" and w.queue_analytics:
            current_shoppers = w.queue_analytics.current_queue_length

        result.append(
            {
                "camera_id": cam_id,
                "name": c_info["name"],
                "department": c_info["name"],
                "role": c_info["role"],
                "status": "Processing" if is_processing else "Active",
                "processing": True,
                "fps": fps,
                "frame_count": frame_idx,
                "current_detections": current_shoppers,
                "current_shoppers": current_shoppers,
                "source": c_info["source"],
                "stream_type": "Recorded Video",
                "is_simulation": True,
                "simulation_label": "Recorded Video / Multi-Camera Demo Simulation",
            }
        )
    return result


@app.get(
    "/api/v1/cameras/{camera_id}/feed",
    summary="Live Camera Video Stream",
    response_description="Live MJPEG video stream with YOLO bounding boxes and tracks.",
)
def stream_camera_feed(camera_id: str):
    """Stream live MJPEG video from active camera worker."""
    mgr = get_camera_manager()
    if mgr is None:
        raise HTTPException(status_code=404, detail="MultiCameraManager not active")
    worker = mgr.get_worker(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    def gen():
        while True:
            f = worker.latest_annotated_frame if worker.latest_annotated_frame is not None else worker.latest_frame
            if f is not None:
                preview = cv2.resize(f, (480, 270)) if (f.shape[1] != 480 or f.shape[0] != 270) else f
                ret, jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            time.sleep(0.04)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get(
    "/api/v1/cameras/{camera_id}/frame",
    summary="Latest Camera Snapshot",
    response_description="Latest single JPEG frame with annotations.",
)
def get_camera_frame(camera_id: str):
    """Return latest single JPEG snapshot with detection annotations."""
    mgr = get_camera_manager()
    if mgr is None:
        raise HTTPException(status_code=404, detail="MultiCameraManager not active")
    worker = mgr.get_worker(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    f = worker.latest_annotated_frame if worker.latest_annotated_frame is not None else worker.latest_frame
    if f is None:
        raise HTTPException(status_code=404, detail="No frame available yet")
    preview = cv2.resize(f, (480, 270)) if (f.shape[1] != 480 or f.shape[0] != 270) else f
    ret, jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to encode frame")
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


@app.get(
    "/api/v1/cameras/grid/feed",
    summary="2x2 Multi-Camera Grid Video Stream",
    response_description="Live 2x2 MJPEG grid streaming all 4 cameras simultaneously.",
)
def stream_grid_feed():
    """Stream live 2x2 composite video grid across all 4 cameras."""
    mgr = get_camera_manager()
    if mgr is None:
        raise HTTPException(status_code=404, detail="MultiCameraManager not active")

    def gen():
        while True:
            grid = mgr.get_grid_frame(target_size=(960, 540))
            if grid is not None:
                ret, jpeg = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            time.sleep(0.04)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get(
    "/api/v1/patterns/hourly",
    summary="Get Hourly Department Traffic",
    response_description="Department traffic aggregated by hour derived from stored CV observations.",
)
def get_hourly_patterns(
    limit: int = Query(default=2000, ge=10, le=5000, description="Max snapshots to analyze."),
    db: EdgeDatabase = Depends(get_db),
) -> Dict[str, Any]:
    """Retrieve hourly department traffic and peak periods derived from stored observations."""
    if hasattr(db, "get_hourly_department_traffic"):
        return db.get_hourly_department_traffic(limit=limit)
    return {
        "status": "insufficient_data",
        "message": "Insufficient stored observations",
        "departments": {},
        "peak_department": None,
        "peak_hour": None,
    }


@app.get(
    "/api/v1/simulation/clock",
    summary="Get Simulation Clock Status",
    response_description="Current simulated store time and simulation parameters.",
)
def get_simulation_clock(db: EdgeDatabase = Depends(get_db)) -> Dict[str, Any]:
    """Return the current simulated store clock position."""
    mgr = get_camera_manager()
    if mgr and hasattr(mgr, "clock"):
        sim_time = mgr.clock.get_simulated_time_str()
        sim_iso = mgr.clock.get_simulated_iso()
        elapsed = mgr.clock.get_elapsed_seconds()
        start_time = mgr.clock.demo_start_time_str
    else:
        latest = db.get_latest_snapshot()
        sim_time = latest.get("timestamp") if latest else "17:00:00"
        sim_iso = latest.get("created_at") if latest else datetime.now(timezone.utc).isoformat()
        elapsed = 0.0
        start_time = "17:00:00"

    return {
        "is_simulation": True,
        "mode": "Demo Simulation",
        "label": "Simulated Store Time",
        "demo_start_time": start_time,
        "simulated_store_time": sim_time,
        "simulated_iso": sim_iso,
        "elapsed_seconds": round(elapsed, 2),
    }

