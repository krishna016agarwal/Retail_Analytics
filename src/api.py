"""FastAPI Edge REST API for Offline-First Retail Telemetry (SIH 179).

Exposes local SQLite telemetry collected at the retail edge for local dashboards,
diagnostics, and monitoring without transmitting images, faces, or PII.
"""

from typing import Any, Dict, List, Optional
from fastapi import Depends, FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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

