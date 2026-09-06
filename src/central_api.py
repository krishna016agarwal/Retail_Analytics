"""FastAPI Central REST API for Retail Analytics (SIH 179).

Central cloud backend hosted on Render, backed by PostgreSQL.
Receives batch edge telemetry synchronizations, provides central querying,
and enables multi-store retail intelligence aggregation.
Stores zero facial, biometric, or personal identity data.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, status

# Ensure .env variables (CORS_ORIGINS, etc.) are loaded
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.central_db import CentralDatabase

app = FastAPI(
    title="Retail Analytics Central Cloud API",
    description="Central cloud REST API hosted on Render for aggregating edge retail telemetry.",
    version="1.0.0",
)

# Configure CORS via CORS_ORIGINS environment variable
cors_env = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
allowed_origins = [orig.strip() for orig in cors_env.split(",") if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_central_db_instance: Optional[CentralDatabase] = None


def get_central_db() -> CentralDatabase:
    """Dependency provider for CentralDatabase instance."""
    global _central_db_instance
    if _central_db_instance is None:
        _central_db_instance = CentralDatabase()
    return _central_db_instance


def set_central_db(db: CentralDatabase) -> None:
    """Set custom CentralDatabase instance (useful for unit tests and mocking)."""
    global _central_db_instance
    _central_db_instance = db


# Pydantic Schemas for Batch Ingestion
class SnapshotItem(BaseModel):
    """Schema for an individual telemetry snapshot in a sync batch."""

    local_id: int = Field(..., description="SQLite primary key row ID on the edge device.")
    store_id: str = Field(default="store_001", description="Store identifier.")
    device_id: Optional[str] = Field(default=None, description="Device identifier (defaults to batch device_id).")
    timestamp: Union[str, float] = Field(..., description="Timeline position or timestamp string.")
    entries: int = Field(default=0, description="Cumulative store entries.")
    exits: int = Field(default=0, description="Cumulative store exits.")
    occupancy: int = Field(default=0, description="Current occupancy count.")
    peak_occupancy: int = Field(default=0, description="Peak occupancy count.")
    queue_length: int = Field(default=0, description="Active queue count.")
    peak_queue: int = Field(default=0, description="Peak queue count.")
    avg_dwell: float = Field(default=0.0, description="Average dwell time in seconds.")
    max_dwell: float = Field(default=0.0, description="Maximum dwell time in seconds.")
    avg_wait: float = Field(default=0.0, description="Average queue wait time in seconds.")
    created_at: Optional[str] = Field(default=None, description="ISO-8601 UTC creation timestamp.")


class ZoneSnapshotItem(BaseModel):
    """Schema for an individual zone snapshot."""

    local_id: Optional[int] = Field(default=None, description="SQLite primary key on edge.")
    snapshot_id: Optional[str] = Field(default=None, description="Unique snapshot ID.")
    store_id: str = Field(default="store_001", description="Store identifier.")
    device_id: Optional[str] = Field(default=None, description="Edge device identifier.")
    camera_id: str = Field(default="CAM_01", description="Camera identifier.")
    zone_id: str = Field(..., description="Zone identifier.")
    zone_name: str = Field(..., description="Human-readable zone name.")
    timestamp: str = Field(..., description="Timeline position or timestamp string.")
    current_shoppers: int = Field(default=0, description="Shoppers currently in zone.")
    peak_shoppers: int = Field(default=0, description="Peak shoppers observed.")
    avg_dwell: float = Field(default=0.0, description="Average dwell time in seconds.")
    traffic_level: str = Field(default="LOW", description="Traffic classification.")
    expected_staff: int = Field(default=1, description="Configured staff capacity.")
    created_at: Optional[str] = Field(default=None, description="ISO-8601 UTC timestamp.")


class AlertItem(BaseModel):
    """Schema for an operational retail alert."""

    local_id: Optional[int] = Field(default=None, description="SQLite primary key on edge.")
    alert_id: str = Field(..., description="Unique alert identifier.")
    store_id: str = Field(default="store_001", description="Store identifier.")
    device_id: Optional[str] = Field(default=None, description="Edge device identifier.")
    camera_id: str = Field(default="CAM_01", description="Camera identifier.")
    zone_id: str = Field(default="store", description="Zone or section identifier.")
    type: str = Field(..., description="Alert classification type.")
    severity: str = Field(..., description="Severity level.")
    title: str = Field(..., description="Alert title.")
    message: str = Field(..., description="Alert message.")
    current_value: float = Field(default=0.0, description="Current observed value.")
    predicted_value: Optional[float] = Field(default=None, description="Predicted future value.")
    threshold: float = Field(default=0.0, description="Triggering threshold.")
    recommendation: str = Field(..., description="Actionable recommendation.")
    status: str = Field(default="ACTIVE", description="Alert status ('ACTIVE', 'RESOLVED').")
    created_at: Optional[str] = Field(default=None, description="ISO-8601 UTC timestamp.")


class BatchSyncRequest(BaseModel):
    """Schema for multi-snapshot batch ingestion payload."""

    device_id: str = Field(..., description="Edge processing node identifier sending the batch.")
    snapshots: List[SnapshotItem] = Field(default_factory=list, description="List of pending telemetry snapshots.")
    zone_snapshots: List[ZoneSnapshotItem] = Field(default_factory=list, description="List of pending zone snapshots.")
    alerts: List[AlertItem] = Field(default_factory=list, description="List of pending operational alerts.")



# Endpoints
@app.get(
    "/",
    summary="Root Cloud Service Banner",
    response_description="Central service identification and status.",
)
def get_root() -> Dict[str, str]:
    """Return central cloud service banner."""
    return {
        "service": "Retail Analytics Central Cloud API",
        "status": "online",
        "platform": "Render Cloud + PostgreSQL",
    }


@app.get(
    "/health",
    summary="Central Health Check",
    response_description="Verifies central cloud API and PostgreSQL connectivity.",
    responses={
        200: {
            "description": "System and PostgreSQL database are healthy.",
            "content": {
                "application/json": {
                    "example": {"status": "healthy", "database": "connected"}
                }
            },
        },
        503: {
            "description": "PostgreSQL database is disconnected or unreachable.",
            "content": {
                "application/json": {
                    "example": {"status": "unhealthy", "database": "disconnected"}
                }
            },
        },
    },
)
def get_health(db: CentralDatabase = Depends(get_central_db)) -> JSONResponse:
    """Verify central cloud API availability and PostgreSQL connectivity."""
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


@app.post(
    "/api/v1/analytics/batch",
    summary="Ingest Edge Telemetry Batch",
    response_description="Confirmation of received, newly inserted, and already synced records.",
    status_code=status.HTTP_200_OK,
)
def post_analytics_batch(
    payload: BatchSyncRequest,
    db: CentralDatabase = Depends(get_central_db),
) -> Dict[str, Any]:
    """Receive a batch of telemetry snapshots, zones, and alerts from an edge node.

    Idempotent: Uses ON CONFLICT DO NOTHING so retried uploads safely succeed.
    """
    resp: Dict[str, Any] = {
        "success": True,
        "received": 0,
        "inserted": 0,
        "already_synced": 0,
        "failed": 0,
        "synced_ids": [],
        "synced_zone_ids": [],
        "synced_alert_ids": [],
    }

    try:
        if payload.snapshots:
            snapshots_data = [item.model_dump() for item in payload.snapshots]
            result = db.insert_batch(device_id=payload.device_id, snapshots=snapshots_data)
            resp["received"] += result["received"]
            resp["inserted"] += result["inserted"]
            resp["already_synced"] += result["already_synced"]
            resp["failed"] += result["failed"]
            resp["synced_ids"] = result["synced_ids"]

        if payload.zone_snapshots and hasattr(db, "insert_zones_batch"):
            zones_data = [item.model_dump() for item in payload.zone_snapshots]
            z_res = db.insert_zones_batch(device_id=payload.device_id, zone_snapshots=zones_data)
            resp["received"] += z_res["received"]
            resp["inserted"] += z_res["inserted"]
            resp["already_synced"] += z_res["already_synced"]
            resp["failed"] += z_res["failed"]
            resp["synced_zone_ids"] = z_res.get("synced_ids", [])

        if payload.alerts and hasattr(db, "insert_alerts_batch"):
            alerts_data = [item.model_dump() for item in payload.alerts]
            a_res = db.insert_alerts_batch(device_id=payload.device_id, alerts=alerts_data)
            resp["received"] += a_res["received"]
            resp["inserted"] += a_res["inserted"]
            resp["already_synced"] += a_res["already_synced"]
            resp["failed"] += a_res["failed"]
            resp["synced_alert_ids"] = a_res.get("synced_ids", [])

        return resp
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database batch insertion failed: {str(e)}",
        )


@app.get(
    "/api/v1/analytics/latest",
    summary="Get Latest Central Snapshot",
    response_description="Most recently ingested retail telemetry snapshot.",
)
def get_latest_analytics(
    db: CentralDatabase = Depends(get_central_db),
) -> Dict[str, Any]:
    """Retrieve the most recently recorded telemetry snapshot from PostgreSQL."""
    snapshot = db.get_latest_snapshot()
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No snapshots found",
        )
    return snapshot


@app.get(
    "/api/v1/analytics",
    summary="Query Historical Analytics",
    response_description="List of historical telemetry snapshots.",
)
def get_analytics(
    store_id: Optional[str] = Query(default=None, description="Filter by store ID."),
    device_id: Optional[str] = Query(default=None, description="Filter by edge device ID."),
    limit: int = Query(default=100, ge=1, le=1000, description="Max records to return."),
    db: CentralDatabase = Depends(get_central_db),
) -> List[Dict[str, Any]]:
    """Retrieve historical telemetry snapshots from PostgreSQL with optional filtering."""
    return db.get_snapshots(store_id=store_id, device_id=device_id, limit=limit)


@app.get(
    "/api/v1/sync/status",
    summary="Get Central Sync Status",
    response_description="Central aggregation metrics and sync telemetry status.",
)
def get_sync_status(
    db: CentralDatabase = Depends(get_central_db),
) -> Dict[str, Any]:
    """Retrieve overall sync status metrics from central PostgreSQL."""
    return db.get_sync_status()


# ==========================================
# Phase 8: Retail Intelligence Endpoints
# ==========================================
def reconcile_active_alerts(
    db: CentralDatabase,
    latest_snapshot: Optional[Dict[str, Any]],
    recent_snapshots: List[Dict[str, Any]],
    zones: List[Dict[str, Any]],
) -> None:
    """Reconcile alert states in central PostgreSQL based on currently observed conditions."""
    if latest_snapshot is None:
        return

    from src.retail_intelligence import RetailIntelligenceEngine

    queue_intel, crowd_intel = RetailIntelligenceEngine.derive_live_summaries(
        latest_snapshot=latest_snapshot,
        recent_snapshots=recent_snapshots,
    )

    pred_q = queue_intel.get("predicted_queue_3min", 0)
    current_q = queue_intel.get("current_queue", 0)
    growth_rate = queue_intel.get("growth_rate_per_min", 0.0)
    trend = queue_intel.get("trend", "STABLE")

    queue_congested = (pred_q >= 6 and growth_rate > 0.0) or (current_q >= 6 and trend != "SHRINKING")
    if not queue_congested and hasattr(db, "resolve_alerts_by_type"):
        db.resolve_alerts_by_type("QUEUE_CONGESTION")

    is_spike = crowd_intel.get("is_spike", False)
    if not is_spike and hasattr(db, "resolve_alerts_by_type"):
        db.resolve_alerts_by_type("CROWD_SPIKE")

    if hasattr(db, "resolve_alerts_by_type"):
        for z in (zones or []):
            load = z.get("shopper_load_per_staff", 0.0)
            shoppers = z.get("current_shoppers", 0)
            if load < 3.0 or shoppers < 2:
                db.resolve_alerts_by_type("STAFFING", zone_id=z.get("zone_id"))


@app.get(
    "/api/v1/intelligence/latest",
    summary="Get Latest Central Intelligence",
    response_description="Real-time multi-camera intelligence, active alerts, and zone metrics.",
)
def get_latest_intelligence(
    db: CentralDatabase = Depends(get_central_db),
) -> Dict[str, Any]:
    """Return the most recent live intelligence view across all store cameras and zones."""
    from src.retail_intelligence import RetailIntelligenceEngine

    latest_snapshot = db.get_latest_snapshot()
    recent_snapshots = db.get_snapshots(limit=10)
    zones = db.get_latest_zones() if hasattr(db, "get_latest_zones") else []

    reconcile_active_alerts(db, latest_snapshot, recent_snapshots, zones)
    alerts = db.get_alerts(status="ACTIVE", limit=50) if hasattr(db, "get_alerts") else []

    queue_intel, crowd_intel = RetailIntelligenceEngine.derive_live_summaries(
        latest_snapshot=latest_snapshot,
        recent_snapshots=recent_snapshots,
    )

    return {
        "status": "online",
        "platform": "Central Cloud API",
        "latest_snapshot": latest_snapshot,
        "queue": queue_intel,
        "crowd": crowd_intel,
        "zones": zones,
        "active_alerts": alerts,
        "active_alert_count": len(alerts),
    }


@app.get(
    "/api/v1/alerts",
    summary="Query Operational Alerts",
    response_description="List of retail operational alerts filtered by severity or type.",
)
def get_alerts(
    severity: Optional[str] = Query(default=None, description="Filter by severity ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')."),
    type: Optional[str] = Query(default=None, description="Filter by alert type ('QUEUE_CONGESTION', 'STAFFING', 'CROWD_SPIKE')."),
    status: Optional[str] = Query(default=None, description="Filter by status ('ACTIVE', 'RESOLVED')."),
    limit: int = Query(default=100, ge=1, le=1000, description="Max alerts to return."),
    db: CentralDatabase = Depends(get_central_db),
) -> List[Dict[str, Any]]:
    """Retrieve operational alerts from central storage."""
    latest_snapshot = db.get_latest_snapshot()
    recent_snapshots = db.get_snapshots(limit=10)
    zones = db.get_latest_zones() if hasattr(db, "get_latest_zones") else []
    reconcile_active_alerts(db, latest_snapshot, recent_snapshots, zones)

    if hasattr(db, "get_alerts"):
        return db.get_alerts(severity=severity, alert_type=type, status=status, limit=limit)
    return []


@app.get(
    "/api/v1/zones",
    summary="Get Store Zones Analytics",
    response_description="Latest status and metrics for all monitored retail zones.",
)
def get_zones(
    db: CentralDatabase = Depends(get_central_db),
) -> List[Dict[str, Any]]:
    """Retrieve the latest metrics across store zones."""
    if hasattr(db, "get_latest_zones"):
        return db.get_latest_zones()
    return []


@app.get(
    "/api/v1/patterns",
    summary="Analyze Historical Traffic Patterns",
    response_description="Hourly averages, busiest hour, busiest zone, and peak queue periods.",
)
def get_patterns(
    store_id: Optional[str] = Query(default=None, description="Optional store ID filter."),
    limit: int = Query(default=500, ge=10, le=2000, description="Historical records sample limit."),
    db: CentralDatabase = Depends(get_central_db),
) -> Dict[str, Any]:
    """Calculate hourly footfall, occupancy, and queue patterns from real historical data."""
    from src.retail_intelligence import RetailIntelligenceEngine

    snapshots = db.get_snapshots(store_id=store_id, limit=limit)
    zone_snapshots = db.get_zone_snapshots(limit=limit) if hasattr(db, "get_zone_snapshots") else []
    return RetailIntelligenceEngine.calculate_historical_patterns(
        snapshots=snapshots,
        zone_snapshots=zone_snapshots,
    )


# ==========================================
# 4-Camera Multi-Stream & Department Analytics Endpoints (Central)
# ==========================================
@app.get(
    "/api/v1/departments",
    summary="Get Synchronized Department Status",
    response_description="Live computer vision metrics for Food, Electronics, and Grocery departments.",
)
def get_central_departments(
    db: CentralDatabase = Depends(get_central_db),
) -> List[Dict[str, Any]]:
    """Retrieve actual computer vision analytics across store departments from central storage."""
    if hasattr(db, "get_latest_departments"):
        db_depts = db.get_latest_departments()
        if db_depts:
            return db_depts

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
    summary="Get 4-Camera Ingestion Status",
    response_description="Camera stream status across CAM_01 to CAM_04.",
)
def get_central_cameras(
    db: CentralDatabase = Depends(get_central_db),
) -> List[Dict[str, Any]]:
    """Retrieve operational status for the 4 concurrent video processing camera streams."""
    cams = [
        {"id": "CAM_01", "name": "Food", "role": "Department Analytics", "source": "videos/food/food.mp4", "staff": 2},
        {"id": "CAM_02", "name": "Electronics", "role": "Department Analytics", "source": "videos/electronics/electronics.mp4", "staff": 1},
        {"id": "CAM_03", "name": "Grocery", "role": "Department Analytics", "source": "videos/grocery/grocery.mp4", "staff": 2},
        {"id": "CAM_04", "name": "Checkout", "role": "Queue Analytics", "source": "videos/checkout/checkout.mp4", "staff": 2},
    ]

    latest_zones = db.get_latest_zones() if hasattr(db, "get_latest_zones") else []
    zone_by_name = {z.get("zone_name", "").lower(): z for z in latest_zones}

    results = []
    for c in cams:
        z = zone_by_name.get(c["name"].lower())
        shoppers = int(z.get("current_shoppers", 0)) if z else 0
        results.append(
            {
                "camera_id": c["id"],
                "name": c["name"],
                "department": c["name"],
                "role": c["role"],
                "status": "Processing",
                "processing": True,
                "fps": 30.0,
                "frame_count": 0,
                "current_detections": shoppers,
                "current_shoppers": shoppers,
                "source": c["source"],
                "stream_type": "Recorded Video",
                "is_simulation": True,
                "simulation_label": "Recorded Video / Multi-Camera Demo Simulation",
            }
        )
    return results


@app.get(
    "/api/v1/patterns/hourly",
    summary="Get Hourly Department Traffic (Central)",
    response_description="Department traffic aggregated by hour derived from synchronized CV observations.",
)
def get_central_hourly_patterns(
    limit: int = Query(default=2000, ge=10, le=5000, description="Max snapshots to analyze."),
    db: CentralDatabase = Depends(get_central_db),
) -> Dict[str, Any]:
    """Retrieve hourly department traffic and peak periods derived from stored PostgreSQL observations."""
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
    summary="Get Central Simulation Clock Status",
    response_description="Current simulated store time position.",
)
def get_central_simulation_clock(
    db: CentralDatabase = Depends(get_central_db),
) -> Dict[str, Any]:
    """Return the simulated store clock position from central telemetry."""
    latest = db.get_latest_snapshot()
    sim_time = latest.get("timestamp") if latest else "17:00:00"
    sim_iso = latest.get("created_at") if latest else None
    return {
        "is_simulation": True,
        "mode": "Demo Simulation",
        "label": "Simulated Store Time",
        "demo_start_time": "17:00:00",
        "simulated_store_time": sim_time or "17:00:00",
        "simulated_iso": sim_iso,
        "elapsed_seconds": 0.0,
    }

