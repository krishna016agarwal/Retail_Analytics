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
