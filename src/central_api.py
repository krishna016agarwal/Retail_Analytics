"""FastAPI Central REST API for Retail Analytics (SIH 179).

Central cloud backend hosted on Render, backed by PostgreSQL.
Receives batch edge telemetry synchronizations, provides central querying,
and enables multi-store retail intelligence aggregation.
Stores zero facial, biometric, or personal identity data.
"""

import os
from typing import Any, Dict, List, Optional, Union
from fastapi import Depends, FastAPI, HTTPException, Query, status
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


class BatchSyncRequest(BaseModel):
    """Schema for multi-snapshot batch ingestion payload."""

    device_id: str = Field(..., description="Edge processing node identifier sending the batch.")
    snapshots: List[SnapshotItem] = Field(default_factory=list, description="List of pending telemetry snapshots.")


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
    """Receive a batch of telemetry snapshots from an edge node.

    Idempotent: Uses ON CONFLICT (snapshot_id) DO NOTHING so retried uploads
    safely succeed without duplication. Confirmed local IDs are returned
    so the edge device can safely transition them from PENDING to SYNCED.
    """
    if not payload.snapshots:
        return {
            "success": True,
            "received": 0,
            "inserted": 0,
            "already_synced": 0,
            "failed": 0,
            "synced_ids": [],
        }

    snapshots_data = [item.model_dump() for item in payload.snapshots]
    try:
        result = db.insert_batch(
            device_id=payload.device_id,
            snapshots=snapshots_data,
        )
        return {
            "success": True,
            "received": result["received"],
            "inserted": result["inserted"],
            "already_synced": result["already_synced"],
            "failed": result["failed"],
            "synced_ids": result["synced_ids"],
        }
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
