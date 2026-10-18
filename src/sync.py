"""Edge Synchronization Client for Retail Analytics (SIH 179).

Provides robust, offline-first Internet synchronization between local edge SQLite
and the Render-hosted Central FastAPI service.
Guarantees zero data loss: pending records remain PENDING whenever the network
or central server is unavailable, and are marked SYNCED only after central confirmation.
"""

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
import requests

from src.database import EdgeDatabase

# Ensure .env variables are loaded when sync client is used
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
load_dotenv()

logger = logging.getLogger("sync_client")


@dataclass
class SyncResult:
    """Encapsulates the outcome of an edge-to-cloud synchronization operation."""

    success: bool
    pending_count: int = 0
    uploaded_count: int = 0
    inserted: int = 0
    already_synced: int = 0
    failed: int = 0
    synced_ids: List[int] = field(default_factory=list)
    remaining_pending: int = 0
    error_message: Optional[str] = None
    offline_mode: bool = False


class SyncClient:
    """Client for synchronizing pending edge telemetry to the central cloud API."""

    def __init__(
        self,
        central_api_url: Optional[str] = None,
        edge_db: Optional[EdgeDatabase] = None,
        db_path: str = "data/retail_edge.db",
        timeout: float = 10.0,
    ):
        """Initialize SyncClient.

        Args:
            central_api_url: Base URL of Central API. Defaults to CENTRAL_API_URL env var.
            edge_db: Optional EdgeDatabase instance. If None, instantiates one with db_path.
            db_path: Path to local SQLite database file if edge_db not provided.
            timeout: HTTP request timeout in seconds.
        """
        raw_url = central_api_url or os.environ.get(
            "CENTRAL_API_URL", "http://127.0.0.1:8000"
        )
        self.central_api_url = raw_url.rstrip("/")
        self.edge_db = edge_db or EdgeDatabase(db_path=db_path)
        self.timeout = float(timeout)

    def check_server_health(self) -> bool:
        """Ping central /health endpoint to check server & PostgreSQL status.

        Returns:
            True if central server is reachable and reports healthy database.
        """
        try:
            url = f"{self.central_api_url}/health"
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("status") == "healthy"
            return False
        except Exception:
            return False

    def sync_batch(
        self,
        snapshots: List[Dict[str, Any]],
        fallback_device_id: str = "edge_device_01",
    ) -> SyncResult:
        """Upload a single batch of snapshots to the Central API.

        Guarantees offline-first safety: on any network or server exception,
        all snapshots remain PENDING in the edge SQLite database.

        Args:
            snapshots: List of SQLite snapshot rows.
            fallback_device_id: Device ID used if not specified in snapshot.

        Returns:
            SyncResult summarizing the batch sync operation.
        """
        if not snapshots:
            unsynced = self.edge_db.get_unsynced_count()
            return SyncResult(
                success=True,
                pending_count=0,
                remaining_pending=unsynced,
            )

        total_pending = len(snapshots)
        device_id = snapshots[0].get("device_id") or fallback_device_id

        # Format snapshots for central batch API schema
        payload_snapshots = []
        for s in snapshots:
            payload_snapshots.append(
                {
                    "local_id": s["id"],
                    "store_id": s.get("store_id", "store_001"),
                    "device_id": s.get("device_id", device_id),
                    "timestamp": s.get("timestamp", ""),
                    "entries": int(s.get("entries", 0)),
                    "exits": int(s.get("exits", 0)),
                    "occupancy": int(s.get("occupancy", 0)),
                    "peak_occupancy": int(s.get("peak_occupancy", 0)),
                    "queue_length": int(s.get("queue_length", 0)),
                    "peak_queue": int(s.get("peak_queue", 0)),
                    "avg_dwell": float(s.get("avg_dwell", 0.0)),
                    "max_dwell": float(s.get("max_dwell", 0.0)),
                    "avg_wait": float(s.get("avg_wait", 0.0)),
                    "created_at": s.get("created_at"),
                }
            )

        # Include pending zone snapshots (multi-camera department observations)
        payload_zones = []
        if hasattr(self.edge_db, "get_pending_zone_snapshots"):
            pending_zones = self.edge_db.get_pending_zone_snapshots(limit=max(10, total_pending * 4))
            for z in pending_zones:
                payload_zones.append(
                    {
                        "local_id": z["id"],
                        "snapshot_id": z.get("snapshot_id"),
                        "store_id": z.get("store_id", "store_001"),
                        "device_id": z.get("device_id", device_id),
                        "camera_id": z.get("camera_id", "CAM_01"),
                        "zone_id": z.get("zone_id", ""),
                        "zone_name": z.get("zone_name", ""),
                        "timestamp": str(z.get("timestamp", "")),
                        "current_shoppers": int(z.get("current_shoppers", 0)),
                        "peak_shoppers": int(z.get("peak_shoppers", 0)),
                        "avg_dwell": float(z.get("avg_dwell", 0.0)),
                        "traffic_level": str(z.get("traffic_level", "LOW")),
                        "expected_staff": int(z.get("expected_staff", 1)),
                        "created_at": z.get("created_at"),
                    }
                )

        # Include pending operational alerts
        payload_alerts = []
        if hasattr(self.edge_db, "get_pending_alerts"):
            pending_alerts = self.edge_db.get_pending_alerts(limit=50)
            for a in pending_alerts:
                payload_alerts.append(
                    {
                        "local_id": a["id"],
                        "alert_id": a.get("alert_id", ""),
                        "store_id": a.get("store_id", "store_001"),
                        "device_id": a.get("device_id", device_id),
                        "camera_id": a.get("camera_id", "CAM_01"),
                        "zone_id": a.get("zone_id", "store"),
                        "type": a.get("type", "UNKNOWN"),
                        "severity": a.get("severity", "MEDIUM"),
                        "title": a.get("title", ""),
                        "message": a.get("message", ""),
                        "current_value": float(a.get("current_value", 0.0) or 0.0),
                        "predicted_value": float(a["predicted_value"]) if a.get("predicted_value") is not None else None,
                        "threshold": float(a.get("threshold", 0.0) or 0.0),
                        "recommendation": a.get("recommendation", ""),
                        "status": a.get("status", "ACTIVE"),
                        "created_at": a.get("created_at"),
                    }
                )

        payload = {
            "device_id": device_id,
            "snapshots": payload_snapshots,
        }
        if payload_zones:
            payload["zone_snapshots"] = payload_zones
        if payload_alerts:
            payload["alerts"] = payload_alerts

        endpoint = f"{self.central_api_url}/api/v1/analytics/batch"
        try:
            response = requests.post(
                endpoint,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()

            synced_ids = data.get("synced_ids", [])
            synced_zone_ids = data.get("synced_zone_ids", [])
            synced_alert_ids = data.get("synced_alert_ids", [])
            inserted = data.get("inserted", 0)
            already_synced = data.get("already_synced", 0)
            failed = data.get("failed", 0)

            # Mark only confirmed IDs as SYNCED in local SQLite
            if synced_ids:
                self.edge_db.mark_synced(synced_ids)
            if synced_zone_ids and hasattr(self.edge_db, "mark_zone_snapshots_synced"):
                self.edge_db.mark_zone_snapshots_synced(synced_zone_ids)
            if synced_alert_ids and hasattr(self.edge_db, "mark_alerts_synced"):
                self.edge_db.mark_alerts_synced(synced_alert_ids)

            remaining = self.edge_db.get_unsynced_count()
            return SyncResult(
                success=True,
                pending_count=total_pending,
                uploaded_count=len(payload_snapshots),
                inserted=inserted,
                already_synced=already_synced,
                failed=failed,
                synced_ids=synced_ids,
                remaining_pending=remaining,
            )

        except requests.exceptions.RequestException as e:
            # Network failure, timeout, 5xx error, or server unreachable
            # Data remains PENDING in SQLite; NO data loss
            remaining = self.edge_db.get_unsynced_count()
            return SyncResult(
                success=False,
                pending_count=total_pending,
                uploaded_count=0,
                remaining_pending=remaining,
                error_message=str(e),
                offline_mode=True,
            )
        except Exception as e:
            remaining = self.edge_db.get_unsynced_count()
            return SyncResult(
                success=False,
                pending_count=total_pending,
                uploaded_count=0,
                remaining_pending=remaining,
                error_message=f"Unexpected error: {str(e)}",
                offline_mode=True,
            )

    def sync_all(
        self,
        batch_size: int = 100,
        verbose: bool = False,
    ) -> SyncResult:
        """Retrieve and synchronize all pending snapshots in batches.

        Args:
            batch_size: Maximum records per HTTPS batch request.
            verbose: If True, prints batch-by-batch progress.

        Returns:
            Aggregate SyncResult for all batches.
        """
        pending = self.edge_db.get_pending_snapshots()
        total_pending = len(pending)

        if total_pending == 0:
            if verbose:
                print("Sync started")
                print("Pending snapshots: 0")
                print("Sync completed")
                print("Remaining pending: 0")
            return SyncResult(
                success=True,
                pending_count=0,
                remaining_pending=0,
            )

        if verbose:
            print("Sync started")
            print(f"Pending snapshots: {total_pending}")

        total_inserted = 0
        total_already_synced = 0
        total_failed = 0
        all_synced_ids: List[int] = []

        # Process in batches
        for start_idx in range(0, total_pending, batch_size):
            batch = pending[start_idx : start_idx + batch_size]
            end_idx = min(start_idx + len(batch), total_pending)
            if verbose:
                print(f"Uploading batch: {start_idx + 1}-{end_idx}")

            result = self.sync_batch(batch)

            if not result.success:
                if verbose:
                    print("Central server unavailable")
                    print(f"Offline mode: keeping {result.remaining_pending} records PENDING")
                    print("Sync failed safely")
                return SyncResult(
                    success=False,
                    pending_count=total_pending,
                    uploaded_count=len(all_synced_ids),
                    inserted=total_inserted,
                    already_synced=total_already_synced,
                    failed=total_failed,
                    synced_ids=all_synced_ids,
                    remaining_pending=result.remaining_pending,
                    error_message=result.error_message,
                    offline_mode=True,
                )

            total_inserted += result.inserted
            total_already_synced += result.already_synced
            total_failed += result.failed
            all_synced_ids.extend(result.synced_ids)

        remaining = self.edge_db.get_unsynced_count()
        if verbose:
            print(f"Inserted: {total_inserted}")
            print(f"Already synced: {total_already_synced}")
            print(f"Failed: {total_failed}")
            print("Sync completed")
            print(f"Remaining pending: {remaining}")

        return SyncResult(
            success=True,
            pending_count=total_pending,
            uploaded_count=len(all_synced_ids),
            inserted=total_inserted,
            already_synced=total_already_synced,
            failed=total_failed,
            synced_ids=all_synced_ids,
            remaining_pending=remaining,
        )


class BackgroundSyncThread:
    """Non-blocking background thread for periodic automatic edge telemetry synchronization."""

    def __init__(
        self,
        sync_client: SyncClient,
        interval_seconds: float = 30.0,
        batch_size: int = 100,
    ):
        """Initialize background sync worker.

        Args:
            sync_client: Configured SyncClient instance.
            interval_seconds: Polling interval between sync attempts.
            batch_size: Batch size for uploads.
        """
        self.sync_client = sync_client
        self.interval_seconds = max(5.0, float(interval_seconds))
        self.batch_size = int(batch_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _worker(self) -> None:
        """Background loop executing periodic sync."""
        while not self._stop_event.is_set():
            # Wait for interval or stop signal
            if self._stop_event.wait(timeout=self.interval_seconds):
                break

            try:
                # Check if there are pending records before attempting network call
                unsynced = self.sync_client.edge_db.get_unsynced_count()
                if unsynced > 0:
                    res = self.sync_client.sync_all(batch_size=self.batch_size, verbose=False)
                    if res.success and (res.inserted > 0 or res.uploaded_count > 0):
                        print(f"\n[Cloud Sync] Uploaded {res.inserted} records to Central Cloud ({self.sync_client.central_api_url}) | {res.remaining_pending} pending")
                    elif not res.success:
                        print(f"\n[!] Cloud Sync: Server offline or waking up (Render cold start) — {res.remaining_pending} records safely queued in Edge SQLite")
            except Exception as e:
                logger.debug(f"Background sync iteration encountered error: {e}")

    def start(self) -> None:
        """Start the background synchronization thread."""
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._worker,
                daemon=True,
                name="AutoSyncThread",
            )
            self._thread.start()
            logger.info(f"Background auto-sync started (interval={self.interval_seconds}s)")

    def stop(self) -> None:
        """Signal worker to stop and wait for termination."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None
            logger.info("Background auto-sync stopped")


def run_sync_cli(
    db_path: str = "data/retail_edge.db",
    central_url: Optional[str] = None,
    batch_size: int = 100,
) -> int:
    """CLI execution entrypoint for manual synchronization.

    Args:
        db_path: Path to edge SQLite database.
        central_url: Optional override for CENTRAL_API_URL.
        batch_size: Number of records per batch.

    Returns:
        Exit code (0 for clean execution).
    """
    client = SyncClient(central_api_url=central_url, db_path=db_path)
    client.sync_all(batch_size=batch_size, verbose=True)
    return 0
