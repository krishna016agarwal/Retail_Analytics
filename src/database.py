"""Edge Database Module for Offline-First Retail Analytics (SIH 179).

Provides local SQLite storage for anonymous, aggregated retail telemetry snapshots.
Uses Python's built-in sqlite3 standard library with zero external dependencies.
Stores zero facial, biometric, image, or personally identifiable information.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Union


@dataclass
class AnalyticsSnapshot:
    """Represents a single telemetry snapshot of retail edge intelligence."""

    store_id: str
    device_id: str
    timestamp: str
    entries: int
    exits: int
    occupancy: int
    peak_occupancy: int
    queue_length: int
    peak_queue: int
    avg_dwell: float
    max_dwell: float
    avg_wait: float
    created_at: str
    sync_status: str = "PENDING"
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return asdict(self)


class EdgeDatabase:
    """Local SQLite database manager for offline-first retail edge intelligence."""

    def __init__(self, db_path: Union[str, Path] = "data/retail_edge.db"):
        """Initialize database manager and ensure parent directories exist.

        Args:
            db_path: Filesystem path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self.initialize()

    def _get_connection(self) -> sqlite3.Connection:
        """Obtain a reusable SQLite connection configured with sqlite3.Row."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=10.0,
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def initialize(self) -> None:
        """Create analytics_snapshots table and indexes if they do not already exist.

        Safe to call multiple times on existing databases.
        """
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analytics_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    store_id TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    entries INTEGER NOT NULL DEFAULT 0,
                    exits INTEGER NOT NULL DEFAULT 0,
                    occupancy INTEGER NOT NULL DEFAULT 0,
                    peak_occupancy INTEGER NOT NULL DEFAULT 0,
                    queue_length INTEGER NOT NULL DEFAULT 0,
                    peak_queue INTEGER NOT NULL DEFAULT 0,
                    avg_dwell REAL NOT NULL DEFAULT 0.0,
                    max_dwell REAL NOT NULL DEFAULT 0.0,
                    avg_wait REAL NOT NULL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    sync_status TEXT NOT NULL DEFAULT 'PENDING'
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp
                ON analytics_snapshots (timestamp);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_snapshots_sync
                ON analytics_snapshots (sync_status);
                """
            )

    def insert_snapshot(
        self,
        store_id: str,
        device_id: str,
        timestamp: str,
        entries: int,
        exits: int,
        occupancy: int,
        peak_occupancy: int,
        queue_length: int,
        peak_queue: int,
        avg_dwell: float,
        max_dwell: float,
        avg_wait: float,
        sync_status: str = "PENDING",
        created_at: Optional[str] = None,
    ) -> int:
        """Insert a new periodic telemetry snapshot into SQLite.

        Args:
            store_id: Retail branch / store identifier.
            device_id: Edge processing node identifier.
            timestamp: Timeline position or formatted video session timestamp.
            entries: Cumulative store entries.
            exits: Cumulative store exits.
            occupancy: Current active store occupancy.
            peak_occupancy: Maximum simultaneous occupancy observed.
            queue_length: Active persons in queue zone.
            peak_queue: Maximum queue length observed.
            avg_dwell: Average dwell time in seconds for completed visits.
            max_dwell: Maximum dwell time in seconds for completed visits.
            avg_wait: Average queue waiting time in seconds.
            sync_status: Sync state ('PENDING' or 'SYNCED').
            created_at: Optional ISO-8601 UTC timestamp. Defaults to current UTC time.

        Returns:
            The integer primary key (id) of the inserted row.
        """
        if created_at is None:
            created_at = datetime.now(timezone.utc).isoformat()

        conn = self._get_connection()
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO analytics_snapshots (
                    store_id, device_id, timestamp,
                    entries, exits, occupancy, peak_occupancy,
                    queue_length, peak_queue,
                    avg_dwell, max_dwell, avg_wait,
                    created_at, sync_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    str(store_id),
                    str(device_id),
                    str(timestamp),
                    int(entries),
                    int(exits),
                    int(occupancy),
                    int(peak_occupancy),
                    int(queue_length),
                    int(peak_queue),
                    float(avg_dwell),
                    float(max_dwell),
                    float(avg_wait),
                    str(created_at),
                    str(sync_status),
                ),
            )
            return cursor.lastrowid

    def get_pending_snapshots(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve telemetry snapshots with sync_status = 'PENDING'.

        Args:
            limit: Optional maximum number of pending snapshots to return.

        Returns:
            List of snapshot records as dictionaries ordered by id ASC.
        """
        conn = self._get_connection()
        if limit is not None and int(limit) > 0:
            cursor = conn.execute(
                """
                SELECT * FROM analytics_snapshots
                WHERE sync_status = 'PENDING'
                ORDER BY id ASC
                LIMIT ?;
                """,
                (int(limit),),
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM analytics_snapshots
                WHERE sync_status = 'PENDING'
                ORDER BY id ASC;
                """
            )
        return [dict(row) for row in cursor.fetchall()]


    def mark_synced(self, ids: List[int]) -> int:
        """Update sync_status to 'SYNCED' for the provided row IDs.

        Args:
            ids: List of primary key IDs successfully synchronized.

        Returns:
            Number of rows updated.
        """
        if not ids:
            return 0
        conn = self._get_connection()
        placeholders = ",".join("?" for _ in ids)
        with conn:
            cursor = conn.execute(
                f"""
                UPDATE analytics_snapshots
                SET sync_status = 'SYNCED'
                WHERE id IN ({placeholders});
                """,
                ids,
            )
            return cursor.rowcount

    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """Retrieve the most recently recorded telemetry snapshot.

        Returns:
            Dictionary of the latest snapshot row, or None if the table is empty.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT * FROM analytics_snapshots
            ORDER BY id DESC
            LIMIT 1;
            """
        )
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def get_snapshots(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve up to limit recent telemetry snapshots.

        Args:
            limit: Maximum rows to return (defaults to 100).

        Returns:
            List of snapshot records as dictionaries ordered by id DESC.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT * FROM analytics_snapshots
            ORDER BY id DESC
            LIMIT ?;
            """,
            (int(limit),),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_unsynced_count(self) -> int:
        """Return the total number of snapshots pending synchronization."""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT COUNT(*) FROM analytics_snapshots
            WHERE sync_status = 'PENDING';
            """
        )
        return cursor.fetchone()[0]

    def get_total_count(self) -> int:
        """Return the total number of snapshots stored in the local database."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM analytics_snapshots;")
        return cursor.fetchone()[0]

    def check_connection(self) -> bool:
        """Verify that the SQLite EdgeDatabase is accessible and readable."""
        try:
            conn = self._get_connection()
            cursor = conn.execute("SELECT 1 FROM analytics_snapshots LIMIT 1;")
            cursor.fetchone()
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close database connection cleanly."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "EdgeDatabase":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
