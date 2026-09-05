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
        """Create analytics_snapshots, zone_snapshots, and analytics_alerts tables.

        Safe to call multiple times on existing databases.
        """
        conn = self._get_connection()
        with conn:
            # Phase 6A: Core telemetry snapshots
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

            # Phase 8: Zone intelligence snapshots
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS zone_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_id TEXT UNIQUE NOT NULL,
                    store_id TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    zone_id TEXT NOT NULL,
                    zone_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    current_shoppers INTEGER NOT NULL DEFAULT 0,
                    peak_shoppers INTEGER NOT NULL DEFAULT 0,
                    avg_dwell REAL NOT NULL DEFAULT 0.0,
                    traffic_level TEXT NOT NULL DEFAULT 'LOW',
                    expected_staff INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    sync_status TEXT NOT NULL DEFAULT 'PENDING'
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_zone_snapshots_zone
                ON zone_snapshots (zone_id);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_zone_snapshots_sync
                ON zone_snapshots (sync_status);
                """
            )

            # Phase 8: Operational retail alerts
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analytics_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    store_id TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    zone_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    current_value REAL NOT NULL DEFAULT 0.0,
                    predicted_value REAL,
                    threshold REAL NOT NULL DEFAULT 0.0,
                    recommendation TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    created_at TEXT NOT NULL,
                    sync_status TEXT NOT NULL DEFAULT 'PENDING'
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_alerts_status
                ON analytics_alerts (status);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_alerts_sync
                ON analytics_alerts (sync_status);
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

    # Phase 8: Zone Snapshots and Alert Methods
    def insert_zone_snapshot(
        self,
        snapshot_id: str,
        store_id: str,
        device_id: str,
        camera_id: str,
        zone_id: str,
        zone_name: str,
        timestamp: str,
        current_shoppers: int,
        peak_shoppers: int,
        avg_dwell: float,
        traffic_level: str = "LOW",
        expected_staff: int = 1,
        created_at: Optional[str] = None,
        sync_status: str = "PENDING",
    ) -> int:
        """Insert a zone snapshot into local SQLite."""
        if created_at is None:
            created_at = datetime.now(timezone.utc).isoformat()

        conn = self._get_connection()
        with conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO zone_snapshots (
                    snapshot_id, store_id, device_id, camera_id,
                    zone_id, zone_name, timestamp,
                    current_shoppers, peak_shoppers, avg_dwell,
                    traffic_level, expected_staff, created_at, sync_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    str(snapshot_id),
                    str(store_id),
                    str(device_id),
                    str(camera_id),
                    str(zone_id),
                    str(zone_name),
                    str(timestamp),
                    int(current_shoppers),
                    int(peak_shoppers),
                    float(avg_dwell),
                    str(traffic_level),
                    int(expected_staff),
                    str(created_at),
                    str(sync_status),
                ),
            )
            return cursor.lastrowid

    def insert_alert(
        self,
        alert_id: str,
        store_id: str,
        device_id: str,
        camera_id: str,
        zone_id: str,
        type: str,
        severity: str,
        title: str,
        message: str,
        current_value: float,
        threshold: float,
        recommendation: str,
        predicted_value: Optional[float] = None,
        status: str = "ACTIVE",
        created_at: Optional[str] = None,
        sync_status: str = "PENDING",
    ) -> int:
        """Insert or update an operational retail alert."""
        if created_at is None:
            created_at = datetime.now(timezone.utc).isoformat()

        conn = self._get_connection()
        with conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO analytics_alerts (
                    alert_id, store_id, device_id, camera_id, zone_id,
                    type, severity, title, message, current_value,
                    predicted_value, threshold, recommendation, status,
                    created_at, sync_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    str(alert_id),
                    str(store_id),
                    str(device_id),
                    str(camera_id),
                    str(zone_id),
                    str(type),
                    str(severity),
                    str(title),
                    str(message),
                    float(current_value),
                    float(predicted_value) if predicted_value is not None else None,
                    float(threshold),
                    str(recommendation),
                    str(status),
                    str(created_at),
                    str(sync_status),
                ),
            )
            return cursor.lastrowid

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an active alert as RESOLVED in SQLite."""
        conn = self._get_connection()
        with conn:
            cursor = conn.execute(
                "UPDATE analytics_alerts SET status = 'RESOLVED' WHERE alert_id = ? AND status = 'ACTIVE';",
                (str(alert_id),),
            )
            return cursor.rowcount > 0

    def resolve_alerts_by_type(self, alert_type: str, zone_id: Optional[str] = None) -> int:
        """Mark active alerts of a given type (and optional zone) as RESOLVED."""
        conn = self._get_connection()
        with conn:
            if zone_id:
                cursor = conn.execute(
                    "UPDATE analytics_alerts SET status = 'RESOLVED' WHERE type = ? AND zone_id = ? AND status = 'ACTIVE';",
                    (str(alert_type).upper(), str(zone_id)),
                )
            else:
                cursor = conn.execute(
                    "UPDATE analytics_alerts SET status = 'RESOLVED' WHERE type = ? AND status = 'ACTIVE';",
                    (str(alert_type).upper(),),
                )
            return cursor.rowcount

    def get_active_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve all currently active alerts ordered newest first."""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT * FROM analytics_alerts
            WHERE status = 'ACTIVE'
            ORDER BY id DESC
            LIMIT ?;
            """,
            (int(limit),),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_alerts(
        self,
        severity: Optional[str] = None,
        alert_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve alerts with optional filtering."""
        query = "SELECT * FROM analytics_alerts WHERE 1=1"
        params: List[Any] = []

        if severity:
            query += " AND severity = ?"
            params.append(str(severity).upper())
        if alert_type:
            query += " AND type = ?"
            params.append(str(alert_type).upper())
        if status:
            query += " AND status = ?"
            params.append(str(status).upper())

        query += " ORDER BY id DESC LIMIT ?;"
        params.append(int(limit))

        conn = self._get_connection()
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_latest_zone_snapshots(self) -> List[Dict[str, Any]]:
        """Retrieve the most recent snapshot for each distinct zone."""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT z.* FROM zone_snapshots z
            INNER JOIN (
                SELECT zone_id, MAX(id) AS max_id
                FROM zone_snapshots
                GROUP BY zone_id
            ) grouped ON z.id = grouped.max_id
            ORDER BY z.zone_name ASC;
            """
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_zone_snapshots(
        self, zone_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve recent zone snapshots with optional zone filtering."""
        conn = self._get_connection()
        if zone_id:
            cursor = conn.execute(
                """
                SELECT * FROM zone_snapshots
                WHERE zone_id = ?
                ORDER BY id DESC LIMIT ?;
                """,
                (str(zone_id), int(limit)),
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM zone_snapshots
                ORDER BY id DESC LIMIT ?;
                """,
                (int(limit),),
            )
        return [dict(row) for row in cursor.fetchall()]

    def get_pending_alerts(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve alerts pending cloud synchronization."""
        conn = self._get_connection()
        if limit is not None and int(limit) > 0:
            cursor = conn.execute(
                "SELECT * FROM analytics_alerts WHERE sync_status = 'PENDING' ORDER BY id ASC LIMIT ?;",
                (int(limit),),
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM analytics_alerts WHERE sync_status = 'PENDING' ORDER BY id ASC;"
            )
        return [dict(row) for row in cursor.fetchall()]

    def mark_alerts_synced(self, ids: List[int]) -> int:
        """Mark alert rows as SYNCED."""
        if not ids:
            return 0
        conn = self._get_connection()
        placeholders = ",".join("?" for _ in ids)
        with conn:
            cursor = conn.execute(
                f"UPDATE analytics_alerts SET sync_status = 'SYNCED' WHERE id IN ({placeholders});",
                ids,
            )
            return cursor.rowcount

    def get_pending_zone_snapshots(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve zone snapshots pending cloud synchronization."""
        conn = self._get_connection()
        if limit is not None and int(limit) > 0:
            cursor = conn.execute(
                "SELECT * FROM zone_snapshots WHERE sync_status = 'PENDING' ORDER BY id ASC LIMIT ?;",
                (int(limit),),
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM zone_snapshots WHERE sync_status = 'PENDING' ORDER BY id ASC;"
            )
        return [dict(row) for row in cursor.fetchall()]

    def mark_zone_snapshots_synced(self, ids: List[int]) -> int:
        """Mark zone snapshot rows as SYNCED."""
        if not ids:
            return 0
        conn = self._get_connection()
        placeholders = ",".join("?" for _ in ids)
        with conn:
            cursor = conn.execute(
                f"UPDATE zone_snapshots SET sync_status = 'SYNCED' WHERE id IN ({placeholders});",
                ids,
            )
            return cursor.rowcount

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
