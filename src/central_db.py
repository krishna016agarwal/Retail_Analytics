"""Central PostgreSQL Database Manager for Retail Analytics (SIH 179).

Manages the central PostgreSQL storage hosted on Render for aggregated retail
telemetry snapshots synchronized from edge devices.
Supports idempotent batch inserts via ON CONFLICT (snapshot_id) DO NOTHING.
Zero PII, facial, or biometric data is ever stored.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from dotenv import load_dotenv

# Ensure .env variables (DATABASE_URL) are loaded
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
load_dotenv()

try:
    import psycopg2
    from psycopg2 import pool
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    psycopg2 = None  # type: ignore
    pool = None  # type: ignore
    RealDictCursor = None  # type: ignore
    PSYCOPG2_AVAILABLE = False

logger = logging.getLogger("central_db")


class CentralDatabase:
    """PostgreSQL database manager for the central cloud service."""

    def __init__(self, database_url: Optional[str] = None):
        """Initialize central database manager.

        Args:
            database_url: PostgreSQL connection URL. If None, reads from DATABASE_URL env var.
        """
        raw_url = database_url or os.environ.get("DATABASE_URL", "")
        # Render PostgreSQL URLs may start with postgres:// which psycopg2 prefers as postgresql://
        if raw_url.startswith("postgres://"):
            raw_url = raw_url.replace("postgres://", "postgresql://", 1)
        self.database_url = raw_url
        self._pool: Optional[Any] = None

        if self.database_url and PSYCOPG2_AVAILABLE:
            try:
                self._pool = pool.SimpleConnectionPool(
                    minconn=1,
                    maxconn=10,
                    dsn=self.database_url,
                )
                self.initialize()
            except Exception as e:
                logger.warning(f"Failed to initialize CentralDatabase pool at startup: {e}")

    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        """Obtain a connection from the pool or directly from URL."""
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError("psycopg2-binary is not installed.")
        if not self.database_url:
            raise ValueError("DATABASE_URL is not set.")

        conn = None
        if self._pool is not None:
            conn = self._pool.getconn()
            try:
                yield conn
            finally:
                if conn is not None:
                    self._pool.putconn(conn)
        else:
            conn = psycopg2.connect(self.database_url)
            try:
                yield conn
            finally:
                if conn is not None:
                    conn.close()

    def check_connection(self) -> bool:
        """Verify that the PostgreSQL database is reachable."""
        if not PSYCOPG2_AVAILABLE or not self.database_url:
            return False
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    result = cur.fetchone()
                    return result is not None and result[0] == 1
        except Exception as e:
            logger.debug(f"PostgreSQL health check failed: {e}")
            return False

    def initialize(self) -> None:
        """Create analytics_snapshots table and unique constraints if not present."""
        if not self.check_connection():
            return
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS analytics_snapshots (
                        id SERIAL PRIMARY KEY,
                        snapshot_id TEXT UNIQUE NOT NULL,
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
                        created_at TEXT NOT NULL
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_central_snapshot_id
                    ON analytics_snapshots (snapshot_id);
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_central_timestamp
                    ON analytics_snapshots (timestamp);
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_central_store_device
                    ON analytics_snapshots (store_id, device_id);
                    """
                )

                # Phase 8: Central Zone Snapshots
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS zone_snapshots (
                        id SERIAL PRIMARY KEY,
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
                        created_at TEXT NOT NULL
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_central_zone_snapshot_id
                    ON zone_snapshots (snapshot_id);
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_central_zone_id
                    ON zone_snapshots (zone_id);
                    """
                )

                # Phase 8: Central Analytics Alerts
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS analytics_alerts (
                        id SERIAL PRIMARY KEY,
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
                        created_at TEXT NOT NULL
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_central_alert_id
                    ON analytics_alerts (alert_id);
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_central_alert_status
                    ON analytics_alerts (status);
                    """
                )
            conn.commit()


    def insert_batch(
        self,
        device_id: str,
        snapshots: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Insert a batch of edge snapshots idempotently.

        Uses ON CONFLICT (snapshot_id) DO NOTHING so duplicate uploads
        are safely recognized as already synced without failing the batch.

        Args:
            device_id: Fallback edge device identifier.
            snapshots: List of snapshot dicts containing local_id and metrics.

        Returns:
            Dict containing inserted, already_synced, failed counts and synced_ids list.
        """
        inserted_count = 0
        already_synced_count = 0
        failed_count = 0
        synced_ids: List[int] = []

        if not snapshots:
            return {
                "received": 0,
                "inserted": 0,
                "already_synced": 0,
                "failed": 0,
                "synced_ids": [],
            }

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for item in snapshots:
                    local_id = item.get("local_id")
                    dev_id = item.get("device_id") or device_id
                    # Deterministic idempotency key: "{device_id}:{local_id}"
                    snapshot_id = item.get("snapshot_id") or f"{dev_id}:{local_id}"
                    store_id = str(item.get("store_id", "store_001"))
                    timestamp = str(item.get("timestamp", ""))
                    entries = int(item.get("entries", 0))
                    exits = int(item.get("exits", 0))
                    occupancy = int(item.get("occupancy", 0))
                    peak_occ = int(item.get("peak_occupancy", 0))
                    queue_len = int(item.get("queue_length", 0))
                    peak_q = int(item.get("peak_queue", 0))
                    avg_dwell = float(item.get("avg_dwell", 0.0))
                    max_dwell = float(item.get("max_dwell", 0.0))
                    avg_wait = float(item.get("avg_wait", 0.0))
                    created_at = str(
                        item.get("created_at")
                        or datetime.now(timezone.utc).isoformat()
                    )

                    try:
                        cur.execute(
                            """
                            INSERT INTO analytics_snapshots (
                                snapshot_id, store_id, device_id, timestamp,
                                entries, exits, occupancy, peak_occupancy,
                                queue_length, peak_queue,
                                avg_dwell, max_dwell, avg_wait, created_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (snapshot_id) DO NOTHING
                            RETURNING id;
                            """,
                            (
                                snapshot_id,
                                store_id,
                                dev_id,
                                timestamp,
                                entries,
                                exits,
                                occupancy,
                                peak_occ,
                                queue_len,
                                peak_q,
                                avg_dwell,
                                max_dwell,
                                avg_wait,
                                created_at,
                            ),
                        )
                        result = cur.fetchone()
                        if result is not None:
                            # Newly inserted
                            inserted_count += 1
                        else:
                            # Already existed in PostgreSQL (idempotent duplicate)
                            already_synced_count += 1

                        if local_id is not None:
                            synced_ids.append(local_id)
                    except Exception as e:
                        logger.error(f"Failed to insert snapshot {snapshot_id}: {e}")
                        failed_count += 1
            conn.commit()

        return {
            "received": len(snapshots),
            "inserted": inserted_count,
            "already_synced": already_synced_count,
            "failed": failed_count,
            "synced_ids": synced_ids,
        }

    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent snapshot across all stores."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, snapshot_id, store_id, device_id, timestamp,
                           entries, exits, occupancy, peak_occupancy,
                           queue_length, peak_queue,
                           avg_dwell, max_dwell, avg_wait, created_at
                    FROM analytics_snapshots
                    ORDER BY id DESC
                    LIMIT 1;
                    """
                )
                row = cur.fetchone()
                return dict(row) if row is not None else None

    def get_snapshots(
        self,
        store_id: Optional[str] = None,
        device_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve historical telemetry snapshots with optional filtering."""
        query = """
            SELECT id, snapshot_id, store_id, device_id, timestamp,
                   entries, exits, occupancy, peak_occupancy,
                   queue_length, peak_queue,
                   avg_dwell, max_dwell, avg_wait, created_at
            FROM analytics_snapshots
        """
        params: List[Any] = []
        conditions: List[str] = []

        if store_id:
            conditions.append("store_id = %s")
            params.append(store_id)
        if device_id:
            conditions.append("device_id = %s")
            params.append(device_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY id DESC LIMIT %s;"
        params.append(max(1, min(1000, int(limit))))

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, tuple(params))
                rows = cur.fetchall()
                return [dict(r) for r in rows]

    def get_sync_status(self) -> Dict[str, Any]:
        """Return central status metrics for edge synchronizations."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*),
                        COUNT(DISTINCT store_id),
                        COUNT(DISTINCT device_id),
                        MAX(created_at)
                    FROM analytics_snapshots;
                    """
                )
                total, stores, devices, latest_ts = cur.fetchone()
                return {
                    "total_snapshots": total or 0,
                    "stores": stores or 0,
                    "devices": devices or 0,
                    "latest_timestamp": latest_ts,
                    "status": "receiving" if (total and total > 0) else "ready",
                }

    # Phase 8: Zone Batch Ingestion & Querying
    def insert_zones_batch(
        self,
        device_id: str,
        zone_snapshots: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Insert a batch of zone snapshots idempotently into PostgreSQL."""
        inserted = 0
        already_synced = 0
        failed = 0
        synced_ids: List[int] = []

        if not zone_snapshots:
            return {"received": 0, "inserted": 0, "already_synced": 0, "failed": 0, "synced_ids": []}

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for item in zone_snapshots:
                    local_id = item.get("local_id")
                    dev_id = item.get("device_id") or device_id
                    cam_id = item.get("camera_id") or "CAM_01"
                    z_id = item.get("zone_id") or "zone"
                    ts = str(item.get("timestamp", ""))
                    snapshot_id = item.get("snapshot_id") or f"{dev_id}:{cam_id}:{z_id}:{ts}"
                    created_at = str(item.get("created_at") or datetime.now(timezone.utc).isoformat())

                    try:
                        cur.execute(
                            """
                            INSERT INTO zone_snapshots (
                                snapshot_id, store_id, device_id, camera_id,
                                zone_id, zone_name, timestamp,
                                current_shoppers, peak_shoppers, avg_dwell,
                                traffic_level, expected_staff, created_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (snapshot_id) DO NOTHING
                            RETURNING id;
                            """,
                            (
                                snapshot_id,
                                str(item.get("store_id", "store_001")),
                                dev_id,
                                cam_id,
                                z_id,
                                str(item.get("zone_name", z_id)),
                                ts,
                                int(item.get("current_shoppers", 0)),
                                int(item.get("peak_shoppers", 0)),
                                float(item.get("avg_dwell", 0.0)),
                                str(item.get("traffic_level", "LOW")),
                                int(item.get("expected_staff", 1)),
                                created_at,
                            ),
                        )
                        res = cur.fetchone()
                        if res is not None:
                            inserted += 1
                        else:
                            already_synced += 1
                        if local_id is not None:
                            synced_ids.append(local_id)
                    except Exception as e:
                        logger.error(f"Failed to insert zone snapshot {snapshot_id}: {e}")
                        failed += 1
            conn.commit()

        return {
            "received": len(zone_snapshots),
            "inserted": inserted,
            "already_synced": already_synced,
            "failed": failed,
            "synced_ids": synced_ids,
        }

    # Phase 8: Alert Batch Ingestion & Querying
    def insert_alerts_batch(
        self,
        device_id: str,
        alerts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Insert a batch of operational retail alerts idempotently into PostgreSQL."""
        inserted = 0
        already_synced = 0
        failed = 0
        synced_ids: List[int] = []

        if not alerts:
            return {"received": 0, "inserted": 0, "already_synced": 0, "failed": 0, "synced_ids": []}

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for item in alerts:
                    local_id = item.get("local_id")
                    alert_id = str(item.get("alert_id"))
                    created_at = str(item.get("created_at") or datetime.now(timezone.utc).isoformat())

                    try:
                        cur.execute(
                            """
                            INSERT INTO analytics_alerts (
                                alert_id, store_id, device_id, camera_id, zone_id,
                                type, severity, title, message, current_value,
                                predicted_value, threshold, recommendation, status,
                                created_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (alert_id) DO UPDATE SET
                                status = EXCLUDED.status,
                                current_value = EXCLUDED.current_value,
                                predicted_value = EXCLUDED.predicted_value
                            RETURNING id;
                            """,
                            (
                                alert_id,
                                str(item.get("store_id", "store_001")),
                                str(item.get("device_id", device_id)),
                                str(item.get("camera_id", "CAM_01")),
                                str(item.get("zone_id", "zone")),
                                str(item.get("type", "GENERIC")),
                                str(item.get("severity", "MEDIUM")),
                                str(item.get("title", "")),
                                str(item.get("message", "")),
                                float(item.get("current_value", 0.0)),
                                float(item["predicted_value"]) if item.get("predicted_value") is not None else None,
                                float(item.get("threshold", 0.0)),
                                str(item.get("recommendation", "")),
                                str(item.get("status", "ACTIVE")),
                                created_at,
                            ),
                        )
                        res = cur.fetchone()
                        if res is not None:
                            inserted += 1
                        else:
                            already_synced += 1
                        if local_id is not None:
                            synced_ids.append(local_id)
                    except Exception as e:
                        logger.error(f"Failed to insert alert {alert_id}: {e}")
                        failed += 1
            conn.commit()

        return {
            "received": len(alerts),
            "inserted": inserted,
            "already_synced": already_synced,
            "failed": failed,
            "synced_ids": synced_ids,
        }

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an active alert as RESOLVED in PostgreSQL."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE analytics_alerts SET status = 'RESOLVED' WHERE alert_id = %s AND status = 'ACTIVE';",
                    (str(alert_id),),
                )
                conn.commit()
                return cur.rowcount > 0

    def resolve_alerts_by_type(self, alert_type: str, zone_id: Optional[str] = None) -> int:
        """Mark active alerts of a given type (and optional zone) as RESOLVED."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                if zone_id:
                    cur.execute(
                        "UPDATE analytics_alerts SET status = 'RESOLVED' WHERE type = %s AND zone_id = %s AND status = 'ACTIVE';",
                        (str(alert_type).upper(), str(zone_id)),
                    )
                else:
                    cur.execute(
                        "UPDATE analytics_alerts SET status = 'RESOLVED' WHERE type = %s AND status = 'ACTIVE';",
                        (str(alert_type).upper(),),
                    )
                conn.commit()
                return cur.rowcount

    def get_alerts(
        self,
        severity: Optional[str] = None,
        alert_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve operational alerts from central PostgreSQL."""
        query = "SELECT * FROM analytics_alerts"
        conditions: List[str] = []
        params: List[Any] = []

        if severity:
            conditions.append("severity = %s")
            params.append(str(severity).upper())
        if alert_type:
            conditions.append("type = %s")
            params.append(str(alert_type).upper())
        if status:
            conditions.append("status = %s")
            params.append(str(status).upper())

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY id DESC LIMIT %s;"
        params.append(max(1, min(1000, int(limit))))

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, tuple(params))
                rows = cur.fetchall()
                return [dict(r) for r in rows]

    def get_latest_zones(self) -> List[Dict[str, Any]]:
        """Retrieve the most recent zone snapshots for each distinct zone."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
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
                rows = cur.fetchall()
                return [dict(r) for r in rows]

    def get_zone_snapshots(
        self,
        zone_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve historical zone snapshots with optional zone_id filter."""
        query = "SELECT * FROM zone_snapshots"
        params: List[Any] = []
        if zone_id:
            query += " WHERE zone_id = %s"
            params.append(str(zone_id))
        query += " ORDER BY id DESC LIMIT %s;"
        params.append(max(1, min(1000, int(limit))))

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, tuple(params))
                rows = cur.fetchall()
                return [dict(r) for r in rows]

    def get_latest_departments(self) -> List[Dict[str, Any]]:
        """Retrieve the latest metrics for each department from central PostgreSQL."""
        zones = self.get_latest_zones()
        results = []
        for z in zones:
            d = dict(z)
            staff = max(1, int(d.get("expected_staff", 1) or 1))
            shoppers = int(d.get("current_shoppers", 0) or 0)
            d["shopper_load_per_staff"] = round(shoppers / staff, 2)
            d["department"] = d.get("zone_name") or d.get("zone_id", "").capitalize()
            results.append(d)
        return results

    def get_hourly_department_traffic(self, limit: int = 2000) -> Dict[str, Any]:
        """Calculate hourly traffic per department from stored PostgreSQL observations."""
        snapshots = self.get_zone_snapshots(limit=limit)
        if not snapshots or len(snapshots) < 3:
            return {
                "status": "insufficient_data",
                "message": "Insufficient stored observations to calculate hourly traffic",
                "departments": {},
                "peak_department": None,
                "peak_hour": None,
                "overall_trend": "STABLE",
                "total_observations": len(snapshots) if snapshots else 0,
            }

        from collections import defaultdict

        dept_map: Dict[str, Dict[str, Any]] = {}

        for s in snapshots:
            z_id = s.get("zone_id", "unknown")
            z_name = s.get("zone_name", z_id.capitalize())
            cam_id = s.get("camera_id", "")
            shoppers = int(s.get("current_shoppers", 0) or 0)
            ts_str = s.get("timestamp") or s.get("created_at") or ""

            hour_int: Optional[int] = None
            if ":" in str(ts_str) and not str(ts_str).startswith("T+"):
                try:
                    if "T" in str(ts_str):
                        clean_ts = str(ts_str).replace("Z", "+00:00")
                        dt = datetime.fromisoformat(clean_ts)
                        hour_int = dt.hour
                    else:
                        parts = str(ts_str).split(":")
                        hour_int = int(parts[0]) % 24
                except Exception:
                    pass

            if hour_int is None:
                continue

            if z_id not in dept_map:
                dept_map[z_id] = {
                    "zone_id": z_id,
                    "name": z_name,
                    "camera_id": cam_id,
                    "hourly_shoppers": defaultdict(list),
                    "all_shoppers": [],
                }
            dept_map[z_id]["hourly_shoppers"][hour_int].append(shoppers)
            dept_map[z_id]["all_shoppers"].append(shoppers)

        if not dept_map:
            return {
                "status": "insufficient_data",
                "message": "No valid timestamps found in stored observations",
                "departments": {},
                "peak_department": None,
                "peak_hour": None,
                "overall_trend": "STABLE",
                "total_observations": 0,
            }

        departments_result = {}
        dept_peak_totals = {}
        hour_overall_totals = defaultdict(list)

        def format_hour_label(h: int) -> str:
            period = "AM" if h < 12 else "PM"
            display_h = h % 12
            if display_h == 0:
                display_h = 12
            return f"{display_h} {period}"

        for z_id, info in dept_map.items():
            hourly_summary = {}
            peak_hour_for_dept = None
            dept_max_avg = -1.0
            dept_peak_val = 0

            for h in sorted(info["hourly_shoppers"].keys()):
                vals = info["hourly_shoppers"][h]
                avg_val = round(sum(vals) / len(vals), 1)
                max_val = max(vals)
                dept_peak_val = max(dept_peak_val, max_val)
                h_label = format_hour_label(h)
                hourly_summary[h_label] = {
                    "hour": h,
                    "avg_shoppers": avg_val,
                    "peak_shoppers": max_val,
                    "sample_count": len(vals),
                }
                hour_overall_totals[h].extend(vals)
                if avg_val > dept_max_avg:
                    dept_max_avg = avg_val
                    peak_hour_for_dept = h_label

            recent = info["all_shoppers"][-10:] if len(info["all_shoppers"]) >= 2 else info["all_shoppers"]
            trend = "STABLE"
            if len(recent) >= 2:
                if recent[-1] > recent[0]:
                    trend = "GROWING"
                elif recent[-1] < recent[0]:
                    trend = "SHRINKING"

            dept_peak_totals[info["name"]] = dept_max_avg
            departments_result[z_id] = {
                "zone_id": z_id,
                "name": info["name"],
                "camera_id": info["camera_id"],
                "hourly": hourly_summary,
                "peak_hour": peak_hour_for_dept,
                "peak_shoppers": dept_peak_val,
                "trend": trend,
            }

        peak_dept = max(dept_peak_totals, key=dept_peak_totals.get) if dept_peak_totals else "N/A"

        overall_peak_hour = None
        overall_max_h_avg = -1.0
        for h, vals in hour_overall_totals.items():
            avg_h = sum(vals) / len(vals)
            if avg_h > overall_max_h_avg:
                overall_max_h_avg = avg_h
                overall_peak_hour = format_hour_label(h)

        return {
            "status": "success",
            "departments": departments_result,
            "peak_department": peak_dept,
            "peak_hour": overall_peak_hour,
            "busiest_hour": overall_peak_hour,
            "overall_trend": "GROWING" if any(d.get("trend") == "GROWING" for d in departments_result.values()) else "STABLE",
            "total_observations": len(snapshots),
        }

    def close(self) -> None:

        """Close connection pool cleanly."""
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None
