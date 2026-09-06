"""Automated Unit & Integration Tests for Central FastAPI Cloud API (Phase 6C).

Tests root banner, health check (200 & 503), batch ingestion, idempotent deduplication,
latest analytics, history filtering, and central sync status.
Uses mock / test-double database layer so tests run offline without requiring Render PostgreSQL.
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.central_api import app, set_central_db
from src.central_db import CentralDatabase


class MockCentralDatabase:
    """In-memory test double mimicking CentralDatabase behavior."""

    def __init__(self, healthy: bool = True):
        self.healthy = healthy
        self.records = []
        self._next_id = 1

    def check_connection(self) -> bool:
        return self.healthy

    def insert_batch(self, device_id: str, snapshots: list) -> dict:
        if not self.healthy:
            raise RuntimeError("Database connection error")

        inserted = 0
        already_synced = 0
        synced_ids = []

        for item in snapshots:
            local_id = item.get("local_id")
            dev_id = item.get("device_id") or device_id
            snapshot_id = item.get("snapshot_id") or f"{dev_id}:{local_id}"

            # Check for existing snapshot_id (uniqueness / idempotency)
            existing = any(r.get("snapshot_id") == snapshot_id for r in self.records)
            if existing:
                already_synced += 1
            else:
                record = dict(item)
                record["id"] = self._next_id
                self._next_id += 1
                record["snapshot_id"] = snapshot_id
                record["device_id"] = dev_id
                self.records.append(record)
                inserted += 1

            if local_id is not None:
                synced_ids.append(local_id)

        return {
            "received": len(snapshots),
            "inserted": inserted,
            "already_synced": already_synced,
            "failed": 0,
            "synced_ids": synced_ids,
        }

    def get_latest_snapshot(self):
        if not self.records:
            return None
        return dict(self.records[-1])

    def get_snapshots(self, store_id=None, device_id=None, limit=100):
        res = list(self.records)
        if store_id:
            res = [r for r in res if r.get("store_id") == store_id]
        if device_id:
            res = [r for r in res if r.get("device_id") == device_id]
        res.reverse()  # DESC
        return res[:limit]

    def get_sync_status(self):
        stores = len(set(r.get("store_id") for r in self.records))
        devices = len(set(r.get("device_id") for r in self.records))
        latest_ts = self.records[-1].get("timestamp") if self.records else None
        return {
            "total_snapshots": len(self.records),
            "stores": stores,
            "devices": devices,
            "latest_timestamp": latest_ts,
            "status": "receiving" if self.records else "ready",
        }

    def insert_zones_batch(self, device_id: str, zone_snapshots: list) -> dict:
        if not hasattr(self, "zones_records"):
            self.zones_records = []
        synced_ids = []
        for z in zone_snapshots:
            self.zones_records.append(dict(z))
            if z.get("local_id") is not None:
                synced_ids.append(z["local_id"])
        return {
            "received": len(zone_snapshots),
            "inserted": len(zone_snapshots),
            "already_synced": 0,
            "failed": 0,
            "synced_ids": synced_ids,
        }

    def insert_alerts_batch(self, device_id: str, alerts: list) -> dict:
        if not hasattr(self, "alerts_records"):
            self.alerts_records = []
        synced_ids = []
        for a in alerts:
            self.alerts_records.append(dict(a))
            if a.get("local_id") is not None:
                synced_ids.append(a["local_id"])
        return {
            "received": len(alerts),
            "inserted": len(alerts),
            "already_synced": 0,
            "failed": 0,
            "synced_ids": synced_ids,
        }

    def get_alerts(self, severity=None, alert_type=None, status=None, limit=100):
        if not hasattr(self, "alerts_records"):
            self.alerts_records = []
        res = list(self.alerts_records)
        if severity:
            res = [a for a in res if a.get("severity") == severity]
        if alert_type:
            res = [a for a in res if a.get("type") == alert_type]
        if status:
            res = [a for a in res if a.get("status") == status]
        return res[:limit]

    def get_latest_zones(self):
        if not hasattr(self, "zones_records"):
            self.zones_records = []
        return list(self.zones_records)

    def get_zone_snapshots(self, zone_id=None, limit=100):
        if not hasattr(self, "zones_records"):
            self.zones_records = []
        res = list(self.zones_records)
        if zone_id:
            res = [z for z in res if z.get("zone_id") == zone_id]
        return res[:limit]



class TestCentralAPI(unittest.TestCase):
    """Test suite for Central FastAPI endpoints."""

    def setUp(self):
        self.mock_db = MockCentralDatabase(healthy=True)
        set_central_db(self.mock_db)  # type: ignore
        self.client = TestClient(app)

    def test_root_endpoint(self):
        """Test GET / returns 200 and central service banner."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("service"), "Retail Analytics Central Cloud API")
        self.assertEqual(data.get("status"), "online")
        self.assertIn("Render", data.get("platform", ""))

    def test_health_endpoint_healthy(self):
        """Test GET /health returns 200 when database is connected."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("database"), "connected")

    def test_health_endpoint_unhealthy(self):
        """Test GET /health returns 503 when database is disconnected."""
        unhealthy_db = MockCentralDatabase(healthy=False)
        set_central_db(unhealthy_db)  # type: ignore
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertEqual(data.get("status"), "unhealthy")
        self.assertEqual(data.get("database"), "disconnected")

    def test_batch_insertion_success(self):
        """Test POST /api/v1/analytics/batch inserts new snapshots and returns synced_ids."""
        payload = {
            "device_id": "edge_device_01",
            "snapshots": [
                {
                    "local_id": 1,
                    "store_id": "store_001",
                    "device_id": "edge_device_01",
                    "timestamp": "T+5.00s",
                    "entries": 2,
                    "exits": 0,
                    "occupancy": 2,
                    "peak_occupancy": 2,
                    "queue_length": 1,
                    "peak_queue": 1,
                    "avg_dwell": 0.0,
                    "max_dwell": 0.0,
                    "avg_wait": 2.5,
                },
                {
                    "local_id": 2,
                    "store_id": "store_001",
                    "device_id": "edge_device_01",
                    "timestamp": "T+10.00s",
                    "entries": 4,
                    "exits": 1,
                    "occupancy": 3,
                    "peak_occupancy": 3,
                    "queue_length": 2,
                    "peak_queue": 2,
                    "avg_dwell": 12.0,
                    "max_dwell": 15.0,
                    "avg_wait": 3.0,
                },
            ],
        }
        response = self.client.post("/api/v1/analytics/batch", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("received"), 2)
        self.assertEqual(data.get("inserted"), 2)
        self.assertEqual(data.get("already_synced"), 0)
        self.assertEqual(data.get("synced_ids"), [1, 2])

    def test_batch_insertion_idempotent_duplicate(self):
        """Test sending duplicate snapshots recognizes them as already_synced safely."""
        payload = {
            "device_id": "edge_device_01",
            "snapshots": [
                {
                    "local_id": 10,
                    "store_id": "store_001",
                    "device_id": "edge_device_01",
                    "timestamp": "T+15.00s",
                    "entries": 5,
                    "exits": 2,
                    "occupancy": 3,
                }
            ],
        }
        # First send
        res1 = self.client.post("/api/v1/analytics/batch", json=payload)
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res1.json().get("inserted"), 1)
        self.assertEqual(res1.json().get("already_synced"), 0)

        # Repeated send (e.g. edge retry)
        res2 = self.client.post("/api/v1/analytics/batch", json=payload)
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertTrue(data2.get("success"))
        self.assertEqual(data2.get("inserted"), 0)
        self.assertEqual(data2.get("already_synced"), 1)
        self.assertIn(10, data2.get("synced_ids"))

    def test_latest_snapshot_endpoint(self):
        """Test GET /api/v1/analytics/latest returns 404 when empty and 200 when populated."""
        # Empty DB returns 404
        empty_res = self.client.get("/api/v1/analytics/latest")
        self.assertEqual(empty_res.status_code, 404)

        # Insert one record
        self.client.post(
            "/api/v1/analytics/batch",
            json={
                "device_id": "edge_device_01",
                "snapshots": [
                    {
                        "local_id": 1,
                        "store_id": "store_test",
                        "timestamp": "T+5.00s",
                        "occupancy": 7,
                    }
                ],
            },
        )
        pop_res = self.client.get("/api/v1/analytics/latest")
        self.assertEqual(pop_res.status_code, 200)
        pop_data = pop_res.json()
        self.assertEqual(pop_data.get("store_id"), "store_test")
        self.assertEqual(pop_data.get("occupancy"), 7)

    def test_query_analytics_history_and_filters(self):
        """Test GET /api/v1/analytics with store_id and limit query parameters."""
        payload = {
            "device_id": "dev_A",
            "snapshots": [
                {"local_id": 1, "store_id": "store_1", "timestamp": "1.0"},
                {"local_id": 2, "store_id": "store_2", "timestamp": "2.0"},
                {"local_id": 3, "store_id": "store_1", "timestamp": "3.0"},
            ],
        }
        self.client.post("/api/v1/analytics/batch", json=payload)

        # Filter by store_1
        res = self.client.get("/api/v1/analytics?store_id=store_1")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data), 2)
        for row in data:
            self.assertEqual(row["store_id"], "store_1")

        # Limit enforcement
        limit_res = self.client.get("/api/v1/analytics?limit=1")
        self.assertEqual(limit_res.status_code, 200)
        self.assertEqual(len(limit_res.json()), 1)

    def test_sync_status_endpoint(self):
        """Test GET /api/v1/sync/status returns aggregated store/device metrics."""
        payload = {
            "device_id": "dev_01",
            "snapshots": [
                {"local_id": 1, "store_id": "store_001", "timestamp": "10.0"},
                {"local_id": 2, "store_id": "store_002", "timestamp": "20.0"},
            ],
        }
        self.client.post("/api/v1/analytics/batch", json=payload)

        res = self.client.get("/api/v1/sync/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("total_snapshots"), 2)
        self.assertEqual(data.get("stores"), 2)
        self.assertEqual(data.get("devices"), 1)
        self.assertEqual(data.get("status"), "receiving")

    # Phase 8 Central API Tests
    def test_central_intelligence_latest_endpoint(self):
        """Test GET /api/v1/intelligence/latest returns central intelligence bundle."""
        res = self.client.get("/api/v1/intelligence/latest")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["platform"], "Central Cloud API")
        self.assertIn("zones", data)
        self.assertIn("active_alerts", data)

    def test_central_alerts_endpoint(self):
        """Test GET /api/v1/alerts with filtering."""
        # Post batch with alerts
        payload = {
            "device_id": "edge_01",
            "alerts": [
                {
                    "local_id": 1,
                    "alert_id": "alt_1",
                    "store_id": "store_001",
                    "camera_id": "CAM_05",
                    "zone_id": "checkout",
                    "type": "QUEUE_CONGESTION",
                    "severity": "HIGH",
                    "title": "Congestion predicted",
                    "message": "Growing queue",
                    "current_value": 6.0,
                    "predicted_value": 12.0,
                    "threshold": 6.0,
                    "recommendation": "Open additional counter",
                    "status": "ACTIVE",
                }
            ],
        }
        post_res = self.client.post("/api/v1/analytics/batch", json=payload)
        self.assertEqual(post_res.status_code, 200)

        # Query alerts
        res = self.client.get("/api/v1/alerts?type=QUEUE_CONGESTION")
        self.assertEqual(res.status_code, 200)
        alerts = res.json()
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["title"], "Congestion predicted")

    def test_central_zones_endpoint(self):
        """Test GET /api/v1/zones returns monitored zones."""
        payload = {
            "device_id": "edge_01",
            "zone_snapshots": [
                {
                    "local_id": 1,
                    "snapshot_id": "snap_z_1",
                    "zone_id": "food",
                    "zone_name": "Food Section",
                    "timestamp": "T+5.0s",
                    "current_shoppers": 8,
                    "expected_staff": 2,
                }
            ],
        }
        self.client.post("/api/v1/analytics/batch", json=payload)

        res = self.client.get("/api/v1/zones")
        self.assertEqual(res.status_code, 200)
        zones = res.json()
        self.assertEqual(len(zones), 1)
        self.assertEqual(zones[0]["zone_id"], "food")

    def test_central_patterns_endpoint_insufficient_data(self):
        """Test GET /api/v1/patterns returns insufficient data fallback when empty."""
        res = self.client.get("/api/v1/patterns")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "insufficient_data")
        self.assertEqual(data["message"], "Insufficient historical data")

    def test_central_departments_endpoint(self):
        """Test GET /api/v1/departments returns department list."""
        res = self.client.get("/api/v1/departments")
        self.assertEqual(res.status_code, 200)
        depts = res.json()
        self.assertIsInstance(depts, list)
        self.assertTrue(len(depts) >= 3)
        self.assertTrue(any(d.get("department") == "Food" for d in depts))

    def test_central_cameras_endpoint(self):
        """Test GET /api/v1/cameras returns 4 camera definitions."""
        res = self.client.get("/api/v1/cameras")
        self.assertEqual(res.status_code, 200)
        cameras = res.json()
        self.assertIsInstance(cameras, list)
        self.assertEqual(len(cameras), 4)

    def test_central_hourly_patterns_endpoint(self):
        """Test GET /api/v1/patterns/hourly returns structure."""
        res = self.client.get("/api/v1/patterns/hourly")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("status", data)

    def test_central_simulation_clock_endpoint(self):
        """Test GET /api/v1/simulation/clock returns simulation metadata."""
        res = self.client.get("/api/v1/simulation/clock")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("is_simulation"))
        self.assertIn("simulated_store_time", data)


if __name__ == "__main__":
    unittest.main()

