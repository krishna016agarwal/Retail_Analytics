"""Automated Unit & Integration Tests for Phase 6B Local FastAPI Edge API.

Tests root banner, health check (200 & 503), latest snapshot, snapshot listing,
summary aggregation, and sync status for both populated and empty edge databases.
"""

import os
from pathlib import Path
import sys
import tempfile
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.api import app, set_db
from src.database import EdgeDatabase


class TestEdgeAPIWithExistingDB(unittest.TestCase):
    """Test suite using existing edge database or populated temporary database."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_endpoint(self):
        """Test GET / returns 200 and valid service banner."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("service"), "Retail Analytics Edge API")
        self.assertEqual(data.get("status"), "online")

    def test_health_endpoint_healthy(self):
        """Test GET /health returns 200 and database: connected."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("database"), "connected")

    def test_latest_snapshot(self):
        """Test GET /api/latest returns snapshot record."""
        response = self.client.get("/api/latest")
        # Should be 200 if retail_edge.db has data, or 404 if empty
        self.assertIn(response.status_code, [200, 404])
        if response.status_code == 200:
            data = response.json()
            self.assertIn("store_id", data)
            self.assertIn("device_id", data)
            self.assertIn("timestamp", data)
            self.assertIn("occupancy", data)
            self.assertIn("sync_status", data)

    def test_snapshots_query_limit(self):
        """Test GET /api/snapshots returns list within limit."""
        response = self.client.get("/api/snapshots?limit=5")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertLessEqual(len(data), 5)

    def test_snapshots_max_limit_enforced(self):
        """Test GET /api/snapshots enforces limit <= 1000."""
        # Querying with limit > 1000 should return 422 Unprocessable Entity
        response = self.client.get("/api/snapshots?limit=1001")
        self.assertEqual(response.status_code, 422)

    def test_summary_endpoint(self):
        """Test GET /api/summary returns all required aggregate fields."""
        response = self.client.get("/api/summary")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        required_keys = [
            "latest_timestamp",
            "entries",
            "exits",
            "current_occupancy",
            "peak_occupancy",
            "current_queue",
            "peak_queue",
            "average_dwell",
            "maximum_dwell",
            "average_wait",
            "total_snapshots",
            "pending_sync_count",
        ]
        for key in required_keys:
            self.assertIn(key, data, f"Missing key in summary response: {key}")

    def test_sync_status_endpoint(self):
        """Test GET /api/sync/status returns pending count and sync_required boolean."""
        response = self.client.get("/api/sync/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("pending_snapshots", data)
        self.assertIn("sync_required", data)
        self.assertIsInstance(data["pending_snapshots"], int)
        self.assertIsInstance(data["sync_required"], bool)


class TestEdgeAPIWithEmptyAndUnhealthyDB(unittest.TestCase):
    """Test suite verifying empty-database and disconnected-database edge cases."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_db_path = Path(self.temp_dir.name) / "empty_test.db"
        self.empty_db = EdgeDatabase(db_path=self.temp_db_path)
        set_db(self.empty_db)
        self.client = TestClient(app)

    def tearDown(self):
        self.empty_db.close()
        # Restore default database
        set_db(EdgeDatabase("data/retail_edge.db"))
        self.temp_dir.cleanup()

    def test_empty_db_latest_snapshot_returns_404(self):
        """Verify empty database returns clean 404 for /api/latest."""
        response = self.client.get("/api/latest")
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertEqual(data.get("detail"), "No snapshots found")

    def test_empty_db_snapshots_returns_empty_list(self):
        """Verify empty database returns empty list [] for /api/snapshots."""
        response = self.client.get("/api/snapshots?limit=100")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data, [])

    def test_empty_db_summary_returns_sensible_zeros(self):
        """Verify empty database returns sensible zero/null metrics for /api/summary."""
        response = self.client.get("/api/summary")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsNone(data.get("latest_timestamp"))
        self.assertEqual(data.get("entries"), 0)
        self.assertEqual(data.get("exits"), 0)
        self.assertEqual(data.get("current_occupancy"), 0)
        self.assertEqual(data.get("peak_occupancy"), 0)
        self.assertEqual(data.get("current_queue"), 0)
        self.assertEqual(data.get("peak_queue"), 0)
        self.assertEqual(data.get("average_dwell"), 0.0)
        self.assertEqual(data.get("maximum_dwell"), 0.0)
        self.assertEqual(data.get("average_wait"), 0.0)
        self.assertEqual(data.get("total_snapshots"), 0)
        self.assertEqual(data.get("pending_sync_count"), 0)

    def test_empty_db_sync_status_returns_zero_and_false(self):
        """Verify empty database returns pending_snapshots=0 and sync_required=false."""
        response = self.client.get("/api/sync/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("pending_snapshots"), 0)
        self.assertEqual(data.get("sync_required"), False)

    def test_unhealthy_db_returns_503(self):
        """Verify that when database is inaccessible, /health returns HTTP 503."""
        class DisconnectedDB:
            def check_connection(self):
                return False

        set_db(DisconnectedDB())  # type: ignore
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertEqual(data.get("status"), "unhealthy")
        self.assertEqual(data.get("database"), "disconnected")


if __name__ == "__main__":
    unittest.main()
