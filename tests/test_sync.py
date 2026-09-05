"""Automated Unit & Integration Tests for Edge Sync Client (Phase 6C).

Tests offline-first edge-to-cloud synchronization:
- Network failure safety: SQLite records remain strictly PENDING on connection errors.
- Successful sync: SQLite records transition from PENDING to SYNCED.
- Partial batch confirmation: Only confirmed IDs transition to SYNCED.
- Idempotency & retries.
- Zero data loss: Records are never deleted on failure.
"""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import EdgeDatabase
from src.sync import BackgroundSyncThread, SyncClient, SyncResult


class TestEdgeSyncClient(unittest.TestCase):
    """Test suite verifying SyncClient functionality and offline-first safety."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_db_path = Path(self.temp_dir.name) / "test_edge.db"
        self.edge_db = EdgeDatabase(db_path=self.temp_db_path)
        self.client = SyncClient(
            central_api_url="https://mock-render-api.onrender.com",
            edge_db=self.edge_db,
            timeout=2.0,
        )

    def tearDown(self):
        self.edge_db.close()
        self.temp_dir.cleanup()

    def _insert_sample_records(self, count: int = 5) -> list:
        ids = []
        for i in range(1, count + 1):
            row_id = self.edge_db.insert_snapshot(
                store_id="store_001",
                device_id="edge_device_01",
                timestamp=f"T+{i * 5}.00s",
                entries=i * 2,
                exits=i,
                occupancy=i,
                peak_occupancy=i + 1,
                queue_length=i % 3,
                peak_queue=2,
                avg_dwell=10.0,
                max_dwell=20.0,
                avg_wait=3.5,
                sync_status="PENDING",
            )
            ids.append(row_id)
        return ids

    @patch("requests.get")
    def test_check_server_health_success(self, mock_get):
        """Test health check returns True when central server is reachable and healthy."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "healthy", "database": "connected"}
        mock_get.return_value = mock_resp

        self.assertTrue(self.client.check_server_health())

    @patch("requests.get")
    def test_check_server_health_failure(self, mock_get):
        """Test health check returns False when network connection fails."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        self.assertFalse(self.client.check_server_health())

    @patch("requests.post")
    def test_successful_sync_marks_records_synced(self, mock_post):
        """Test successful sync marks confirmed records SYNCED in SQLite."""
        ids = self._insert_sample_records(count=3)
        self.assertEqual(self.edge_db.get_unsynced_count(), 3)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "success": True,
            "received": 3,
            "inserted": 3,
            "already_synced": 0,
            "failed": 0,
            "synced_ids": ids,
        }
        mock_post.return_value = mock_resp

        result = self.client.sync_all(batch_size=10, verbose=False)

        self.assertTrue(result.success)
        self.assertEqual(result.inserted, 3)
        self.assertEqual(result.remaining_pending, 0)
        self.assertEqual(self.edge_db.get_unsynced_count(), 0)

        # Verify all snapshots now have sync_status = SYNCED
        snapshots = self.edge_db.get_snapshots(limit=10)
        for s in snapshots:
            self.assertEqual(s["sync_status"], "SYNCED")

    @patch("requests.post")
    def test_network_failure_preserves_pending_records(self, mock_post):
        """CRITICAL: Test network failure keeps all SQLite records strictly PENDING."""
        ids = self._insert_sample_records(count=4)
        initial_pending = self.edge_db.get_unsynced_count()
        self.assertEqual(initial_pending, 4)

        # Simulate ConnectionError (offline / Render asleep / no internet)
        mock_post.side_effect = requests.exceptions.ConnectionError(
            "HTTPSConnectionPool: Max retries exceeded"
        )

        result = self.client.sync_all(batch_size=10, verbose=False)

        self.assertFalse(result.success)
        self.assertTrue(result.offline_mode)
        # All 4 records must remain PENDING
        self.assertEqual(self.edge_db.get_unsynced_count(), 4)
        # Zero records deleted
        self.assertEqual(self.edge_db.get_total_count(), 4)

        # Verify each record is still PENDING
        pending_records = self.edge_db.get_pending_snapshots()
        self.assertEqual(len(pending_records), 4)
        for r in pending_records:
            self.assertEqual(r["sync_status"], "PENDING")

    @patch("requests.post")
    def test_http_500_preserves_pending_records(self, mock_post):
        """Test HTTP 500 error from central server safely retains PENDING records."""
        self._insert_sample_records(count=3)
        self.assertEqual(self.edge_db.get_unsynced_count(), 3)

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_post.return_value = mock_resp

        result = self.client.sync_all(batch_size=10, verbose=False)

        self.assertFalse(result.success)
        self.assertEqual(self.edge_db.get_unsynced_count(), 3)

    @patch("requests.post")
    def test_partial_batch_confirmation(self, mock_post):
        """Test only confirmed IDs become SYNCED, unconfirmed IDs remain PENDING."""
        ids = self._insert_sample_records(count=3)  # [1, 2, 3]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # Server only confirms first 2
        mock_resp.json.return_value = {
            "success": True,
            "received": 3,
            "inserted": 2,
            "already_synced": 0,
            "failed": 1,
            "synced_ids": [ids[0], ids[1]],
        }
        mock_post.return_value = mock_resp

        result = self.client.sync_all(batch_size=10, verbose=False)
        self.assertTrue(result.success)
        # 1 record remains PENDING
        self.assertEqual(self.edge_db.get_unsynced_count(), 1)
        remaining = self.edge_db.get_pending_snapshots()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["id"], ids[2])

    @patch("requests.post")
    def test_idempotent_duplicate_upload(self, mock_post):
        """Test already_synced response from server correctly marks local records SYNCED."""
        ids = self._insert_sample_records(count=2)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # Central server returns already_synced: 2, but includes local IDs in synced_ids
        mock_resp.json.return_value = {
            "success": True,
            "received": 2,
            "inserted": 0,
            "already_synced": 2,
            "failed": 0,
            "synced_ids": ids,
        }
        mock_post.return_value = mock_resp

        result = self.client.sync_all(batch_size=10, verbose=False)
        self.assertTrue(result.success)
        self.assertEqual(result.already_synced, 2)
        self.assertEqual(self.edge_db.get_unsynced_count(), 0)

    def test_empty_pending_sync_returns_safely(self):
        """Test syncing when edge database has 0 pending records."""
        result = self.client.sync_all(batch_size=10, verbose=False)
        self.assertTrue(result.success)
        self.assertEqual(result.pending_count, 0)
        self.assertEqual(result.remaining_pending, 0)

    def test_background_sync_worker_start_stop(self):
        """Test BackgroundSyncThread starts and terminates cleanly."""
        worker = BackgroundSyncThread(
            sync_client=self.client,
            interval_seconds=60.0,
            batch_size=50,
        )
        worker.start()
        self.assertIsNotNone(worker._thread)
        self.assertTrue(worker._thread.is_alive())
        worker.stop()
        self.assertIsNone(worker._thread)

    def test_sync_client_uses_env_var_when_no_override(self):
        """Test SyncClient picks up CENTRAL_API_URL from environment when not passed."""
        with patch.dict("os.environ", {"CENTRAL_API_URL": "https://env-url.onrender.com"}):
            client = SyncClient(central_api_url=None, edge_db=self.edge_db)
            self.assertEqual(client.central_api_url, "https://env-url.onrender.com")

    def test_sync_client_cli_override_takes_priority(self):
        """Test explicit central_api_url takes priority over CENTRAL_API_URL in environment."""
        with patch.dict("os.environ", {"CENTRAL_API_URL": "https://env-url.onrender.com"}):
            client = SyncClient(
                central_api_url="https://override-url.onrender.com",
                edge_db=self.edge_db,
            )
            self.assertEqual(client.central_api_url, "https://override-url.onrender.com")

    def test_sync_client_default_fallback_when_no_env_or_arg(self):
        """Test SyncClient falls back to http://127.0.0.1:8000 if neither env nor arg is provided."""
        with patch.dict("os.environ", {}, clear=True):
            client = SyncClient(central_api_url=None, edge_db=self.edge_db)
            self.assertEqual(client.central_api_url, "http://127.0.0.1:8000")


if __name__ == "__main__":
    unittest.main()
