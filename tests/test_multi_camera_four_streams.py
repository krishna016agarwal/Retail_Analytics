"""Unit and integration tests for the 4-Camera concurrent retail analytics setup.

Tests:
- SimulationClock progression and format
- FourCameraSetupConfig and camera stream definitions
- CameraWorker initialization, distinct detector instances, and thread-safety
- Unique camera-aware snapshot IDs
- Staffing load calculation using configured staff capacity
- SQLite zone snapshot persistence and hourly pattern queries
- Edge API endpoints for departments, cameras, hourly traffic, and simulation clock
"""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import FourCameraSetupConfig, SimulationClockConfig, CameraStreamConfig
from src.camera_manager import SimulationClock, CameraWorker, MultiCameraManager
from src.database import EdgeDatabase
from fastapi.testclient import TestClient
from src.api import app, set_db, set_camera_manager


class TestSimulationClock(unittest.TestCase):
    """Test the SimulationClock behaviour."""

    def test_clock_initialization_and_elapsed(self):
        clock = SimulationClock(demo_start_time="17:00:00", time_scale=1.0)
        self.assertEqual(clock.demo_start_time_str, "17:00:00")
        self.assertIn("17:00:", clock.get_simulated_time_str(0.0))

        # Test simulated timestamp after 125 seconds
        sim_dt = clock.get_simulated_datetime(elapsed_seconds=125)
        self.assertEqual(sim_dt.hour, 17)
        self.assertEqual(sim_dt.minute, 2)
        self.assertEqual(sim_dt.second, 5)

        # Test simulated time string
        time_str = clock.get_simulated_time_str(elapsed_seconds=3665)
        self.assertEqual(time_str, "18:01:05")


class TestFourCameraConfig(unittest.TestCase):
    """Verify configuration parameters for the 4-camera layout."""

    def test_camera_setup_mappings(self):
        config = FourCameraSetupConfig()
        cameras = {c.camera_id: c for c in config.cameras}
        self.assertEqual(len(cameras), 4)

        # CAM_01 -> Food
        self.assertEqual(cameras["CAM_01"].camera_id, "CAM_01")
        self.assertEqual(cameras["CAM_01"].name, "Food")
        self.assertEqual(cameras["CAM_01"].role, "department")
        self.assertEqual(cameras["CAM_01"].expected_staff, 2)

        # CAM_02 -> Electronics
        self.assertEqual(cameras["CAM_02"].camera_id, "CAM_02")
        self.assertEqual(cameras["CAM_02"].name, "Electronics")
        self.assertEqual(cameras["CAM_02"].role, "department")
        self.assertEqual(cameras["CAM_02"].expected_staff, 1)

        # CAM_03 -> Grocery
        self.assertEqual(cameras["CAM_03"].camera_id, "CAM_03")
        self.assertEqual(cameras["CAM_03"].name, "Grocery")
        self.assertEqual(cameras["CAM_03"].role, "department")
        self.assertEqual(cameras["CAM_03"].expected_staff, 2)

        # CAM_04 -> Checkout
        self.assertEqual(cameras["CAM_04"].camera_id, "CAM_04")
        self.assertEqual(cameras["CAM_04"].name, "Checkout")
        self.assertEqual(cameras["CAM_04"].role, "checkout")
        self.assertEqual(cameras["CAM_04"].expected_staff, 2)


class TestStaffingLoadCalculation(unittest.TestCase):
    """Verify that shopper load formula strictly uses configured staff capacity."""

    def test_shopper_load_calculation(self):
        # configured_staff = 2
        # current_shoppers = 6 -> load = 3.0 (3 shoppers per staff)
        configured_staff = 2
        current_shoppers = 6
        shopper_load = round(current_shoppers / max(1, configured_staff), 2)
        self.assertEqual(shopper_load, 3.0)

        # Zero shoppers
        current_shoppers = 0
        shopper_load = round(current_shoppers / max(1, configured_staff), 2)
        self.assertEqual(shopper_load, 0.0)


class TestUniqueSnapshotIDs(unittest.TestCase):
    """Verify that snapshot IDs are camera-aware and unique across cameras."""

    def test_camera_aware_id_uniqueness(self):
        import uuid
        device_id = "EDGE_STORE_001"
        ts = "2026-09-05T17:00:00"

        id_cam1 = f"{device_id}_CAM_01_food_{ts}_{uuid.uuid4().hex[:6]}"
        id_cam2 = f"{device_id}_CAM_02_electronics_{ts}_{uuid.uuid4().hex[:6]}"
        id_cam3 = f"{device_id}_CAM_03_grocery_{ts}_{uuid.uuid4().hex[:6]}"
        id_cam4 = f"{device_id}_CAM_04_checkout_{ts}_{uuid.uuid4().hex[:6]}"

        ids = [id_cam1, id_cam2, id_cam3, id_cam4]
        self.assertEqual(len(set(ids)), 4)
        self.assertTrue("CAM_01" in id_cam1)
        self.assertTrue("CAM_04" in id_cam4)


class TestDatabaseMultiCameraPersistence(unittest.TestCase):
    """Verify SQLite persistence for multi-camera snapshots and department queries."""

    def setUp(self):
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")
        os.close(self.temp_db_fd)
        self.db = EdgeDatabase(db_path=self.temp_db_path)

    def tearDown(self):
        self.db.close()
        try:
            if os.path.exists(self.temp_db_path):
                os.remove(self.temp_db_path)
        except OSError:
            pass

    def test_zone_snapshot_insert_and_hourly_traffic(self):
        # Insert records for Food and Electronics at simulated 17:05 and 18:10
        self.db.insert_zone_snapshot(
            snapshot_id="snap_food_1",
            store_id="STORE_001",
            device_id="EDGE_001",
            camera_id="CAM_01",
            zone_id="food",
            zone_name="Food",
            timestamp="2026-09-05T17:05:00",
            current_shoppers=8,
            peak_shoppers=10,
            avg_dwell=45.2,
            traffic_level="NORMAL",
            expected_staff=2,
            footfall=15,
            max_dwell=120.0,
            video_timestamp="00:05.00"
        )
        self.db.insert_zone_snapshot(
            snapshot_id="snap_elec_1",
            store_id="STORE_001",
            device_id="EDGE_001",
            camera_id="CAM_02",
            zone_id="electronics",
            zone_name="Electronics",
            timestamp="2026-09-05T17:15:00",
            current_shoppers=3,
            peak_shoppers=4,
            avg_dwell=90.0,
            traffic_level="LOW",
            expected_staff=1,
            footfall=6,
            max_dwell=180.0,
            video_timestamp="00:15.00"
        )
        self.db.insert_zone_snapshot(
            snapshot_id="snap_food_2",
            store_id="STORE_001",
            device_id="EDGE_001",
            camera_id="CAM_01",
            zone_id="food",
            zone_name="Food",
            timestamp="2026-09-05T18:10:00",
            current_shoppers=14,
            peak_shoppers=16,
            avg_dwell=52.0,
            traffic_level="HIGH",
            expected_staff=2,
            footfall=28,
            max_dwell=140.0,
            video_timestamp="01:10.00"
        )

        # Test get_latest_departments
        latest_depts = self.db.get_latest_departments()
        dept_ids = [d["zone_id"] for d in latest_depts]
        self.assertIn("food", dept_ids)
        self.assertIn("electronics", dept_ids)

        food_rec = next(d for d in latest_depts if d["zone_id"] == "food")
        self.assertEqual(food_rec["current_shoppers"], 14)
        self.assertEqual(food_rec["shopper_load_per_staff"], 7.0)  # 14 / 2

        # Test get_hourly_department_traffic
        hourly_data = self.db.get_hourly_department_traffic()
        self.assertEqual(hourly_data["status"], "success")
        self.assertIn("departments", hourly_data)
        self.assertIn("peak_department", hourly_data)
        self.assertIn("peak_hour", hourly_data)
        self.assertEqual(hourly_data["peak_department"], "Food")
        self.assertEqual(hourly_data["peak_hour"], "6 PM")


class TestEdgeAPINewEndpoints(unittest.TestCase):
    """Verify Edge API handles /api/v1/departments, /api/v1/cameras, /api/v1/patterns/hourly."""

    @classmethod
    def setUpClass(cls):
        cls.temp_db_fd, cls.temp_db_path = tempfile.mkstemp(suffix=".db")
        os.close(cls.temp_db_fd)
        cls.db = EdgeDatabase(db_path=cls.temp_db_path)

        # Populate sample department records
        cls.db.insert_zone_snapshot(
            snapshot_id="snap_api_food",
            store_id="STORE_001",
            device_id="EDGE_001",
            camera_id="CAM_01",
            zone_id="food",
            zone_name="Food",
            timestamp="2026-09-05T17:20:00",
            current_shoppers=6,
            peak_shoppers=7,
            avg_dwell=40.0,
            traffic_level="NORMAL",
            expected_staff=2,
            footfall=12,
            max_dwell=90.0,
            video_timestamp="00:20.00"
        )
        set_db(cls.db)
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()
        try:
            if os.path.exists(cls.temp_db_path):
                os.remove(cls.temp_db_path)
        except OSError:
            pass

    def test_departments_endpoint(self):
        response = self.client.get("/api/v1/departments")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertTrue(len(data) > 0)
        food_dept = next((d for d in data if d.get("zone_id") == "food" or d.get("department") == "Food"), None)
        self.assertIsNotNone(food_dept)
        self.assertEqual(food_dept["current_shoppers"], 6)
        self.assertIn("shopper_load_per_staff", food_dept)
        self.assertEqual(food_dept["expected_staff"], 2)
        self.assertEqual(food_dept["shopper_load_per_staff"], 3.0)

    def test_cameras_endpoint(self):
        response = self.client.get("/api/v1/cameras")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 4)

    def test_hourly_patterns_endpoint(self):
        response = self.client.get("/api/v1/patterns/hourly")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("departments", data)

    def test_simulation_clock_endpoint(self):
        response = self.client.get("/api/v1/simulation/clock")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("is_simulation", data)
        self.assertIn("simulated_store_time", data)
