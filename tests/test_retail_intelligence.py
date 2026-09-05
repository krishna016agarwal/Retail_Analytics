"""Automated Unit & Integration Tests for Phase 8: Retail Intelligence Engine.

Covers all 11 required areas:
1. Queue growth calculation
2. Queue prediction
3. Queue congestion alert
4. Alert cooldown/deduplication
5. Staffing load calculation
6. Staffing alert
7. Zone occupancy
8. Crowd spike anomaly detection
9. Historical pattern calculation (valid and insufficient data)
10. Multi-camera configuration and edge aggregation
11. Integration with SQLite Edge Database and REST APIs
"""

from datetime import datetime, timezone
from pathlib import Path
import sys
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import RetailIntelligenceConfig, ZoneConfig
from src.camera_manager import CameraConfig, CameraWorker, MultiCameraManager
from src.database import EdgeDatabase
from src.detector import Detection, DetectionBatch
from src.queue_analytics import QueueMetrics
from src.retail_intelligence import AlertManager, RetailAlert, RetailIntelligenceEngine
from src.shopper_analytics import FootfallMetrics
from src.tracker import TrackedPerson, TrackingBatch
from src.zone_analytics import ZoneAnalyticsManager, ZoneMetrics


class TestQueueIntelligence(unittest.TestCase):
    """Tests for queue growth calculation, prediction, and congestion alerting."""

    def setUp(self):
        self.config = RetailIntelligenceConfig(
            prediction_minutes=3.0,
            queue_high_threshold=6,
            cooldown_seconds=30.0,
        )
        self.engine = RetailIntelligenceEngine(config=self.config)

    def test_queue_growth_calculation_stable(self):
        """Test queue growth rate calculation when queue is constant."""
        rate, trend = self.engine.calculate_queue_growth(current_queue=4, current_time=0.0)
        self.assertEqual(trend, "STABLE")

        rate, trend = self.engine.calculate_queue_growth(current_queue=4, current_time=10.0)
        self.assertEqual(rate, 0.0)
        self.assertEqual(trend, "STABLE")

    def test_queue_growth_calculation_growing(self):
        """Test queue growth rate when queue increases by 2 people in 30 seconds (+4 people/min)."""
        self.engine.calculate_queue_growth(current_queue=3, current_time=0.0)
        rate, trend = self.engine.calculate_queue_growth(current_queue=5, current_time=30.0)

        self.assertAlmostEqual(rate, 4.0, delta=0.1)
        self.assertEqual(trend, "GROWING")

    def test_queue_prediction_formula(self):
        """Test forward 3-minute prediction formula."""
        # current = 5, growth = +2.0 people/min -> in 3 min: 5 + (2.0 * 3) = 11
        predicted = self.engine.predict_queue(current_queue=5, growth_rate=2.0, prediction_minutes=3.0)
        self.assertEqual(predicted, 11)

        # Shrinking queue: current = 4, growth = -2.0 -> max(0, 4 - 6) = 0
        predicted_shrinking = self.engine.predict_queue(current_queue=4, growth_rate=-2.0, prediction_minutes=3.0)
        self.assertEqual(predicted_shrinking, 0)

    def test_queue_congestion_alert_triggered(self):
        """Test queue congestion alert triggers when predicted queue crosses threshold with positive growth."""
        # Baseline observation at t=0
        qm_t0 = QueueMetrics(current_queue_length=3, peak_queue_length=3)
        self.engine.evaluate_queue_intelligence(qm_t0, current_time=0.0)

        # Observation at t=30 with 5 people (+4 people/min -> predicted in 3m: 5 + 12 = 17 >= threshold 6)
        qm_t30 = QueueMetrics(current_queue_length=5, peak_queue_length=5)
        intel, alert = self.engine.evaluate_queue_intelligence(qm_t30, current_time=30.0)

        self.assertIsNotNone(alert)
        self.assertEqual(alert.type, "QUEUE_CONGESTION")
        self.assertIn(alert.severity, ["HIGH", "CRITICAL"])
        self.assertEqual(alert.title, "Checkout congestion predicted")
        self.assertEqual(alert.recommendation, "Open additional billing counter")
        self.assertGreaterEqual(alert.predicted_value, 6)

    def test_queue_growth_calculation_near_zero_dt(self):
        """Test near-zero dt (e.g. 0.05s) does not produce absurd growth rate."""
        self.engine.calculate_queue_growth(current_queue=2, current_time=10.00)
        # 50 milliseconds later with 1 person difference
        rate, trend = self.engine.calculate_queue_growth(current_queue=3, current_time=10.05)
        self.assertEqual(rate, 0.0)
        self.assertEqual(trend, "STABLE")

    def test_queue_growth_calculation_duplicate_timestamps(self):
        """Test identical/duplicate timestamps do not cause ZeroDivisionError or massive rates."""
        self.engine.calculate_queue_growth(current_queue=2, current_time=15.0)
        rate, trend = self.engine.calculate_queue_growth(current_queue=4, current_time=15.0)
        self.assertEqual(rate, 0.0)
        self.assertEqual(trend, "STABLE")

    def test_queue_growth_calculation_shrinking(self):
        """Test shrinking queue correctly computes negative growth rate and SHRINKING trend."""
        # Baseline at t=0: 6 people
        self.engine.calculate_queue_growth(current_queue=6, current_time=0.0)
        # Intermediate at t=15: 4 people
        self.engine.calculate_queue_growth(current_queue=4, current_time=15.0)
        # At t=30: 2 people (-4 people in 30 seconds = -8 people/min)
        rate, trend = self.engine.calculate_queue_growth(current_queue=2, current_time=30.0)
        self.assertLess(rate, 0.0)
        self.assertEqual(trend, "SHRINKING")

    def test_queue_growth_calculation_rapid_fluctuations(self):
        """Test that single-frame fluctuations (2 -> 3 -> 2 -> 3 -> 2) over time are smoothed by regression."""
        times = [0.0, 5.0, 10.0, 15.0, 20.0]
        queues = [2, 3, 2, 3, 2]
        for t, q in zip(times, queues):
            rate, trend = self.engine.calculate_queue_growth(current_queue=q, current_time=t)
        # Linear slope over symmetric 2, 3, 2, 3, 2 is exactly 0
        self.assertAlmostEqual(rate, 0.0, delta=0.5)
        self.assertEqual(trend, "STABLE")

    def test_queue_prevention_of_absurd_prediction_from_noise(self):
        """Test prevention of absurd predictions (e.g. 302 from current queue 2)."""
        # Simulate high-frequency frame events: queue increases by 1 in 0.6 seconds
        self.engine.calculate_queue_growth(current_queue=2, current_time=100.0)
        rate, trend = self.engine.calculate_queue_growth(current_queue=3, current_time=100.6)
        
        # Because elapsed time < 5.0s, rate must remain 0.0 (STABLE)
        self.assertEqual(rate, 0.0)
        self.assertEqual(trend, "STABLE")
        
        # Prediction formula must never produce 302
        pred = self.engine.predict_queue(current_queue=2, growth_rate=rate, prediction_minutes=3.0)
        self.assertEqual(pred, 2)
        self.assertNotEqual(pred, 302)

    def test_derive_live_summaries_queue_robustness(self):
        """Test derive_live_summaries does not compute 100 people/min or 302 predicted queue."""
        # 2 snapshots close together in time (0.6 seconds apart)
        snapshots = [
            {
                "timestamp": "T+10.60",
                "created_at": "2026-09-05T12:00:00.600000Z",
                "queue_length": 2,
                "peak_queue_length": 2,
                "average_wait_time": 4.5,
                "current_occupancy": 10,
                "store_id": "store_001",
                "device_id": "edge_device_01",
            },
            {
                "timestamp": "T+10.00",
                "created_at": "2026-09-05T12:00:00.000000Z",
                "queue_length": 1,
                "peak_queue_length": 2,
                "average_wait_time": 4.0,
                "current_occupancy": 9,
                "store_id": "store_001",
                "device_id": "edge_device_01",
            }
        ]
        q_intel, c_intel = RetailIntelligenceEngine.derive_live_summaries(
            latest_snapshot=snapshots[0],
            recent_snapshots=snapshots
        )
        
        # Growth rate must NOT be 100.0 people/min and predicted must NOT be 302
        self.assertNotEqual(q_intel["growth_rate_per_min"], 100.0)
        self.assertNotEqual(q_intel["predicted_queue_3min"], 302)
        self.assertEqual(q_intel["growth_rate_per_min"], 0.0)
        self.assertEqual(q_intel["trend"], "STABLE")
        self.assertEqual(q_intel["predicted_queue_3min"], 2)


class TestAlertCooldownAndDeduplication(unittest.TestCase):
    """Tests for alert cooldown deduplication and state tracking."""

    def setUp(self):
        self.manager = AlertManager(cooldown_seconds=40.0)

    def test_cooldown_suppresses_duplicate_alert(self):
        """Test that identical alert within cooldown period does not fire."""
        # First alert at t=10.0 should fire
        self.assertTrue(self.manager.should_fire("QUEUE_CONGESTION", "checkout", "HIGH", current_time=10.0))

        alert = RetailAlert(
            alert_id="a1",
            timestamp="2026-09-05T00:00:00Z",
            store_id="store_001",
            device_id="edge_01",
            camera_id="CAM_01",
            zone_id="checkout",
            type="QUEUE_CONGESTION",
            severity="HIGH",
            title="Congestion",
            message="Growing queue",
            current_value=6,
            predicted_value=10,
            threshold=6,
            recommendation="Open counter",
        )
        self.manager.record_alert(alert, current_time=10.0)

        # Second alert at t=25.0 (15s elapsed < 40s cooldown) should NOT fire
        self.assertFalse(self.manager.should_fire("QUEUE_CONGESTION", "checkout", "HIGH", current_time=25.0))

        # Third alert at t=55.0 (45s elapsed >= 40s cooldown) SHOULD fire
        self.assertTrue(self.manager.should_fire("QUEUE_CONGESTION", "checkout", "HIGH", current_time=55.0))

    def test_severity_escalation_bypasses_cooldown(self):
        """Test that severity escalation (e.g. HIGH -> CRITICAL) immediately fires even inside cooldown."""
        self.manager.should_fire("QUEUE_CONGESTION", "checkout", "HIGH", current_time=10.0)
        alert = RetailAlert(
            alert_id="a1",
            timestamp="2026-09-05T00:00:00Z",
            store_id="store_001",
            device_id="edge_01",
            camera_id="CAM_01",
            zone_id="checkout",
            type="QUEUE_CONGESTION",
            severity="HIGH",
            title="Congestion",
            message="Growing queue",
            current_value=6,
            predicted_value=10,
            threshold=6,
            recommendation="Open counter",
        )
        self.manager.record_alert(alert, current_time=10.0)

        # Escalation to CRITICAL at t=15.0 should fire immediately
        self.assertTrue(self.manager.should_fire("QUEUE_CONGESTION", "checkout", "CRITICAL", current_time=15.0))

    def test_alert_becomes_active_when_condition_is_true(self):
        """Test alert transitions to ACTIVE status when condition is met."""
        alert = RetailAlert(
            alert_id="alert_act_01",
            timestamp="2026-09-05T00:00:00Z",
            store_id="store_001",
            device_id="edge_01",
            camera_id="CAM_01",
            zone_id="checkout",
            type="QUEUE_CONGESTION",
            severity="HIGH",
            title="Congestion",
            message="Queue growing",
            current_value=6.0,
            predicted_value=10.0,
            threshold=6.0,
            recommendation="Open counter",
            status="ACTIVE",
        )
        self.manager.record_alert(alert, current_time=10.0)
        active_alerts = self.manager.get_active_alerts()
        self.assertEqual(len(active_alerts), 1)
        self.assertEqual(active_alerts[0].status, "ACTIVE")
        self.assertEqual(active_alerts[0].alert_id, "alert_act_01")

    def test_same_alert_deduplicated_during_cooldown(self):
        """Test identical alert within cooldown period is deduplicated and not re-fired."""
        # Initial trigger
        self.assertTrue(self.manager.should_fire("STAFFING", "food", "MEDIUM", current_time=10.0))
        alert = RetailAlert(
            alert_id="alert_st_01",
            timestamp="2026-09-05T00:00:00Z",
            store_id="store_001",
            device_id="edge_01",
            camera_id="CAM_01",
            zone_id="food",
            type="STAFFING",
            severity="MEDIUM",
            title="Staffing",
            message="Overloaded",
            current_value=4.0,
            threshold=3.0,
            recommendation="Move staff",
            status="ACTIVE",
        )
        self.manager.record_alert(alert, current_time=10.0)

        # Attempt to fire same alert again at t=20.0 (10s < 40s cooldown) -> suppressed
        self.assertFalse(self.manager.should_fire("STAFFING", "food", "MEDIUM", current_time=20.0))

    def test_alert_resolves_when_condition_becomes_false(self):
        """Test active alert transitions to RESOLVED when condition is no longer true."""
        alert = RetailAlert(
            alert_id="alert_res_01",
            timestamp="2026-09-05T00:00:00Z",
            store_id="store_001",
            device_id="edge_01",
            camera_id="CAM_01",
            zone_id="electronics",
            type="CROWD_SPIKE",
            severity="HIGH",
            title="Crowd spike",
            message="Spike detected",
            current_value=5.0,
            threshold=3.0,
            recommendation="Deploy staff",
            status="ACTIVE",
        )
        self.manager.record_alert(alert, current_time=10.0)
        self.assertEqual(len(self.manager.get_active_alerts()), 1)

        # Condition clears -> resolve alert
        resolved = self.manager.resolve_alert("CROWD_SPIKE", "electronics")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.status, "RESOLVED")
        self.assertEqual(resolved.alert_id, "alert_res_01")

        # Active alerts list must now be empty
        self.assertEqual(len(self.manager.get_active_alerts()), 0)

        # Recently resolved queue contains the resolved alert
        popped = self.manager.pop_recently_resolved()
        self.assertEqual(len(popped), 1)
        self.assertEqual(popped[0].alert_id, "alert_res_01")

    def test_resolved_alert_remains_in_history(self):
        """Test resolved alert is preserved in historical logs with status RESOLVED."""
        alert = RetailAlert(
            alert_id="alert_hist_01",
            timestamp="2026-09-05T00:00:00Z",
            store_id="store_001",
            device_id="edge_01",
            camera_id="CAM_01",
            zone_id="checkout",
            type="QUEUE_CONGESTION",
            severity="HIGH",
            title="Congestion",
            message="Queue growing",
            current_value=6.0,
            predicted_value=12.0,
            threshold=6.0,
            recommendation="Open counter",
            status="ACTIVE",
        )
        self.manager.record_alert(alert, current_time=10.0)
        self.manager.resolve_alert("QUEUE_CONGESTION", "checkout")

        # Must not be active
        self.assertEqual(len(self.manager.get_active_alerts()), 0)

        # Must be present in history and resolved lists
        history = self.manager.get_alert_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].alert_id, "alert_hist_01")
        self.assertEqual(history[0].status, "RESOLVED")

        resolved_list = self.manager.get_resolved_alerts()
        self.assertEqual(len(resolved_list), 1)
        self.assertEqual(resolved_list[0].status, "RESOLVED")


class TestStaffingIntelligence(unittest.TestCase):
    """Tests for configured staff capacity load calculation and staffing alerts."""

    def setUp(self):
        self.config = RetailIntelligenceConfig(max_shoppers_per_staff=4.0)
        self.engine = RetailIntelligenceEngine(config=self.config)

    def test_staffing_load_calculation(self):
        """Test shopper_count / expected_staff calculation."""
        zone_metrics = {
            "food": ZoneMetrics(
                zone_id="food",
                name="Food Section",
                current_shoppers=6,
                expected_staff=2,
                shopper_load_per_staff=3.0,
            ),
            "electronics": ZoneMetrics(
                zone_id="electronics",
                name="Electronics Section",
                current_shoppers=10,
                expected_staff=2,
                shopper_load_per_staff=5.0,
            ),
        }

        summaries, alerts = self.engine.evaluate_staffing_intelligence(zone_metrics, current_time=0.0)

        # Food: 6 / 2 = 3.0 (< threshold 4.0) -> No alert
        self.assertEqual(summaries["food"]["shopper_load_per_staff"], 3.0)
        self.assertFalse(summaries["food"]["is_overloaded"])

        # Electronics: 10 / 2 = 5.0 (>= threshold 4.0) -> Overloaded, alert generated
        self.assertEqual(summaries["electronics"]["shopper_load_per_staff"], 5.0)
        self.assertTrue(summaries["electronics"]["is_overloaded"])

        self.assertEqual(len(alerts), 1)
        alert = alerts[0]
        self.assertEqual(alert.type, "STAFFING")
        self.assertEqual(alert.zone_id, "electronics")
        self.assertEqual(alert.title, "Low staff coverage")
        self.assertEqual(alert.recommendation, "Move 1 staff member to Electronics Section")


class TestZoneOccupancyAndCrowdSpike(unittest.TestCase):
    """Tests for zone occupancy tracking and statistical crowd spike anomaly detection."""

    def setUp(self):
        self.zones = [
            ZoneConfig(id="food", name="Food Section", x1=0, y1=0, x2=100, y2=100, expected_staff=2, max_shopper_capacity=10),
            ZoneConfig(id="clothing", name="Clothing Section", x1=200, y1=200, x2=300, y2=300, expected_staff=1, max_shopper_capacity=10),
        ]
        self.zone_mgr = ZoneAnalyticsManager(zones=self.zones, fps=10.0)
        self.engine = RetailIntelligenceEngine(
            config=RetailIntelligenceConfig(spike_percentage_threshold=0.50, min_occupancy_for_spike=4)
        )

    def test_zone_occupancy_spatial_assignment(self):
        """Test that tracked persons are assigned to the correct zone by ground position."""
        # Person 1 at bottom_center (50, 50) -> Food (0, 0, 100, 100)
        # Person 2 at bottom_center (250, 250) -> Clothing (200, 200, 300, 300)
        # Person 3 at bottom_center (500, 500) -> Outside any zone
        tracks = [
            TrackedPerson(track_id=1, bbox=(40, 20, 60, 50), confidence=0.9),
            TrackedPerson(track_id=2, bbox=(240, 220, 260, 250), confidence=0.9),
            TrackedPerson(track_id=3, bbox=(490, 470, 510, 500), confidence=0.9),
        ]

        batch = TrackingBatch(frame_id=1, tracks=tracks)
        metrics_dict = self.zone_mgr.update(batch)


        self.assertEqual(metrics_dict["food"].current_shoppers, 1)
        self.assertEqual(metrics_dict["clothing"].current_shoppers, 1)
        self.assertIn(1, metrics_dict["food"].active_track_ids)
        self.assertIn(2, metrics_dict["clothing"].active_track_ids)

    def test_crowd_spike_detection(self):
        """Test statistical crowd spike anomaly detection against rolling baseline."""
        # Establish a rolling baseline of 4 shoppers
        for t in range(5):
            zm = {
                "food": ZoneMetrics(
                    zone_id="food",
                    name="Food Section",
                    current_shoppers=4,
                    expected_staff=2,
                )
            }
            self.engine.evaluate_crowd_intelligence(zm, current_time=float(t))

        # Spike to 8 shoppers (+100% over baseline 4.0 >= 50% threshold)
        spike_zm = {
            "food": ZoneMetrics(
                zone_id="food",
                name="Food Section",
                current_shoppers=8,
                expected_staff=2,
            )
        }
        summaries, alerts = self.engine.evaluate_crowd_intelligence(spike_zm, current_time=10.0)

        self.assertTrue(summaries["food"]["is_spike"])
        self.assertEqual(len(alerts), 1)
        alert = alerts[0]
        self.assertEqual(alert.type, "CROWD_SPIKE")
        self.assertEqual(alert.title, "High shopper concentration")
        self.assertEqual(alert.recommendation, "Monitor zone / deploy floor staff")


class TestHistoricalPatterns(unittest.TestCase):
    """Tests for historical traffic pattern calculation and insufficient data fallback."""

    def test_insufficient_historical_data_when_empty(self):
        """Test that empty or sparse snapshots return 'Insufficient historical data'."""
        result = RetailIntelligenceEngine.calculate_historical_patterns([])
        self.assertEqual(result["status"], "insufficient_data")
        self.assertEqual(result["message"], "Insufficient historical data")

        # Two records only is also insufficient
        two_records = [
            {"timestamp": "T+5.00s", "occupancy": 5},
            {"timestamp": "T+10.00s", "occupancy": 6},
        ]
        result2 = RetailIntelligenceEngine.calculate_historical_patterns(two_records)
        self.assertEqual(result2["status"], "insufficient_data")

    def test_valid_historical_pattern_calculation(self):
        """Test historical pattern calculation with realistic ISO timestamps across hours."""
        snapshots = [
            {"created_at": "2026-09-05T10:15:00Z", "occupancy": 10, "queue_length": 2},
            {"created_at": "2026-09-05T10:45:00Z", "occupancy": 14, "queue_length": 3},
            {"created_at": "2026-09-05T14:10:00Z", "occupancy": 28, "queue_length": 8},
            {"created_at": "2026-09-05T14:40:00Z", "occupancy": 32, "queue_length": 10},
            {"created_at": "2026-09-05T18:20:00Z", "occupancy": 18, "queue_length": 4},
        ]
        zone_snapshots = [
            {"zone_name": "Food Section", "current_shoppers": 15},
            {"zone_name": "Food Section", "current_shoppers": 20},
            {"zone_name": "Electronics", "current_shoppers": 8},
        ]

        patterns = RetailIntelligenceEngine.calculate_historical_patterns(snapshots, zone_snapshots)

        self.assertEqual(patterns["status"], "success")
        self.assertEqual(patterns["busiest_hour"], "14:00 - 15:00")
        self.assertEqual(patterns["busiest_zone"], "Food Section")
        self.assertIn("10:00", patterns["average_occupancy_by_hour"])
        self.assertIn("14:00", patterns["average_occupancy_by_hour"])
        self.assertEqual(patterns["average_occupancy_by_hour"]["14:00"], 30.0)
        self.assertEqual(patterns["average_occupancy_by_hour"]["10:00"], 12.0)


class TestMultiCameraArchitecture(unittest.TestCase):
    """Tests for multi-camera stream worker orchestration and edge aggregation."""

    def test_simulated_setup_creation(self):
        """Test creating simulated 5-camera setup."""
        mgr = MultiCameraManager.create_simulated_setup(
            video_path="videos/test.mp4",
            store_id="store_001",
            device_id="edge_device_01",
        )
        workers = mgr.get_workers()

        self.assertEqual(len(workers), 5)
        camera_ids = [w.config.camera_id for w in workers]
        self.assertEqual(camera_ids, ["CAM_01", "CAM_02", "CAM_03", "CAM_04", "CAM_05"])

        # Verify simulation tags
        for w in workers:
            self.assertTrue(w.config.is_simulation)

    def test_multi_camera_edge_aggregation(self):
        """Test multi-camera telemetry aggregation on single logical Edge Device."""
        mgr = MultiCameraManager.create_simulated_setup()

        # Simulate independent analytics from 2 cameras
        from src.camera_manager import CameraAnalytics
        cam1_analytics = CameraAnalytics(
            camera_id="CAM_01",
            name="Food Section",
            role="zone",
            timestamp="T+1.00s",
            is_simulation=True,
            active_persons_detected=5,
            zone_metrics=ZoneMetrics(zone_id="food", name="Food Section", current_shoppers=5, expected_staff=2),
        )
        cam5_analytics = CameraAnalytics(
            camera_id="CAM_05",
            name="Checkout Queue",
            role="checkout",
            timestamp="T+1.00s",
            is_simulation=True,
            active_persons_detected=3,
            queue_metrics=QueueMetrics(current_queue_length=3, peak_queue_length=3),
        )

        aggregated = mgr.aggregate_and_evaluate([cam1_analytics, cam5_analytics], current_time=1.0)

        self.assertEqual(aggregated["edge_device"]["device_id"], "edge_device_01")
        self.assertIn("queue_intelligence", aggregated)
        self.assertIn("staffing_intelligence", aggregated)
        self.assertEqual(aggregated["queue_intelligence"]["current_queue"], 3)


class TestDatabaseAndAPIIntegration(unittest.TestCase):
    """Tests for SQLite database extensions and FastAPI endpoints for Phase 8."""

    def setUp(self):
        import tempfile
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.db = EdgeDatabase(db_path=self.temp_db.name)

    def tearDown(self):
        self.db.close()
        try:
            Path(self.temp_db.name).unlink(missing_ok=True)
        except Exception:
            pass

    def test_sqlite_zone_snapshot_insertion_and_query(self):
        """Test inserting and querying zone snapshots in SQLite."""
        self.db.insert_zone_snapshot(
            snapshot_id="edge_01:CAM_01:food:T+5.00s",
            store_id="store_001",
            device_id="edge_01",
            camera_id="CAM_01",
            zone_id="food",
            zone_name="Food Section",
            timestamp="T+5.00s",
            current_shoppers=7,
            peak_shoppers=8,
            avg_dwell=14.5,
            traffic_level="NORMAL",
            expected_staff=2,
        )

        snapshots = self.db.get_zone_snapshots(zone_id="food")
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]["current_shoppers"], 7)
        self.assertEqual(snapshots[0]["zone_name"], "Food Section")

    def test_sqlite_alert_insertion_and_query(self):
        """Test inserting and querying operational alerts in SQLite."""
        self.db.insert_alert(
            alert_id="alert_q_123",
            store_id="store_001",
            device_id="edge_01",
            camera_id="CAM_05",
            zone_id="checkout",
            type="QUEUE_CONGESTION",
            severity="HIGH",
            title="Checkout congestion predicted",
            message="Queue growing rapidly",
            current_value=6.0,
            predicted_value=12.0,
            threshold=6.0,
            recommendation="Open additional billing counter",
            status="ACTIVE",
        )

        active_alerts = self.db.get_active_alerts()
        self.assertEqual(len(active_alerts), 1)
        self.assertEqual(active_alerts[0]["alert_id"], "alert_q_123")
        self.assertEqual(active_alerts[0]["recommendation"], "Open additional billing counter")

    def test_edge_api_phase8_endpoints(self):
        """Test GET /api/v1/intelligence/latest, /api/v1/alerts, /api/v1/zones, /api/v1/patterns."""
        from fastapi.testclient import TestClient
        from src.api import app, set_db

        set_db(self.db)
        client = TestClient(app)

        # Insert a sample alert and zone
        self.db.insert_alert(
            alert_id="alert_test_01",
            store_id="store_001",
            device_id="edge_01",
            camera_id="CAM_01",
            zone_id="food",
            type="STAFFING",
            severity="MEDIUM",
            title="Low staff coverage",
            message="High load",
            current_value=5.0,
            threshold=4.0,
            recommendation="Move 1 staff member to Food",
            status="ACTIVE",
        )

        # Test /api/v1/intelligence/latest
        res_intel = client.get("/api/v1/intelligence/latest")
        self.assertEqual(res_intel.status_code, 200)
        data = res_intel.json()
        self.assertEqual(data["active_alert_count"], 1)

        # Test /api/v1/alerts
        res_alerts = client.get("/api/v1/alerts?type=STAFFING")
        self.assertEqual(res_alerts.status_code, 200)
        alerts_data = res_alerts.json()
        self.assertEqual(len(alerts_data), 1)
        self.assertEqual(alerts_data[0]["type"], "STAFFING")

        # Test /api/v1/zones
        res_zones = client.get("/api/v1/zones")
        self.assertEqual(res_zones.status_code, 200)

        # Test /api/v1/patterns
        res_patterns = client.get("/api/v1/patterns")
        self.assertEqual(res_patterns.status_code, 200)
        self.assertIn("status", res_patterns.json())

    def test_active_endpoint_returns_only_currently_active_alerts(self):
        """Test /api/v1/alerts?status=ACTIVE returns only active alerts and distinguishes resolved ones."""
        from fastapi.testclient import TestClient
        from src.api import app, set_db

        set_db(self.db)
        client = TestClient(app)

        # Insert 1 active alert and 1 resolved alert
        self.db.insert_alert(
            alert_id="alert_act_only",
            store_id="store_001",
            device_id="edge_01",
            camera_id="CAM_01",
            zone_id="electronics",
            type="CROWD_SPIKE",
            severity="HIGH",
            title="Crowd spike",
            message="Active crowd spike",
            current_value=6.0,
            threshold=4.0,
            recommendation="Monitor",
            status="ACTIVE",
        )
        self.db.insert_alert(
            alert_id="alert_res_only",
            store_id="store_001",
            device_id="edge_01",
            camera_id="CAM_01",
            zone_id="checkout",
            type="QUEUE_CONGESTION",
            severity="HIGH",
            title="Old congestion",
            message="Resolved congestion",
            current_value=2.0,
            threshold=6.0,
            recommendation="Normal",
            status="RESOLVED",
        )

        # Query active alerts
        res_active = client.get("/api/v1/alerts?status=ACTIVE")
        self.assertEqual(res_active.status_code, 200)
        active_list = res_active.json()
        self.assertEqual(len(active_list), 1)
        self.assertEqual(active_list[0]["alert_id"], "alert_act_only")
        self.assertEqual(active_list[0]["status"], "ACTIVE")

        # Query resolved alerts
        res_resolved = client.get("/api/v1/alerts?status=RESOLVED")
        self.assertEqual(res_resolved.status_code, 200)
        resolved_list = res_resolved.json()
        self.assertEqual(len(resolved_list), 1)
        self.assertEqual(resolved_list[0]["alert_id"], "alert_res_only")
        self.assertEqual(resolved_list[0]["status"], "RESOLVED")

        # Query all alerts (no filter)
        res_all = client.get("/api/v1/alerts")
        self.assertEqual(res_all.status_code, 200)
        all_list = res_all.json()
        self.assertGreaterEqual(len(all_list), 2)


if __name__ == "__main__":
    unittest.main()
