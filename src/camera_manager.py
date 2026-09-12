"""Multi-Camera Architecture and Stream Abstraction for Edge Retail Computing (SIH 179).

Provides multi-stream camera coordination:
- Individual CameraWorker instances handle video reading, detection, tracking, and localized analytics.
- MultiCameraManager aggregates independent camera telemetry on ONE logical Edge Device.
- Feeds aggregated telemetry into RetailIntelligenceEngine to generate centralized store insights and alerts.
- Dedicated SimulationClock maps video playback to simulated store time (e.g. 17:00).
- Pure computer vision: all metrics originate from YOLO11 + ByteTrack on actual video feeds.
"""

from collections import deque
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
import uuid

import cv2
import numpy as np

from configs.config import AnalyticsConfig, DetectorConfig, QueueConfig, TrackerConfig, ZoneConfig
from src.detector import PersonDetector
from src.queue_analytics import QueueAnalytics, QueueMetrics
from src.retail_intelligence import RetailIntelligenceEngine
from src.shopper_analytics import EntryExitCounter, FootfallMetrics
from src.tracker import PersonTracker
from src.visualizer import Visualizer
from src.zone_analytics import ZoneAnalyticsManager, ZoneMetrics

logger = logging.getLogger("camera_manager")


IST = timezone(timedelta(hours=5, minutes=30))


class SimulationClock:
    """Provides a consistent simulated store time for multi-camera demo replay in IST (UTC+5:30)."""

    def __init__(self, demo_start_time: str = "now", time_scale: float = 1.0):
        """Initialize simulation clock in Indian Standard Time (IST).

        Args:
            demo_start_time: Initial store time string (HH:MM:SS), 'now' for current IST time, or e.g. '17:00:00'.
            time_scale: Acceleration factor (1.0 = real-time).
        """
        self.time_scale = max(0.1, float(time_scale))
        now_ist = datetime.now(IST)
        today = now_ist.date()

        if not demo_start_time or str(demo_start_time).strip().lower() in ("now", "current"):
            hour = now_ist.hour
            minute = now_ist.minute
            second = now_ist.second
            self.demo_start_time_str = f"{hour:02d}:{minute:02d}:{second:02d}"
        else:
            self.demo_start_time_str = str(demo_start_time).strip()
            parts = [int(p) for p in self.demo_start_time_str.split(":")]
            hour = parts[0] if len(parts) > 0 else 17
            minute = parts[1] if len(parts) > 1 else 0
            second = parts[2] if len(parts) > 2 else 0

        self.base_datetime = datetime(
            today.year, today.month, today.day, hour, minute, second, tzinfo=IST
        )
        self.start_wall_time = time.time()
        self.simulated_elapsed_seconds = 0.0

    def update_elapsed(self, elapsed_seconds: float) -> None:
        """Manually advance simulated elapsed seconds."""
        self.simulated_elapsed_seconds = max(self.simulated_elapsed_seconds, float(elapsed_seconds))

    def get_elapsed_seconds(self) -> float:
        """Return current simulated elapsed seconds."""
        if self.simulated_elapsed_seconds > 0.0:
            return self.simulated_elapsed_seconds
        return (time.time() - self.start_wall_time) * self.time_scale

    def get_simulated_datetime(self, elapsed_seconds: Optional[float] = None) -> datetime:
        """Return full simulated IST datetime."""
        sec = self.get_elapsed_seconds() if elapsed_seconds is None else float(elapsed_seconds)
        return self.base_datetime + timedelta(seconds=sec)

    def get_simulated_time_str(self, elapsed_seconds: Optional[float] = None) -> str:
        """Return simulated time string (HH:MM:SS), e.g. '17:05:23'."""
        return self.get_simulated_datetime(elapsed_seconds).strftime("%H:%M:%S")

    def get_simulated_iso(self, elapsed_seconds: Optional[float] = None) -> str:
        """Return simulated ISO-8601 IST timestamp."""
        return self.get_simulated_datetime(elapsed_seconds).isoformat()


@dataclass
class CameraConfig:
    """Configuration for an individual camera stream.

    Attributes:
        camera_id: Unique camera ID (e.g., 'CAM_01', 'CAM_02').
        name: Human-readable camera label (e.g., 'Food Section').
        zone_id: Store zone or department associated with this camera.
        video_source: Path to video file or RTSP stream URI.
        role: Primary analytical role ('department', 'zone', 'entrance', 'checkout').
        expected_staff: Configured staff capacity for this department/section.
        is_simulation: True if using recorded video for demo simulation.
        zone_bbox: Optional spatial bounding box (x1, y1, x2, y2) in frame.
        loop: Whether to loop video continuously when stream reaches the end.
    """

    camera_id: str
    name: str
    zone_id: str
    video_source: str
    role: str = "department"
    expected_staff: int = 1
    is_simulation: bool = True
    zone_bbox: Optional[Tuple[int, int, int, int]] = None
    loop: bool = True


@dataclass
class CameraAnalytics:
    """Frame-level analytics produced by a single camera worker."""

    camera_id: str
    name: str
    role: str
    timestamp: str  # Video timeline timestamp (e.g. '00:05.23')
    is_simulation: bool
    active_persons_detected: int = 0
    current_shoppers: int = 0
    peak_shoppers: int = 0
    footfall: int = 0
    avg_dwell: float = 0.0
    max_dwell: float = 0.0
    traffic_level: str = "LOW"
    traffic_trend: str = "STABLE"
    expected_staff: int = 1
    shopper_load_per_staff: float = 0.0
    fps: float = 25.0
    frame_idx: int = 0
    simulated_store_timestamp: str = "17:00:00"
    simulated_iso: str = ""
    footfall_metrics: Optional[FootfallMetrics] = None
    queue_metrics: Optional[QueueMetrics] = None
    zone_metrics: Optional[ZoneMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert camera analytics to dictionary."""
        res: Dict[str, Any] = {
            "camera_id": self.camera_id,
            "name": self.name,
            "role": self.role,
            "video_timestamp": self.timestamp,
            "timestamp": self.timestamp,
            "simulated_store_timestamp": self.simulated_store_timestamp,
            "simulated_iso": self.simulated_iso,
            "is_simulation": self.is_simulation,
            "active_persons_detected": self.active_persons_detected,
            "current_shoppers": self.current_shoppers,
            "peak_shoppers": self.peak_shoppers,
            "footfall": self.footfall,
            "avg_dwell": self.avg_dwell,
            "max_dwell": self.max_dwell,
            "traffic_level": self.traffic_level,
            "traffic_trend": self.traffic_trend,
            "expected_staff": self.expected_staff,
            "shopper_load_per_staff": self.shopper_load_per_staff,
            "fps": round(self.fps, 2),
            "frame_idx": self.frame_idx,
        }
        if self.footfall_metrics:
            res["footfall_metrics"] = {
                "entries": self.footfall_metrics.total_entries,
                "exits": self.footfall_metrics.total_exits,
                "occupancy": self.footfall_metrics.current_occupancy,
                "peak_occupancy": self.footfall_metrics.peak_occupancy,
            }
        if self.queue_metrics:
            res["queue"] = {
                "current_length": self.queue_metrics.current_queue_length,
                "peak_length": self.queue_metrics.peak_queue_length,
                "congestion_level": self.queue_metrics.congestion_level,
                "average_wait_time": self.queue_metrics.average_wait_time,
                "maximum_wait_time": self.queue_metrics.maximum_wait_time,
                "recommendation": self.queue_metrics.recommendation,
            }
        if self.zone_metrics:
            res["zone"] = self.zone_metrics.to_dict()
        return res


class CameraWorker:
    """Worker responsible for video reading, detection, tracking, and localized analytics.

    Maintains its own video capture, its own FPS, its own video timestamp, and its own
    detector/tracker to ensure zero multi-threading race conditions.
    """

    def __init__(
        self,
        config: CameraConfig,
        detector: Optional[PersonDetector] = None,
        tracker: Optional[PersonTracker] = None,
        fps: Optional[float] = None,
    ):
        """Initialize camera worker with components."""
        self.config = config
        self.detector = detector
        self.tracker = tracker
        self.fps = float(fps) if fps is not None and fps > 0 else 25.0

        self.counter: Optional[EntryExitCounter] = None
        self.queue_analytics: Optional[QueueAnalytics] = None
        self.zone_manager: Optional[ZoneAnalyticsManager] = None

        self._cap: Optional[cv2.VideoCapture] = None
        self.frame_idx = 0
        self.loop = config.loop

        # Department metrics tracking
        self._unique_track_ids: Set[int] = set()
        self._peak_shoppers: int = 0
        self._track_enter_frames: Dict[int, int] = {}
        self._completed_dwells: List[float] = []
        self._recent_shoppers: Deque[int] = deque(maxlen=20)
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.visualizer: Optional[Visualizer] = Visualizer()

        self._inference_lock = threading.Lock()
        self._setup_analytics()

    def _setup_analytics(self) -> None:
        """Configure analytics according to the camera's designated role."""
        if self.config.role == "entrance":
            self.counter = EntryExitCounter(
                config=AnalyticsConfig(entrance_line_y=300, entry_direction="down"),
                fps=self.fps,
            )
        elif self.config.role == "checkout":
            q_bbox = self.config.zone_bbox
            if q_bbox is None:
                q_bbox = QueueAnalytics.load_zone_config("configs/queue_config.json") or (0, 0, 1920, 1080)
            self.queue_analytics = QueueAnalytics(
                config=QueueConfig(zone_bbox=q_bbox, medium_threshold=3, high_threshold=6),
                fps=self.fps,
            )
        else:
            # Department / zone camera
            z_bbox = self.config.zone_bbox or (0, 0, 1920, 1080)
            zone_cfg = ZoneConfig(
                id=self.config.zone_id,
                name=self.config.name,
                x1=z_bbox[0],
                y1=z_bbox[1],
                x2=z_bbox[2],
                y2=z_bbox[3],
                expected_staff=self.config.expected_staff,
            )
            self.zone_manager = ZoneAnalyticsManager(zones=[zone_cfg], fps=self.fps)

    def open_stream(self) -> bool:
        """Open the video capture stream and calibrate actual FPS."""
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(str(self.config.video_source))
            if self._cap.isOpened():
                detected_fps = self._cap.get(cv2.CAP_PROP_FPS)
                if detected_fps and detected_fps > 0 and not np.isnan(detected_fps):
                    self.fps = float(detected_fps)
                    if self.counter:
                        self.counter.set_fps(self.fps)
                    if self.queue_analytics:
                        self.queue_analytics.set_fps(self.fps)
                    if self.zone_manager:
                        self.zone_manager.set_fps(self.fps)
        return self._cap.isOpened() if self._cap is not None else False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame from the camera stream, seamlessly looping if enabled."""
        if self._cap is None:
            if not self.open_stream():
                return False, None

        ret, frame = self._cap.read()
        if not ret and self.loop and self._cap is not None:
            # Rewind to start of video
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()

        if ret and frame is not None:
            self.latest_frame = frame

        return ret, frame

    def _determine_traffic_level(self, current: int) -> str:
        """Classify traffic density based on active detections."""
        if current >= 8:
            return "PEAK"
        elif current >= 5:
            return "HIGH"
        elif current >= 2:
            return "NORMAL"
        return "LOW"

    def _determine_traffic_trend(self, current: int) -> str:
        """Determine trend from recent occupancy window."""
        self._recent_shoppers.append(current)
        if len(self._recent_shoppers) >= 6:
            first_half = list(self._recent_shoppers)[: len(self._recent_shoppers) // 2]
            second_half = list(self._recent_shoppers)[len(self._recent_shoppers) // 2 :]
            avg1 = sum(first_half) / len(first_half)
            avg2 = sum(second_half) / len(second_half)
            diff = avg2 - avg1
            if diff >= 0.8:
                return "GROWING"
            elif diff <= -0.8:
                return "SHRINKING"
        return "STABLE"

    def process_frame(
        self,
        frame: np.ndarray,
        simulated_time_str: str = "17:00:00",
        simulated_iso: str = "",
    ) -> CameraAnalytics:
        """Run detection, tracking, and localized analytics on a single frame."""
        self.frame_idx += 1
        video_time_sec = self.frame_idx / self.fps
        mins = int(video_time_sec // 60)
        secs = video_time_sec % 60
        formatted_video_ts = f"{mins:02d}:{secs:05.2f}"

        footfall_m: Optional[FootfallMetrics] = None
        queue_m: Optional[QueueMetrics] = None
        zone_m: Optional[ZoneMetrics] = None
        person_count = 0
        active_track_ids: Set[int] = set()

        if self.detector is not None:
            with self._inference_lock:
                batch_det = self.detector.detect(frame)
            person_count = len(batch_det.detections)

            if self.tracker is not None:
                track_batch = self.tracker.update(batch_det, frame)

                for trk in track_batch.tracks:
                    t_id = trk.track_id
                    active_track_ids.add(t_id)
                    self._unique_track_ids.add(t_id)
                    if t_id not in self._track_enter_frames:
                        self._track_enter_frames[t_id] = self.frame_idx

                # Check disappeared tracks to finalize dwell times
                departed_ids = [t for t in self._track_enter_frames if t not in active_track_ids]
                for d_id in departed_ids:
                    dwell = (self.frame_idx - self._track_enter_frames.pop(d_id)) / self.fps
                    if dwell > 0.2:
                        self._completed_dwells.append(dwell)

                # Localized analytics by role
                if self.counter is not None:
                    footfall_m = self.counter.update(track_batch, self.tracker.trajectory_manager)
                if self.queue_analytics is not None:
                    queue_m = self.queue_analytics.update(track_batch, self.tracker.trajectory_manager)
                if self.zone_manager is not None:
                    z_dict = self.zone_manager.update(track_batch, self.tracker.trajectory_manager)
                    zone_m = z_dict.get(self.config.zone_id)

                # Generate annotated frame with bounding boxes and tracks
                if self.visualizer is not None and track_batch is not None:
                    try:
                        self.latest_annotated_frame = self.visualizer.draw_tracks(
                            frame,
                            batch=track_batch,
                            trajectory_manager=self.tracker.trajectory_manager if self.tracker else None,
                            fps=self.fps,
                            model_name="yolo11n",
                            queue_metrics=queue_m,
                        )
                    except Exception:
                        self.latest_annotated_frame = frame
                else:
                    self.latest_annotated_frame = frame

        # Shopper count
        current_shoppers = len(active_track_ids) if active_track_ids else person_count
        self._peak_shoppers = max(self._peak_shoppers, current_shoppers)

        # Calculate dwell metrics
        all_dwells = list(self._completed_dwells)
        for t_id, enter_f in self._track_enter_frames.items():
            all_dwells.append((self.frame_idx - enter_f) / self.fps)

        avg_dwell = round(sum(all_dwells) / len(all_dwells), 1) if all_dwells else 0.0
        max_dwell = round(max(all_dwells), 1) if all_dwells else 0.0

        # Footfall count
        footfall_count = len(self._unique_track_ids) if self._unique_track_ids else current_shoppers

        # Staff load
        expected_staff = max(1, self.config.expected_staff)
        shopper_load = round(current_shoppers / expected_staff, 2)

        traffic_level = self._determine_traffic_level(current_shoppers)
        traffic_trend = self._determine_traffic_trend(current_shoppers)

        # For checkout camera, derive metrics from queue_analytics
        if self.config.role == "checkout" and queue_m is not None:
            current_shoppers = queue_m.current_queue_length
            self._peak_shoppers = queue_m.peak_queue_length
            avg_dwell = round(queue_m.average_wait_time, 1)
            max_dwell = round(queue_m.maximum_wait_time, 1)
            traffic_level = queue_m.congestion_level

        return CameraAnalytics(
            camera_id=self.config.camera_id,
            name=self.config.name,
            role=self.config.role,
            timestamp=formatted_video_ts,
            simulated_store_timestamp=simulated_time_str,
            simulated_iso=simulated_iso or datetime.now(timezone.utc).isoformat(),
            is_simulation=self.config.is_simulation,
            active_persons_detected=person_count,
            current_shoppers=current_shoppers,
            peak_shoppers=self._peak_shoppers,
            footfall=footfall_count,
            avg_dwell=avg_dwell,
            max_dwell=max_dwell,
            traffic_level=traffic_level,
            traffic_trend=traffic_trend,
            expected_staff=expected_staff,
            shopper_load_per_staff=shopper_load,
            fps=self.fps,
            frame_idx=self.frame_idx,
            footfall_metrics=footfall_m,
            queue_metrics=queue_m,
            zone_metrics=zone_m,
        )

    def step(
        self,
        simulated_time_str: str = "17:00:00",
        simulated_iso: str = "",
    ) -> Optional[CameraAnalytics]:
        """Read next frame and process it."""
        ret, frame = self.read_frame()
        if not ret or frame is None:
            return None
        return self.process_frame(
            frame, simulated_time_str=simulated_time_str, simulated_iso=simulated_iso
        )

    def close(self) -> None:
        """Release OpenCV capture stream."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class MultiCameraManager:
    """Manages multi-camera streams on ONE logical Edge Device.

    Orchestrates camera workers, aggregates multi-camera telemetry into a unified
    store state, and feeds it into the RetailIntelligenceEngine.
    """

    def __init__(
        self,
        cameras: Optional[List[CameraConfig]] = None,
        intelligence_engine: Optional[RetailIntelligenceEngine] = None,
        store_id: str = "store_001",
        device_id: str = "edge_device_01",
        simulation_clock: Optional[SimulationClock] = None,
    ):
        """Initialize MultiCameraManager."""
        self.store_id = str(store_id)
        self.device_id = str(device_id)
        self.workers: Dict[str, CameraWorker] = {}
        self.clock = simulation_clock or SimulationClock()
        self.intelligence_engine = intelligence_engine or RetailIntelligenceEngine(
            store_id=self.store_id, device_id=self.device_id
        )

        self._thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(4, len(cameras) if cameras else 4),
            thread_name_prefix="CamWorker",
        )

        if cameras:
            for cam in cameras:
                self.add_camera(cam)

    def add_camera(
        self,
        config: CameraConfig,
        detector: Optional[PersonDetector] = None,
        tracker: Optional[PersonTracker] = None,
        fps: float = 25.0,
    ) -> CameraWorker:
        """Register a new camera worker."""
        worker = CameraWorker(config=config, detector=detector, tracker=tracker, fps=fps)
        self.workers[config.camera_id] = worker
        return worker

    def get_workers(self) -> List[CameraWorker]:
        """Return list of active camera workers."""
        return list(self.workers.values())

    def get_worker(self, camera_id: str) -> Optional[CameraWorker]:
        """Return worker by camera ID."""
        return self.workers.get(camera_id)

    @classmethod
    def create_four_camera_setup(
        cls,
        video_dir: str = "videos",
        store_id: str = "store_001",
        device_id: str = "edge_device_01",
        detector_cfg: Optional[DetectorConfig] = None,
        demo_start_time: str = "now",
        time_scale: float = 1.0,
        queue_config_path: str = "configs/queue_config.json",
        queue_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> "MultiCameraManager":
        """Build standard 4-camera recorded video store layout.

        CAM_01 = Food (videos/food/food.mp4)
        CAM_02 = Electronics (videos/electronics/electronics.mp4)
        CAM_03 = Grocery (videos/grocery/grocery.mp4)
        CAM_04 = Checkout (videos/checkout/checkout.mp4)
        """
        v_base = Path(video_dir)

        # Fallback to test.mp4 if specific file not found
        fallback_test = str(v_base / "test.mp4") if (v_base / "test.mp4").exists() else "test.mp4"

        food_path = str(v_base / "food" / "food.mp4")
        if not Path(food_path).exists():
            food_path = fallback_test

        elec_path = str(v_base / "electronics" / "electronics.mp4")
        if not Path(elec_path).exists():
            elec_path = fallback_test

        groc_path = str(v_base / "grocery" / "grocery.mp4")
        if not Path(groc_path).exists():
            groc_path = fallback_test

        check_path = str(v_base / "checkout" / "checkout.mp4")
        if not Path(check_path).exists():
            check_path = fallback_test

        # Resolve queue zone bounding box for checkout camera
        effective_queue_bbox = queue_bbox
        if effective_queue_bbox is None and queue_config_path:
            effective_queue_bbox = QueueAnalytics.load_zone_config(queue_config_path)

        camera_configs = [
            CameraConfig(
                camera_id="CAM_01",
                name="Food",
                zone_id="food",
                video_source=food_path,
                role="department",
                expected_staff=2,
                is_simulation=True,
            ),
            CameraConfig(
                camera_id="CAM_02",
                name="Electronics",
                zone_id="electronics",
                video_source=elec_path,
                role="department",
                expected_staff=1,
                is_simulation=True,
            ),
            CameraConfig(
                camera_id="CAM_03",
                name="Grocery",
                zone_id="grocery",
                video_source=groc_path,
                role="department",
                expected_staff=2,
                is_simulation=True,
            ),
            CameraConfig(
                camera_id="CAM_04",
                name="Checkout",
                zone_id="checkout",
                video_source=check_path,
                role="checkout",
                expected_staff=2,
                is_simulation=True,
                zone_bbox=effective_queue_bbox,
            ),
        ]

        clock = SimulationClock(demo_start_time=demo_start_time, time_scale=time_scale)
        manager = cls(
            cameras=camera_configs,
            store_id=store_id,
            device_id=device_id,
            simulation_clock=clock,
        )

        det_cfg = detector_cfg or DetectorConfig(imgsz=640, device="cpu")

        # Give each worker its own detector and tracker instance to ensure thread safety
        for worker in manager.get_workers():
            worker.detector = PersonDetector(config=det_cfg)
            worker.tracker = PersonTracker()
            worker.open_stream()

        return manager

    @classmethod
    def create_simulated_setup(
        cls,
        video_path: str = "videos/test.mp4",
        store_id: str = "store_001",
        device_id: str = "edge_device_01",
    ) -> "MultiCameraManager":
        """Build simulated setup (preserved for backward compatibility)."""
        camera_configs = [
            CameraConfig(
                camera_id="CAM_01",
                name="Food Section",
                zone_id="food",
                video_source=str(video_path),
                role="zone",
                expected_staff=2,
                is_simulation=True,
                zone_bbox=(50, 100, 300, 450),
            ),
            CameraConfig(
                camera_id="CAM_02",
                name="Electronics Section",
                zone_id="electronics",
                video_source=str(video_path),
                role="zone",
                expected_staff=2,
                is_simulation=True,
                zone_bbox=(300, 100, 580, 450),
            ),
            CameraConfig(
                camera_id="CAM_03",
                name="Clothing Section",
                zone_id="clothing",
                video_source=str(video_path),
                role="zone",
                expected_staff=1,
                is_simulation=True,
                zone_bbox=(50, 200, 400, 450),
            ),
            CameraConfig(
                camera_id="CAM_04",
                name="Store Entrance",
                zone_id="entrance",
                video_source=str(video_path),
                role="entrance",
                expected_staff=1,
                is_simulation=True,
            ),
            CameraConfig(
                camera_id="CAM_05",
                name="Checkout Queue",
                zone_id="checkout",
                video_source=str(video_path),
                role="checkout",
                expected_staff=2,
                is_simulation=True,
                zone_bbox=(250, 150, 550, 400),
            ),
        ]
        return cls(cameras=camera_configs, store_id=store_id, device_id=device_id)

    def process_concurrent_step(self) -> Tuple[List[CameraAnalytics], Dict[str, Any]]:
        """Run one frame processing step concurrently across all active camera workers.

        Returns:
            Tuple of (list of CameraAnalytics, aggregated store intelligence dict).
        """
        sim_time_str = self.clock.get_simulated_time_str()
        sim_iso = self.clock.get_simulated_iso()

        workers_list = list(self.workers.values())
        futures = {
            self._thread_pool.submit(
                w.step, simulated_time_str=sim_time_str, simulated_iso=sim_iso
            ): w
            for w in workers_list
        }

        analytics_list: List[CameraAnalytics] = []
        for fut in concurrent.futures.as_completed(futures):
            try:
                res = fut.result()
                if res is not None:
                    analytics_list.append(res)
            except Exception as e:
                w = futures[fut]
                logger.error(f"Error processing camera {w.config.camera_id}: {e}")

        # Sort by camera_id so order is deterministic: CAM_01, CAM_02, CAM_03, CAM_04
        analytics_list.sort(key=lambda a: a.camera_id)

        # Evaluate complete intelligence on aggregated camera metrics
        eval_result = self.aggregate_and_evaluate(
            analytics_list,
            current_time=self.clock.get_elapsed_seconds(),
            timestamp_iso=sim_iso,
        )

        return analytics_list, eval_result

    def aggregate_and_evaluate(
        self,
        camera_analytics_list: List[CameraAnalytics],
        current_time: Optional[float] = None,
        timestamp_iso: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Aggregate independent camera outputs and evaluate store intelligence."""
        aggregated_zones: Dict[str, ZoneMetrics] = {}
        primary_queue: Optional[QueueMetrics] = None
        primary_footfall: Optional[FootfallMetrics] = None

        total_live_shoppers = 0
        total_footfall = 0

        for a in camera_analytics_list:
            total_live_shoppers += a.current_shoppers
            total_footfall += a.footfall

            if a.zone_metrics is not None:
                aggregated_zones[a.zone_metrics.zone_id] = a.zone_metrics
            else:
                # Wrap department metrics into ZoneMetrics for intelligence engine evaluation
                z_m = ZoneMetrics(
                    zone_id=a.config.zone_id if hasattr(a, "config") else a.camera_id.lower(),
                    name=a.name,
                    current_shoppers=a.current_shoppers,
                    peak_shoppers=a.peak_shoppers,
                    avg_dwell=a.avg_dwell,
                    traffic_level=a.traffic_level,
                    expected_staff=a.expected_staff,
                    shopper_load_per_staff=a.shopper_load_per_staff,
                )
                aggregated_zones[z_m.zone_id] = z_m

            if a.queue_metrics is not None:
                primary_queue = a.queue_metrics
            elif a.role == "checkout":
                # Synthesize QueueMetrics from camera analytics
                primary_queue = QueueMetrics(
                    current_queue_length=a.current_shoppers,
                    peak_queue_length=a.peak_shoppers,
                    average_wait_time=a.avg_dwell,
                    maximum_wait_time=a.max_dwell,
                    congestion_level=a.traffic_level,
                )

            if a.footfall_metrics is not None:
                primary_footfall = a.footfall_metrics

        now_time = current_time if current_time is not None else self.clock.get_elapsed_seconds()
        now_iso = timestamp_iso or self.clock.get_simulated_iso()

        # Evaluate complete intelligence on edge
        intelligence_result = self.intelligence_engine.process_analytics(
            queue_metrics=primary_queue,
            zone_metrics_dict=aggregated_zones,
            footfall_metrics=primary_footfall,
            current_time=now_time,
            timestamp_iso=now_iso,
        )

        intelligence_result["edge_device"] = {
            "store_id": self.store_id,
            "device_id": self.device_id,
            "monitored_cameras": len(self.workers),
            "camera_ids": list(self.workers.keys()),
        }
        intelligence_result["camera_analytics"] = [a.to_dict() for a in camera_analytics_list]
        intelligence_result["simulation_clock"] = {
            "simulated_store_time": self.clock.get_simulated_time_str(now_time),
            "simulated_iso": now_iso,
            "demo_start_time": self.clock.demo_start_time_str,
            "elapsed_seconds": round(now_time, 2),
            "is_simulation": True,
        }
        intelligence_result["store_totals"] = {
            "total_live_shoppers": total_live_shoppers,
            "total_footfall": total_footfall,
        }
        return intelligence_result

    def save_snapshots_to_db(
        self,
        edge_db: Any,
        analytics_list: List[CameraAnalytics],
        intelligence_result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save camera snapshots with collision-free, camera-aware snapshot IDs.

        Guarantees snapshot IDs are unique across cameras and preserves PostgreSQL idempotency.
        """
        if edge_db is None or not analytics_list:
            return

        sim_time = self.clock.get_simulated_time_str()
        sim_iso = self.clock.get_simulated_iso()
        now_ist = datetime.now(IST).isoformat()

        # Insert per-camera department snapshots
        for a in analytics_list:
            # Camera-aware unique snapshot ID
            snapshot_id = f"{self.device_id}_{a.camera_id}_{a.name.lower()}_{sim_time.replace(':', '')}_{uuid.uuid4().hex[:6]}"
            try:
                edge_db.insert_zone_snapshot(
                    snapshot_id=snapshot_id,
                    store_id=self.store_id,
                    device_id=self.device_id,
                    camera_id=a.camera_id,
                    zone_id=a.name.lower(),
                    zone_name=a.name,
                    timestamp=sim_time,
                    current_shoppers=a.current_shoppers,
                    peak_shoppers=a.peak_shoppers,
                    avg_dwell=a.avg_dwell,
                    traffic_level=a.traffic_level,
                    expected_staff=a.expected_staff,
                    footfall=a.footfall,
                    max_dwell=a.max_dwell,
                    video_timestamp=a.timestamp,
                    created_at=sim_iso,
                    sync_status="PENDING",
                )
            except Exception as e:
                logger.debug(f"Snapshot insert error: {e}")

        # Insert store-level aggregate telemetry snapshot
        total_live = sum(a.current_shoppers for a in analytics_list if a.role != "checkout")
        total_footfall = sum(a.footfall for a in analytics_list if a.role != "checkout")
        q_cam = next((a for a in analytics_list if a.role == "checkout"), None)
        q_len = q_cam.current_shoppers if q_cam else 0
        q_peak = q_cam.peak_shoppers if q_cam else 0
        q_avg_wait = q_cam.avg_dwell if q_cam else 0.0

        all_dwells = [a.avg_dwell for a in analytics_list if a.avg_dwell > 0]
        avg_dwell = round(sum(all_dwells) / len(all_dwells), 1) if all_dwells else 0.0
        max_dwell = max([a.max_dwell for a in analytics_list], default=0.0)

        edge_db.insert_snapshot(
            store_id=self.store_id,
            device_id=self.device_id,
            timestamp=sim_time,
            entries=total_footfall,
            exits=max(0, total_footfall - total_live),
            occupancy=total_live,
            peak_occupancy=max(total_live, max([a.peak_shoppers for a in analytics_list], default=0)),
            queue_length=q_len,
            peak_queue=q_peak,
            avg_dwell=avg_dwell,
            max_dwell=max_dwell,
            avg_wait=q_avg_wait,
            sync_status="PENDING",
            created_at=sim_iso,
        )

        # Save active alerts
        if intelligence_result and "active_alerts" in intelligence_result:
            for alt in intelligence_result["active_alerts"]:
                try:
                    edge_db.insert_alert(
                        alert_id=alt.get("alert_id", f"alt_{uuid.uuid4().hex[:8]}"),
                        store_id=self.store_id,
                        device_id=self.device_id,
                        camera_id=alt.get("camera_id", "CAM_ALL"),
                        zone_id=alt.get("zone_id", "store"),
                        type=alt.get("type", "GENERAL"),
                        severity=alt.get("severity", "MEDIUM"),
                        title=alt.get("title", ""),
                        message=alt.get("message", ""),
                        current_value=float(alt.get("current_value", 0.0)),
                        threshold=float(alt.get("threshold", 0.0)),
                        recommendation=alt.get("recommendation", ""),
                        predicted_value=float(alt["predicted_value"]) if alt.get("predicted_value") is not None else None,
                        status=alt.get("status", "ACTIVE"),
                        created_at=sim_iso,
                        sync_status="PENDING",
                    )
                except Exception as e:
                    logger.debug(f"Alert insert error: {e}")

    def get_grid_frame(self, target_size: Tuple[int, int] = (1280, 720)) -> Optional[np.ndarray]:
        """Combine latest annotated frames from all 4 cameras into a 2x2 video grid."""
        workers = self.get_workers()
        if not workers:
            return None
        half_w = target_size[0] // 2
        half_h = target_size[1] // 2
        quadrants = []
        for w in workers[:4]:
            f = w.latest_annotated_frame if w.latest_annotated_frame is not None else w.latest_frame
            if f is None:
                f = np.zeros((half_h, half_w, 3), dtype=np.uint8)
            else:
                f = cv2.resize(f, (half_w, half_h))
            label = f"{w.config.camera_id} {w.config.name} ({len(w._track_enter_frames)} live)"
            cv2.putText(f, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2, cv2.LINE_AA)
            quadrants.append(f)
        while len(quadrants) < 4:
            quadrants.append(np.zeros((half_h, half_w, 3), dtype=np.uint8))
        top_row = np.hstack([quadrants[0], quadrants[1]])
        bottom_row = np.hstack([quadrants[2], quadrants[3]])
        return np.vstack([top_row, bottom_row])

    def close(self) -> None:
        """Release resources across all camera workers."""
        for worker in self.workers.values():
            worker.close()
        self._thread_pool.shutdown(wait=False)
