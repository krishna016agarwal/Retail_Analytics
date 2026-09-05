"""Multi-Camera Architecture and Stream Abstraction for Edge Retail Computing (SIH 179).

Provides multi-stream camera coordination:
- Individual CameraWorker instances handle video reading, detection, tracking, and localized analytics.
- MultiCameraManager aggregates independent camera telemetry on ONE logical Edge Device.
- Feeds aggregated telemetry into RetailIntelligenceEngine to generate centralized store insights and alerts.
- Includes simulation mode for testing concurrency without fabricating artificial metrics.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from configs.config import AnalyticsConfig, DetectorConfig, QueueConfig, TrackerConfig, ZoneConfig
from src.detector import PersonDetector
from src.queue_analytics import QueueAnalytics, QueueMetrics
from src.retail_intelligence import RetailIntelligenceEngine
from src.shopper_analytics import EntryExitCounter, FootfallMetrics
from src.tracker import PersonTracker
from src.zone_analytics import ZoneAnalyticsManager, ZoneMetrics

logger = logging.getLogger("camera_manager")


@dataclass
class CameraConfig:
    """Configuration for an individual camera stream.

    Attributes:
        camera_id: Unique camera ID (e.g., 'CAM_01', 'CAM_02').
        name: Human-readable camera label (e.g., 'CAM_01: Food Section').
        zone_id: Store zone or role associated with this camera.
        video_source: Path to video file or RTSP stream URI.
        role: Primary analytical role ('zone', 'entrance', 'checkout').
        expected_staff: Configured staff capacity for this section.
        is_simulation: True if using shared test video for concurrency testing.
        zone_bbox: Optional spatial bounding box (x1, y1, x2, y2) in frame.
    """

    camera_id: str
    name: str
    zone_id: str
    video_source: str
    role: str = "zone"
    expected_staff: int = 1
    is_simulation: bool = False
    zone_bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass
class CameraAnalytics:
    """Frame-level analytics produced by a single camera worker."""

    camera_id: str
    name: str
    role: str
    timestamp: str
    is_simulation: bool
    active_persons_detected: int = 0
    footfall_metrics: Optional[FootfallMetrics] = None
    queue_metrics: Optional[QueueMetrics] = None
    zone_metrics: Optional[ZoneMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert camera analytics to dictionary."""
        res: Dict[str, Any] = {
            "camera_id": self.camera_id,
            "name": self.name,
            "role": self.role,
            "timestamp": self.timestamp,
            "is_simulation": self.is_simulation,
            "active_persons_detected": self.active_persons_detected,
        }
        if self.footfall_metrics:
            res["footfall"] = {
                "entries": self.footfall_metrics.total_entries,
                "exits": self.footfall_metrics.total_exits,
                "occupancy": self.footfall_metrics.current_occupancy,
            }
        if self.queue_metrics:
            res["queue"] = {
                "current_length": self.queue_metrics.current_queue_length,
                "congestion_level": self.queue_metrics.congestion_level,
            }
        if self.zone_metrics:
            res["zone"] = self.zone_metrics.to_dict()
        return res


class CameraWorker:
    """Worker responsible for frame capture, detection, tracking, and localized analytics.

    Note: CameraWorker does NOT run high-level business intelligence. It produces
    raw localized analytics which are passed to MultiCameraManager and the centralized
    RetailIntelligenceEngine.
    """

    def __init__(
        self,
        config: CameraConfig,
        detector: Optional[PersonDetector] = None,
        tracker: Optional[PersonTracker] = None,
        fps: float = 25.0,
    ):
        """Initialize camera worker with components."""
        self.config = config
        self.fps = max(1.0, float(fps))
        self.detector = detector
        self.tracker = tracker

        self.counter: Optional[EntryExitCounter] = None
        self.queue_analytics: Optional[QueueAnalytics] = None
        self.zone_manager: Optional[ZoneAnalyticsManager] = None

        self._cap: Optional[cv2.VideoCapture] = None
        self.frame_idx = 0

        self._setup_analytics()

    def _setup_analytics(self) -> None:
        """Configure analytics according to the camera's designated role."""
        if self.config.role == "entrance":
            self.counter = EntryExitCounter(
                config=AnalyticsConfig(entrance_line_y=300, entry_direction="down"),
                fps=self.fps,
            )
        elif self.config.role == "checkout":
            q_bbox = self.config.zone_bbox or (300, 200, 600, 400)
            self.queue_analytics = QueueAnalytics(
                config=QueueConfig(zone_bbox=q_bbox, medium_threshold=3, high_threshold=6),
                fps=self.fps,
            )
        else:
            # Default zone camera
            z_bbox = self.config.zone_bbox or (50, 50, 600, 450)
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
        """Open the video capture stream."""
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(str(self.config.video_source))
        return self._cap.isOpened()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame from the camera stream."""
        if self._cap is None:
            if not self.open_stream():
                return False, None
        ret, frame = self._cap.read()
        return ret, frame

    def process_frame(self, frame: np.ndarray) -> CameraAnalytics:
        """Run detection, tracking, and localized analytics on a single frame."""
        self.frame_idx += 1
        video_time = self.frame_idx / self.fps
        ts = f"T+{video_time:.2f}s"

        footfall_m: Optional[FootfallMetrics] = None
        queue_m: Optional[QueueMetrics] = None
        zone_m: Optional[ZoneMetrics] = None
        person_count = 0

        if self.detector is not None:
            batch_det = self.detector.detect(frame)
            person_count = len(batch_det.detections)


            if self.tracker is not None:
                track_batch = self.tracker.update(batch_det, frame)


                # Localized analytics by role
                if self.counter is not None:
                    footfall_m = self.counter.update(track_batch, self.tracker.trajectory_manager)
                if self.queue_analytics is not None:
                    queue_m = self.queue_analytics.update(track_batch, self.tracker.trajectory_manager)
                if self.zone_manager is not None:
                    z_dict = self.zone_manager.update(track_batch, self.tracker.trajectory_manager)
                    zone_m = z_dict.get(self.config.zone_id)

        return CameraAnalytics(
            camera_id=self.config.camera_id,
            name=self.config.name,
            role=self.config.role,
            timestamp=ts,
            is_simulation=self.config.is_simulation,
            active_persons_detected=person_count,
            footfall_metrics=footfall_m,
            queue_metrics=queue_m,
            zone_metrics=zone_m,
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
    ):
        """Initialize MultiCameraManager."""
        self.store_id = str(store_id)
        self.device_id = str(device_id)
        self.workers: Dict[str, CameraWorker] = {}
        self.intelligence_engine = intelligence_engine or RetailIntelligenceEngine(
            store_id=self.store_id, device_id=self.device_id
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

    @classmethod
    def create_simulated_setup(
        cls,
        video_path: str = "videos/test.mp4",
        store_id: str = "store_001",
        device_id: str = "edge_device_01",
    ) -> "MultiCameraManager":
        """Build the required multi-camera architecture using simulated test feeds.

        Note: As real multi-angle retail recordings are unavailable, this assigns
        the existing test video to multiple workers strictly for concurrency/architecture
        testing, clearly tagging each with is_simulation: True.
        """
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

    def aggregate_and_evaluate(
        self,
        camera_analytics_list: List[CameraAnalytics],
        current_time: Optional[float] = None,
        timestamp_iso: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Aggregate independent camera outputs and evaluate store intelligence.

        Flow:
        Camera Workers -> Camera Analytics -> MultiCameraManager -> RetailIntelligenceEngine -> Alerts
        """
        aggregated_zones: Dict[str, ZoneMetrics] = {}
        primary_queue: Optional[QueueMetrics] = None
        primary_footfall: Optional[FootfallMetrics] = None

        for a in camera_analytics_list:
            if a.zone_metrics is not None:
                aggregated_zones[a.zone_metrics.zone_id] = a.zone_metrics
            if a.queue_metrics is not None:
                primary_queue = a.queue_metrics
            if a.footfall_metrics is not None:
                primary_footfall = a.footfall_metrics

        # Evaluate complete intelligence on the edge
        intelligence_result = self.intelligence_engine.process_analytics(
            queue_metrics=primary_queue,
            zone_metrics_dict=aggregated_zones,
            footfall_metrics=primary_footfall,
            current_time=current_time,
            timestamp_iso=timestamp_iso,
        )

        intelligence_result["edge_device"] = {
            "store_id": self.store_id,
            "device_id": self.device_id,
            "monitored_cameras": len(self.workers),
            "camera_ids": list(self.workers.keys()),
        }
        intelligence_result["camera_analytics"] = [
            a.to_dict() for a in camera_analytics_list
        ]
        return intelligence_result

    def close(self) -> None:
        """Release resources across all camera workers."""
        for worker in self.workers.values():
            worker.close()
