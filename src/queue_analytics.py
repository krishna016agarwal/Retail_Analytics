"""Queue Intelligence and Waiting Time Analytics Module (SIH Problem Statement 179).

Provides real-time queue length tracking, congestion status assessment (LOW, MEDIUM, HIGH),
shopper waiting time estimation, and automated counter recommendations for retail checkout lines.
Guarantees privacy: operates strictly on anonymous tracking IDs and 2D ground coordinates (bottom_center).
"""

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from configs.config import QueueConfig
from src.tracker import TrackingBatch, TrajectoryManager


class CongestionLevel(Enum):
    """Congestion classification based on queue occupancy."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class QueueTrackSession:
    """Represents an active or completed queue waiting session for a tracked person.

    Attributes:
        track_id: Anonymous tracking identifier.
        enter_frame: Frame index when person first entered queue zone.
        enter_time: Video timestamp (seconds) when person entered queue zone.
        last_seen_frame: Most recent frame index person was seen inside zone.
        leave_frame: Frame index when person left queue zone.
        leave_time: Video timestamp (seconds) when person left queue zone.
        wait_time: Total waiting duration in seconds (leave_time - enter_time).
    """

    track_id: int
    enter_frame: int
    enter_time: float
    last_seen_frame: int
    leave_frame: Optional[int] = None
    leave_time: Optional[float] = None
    wait_time: Optional[float] = None


@dataclass
class QueueMetrics:
    """Frame-level metrics for queue intelligence and HUD display.

    Attributes:
        current_queue_length: Number of actively tracked persons currently inside queue zone.
        peak_queue_length: Maximum queue length observed during the session.
        average_wait_time: Average waiting time (seconds) across completed queue visits.
        maximum_wait_time: Maximum waiting time (seconds) across completed queue visits.
        congestion_level: Current congestion category ('LOW', 'MEDIUM', 'HIGH').
        recommendation: Operational recommendation ('QUEUE NORMAL', 'MONITOR QUEUE', 'OPEN ADDITIONAL COUNTER').
        zone_bbox: Spatial boundary of queue zone (x1, y1, x2, y2).
        active_track_ids: List of track IDs currently inside the queue zone.
        completed_visits: Total number of persons who completed waiting in queue.
    """

    current_queue_length: int = 0
    peak_queue_length: int = 0
    average_wait_time: float = 0.0
    maximum_wait_time: float = 0.0
    congestion_level: str = "LOW"
    recommendation: str = "QUEUE NORMAL"
    zone_bbox: Tuple[int, int, int, int] = (300, 200, 600, 400)
    active_track_ids: List[int] = field(default_factory=list)
    completed_visits: int = 0


class QueueAnalytics:
    """Monitors rectangular queue zones, tracking queue occupancy and waiting times."""

    def __init__(
        self,
        config: Optional[QueueConfig] = None,
        fps: float = 25.0,
    ):
        """Initialize queue analytics module.

        Args:
            config: QueueConfig instance. If None, default settings are used.
            fps: Video frames per second for accurate time calculations.
        """
        self.config = config or QueueConfig()
        self.fps = max(1.0, float(fps))
        self.zone_bbox = self.config.zone_bbox

        # Metrics
        self.current_queue_length = 0
        self.peak_queue_length = 0
        self.completed_queue_visits = 0
        self.completed_wait_times: List[float] = []

        # Active queue sessions mapped by track_id
        self._active_sessions: Dict[int, QueueTrackSession] = {}

    def set_fps(self, fps: float) -> None:
        """Update video FPS for accurate dwell/waiting time calculation."""
        self.fps = max(1.0, float(fps))

    def set_zone_bbox(self, zone_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Dynamically update the queue zone bounding box (x1, y1, x2, y2).
        
        Guarantees normalized coordinates (x1 < x2, y1 < y2).
        """
        x1, y1, x2, y2 = zone_bbox
        norm_bbox = (min(int(x1), int(x2)), min(int(y1), int(y2)), max(int(x1), int(x2)), max(int(y1), int(y2)))
        self.zone_bbox = norm_bbox
        return norm_bbox

    @staticmethod
    def save_zone_config(
        zone_bbox: Tuple[int, int, int, int],
        config_path: str = "configs/queue_config.json",
        medium_threshold: int = 3,
        high_threshold: int = 6,
    ) -> bool:
        """Persist calibrated queue zone coordinates to a JSON configuration file."""
        try:
            p = Path(config_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "zone_bbox": [int(zone_bbox[0]), int(zone_bbox[1]), int(zone_bbox[2]), int(zone_bbox[3])],
                "medium_threshold": medium_threshold,
                "high_threshold": high_threshold,
            }
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception:
            return False

    @staticmethod
    def load_zone_config(
        config_path: str = "configs/queue_config.json",
    ) -> Optional[Tuple[int, int, int, int]]:
        """Load saved queue zone coordinates from a JSON configuration file if present."""
        try:
            p = Path(config_path)
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    bbox = data.get("zone_bbox")
                    if bbox and len(bbox) == 4:
                        return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        except Exception:
            pass
        return None

    def _point_in_zone(self, point: Tuple[int, int]) -> bool:
        """Check whether coordinate (x, y) lies inside the rectangular queue zone."""
        x, y = point
        x1, y1, x2, y2 = self.zone_bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    @property
    def average_wait_time(self) -> float:
        """Average waiting duration in seconds for completed queue visits."""
        if not self.completed_wait_times:
            return 0.0
        return sum(self.completed_wait_times) / len(self.completed_wait_times)

    @property
    def maximum_wait_time(self) -> float:
        """Maximum waiting duration in seconds for completed queue visits."""
        if not self.completed_wait_times:
            return 0.0
        return max(self.completed_wait_times)

    def _determine_congestion_and_recommendation(
        self, queue_length: int
    ) -> Tuple[str, str]:
        """Determine congestion category and actionable staff recommendation."""
        if queue_length >= self.config.high_threshold:
            return CongestionLevel.HIGH.value, "OPEN ADDITIONAL COUNTER"
        elif queue_length >= self.config.medium_threshold:
            return CongestionLevel.MEDIUM.value, "MONITOR QUEUE"
        return CongestionLevel.LOW.value, "QUEUE NORMAL"

    def update(
        self,
        batch: TrackingBatch,
        trajectory_manager: Optional[TrajectoryManager] = None,
    ) -> QueueMetrics:
        """Process active tracks for current frame and update queue intelligence.

        Args:
            batch: TrackingBatch containing active TrackedPerson instances.
            trajectory_manager: Optional TrajectoryManager to check track lifecycle termination.

        Returns:
            QueueMetrics with current queue length, wait times, congestion, and recommendation.
        """
        frame_id = batch.frame_id
        current_time = frame_id / self.fps
        observed_track_ids = set()
        active_in_zone_ids: List[int] = []

        for track in batch.tracks:
            t_id = track.track_id
            observed_track_ids.add(t_id)
            in_zone = self._point_in_zone(track.bottom_center)

            if in_zone:
                active_in_zone_ids.append(t_id)
                if t_id not in self._active_sessions:
                    # New arrival inside the queue zone
                    self._active_sessions[t_id] = QueueTrackSession(
                        track_id=t_id,
                        enter_frame=frame_id,
                        enter_time=current_time,
                        last_seen_frame=frame_id,
                    )
                else:
                    # Ongoing presence inside the queue zone
                    self._active_sessions[t_id].last_seen_frame = frame_id
            else:
                # Track is actively observed, BUT foot position is OUTSIDE the queue zone.
                # If it was previously queued, this is a confirmed exit from the queue zone.
                if t_id in self._active_sessions:
                    sess = self._active_sessions.pop(t_id)
                    sess.leave_frame = frame_id
                    sess.leave_time = current_time
                    sess.wait_time = max(0.0, sess.leave_time - sess.enter_time)
                    self.completed_wait_times.append(sess.wait_time)
                    self.completed_queue_visits += 1

        # Check for disappeared tracks that were in the queue zone.
        # Temporary occlusions/missing detections do NOT close the queue session immediately.
        # Only close if tracker lifecycle considers the track terminated (exceeded buffer frames).
        disappeared_ids = [
            t_id for t_id in self._active_sessions if t_id not in observed_track_ids
        ]
        for t_id in disappeared_ids:
            sess = self._active_sessions[t_id]
            frames_since_seen = frame_id - sess.last_seen_frame

            is_terminated = False
            if trajectory_manager is not None:
                hist = trajectory_manager.get_history(t_id)
                if hist is not None and not hist.is_active:
                    if frames_since_seen >= self.config.disappear_buffer_frames:
                        is_terminated = True
            elif frames_since_seen >= self.config.disappear_buffer_frames:
                is_terminated = True

            if is_terminated:
                # Track is officially confirmed terminated by tracker lifecycle
                self._active_sessions.pop(t_id)
                sess.leave_frame = sess.last_seen_frame
                sess.leave_time = sess.last_seen_frame / self.fps
                sess.wait_time = max(0.0, sess.leave_time - sess.enter_time)
                self.completed_wait_times.append(sess.wait_time)
                self.completed_queue_visits += 1

        self.current_queue_length = len(active_in_zone_ids)
        self.peak_queue_length = max(self.peak_queue_length, self.current_queue_length)

        congestion, recommendation = self._determine_congestion_and_recommendation(
            self.current_queue_length
        )

        return QueueMetrics(
            current_queue_length=self.current_queue_length,
            peak_queue_length=self.peak_queue_length,
            average_wait_time=self.average_wait_time,
            maximum_wait_time=self.maximum_wait_time,
            congestion_level=congestion,
            recommendation=recommendation,
            zone_bbox=self.zone_bbox,
            active_track_ids=active_in_zone_ids,
            completed_visits=self.completed_queue_visits,
        )

    def finalize(self, last_frame_id: int) -> None:
        """Close any remaining active sessions at video completion."""
        for sess in list(self._active_sessions.values()):
            sess.leave_frame = sess.last_seen_frame
            sess.leave_time = sess.last_seen_frame / self.fps
            sess.wait_time = max(0.0, sess.leave_time - sess.enter_time)
            self.completed_wait_times.append(sess.wait_time)
            self.completed_queue_visits += 1
        self._active_sessions.clear()

    def reset(self) -> None:
        """Reset all counters and state tracking."""
        self.current_queue_length = 0
        self.peak_queue_length = 0
        self.completed_queue_visits = 0
        self.completed_wait_times.clear()
        self._active_sessions.clear()
