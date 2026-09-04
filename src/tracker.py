"""Person Tracking and Trajectory Management module using ByteTrack.

This module provides multi-object tracking for the 'person' class, assigns
persistent anonymous session IDs (Person 1, Person 2, ...), and records spatial
trajectories in memory for downstream retail analytics (dwell time, zones, heatmaps).
"""

from collections import deque
from dataclasses import dataclass, field
import time
from types import SimpleNamespace
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker, BaseTrack
import torch

from configs.config import TrackerConfig
from src.detector import Detection, DetectionBatch


@dataclass
class TrackedPerson:
    """Represents a single tracked person with a persistent session ID.

    Attributes:
        track_id: Anonymous session-level tracking identifier (1, 2, 3...).
        bbox: Bounding box coordinates (x1, y1, x2, y2) in integer pixels.
        confidence: Detection/tracking confidence score in range [0.0, 1.0].
        class_id: Target class ID (0 for COCO 'person').
        class_name: Human-readable class name ('person').
    """

    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int = 0
    class_name: str = "person"

    @property
    def center(self) -> Tuple[int, int]:
        """Return center point (cx, cy) of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def bottom_center(self) -> Tuple[int, int]:
        """Return bottom-center point (cx, y2) representing ground/foot position.

        Essential for floor-plane mapping, zone intersection, and spatial heatmaps.
        """
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) // 2, y2)

    @property
    def display_id(self) -> str:
        """Formatted label for visualization (e.g., 'Person 1')."""
        return f"Person {self.track_id}"


@dataclass
class TrajectoryHistory:
    """Historical spatial trajectory for an individual tracked person."""

    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    recent_centers: deque = field(default_factory=deque)
    recent_bottom_centers: deque = field(default_factory=deque)
    all_centers: List[Tuple[int, int]] = field(default_factory=list)
    total_frames_active: int = 0
    is_active: bool = True

    def add_point(
        self,
        center: Tuple[int, int],
        bottom_center: Tuple[int, int],
        frame_id: int,
    ) -> None:
        """Append a new coordinate observation to the trajectory."""
        self.recent_centers.append(center)
        self.recent_bottom_centers.append(bottom_center)
        self.all_centers.append(center)
        self.last_seen_frame = frame_id
        self.total_frames_active += 1
        self.is_active = True


class TrajectoryManager:
    """Manages in-memory trajectory histories for all observed tracks.

    Designed to feed directly into downstream analytics modules:
    - Entry/Exit line crossing
    - Area/Zone occupancy
    - Dwell-time calculation
    - Spatial heatmaps
    """

    def __init__(self, max_trail_points: int = 40):
        """Initialize manager.

        Args:
            max_trail_points: Maximum recent points to keep in memory for visual trail.
        """
        self.max_trail_points = max_trail_points
        self._histories: Dict[int, TrajectoryHistory] = {}
        self._unique_track_ids: Set[int] = set()

    def update(self, tracks: List[TrackedPerson], frame_id: int) -> None:
        """Update trajectories with the current frame's tracked persons.

        Args:
            tracks: List of TrackedPerson objects active in the current frame.
            frame_id: Monotonically increasing frame index.
        """
        current_active_ids = {t.track_id for t in tracks}

        for track in tracks:
            t_id = track.track_id
            self._unique_track_ids.add(t_id)

            if t_id not in self._histories:
                self._histories[t_id] = TrajectoryHistory(
                    track_id=t_id,
                    first_seen_frame=frame_id,
                    last_seen_frame=frame_id,
                    recent_centers=deque(maxlen=self.max_trail_points),
                    recent_bottom_centers=deque(maxlen=self.max_trail_points),
                )

            self._histories[t_id].add_point(track.center, track.bottom_center, frame_id)

        # Mark missing tracks as inactive
        for t_id, hist in self._histories.items():
            if t_id not in current_active_ids:
                hist.is_active = False

    def get_recent_trail(self, track_id: int) -> List[Tuple[int, int]]:
        """Return list of recent center points for rendering a motion trail."""
        hist = self._histories.get(track_id)
        if hist is None:
            return []
        return list(hist.recent_centers)

    def get_history(self, track_id: int) -> Optional[TrajectoryHistory]:
        """Return the complete trajectory history object for a track ID."""
        return self._histories.get(track_id)

    def get_all_histories(self) -> Dict[int, TrajectoryHistory]:
        """Return entire in-memory trajectory dictionary (for heatmap & analytics)."""
        return self._histories

    @property
    def unique_track_count(self) -> int:
        """Return total count of distinct track IDs observed across the session."""
        return len(self._unique_track_ids)

    def reset(self) -> None:
        """Clear all stored trajectories and ID sets."""
        self._histories.clear()
        self._unique_track_ids.clear()


@dataclass
class TrackingBatch:
    """Container for tracked persons and performance metadata for a frame."""

    tracks: List[TrackedPerson] = field(default_factory=list)
    detections: List[Detection] = field(default_factory=list)
    frame_id: int = 0
    tracking_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    total_unique_ids: int = 0

    @property
    def count(self) -> int:
        """Number of active tracks in this frame."""
        return len(self.tracks)

    @property
    def active_ids(self) -> List[int]:
        """List of active track IDs in this frame."""
        return [t.track_id for t in self.tracks]


class PersonTracker:
    """ByteTrack-based multi-object tracker for person detection batches."""

    def __init__(self, config: Optional[TrackerConfig] = None):
        """Initialize ByteTrack tracker with specified or default configuration.

        Args:
            config: TrackerConfig options. If None, default settings are used.
        """
        self.config = config or TrackerConfig()
        self.frame_id = 0
        BaseTrack._count = 0

        # Construct configuration namespace for Ultralytics BYTETracker
        tracker_args = SimpleNamespace(
            track_high_thresh=self.config.track_high_thresh,
            track_low_thresh=self.config.track_low_thresh,
            new_track_thresh=self.config.new_track_thresh,
            track_buffer=self.config.track_buffer,
            match_thresh=self.config.match_thresh,
            fuse_score=self.config.fuse_score,
        )
        self._byte_tracker = BYTETracker(tracker_args)
        self.trajectory_manager = TrajectoryManager(
            max_trail_points=self.config.max_trajectory_points
        )

    def update(
        self,
        batch: DetectionBatch,
        frame: np.ndarray,
    ) -> TrackingBatch:
        """Update tracks using detections from the current frame.

        Args:
            batch: DetectionBatch produced by PersonDetector.
            frame: Current BGR image array (H, W, 3).

        Returns:
            TrackingBatch with active TrackedPerson objects and performance metrics.
        """
        self.frame_id += 1
        start_time = time.perf_counter()

        # Prepare boxes for Ultralytics ByteTracker
        raw_boxes = batch.raw_boxes
        h, w = frame.shape[:2]

        if raw_boxes is None:
            # Construct empty Boxes container if no raw detection container is available
            raw_boxes = Boxes(torch.empty((0, 6)), (h, w))

        # Run ByteTrack update step on CPU
        tracked_results = self._byte_tracker.update(raw_boxes, frame)
        tracking_time_ms = (time.perf_counter() - start_time) * 1000.0

        active_tracks: List[TrackedPerson] = []
        if len(tracked_results) > 0:
            for item in tracked_results:
                # Format: [x1, y1, x2, y2, track_id, score, cls, idx]
                x1, y1, x2, y2 = map(int, item[:4])
                track_id = int(item[4])
                score = float(item[5])
                cls_id = int(item[6]) if len(item) > 6 else 0

                # Ensure exclusively person class
                if cls_id == 0:
                    active_tracks.append(
                        TrackedPerson(
                            track_id=track_id,
                            bbox=(x1, y1, x2, y2),
                            confidence=score,
                            class_id=0,
                            class_name="person",
                        )
                    )

        # Update in-memory trajectory manager
        self.trajectory_manager.update(active_tracks, self.frame_id)

        return TrackingBatch(
            tracks=active_tracks,
            detections=batch.detections,
            frame_id=self.frame_id,
            tracking_time_ms=tracking_time_ms,
            inference_time_ms=batch.inference_time_ms,
            total_unique_ids=self.trajectory_manager.unique_track_count,
        )

    def reset(self) -> None:
        """Reset internal frame counter, tracker state, and trajectory memory."""
        self.frame_id = 0
        BaseTrack._count = 0
        tracker_args = SimpleNamespace(
            track_high_thresh=self.config.track_high_thresh,
            track_low_thresh=self.config.track_low_thresh,
            new_track_thresh=self.config.new_track_thresh,
            track_buffer=self.config.track_buffer,
            match_thresh=self.config.match_thresh,
            fuse_score=self.config.fuse_score,
        )
        self._byte_tracker = BYTETracker(tracker_args)
        self.trajectory_manager.reset()
