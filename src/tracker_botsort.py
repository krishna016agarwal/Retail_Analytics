"""Experimental BoT-SORT Tracker implementation using Ultralytics native BOTSORT.

Provides multi-object tracking for the 'person' class using BoT-SORT with
optional Global Motion Compensation (GMC) and appearance ReID integration,
while preserving anonymous tracking and trajectory management.
"""

from dataclasses import dataclass, field
import time
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BaseTrack

from src.detector import DetectionBatch
from src.tracker import TrackedPerson, TrackingBatch, TrajectoryManager


@dataclass
class BotSortTrackerConfig:
    """Configuration options for Ultralytics BoT-SORT tracker."""

    # Association thresholds
    track_high_thresh: float = 0.35
    track_low_thresh: float = 0.05
    new_track_thresh: float = 0.35
    track_buffer: int = 60
    match_thresh: float = 0.80
    fuse_score: bool = True

    # BoT-SORT specific parameters
    gmc_method: str = "sparseOptFlow"  # 'sparseOptFlow', 'none', 'orb', 'sift', 'ecc'
    proximity_thresh: float = 0.50
    appearance_thresh: float = 0.80
    with_reid: bool = False
    reid_model: Optional[str] = "auto"
    device: str = "cpu"

    # Trajectory configuration
    max_trajectory_points: int = 40
    draw_trajectories: bool = True


class BotSortTracker:
    """BoT-SORT based multi-object tracker for person detection batches."""

    def __init__(self, config: Optional[BotSortTrackerConfig] = None):
        """Initialize BoT-SORT tracker with specified or default configuration.

        Args:
            config: BotSortTrackerConfig options. If None, default settings are used.
        """
        self.config = config or BotSortTrackerConfig()
        self.frame_id = 0
        BaseTrack._count = 0

        # Construct configuration namespace for Ultralytics BOTSORT
        tracker_args = SimpleNamespace(
            track_high_thresh=self.config.track_high_thresh,
            track_low_thresh=self.config.track_low_thresh,
            new_track_thresh=self.config.new_track_thresh,
            track_buffer=self.config.track_buffer,
            match_thresh=self.config.match_thresh,
            fuse_score=self.config.fuse_score,
            gmc_method=self.config.gmc_method,
            proximity_thresh=self.config.proximity_thresh,
            appearance_thresh=self.config.appearance_thresh,
            with_reid=self.config.with_reid,
            model=self.config.reid_model if self.config.with_reid else None,
            device=self.config.device,
        )
        self._botsort_tracker = BOTSORT(tracker_args)
        self.trajectory_manager = TrajectoryManager(
            max_trail_points=self.config.max_trajectory_points
        )

    def update(
        self,
        batch: DetectionBatch,
        frame: np.ndarray,
    ) -> TrackingBatch:
        """Update tracks using detections from the current frame via BoT-SORT.

        Args:
            batch: DetectionBatch produced by PersonDetector.
            frame: Current BGR image array (H, W, 3).

        Returns:
            TrackingBatch with active TrackedPerson objects and performance metrics.
        """
        self.frame_id += 1
        start_time = time.perf_counter()

        # Prepare boxes for Ultralytics BOTSORT
        raw_boxes = batch.raw_boxes
        h, w = frame.shape[:2]

        if raw_boxes is None:
            raw_boxes = Boxes(torch.empty((0, 6)), (h, w))

        # Run BoT-SORT update step on CPU
        tracked_results = self._botsort_tracker.update(raw_boxes, frame)
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
        """Reset frame counter, tracker state, and trajectory memory."""
        self.frame_id = 0
        BaseTrack._count = 0
        tracker_args = SimpleNamespace(
            track_high_thresh=self.config.track_high_thresh,
            track_low_thresh=self.config.track_low_thresh,
            new_track_thresh=self.config.new_track_thresh,
            track_buffer=self.config.track_buffer,
            match_thresh=self.config.match_thresh,
            fuse_score=self.config.fuse_score,
            gmc_method=self.config.gmc_method,
            proximity_thresh=self.config.proximity_thresh,
            appearance_thresh=self.config.appearance_thresh,
            with_reid=self.config.with_reid,
            model=self.config.reid_model if self.config.with_reid else None,
            device=self.config.device,
        )
        self._botsort_tracker = BOTSORT(tracker_args)
        self.trajectory_manager.reset()
