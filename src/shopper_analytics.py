"""Shopper Analytics Module for Retail Intelligence (SIH Problem Statement 179).

Provides footfall counting, entry/exit boundary crossing detection,
current store occupancy, and peak occupancy tracking using ground-contact
person foot positions (bottom_center).
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Dict, List, Optional, Tuple

from configs.config import AnalyticsConfig
from src.tracker import TrackedPerson, TrackingBatch, TrajectoryManager


class LineSide(Enum):
    """Spatial position of a track relative to the boundary line."""

    ABOVE = "ABOVE"
    BELOW = "BELOW"
    ON_LINE = "ON_LINE"


@dataclass
class CrossingEvent:
    """Represents an individual entry or exit crossing event.

    Attributes:
        track_id: Anonymous tracking identifier.
        direction: Direction of crossing ('ENTRY' or 'EXIT').
        frame_id: Frame index at which crossing occurred.
        timestamp: Time of crossing event.
        coordinate: Ground-contact coordinate (x, y) where crossing occurred.
    """

    track_id: int
    direction: str
    frame_id: int
    timestamp: float
    coordinate: Tuple[int, int]


@dataclass
class ShopperDwellRecord:
    """Tracks frame observations, entry/exit events, and dwell duration for a tracked person.

    Attributes:
        track_id: Anonymous session-level track identifier.
        first_seen_frame: Frame index when first detected/tracked.
        last_seen_frame: Frame index of the most recent observation.
        total_observed_frames: Cumulative frames this person was actively tracked.
        entry_frame: Frame index of valid store entry crossing (if any).
        entry_time: Video timestamp (seconds) of entry crossing.
        exit_frame: Frame index of valid store exit crossing (if any).
        exit_time: Video timestamp (seconds) of exit crossing.
        dwell_time: Calculated dwell time in seconds (exit_time - entry_time for completed visits).
        is_completed_visit: Whether the person completed a valid entry -> exit cycle.
    """

    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    total_observed_frames: int = 1
    entry_frame: Optional[int] = None
    entry_time: Optional[float] = None
    exit_frame: Optional[int] = None
    exit_time: Optional[float] = None
    dwell_time: Optional[float] = None
    is_completed_visit: bool = False


@dataclass
class FootfallMetrics:
    """Current footfall, occupancy, and dwell time metrics.

    Attributes:
        total_entries: Cumulative count of entries into the retail space.
        total_exits: Cumulative count of exits from the retail space.
        current_occupancy: Estimated number of people currently inside (entries - exits).
        peak_occupancy: Maximum simultaneous occupancy observed during the session.
        recent_events: List of recent crossing events for UI notifications.
        completed_visits: Number of shoppers with a confirmed entry and subsequent exit.
        avg_dwell_time: Average dwell time in seconds across completed visits.
        max_dwell_time: Maximum dwell time in seconds across completed visits.
        min_dwell_time: Minimum dwell time in seconds across completed visits.
    """

    total_entries: int = 0
    total_exits: int = 0
    current_occupancy: int = 0
    peak_occupancy: int = 0
    recent_events: List[CrossingEvent] = field(default_factory=list)
    completed_visits: int = 0
    avg_dwell_time: float = 0.0
    max_dwell_time: float = 0.0
    min_dwell_time: float = 0.0


class EntryExitCounter:
    """Virtual line crossing detector and footfall counter.

    Uses the bottom-center (foot ground-plane position) of tracked people
    to detect crossings across a configurable horizontal boundary line.
    Prevents duplicate counting using trajectory state history and spatial hysteresis.
    """

    def __init__(
        self,
        config: Optional[AnalyticsConfig] = None,
        fps: float = 25.0,
    ):
        """Initialize counter with configuration options and video framerate.

        Args:
            config: AnalyticsConfig instance. If None, default settings are used.
            fps: Video frames per second for converting frame intervals to seconds.
        """
        self.config = config or AnalyticsConfig()
        self.fps = max(1.0, float(fps))
        self.line_y = self.config.entrance_line_y
        self.entry_direction = self.config.entry_direction.lower()
        self.buffer_margin = self.config.buffer_margin
        self.debounce_frames = self.config.debounce_frames

        # Metrics
        self.total_entries = 0
        self.total_exits = 0
        self.current_occupancy = 0
        self.peak_occupancy = 0
        self.events: List[CrossingEvent] = []

        # Dwell Time Analytics (Phase 4)
        self.dwell_records: Dict[int, ShopperDwellRecord] = {}
        self.completed_visits = 0
        self.completed_dwell_times: List[float] = []

        # State tracking per track_id
        self._last_side: Dict[int, LineSide] = {}
        self._last_crossing_frame: Dict[int, int] = {}
        self._counted_crossings: Dict[int, List[str]] = {}

    def set_fps(self, fps: float) -> None:
        """Update video FPS for accurate dwell-time calculation."""
        self.fps = max(1.0, float(fps))

    @property
    def avg_dwell_time(self) -> float:
        """Average dwell time in seconds across completed visits."""
        if not self.completed_dwell_times:
            return 0.0
        return sum(self.completed_dwell_times) / len(self.completed_dwell_times)

    @property
    def max_dwell_time(self) -> float:
        """Maximum dwell time in seconds across completed visits."""
        if not self.completed_dwell_times:
            return 0.0
        return max(self.completed_dwell_times)

    @property
    def min_dwell_time(self) -> float:
        """Minimum dwell time in seconds across completed visits."""
        if not self.completed_dwell_times:
            return 0.0
        return min(self.completed_dwell_times)

    def _determine_side(self, y: int) -> LineSide:
        """Determine which side of the boundary line a coordinate lies on."""
        if y < self.line_y - self.buffer_margin:
            return LineSide.ABOVE
        elif y > self.line_y + self.buffer_margin:
            return LineSide.BELOW
        return LineSide.ON_LINE

    def update(
        self,
        batch: TrackingBatch,
        trajectory_manager: Optional[TrajectoryManager] = None,
    ) -> FootfallMetrics:
        """Process active tracks in the frame and update crossing and dwell metrics.

        Args:
            batch: TrackingBatch from PersonTracker / BotSortTracker.
            trajectory_manager: Optional TrajectoryManager to inspect coordinate history.

        Returns:
            FootfallMetrics containing entries, exits, occupancy, and dwell metrics.
        """
        frame_id = batch.frame_id
        now = time.time()
        new_events: List[CrossingEvent] = []

        # Update observation records for every active track
        for track in batch.tracks:
            t_id = track.track_id
            if t_id not in self.dwell_records:
                self.dwell_records[t_id] = ShopperDwellRecord(
                    track_id=t_id,
                    first_seen_frame=frame_id,
                    last_seen_frame=frame_id,
                    total_observed_frames=1,
                )
            else:
                rec = self.dwell_records[t_id]
                rec.last_seen_frame = frame_id
                rec.total_observed_frames += 1

            _, foot_y = track.bottom_center
            current_side = self._determine_side(foot_y)

            # If track is seen for the first time, record initial side without counting
            if t_id not in self._last_side:
                if current_side != LineSide.ON_LINE:
                    self._last_side[t_id] = current_side
                continue

            previous_side = self._last_side[t_id]

            # Only evaluate when person has clearly transitioned to a definite side
            if current_side == LineSide.ON_LINE or current_side == previous_side:
                continue

            # Check debounce time to prevent oscillation on the line
            last_frame = self._last_crossing_frame.get(t_id, -self.debounce_frames)
            if frame_id - last_frame < self.debounce_frames:
                self._last_side[t_id] = current_side
                continue

            # Verify trajectory history: ensure track was on previous side in recent frames
            has_history = True
            if trajectory_manager is not None:
                trail = trajectory_manager.get_recent_trail(t_id)
                if len(trail) >= 2:
                    pass

            # Detect direction:
            # Transition: ABOVE -> BELOW
            if previous_side == LineSide.ABOVE and current_side == LineSide.BELOW:
                direction = "ENTRY" if self.entry_direction == "down" else "EXIT"
            # Transition: BELOW -> ABOVE
            elif previous_side == LineSide.BELOW and current_side == LineSide.ABOVE:
                direction = "EXIT" if self.entry_direction == "down" else "ENTRY"
            else:
                direction = None

            if direction and has_history:
                event = CrossingEvent(
                    track_id=t_id,
                    direction=direction,
                    frame_id=frame_id,
                    timestamp=now,
                    coordinate=track.bottom_center,
                )
                self.events.append(event)
                new_events.append(event)

                rec = self.dwell_records.get(t_id)

                if direction == "ENTRY":
                    self.total_entries += 1
                    if rec is not None and rec.entry_frame is None:
                        rec.entry_frame = frame_id
                        rec.entry_time = frame_id / self.fps
                else:
                    self.total_exits += 1
                    if rec is not None:
                        rec.exit_frame = frame_id
                        rec.exit_time = frame_id / self.fps
                        # Only count completed store visit if preceded by a valid ENTRY event
                        if rec.entry_frame is not None and not rec.is_completed_visit:
                            rec.dwell_time = rec.exit_time - rec.entry_time
                            rec.is_completed_visit = True
                            self.completed_visits += 1
                            self.completed_dwell_times.append(rec.dwell_time)

                # Update occupancy metrics
                self.current_occupancy = max(0, self.total_entries - self.total_exits)
                self.peak_occupancy = max(self.peak_occupancy, self.current_occupancy)

                # Update track state
                self._last_crossing_frame[t_id] = frame_id
                if t_id not in self._counted_crossings:
                    self._counted_crossings[t_id] = []
                self._counted_crossings[t_id].append(direction)

            self._last_side[t_id] = current_side

        return FootfallMetrics(
            total_entries=self.total_entries,
            total_exits=self.total_exits,
            current_occupancy=self.current_occupancy,
            peak_occupancy=self.peak_occupancy,
            recent_events=new_events,
            completed_visits=self.completed_visits,
            avg_dwell_time=self.avg_dwell_time,
            max_dwell_time=self.max_dwell_time,
            min_dwell_time=self.min_dwell_time,
        )

    def reset(self) -> None:
        """Reset all counters, dwell records, and state tracking."""
        self.total_entries = 0
        self.total_exits = 0
        self.current_occupancy = 0
        self.peak_occupancy = 0
        self.events.clear()
        self.dwell_records.clear()
        self.completed_visits = 0
        self.completed_dwell_times.clear()
        self._last_side.clear()
        self._last_crossing_frame.clear()
        self._counted_crossings.clear()
