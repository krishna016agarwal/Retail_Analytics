"""Zone Analytics Module for Retail Spatial Intelligence (SIH Problem Statement 179).

Provides real-time multi-zone occupancy monitoring, shopper dwell calculation,
traffic density classification, and staff coverage ratios per physical zone.
Guarantees privacy: operates strictly on anonymous tracking IDs and 2D ground-contact coordinates.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from configs.config import ZoneConfig
from src.tracker import TrackingBatch, TrajectoryManager


class TrafficLevel(Enum):
    """Traffic density classification for store zones."""

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    PEAK = "PEAK"


class ZoneStatus(Enum):
    """Operational status of a store zone."""

    OPTIMAL = "OPTIMAL"
    ATTENTION = "ATTENTION"
    CONGESTED = "CONGESTED"
    UNDERSTAFFED = "UNDERSTAFFED"


@dataclass
class ZoneTrackSession:
    """Represents an active or completed visit in a specific zone.

    Attributes:
        track_id: Anonymous tracking identifier.
        zone_id: Identifier of the zone.
        enter_frame: Frame index when person first entered the zone.
        enter_time: Video timestamp (seconds) when person entered zone.
        last_seen_frame: Most recent frame index person was detected inside zone.
        leave_frame: Frame index when person left zone.
        leave_time: Video timestamp (seconds) when person left zone.
        dwell_time: Total dwell time in seconds inside the zone.
    """

    track_id: int
    zone_id: str
    enter_frame: int
    enter_time: float
    last_seen_frame: int
    leave_frame: Optional[int] = None
    leave_time: Optional[float] = None
    dwell_time: Optional[float] = None


@dataclass
class ZoneMetrics:
    """Frame-level metrics for an individual store zone.

    Attributes:
        zone_id: Unique zone identifier (e.g., 'food', 'electronics').
        name: Human-readable zone name (e.g., 'Food & Snacks').
        current_shoppers: Actively tracked shoppers currently inside the zone.
        peak_shoppers: Maximum simultaneous shopper count observed.
        avg_dwell: Average dwell duration in seconds for completed visits.
        traffic_level: Traffic category ('LOW', 'NORMAL', 'HIGH', 'PEAK').
        zone_status: Operational status ('OPTIMAL', 'ATTENTION', 'CONGESTED', 'UNDERSTAFFED').
        expected_staff: Configured staff capacity for this zone.
        shopper_load_per_staff: Ratio of active shoppers to expected staff.
        active_track_ids: List of track IDs currently inside the zone.
        completed_visits: Total number of completed shopper visits in this zone.
        bbox: Spatial boundary (x1, y1, x2, y2).
    """

    zone_id: str
    name: str
    current_shoppers: int = 0
    peak_shoppers: int = 0
    avg_dwell: float = 0.0
    traffic_level: str = "LOW"
    zone_status: str = "OPTIMAL"
    expected_staff: int = 1
    shopper_load_per_staff: float = 0.0
    active_track_ids: List[int] = field(default_factory=list)
    completed_visits: int = 0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


class ZoneAnalyticsManager:
    """Manages spatial zone boundaries, occupancy tracking, and dwell time analysis."""

    def __init__(
        self,
        zones: Optional[List[ZoneConfig]] = None,
        fps: float = 25.0,
        max_load_threshold: float = 4.0,
        disappear_buffer_frames: int = 60,
    ):
        """Initialize zone analytics manager.

        Args:
            zones: List of ZoneConfig configurations. If None, empty list is used.
            fps: Video frames per second for accurate dwell timing.
            max_load_threshold: Shoppers-per-staff threshold for UNDERSTAFFED status.
            disappear_buffer_frames: Frame tolerance before a missing track is finalized.
        """
        self.fps = max(1.0, float(fps))
        self.max_load_threshold = float(max_load_threshold)
        self.disappear_buffer_frames = int(disappear_buffer_frames)

        # Zones registry: zone_id -> ZoneConfig
        self._zones: Dict[str, ZoneConfig] = {}

        # State per zone: zone_id -> {metrics, sessions, completed_dwells}
        self._current_occupancy: Dict[str, int] = {}
        self._peak_occupancy: Dict[str, int] = {}
        self._completed_visits: Dict[str, int] = {}
        self._completed_dwell_times: Dict[str, List[float]] = {}
        # Active sessions: (zone_id, track_id) -> ZoneTrackSession
        self._active_sessions: Dict[Tuple[str, int], ZoneTrackSession] = {}

        if zones:
            for z in zones:
                self.add_zone(z)

    def set_fps(self, fps: float) -> None:
        """Update video FPS for accurate dwell time calculation."""
        self.fps = max(1.0, float(fps))

    def add_zone(self, zone: ZoneConfig) -> None:
        """Register or update a store zone configuration."""
        self._zones[zone.id] = zone
        if zone.id not in self._current_occupancy:
            self._current_occupancy[zone.id] = 0
            self._peak_occupancy[zone.id] = 0
            self._completed_visits[zone.id] = 0
            self._completed_dwell_times[zone.id] = []

    def remove_zone(self, zone_id: str) -> None:
        """Remove a zone from monitoring."""
        self._zones.pop(zone_id, None)
        self._current_occupancy.pop(zone_id, None)
        self._peak_occupancy.pop(zone_id, None)
        self._completed_visits.pop(zone_id, None)
        self._completed_dwell_times.pop(zone_id, None)
        keys_to_remove = [k for k in self._active_sessions if k[0] == zone_id]
        for k in keys_to_remove:
            self._active_sessions.pop(k, None)

    def get_zones(self) -> List[ZoneConfig]:
        """Return list of configured zones."""
        return list(self._zones.values())

    @staticmethod
    def _point_in_zone(point: Tuple[int, int], zone: ZoneConfig) -> bool:
        """Check whether bottom-center foot coordinate (x, y) lies inside the zone."""
        x, y = point
        return zone.x1 <= x <= zone.x2 and zone.y1 <= y <= zone.y2

    def _determine_traffic_level(self, current: int, max_capacity: int) -> str:
        """Classify traffic level based on occupancy and zone capacity."""
        if max_capacity <= 0:
            max_capacity = 10
        ratio = current / max_capacity
        if ratio >= 0.90:
            return TrafficLevel.PEAK.value
        elif ratio >= 0.60:
            return TrafficLevel.HIGH.value
        elif ratio >= 0.25:
            return TrafficLevel.NORMAL.value
        return TrafficLevel.LOW.value

    def _determine_zone_status(
        self, current: int, expected_staff: int, max_capacity: int
    ) -> str:
        """Determine operational status based on staff load and congestion."""
        staff = max(1, expected_staff)
        load = current / staff
        if load >= self.max_load_threshold and current >= 2:
            return ZoneStatus.UNDERSTAFFED.value
        if max_capacity > 0 and current >= max_capacity:
            return ZoneStatus.CONGESTED.value
        if current >= (max_capacity * 0.75):
            return ZoneStatus.ATTENTION.value
        return ZoneStatus.OPTIMAL.value

    def update(
        self,
        batch: TrackingBatch,
        trajectory_manager: Optional[TrajectoryManager] = None,
    ) -> Dict[str, ZoneMetrics]:
        """Process active tracks for the current frame and update all zone metrics.

        Args:
            batch: TrackingBatch containing active TrackedPerson instances.
            trajectory_manager: Optional TrajectoryManager to check track lifecycle termination.

        Returns:
            Dictionary mapping zone_id -> ZoneMetrics.
        """
        frame_id = batch.frame_id
        current_time = frame_id / self.fps
        observed_track_ids = set()

        # Map zone_id -> list of track_ids present in this frame
        active_in_zones: Dict[str, List[int]] = {z_id: [] for z_id in self._zones}

        for track in batch.tracks:
            t_id = track.track_id
            observed_track_ids.add(t_id)
            pt = track.bottom_center

            for z_id, zone in self._zones.items():
                if self._point_in_zone(pt, zone):
                    active_in_zones[z_id].append(t_id)
                    key = (z_id, t_id)
                    if key not in self._active_sessions:
                        # New zone arrival
                        self._active_sessions[key] = ZoneTrackSession(
                            track_id=t_id,
                            zone_id=z_id,
                            enter_frame=frame_id,
                            enter_time=current_time,
                            last_seen_frame=frame_id,
                        )
                    else:
                        # Continuing zone presence
                        self._active_sessions[key].last_seen_frame = frame_id
                else:
                    # Track is observed in frame, but NOT in this zone.
                    # If it was previously in this zone, close session.
                    key = (z_id, t_id)
                    if key in self._active_sessions:
                        sess = self._active_sessions.pop(key)
                        sess.leave_frame = frame_id
                        sess.leave_time = current_time
                        sess.dwell_time = max(0.0, sess.leave_time - sess.enter_time)
                        self._completed_dwell_times[z_id].append(sess.dwell_time)
                        self._completed_visits[z_id] += 1

        # Check for disappeared tracks that were in zones
        disappeared_keys = [
            k for k in self._active_sessions if k[1] not in observed_track_ids
        ]
        for key in disappeared_keys:
            sess = self._active_sessions[key]
            z_id, t_id = key
            frames_since_seen = frame_id - sess.last_seen_frame

            is_terminated = False
            if trajectory_manager is not None:
                hist = trajectory_manager.get_history(t_id)
                if hist is not None and not hist.is_active:
                    if frames_since_seen >= self.disappear_buffer_frames:
                        is_terminated = True
            elif frames_since_seen >= self.disappear_buffer_frames:
                is_terminated = True

            if is_terminated:
                self._active_sessions.pop(key, None)
                sess.leave_frame = sess.last_seen_frame
                sess.leave_time = sess.last_seen_frame / self.fps
                sess.dwell_time = max(0.0, sess.leave_time - sess.enter_time)
                self._completed_dwell_times[z_id].append(sess.dwell_time)
                self._completed_visits[z_id] += 1

        # Compute metrics per zone
        results: Dict[str, ZoneMetrics] = {}
        for z_id, zone in self._zones.items():
            occupancy = len(active_in_zones[z_id])
            self._current_occupancy[z_id] = occupancy
            self._peak_occupancy[z_id] = max(self._peak_occupancy[z_id], occupancy)

            dwells = self._completed_dwell_times[z_id]
            avg_dwell = sum(dwells) / len(dwells) if dwells else 0.0

            staff = max(1, zone.expected_staff)
            load = round(occupancy / staff, 2)
            traffic = self._determine_traffic_level(occupancy, zone.max_shopper_capacity)
            status = self._determine_zone_status(occupancy, zone.expected_staff, zone.max_shopper_capacity)

            metrics = ZoneMetrics(
                zone_id=z_id,
                name=zone.name,
                current_shoppers=occupancy,
                peak_shoppers=self._peak_occupancy[z_id],
                avg_dwell=round(avg_dwell, 1),
                traffic_level=traffic,
                zone_status=status,
                expected_staff=zone.expected_staff,
                shopper_load_per_staff=load,
                active_track_ids=active_in_zones[z_id],
                completed_visits=self._completed_visits[z_id],
                bbox=(zone.x1, zone.y1, zone.x2, zone.y2),
            )
            results[z_id] = metrics

        return results

    def get_zone_snapshot_records(
        self,
        timestamp: str,
        store_id: str,
        device_id: str,
        camera_id: str = "CAM_01",
    ) -> List[Dict[str, Any]]:
        """Generate structured records suitable for database insertion.

        Args:
            timestamp: Video timeline position or ISO string.
            store_id: Retail branch identifier.
            device_id: Edge processing node identifier.
            camera_id: Camera identifier.

        Returns:
            List of snapshot dictionary records for each monitored zone.
        """
        records = []
        for z_id, zone in self._zones.items():
            occ = self._current_occupancy.get(z_id, 0)
            peak = self._peak_occupancy.get(z_id, 0)
            dwells = self._completed_dwell_times.get(z_id, [])
            avg_d = sum(dwells) / len(dwells) if dwells else 0.0
            traffic = self._determine_traffic_level(occ, zone.max_shopper_capacity)

            snapshot_id = f"{device_id}:{camera_id}:{z_id}:{timestamp}"
            records.append(
                {
                    "snapshot_id": snapshot_id,
                    "store_id": str(store_id),
                    "device_id": str(device_id),
                    "camera_id": str(camera_id),
                    "zone_id": str(z_id),
                    "zone_name": str(zone.name),
                    "timestamp": str(timestamp),
                    "current_shoppers": int(occ),
                    "peak_shoppers": int(peak),
                    "avg_dwell": float(round(avg_d, 2)),
                    "traffic_level": str(traffic),
                    "expected_staff": int(zone.expected_staff),
                }
            )
        return records

    def reset(self) -> None:
        """Reset all zone counts and active tracking sessions."""
        for z_id in self._zones:
            self._current_occupancy[z_id] = 0
            self._peak_occupancy[z_id] = 0
            self._completed_visits[z_id] = 0
            self._completed_dwell_times[z_id].clear()
        self._active_sessions.clear()
