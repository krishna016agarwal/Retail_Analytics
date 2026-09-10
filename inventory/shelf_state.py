"""Temporal shelf-state stability tracker.

Analyses a rolling window of frame observations to:
- Distinguish genuine shelf-state changes from transient occlusions.
- Mark frames as UNCERTAIN when a person blocks the shelf.
- Report persistent STATE_CHANGED only after sufficient stable evidence.
- Avoid claiming inventory changes from single blocked frames.

State transitions
-----------------
    INITIALIZING      Not enough frames in window yet.
    STABLE            Consistent visible count across clear frames.
    UNCERTAIN         Person detected OR sudden detection drop (possible occlusion).
    POSSIBLY_CHANGING Gradual drift in clear-frame count; needs more evidence.
    STATE_CHANGED     Persistent significant change confirmed across stable frames.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

from inventory.config import InventoryTemporalConfig
from inventory.shelf_detector import ShelfDetectionBatch


class ObservationStatus(str, Enum):
    """Frame-level and window-level shelf observation status labels."""

    INITIALIZING = "INITIALIZING"
    STABLE = "STABLE"
    UNCERTAIN = "UNCERTAIN"
    POSSIBLY_CHANGING = "POSSIBLY_CHANGING"
    STATE_CHANGED = "STATE_CHANGED"


@dataclass
class ShelfObservation:
    """Internal record for a single-frame observation."""

    frame_index: int
    timestamp_sec: float
    visible_count: int
    person_detected: bool
    raw_status: ObservationStatus


@dataclass
class ShelfStateSnapshot:
    """Output of temporal analysis for the current frame.

    This is the authoritative shelf-state output consumed by the visualizer,
    pipeline, and report.

    Attributes:
        observation_status: Current window-level status.
        stable_visible_count: Median visible count from reliable (non-occluded)
                              frames in the rolling window. This is the best
                              estimate of current visible facings.
                              NOT total physical inventory quantity.
        visible_count_this_frame: Raw visible count from the current frame only.
        state_change_detected: True ONLY if a persistent change has been
                               confirmed across min_stable_frames consecutive
                               reliable frames. A single blocked frame does
                               NOT set this flag.
        change_description: Human-readable description of the confirmed change,
                            e.g. "Visible facings decreased from 12 to 6".
        uncertainty_reason: Why the observation is UNCERTAIN (if applicable),
                            e.g. "Person detected — shelf partially blocked".
        window_size: Total observations currently in the rolling window.
        clear_frame_count: Frames in window classified as reliable (no occlusion).
        uncertain_frame_count: Frames in window classified as uncertain.
        previous_stable_count: Stable baseline before the current state change.
        shelf_occupancy_pct: Rough visible occupancy relative to window peak.
    """

    observation_status: ObservationStatus = ObservationStatus.INITIALIZING
    stable_visible_count: int = 0
    visible_count_this_frame: int = 0
    state_change_detected: bool = False
    change_description: str = ""
    uncertainty_reason: str = ""
    window_size: int = 0
    clear_frame_count: int = 0
    uncertain_frame_count: int = 0
    previous_stable_count: int = 0
    shelf_occupancy_pct: float = 0.0


def _median(values: List[int]) -> float:
    """Compute the median of a non-empty integer list."""
    if not values:
        return 0.0
    sv = sorted(values)
    n = len(sv)
    mid = n // 2
    return float(sv[mid]) if n % 2 == 1 else (sv[mid - 1] + sv[mid]) / 2.0


class ShelfStateTracker:
    """Temporal stability tracker for shelf observation windows.

    Uses a rolling deque of ShelfObservation records. Key guarantees:

    - UNCERTAIN frames do NOT update the stable baseline.
    - STATE_CHANGED is raised only when stable_visible_count changes by
      >= change_threshold for >= min_stable_frames consecutive frames.
    - A sudden drop caused by a person blocking the shelf triggers UNCERTAIN,
      not STATE_CHANGED.
    - Single-frame anomalies (lighting glitch, motion blur) are absorbed by
      the window without triggering false state changes.
    """

    def __init__(self, config: Optional[InventoryTemporalConfig] = None) -> None:
        self.config = config or InventoryTemporalConfig()
        self._window: Deque[ShelfObservation] = deque(maxlen=self.config.window_size)
        self._stable_baseline: Optional[int] = None
        self._previous_baseline: Optional[int] = None
        self._consecutive_change_frames: int = 0
        self._state_changed_flag: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        batch: ShelfDetectionBatch,
        timestamp_sec: float = 0.0,
    ) -> ShelfStateSnapshot:
        """Process a new frame detection batch and return the current state.

        Args:
            batch: ShelfDetectionBatch from ShelfProductDetector.
            timestamp_sec: Video timestamp of this frame in seconds.

        Returns:
            ShelfStateSnapshot representing the analysed shelf state.
        """
        visible_count = batch.visible_count
        person_detected = batch.person_present

        raw_status = self._classify_frame(visible_count, person_detected)
        obs = ShelfObservation(
            frame_index=batch.frame_index,
            timestamp_sec=timestamp_sec,
            visible_count=visible_count,
            person_detected=person_detected,
            raw_status=raw_status,
        )
        self._window.append(obs)

        # Partition window into reliable vs uncertain observations
        clear_obs = [
            o for o in self._window if o.raw_status != ObservationStatus.UNCERTAIN
        ]
        uncertain_obs = [
            o for o in self._window if o.raw_status == ObservationStatus.UNCERTAIN
        ]
        clear_counts = [o.visible_count for o in clear_obs]
        window_median = int(_median(clear_counts)) if clear_counts else visible_count

        # Initialise stable baseline once enough reliable frames exist
        if (
            self._stable_baseline is None
            and len(clear_obs) >= self.config.min_stable_frames
        ):
            self._stable_baseline = window_median

        window_status, change_detected, change_desc, uncertainty_reason = (
            self._evaluate_window(
                window_median, person_detected, clear_obs, uncertain_obs
            )
        )

        # Shelf occupancy: visible vs window peak (rough heuristic)
        all_counts = [o.visible_count for o in self._window]
        max_ever = max(all_counts) if all_counts else 1
        occupancy_pct = min(100.0, (window_median / max(max_ever, 1)) * 100.0)

        return ShelfStateSnapshot(
            observation_status=window_status,
            stable_visible_count=(
                self._stable_baseline
                if self._stable_baseline is not None
                else window_median
            ),
            visible_count_this_frame=visible_count,
            state_change_detected=change_detected,
            change_description=change_desc,
            uncertainty_reason=uncertainty_reason,
            window_size=len(self._window),
            clear_frame_count=len(clear_obs),
            uncertain_frame_count=len(uncertain_obs),
            previous_stable_count=self._previous_baseline or 0,
            shelf_occupancy_pct=round(occupancy_pct, 1),
        )

    def reset(self) -> None:
        """Reset tracker state (use when switching to a new scene or ROI)."""
        self._window.clear()
        self._stable_baseline = None
        self._previous_baseline = None
        self._consecutive_change_frames = 0
        self._state_changed_flag = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_frame(
        self, visible_count: int, person_detected: bool
    ) -> ObservationStatus:
        """Classify a single frame before temporal analysis."""
        if person_detected:
            return ObservationStatus.UNCERTAIN

        if self._stable_baseline is not None and self._stable_baseline > 0:
            drop_ratio = visible_count / self._stable_baseline
            abs_drop = self._stable_baseline - visible_count
            if (
                drop_ratio < self.config.occlusion_drop_threshold
                and abs_drop > 1
            ):
                # Sudden large drop without a person: likely non-person occlusion
                # (e.g. hand, trolley). Mark UNCERTAIN rather than a real change.
                return ObservationStatus.UNCERTAIN

        return ObservationStatus.STABLE

    def _evaluate_window(
        self,
        window_median: int,
        person_detected: bool,
        clear_obs: List[ShelfObservation],
        uncertain_obs: List[ShelfObservation],
    ):
        """Evaluate window-level status and detect persistent state changes.

        Returns:
            (window_status, change_detected, change_description, uncertainty_reason)
        """
        n = len(self._window)
        n_uncertain = len(uncertain_obs)
        n_clear = len(clear_obs)

        # Not enough data
        if n < self.config.min_stable_frames:
            return ObservationStatus.INITIALIZING, False, "", ""

        # Majority of window is uncertain
        if n_uncertain > n * 0.6:
            reason = (
                "Person detected — shelf partially or fully blocked"
                if person_detected
                else "Multiple frames with low detection count (possible occlusion or lighting issue)"
            )
            return ObservationStatus.UNCERTAIN, False, "", reason

        # Need enough reliable frames to evaluate a state change
        if n_clear < self.config.min_stable_frames or self._stable_baseline is None:
            return ObservationStatus.INITIALIZING, False, "", ""

        # Check for persistent change
        delta = abs(window_median - self._stable_baseline)
        if delta >= self.config.change_threshold:
            self._consecutive_change_frames += 1
            if self._consecutive_change_frames >= self.config.min_stable_frames:
                # Confirmed persistent change
                old_baseline = self._stable_baseline
                self._previous_baseline = old_baseline
                self._stable_baseline = window_median
                self._consecutive_change_frames = 0
                self._state_changed_flag = True
                direction = "increased" if window_median > old_baseline else "decreased"
                desc = (
                    f"Visible facings {direction} from "
                    f"{old_baseline} to {window_median}"
                )
                return ObservationStatus.STATE_CHANGED, True, desc, ""
            else:
                return ObservationStatus.POSSIBLY_CHANGING, False, "", ""
        else:
            self._consecutive_change_frames = 0
            self._state_changed_flag = False

        return ObservationStatus.STABLE, False, "", ""


# ---------------------------------------------------------------------------
# Per-Track Temporal Inventory State Estimation
# ---------------------------------------------------------------------------


class ProductTemporalState(str, Enum):
    """Temporal facing state for an individual tracked product."""

    UNCERTAIN = "UNCERTAIN"                 # Awaiting confirmation or temporarily lost/unstable
    STABLE = "STABLE"                       # Confirmed persistent facing on shelf
    POSSIBLY_CHANGING = "POSSIBLY_CHANGING" # Significant bbox shift/displacement detected


@dataclass
class ProductTrackRecord:
    """Temporal tracking state and history for an individual product track."""

    track_id: int
    state: ProductTemporalState = ProductTemporalState.UNCERTAIN
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    total_hits: int = 1
    consecutive_hits: int = 1
    consecutive_misses: int = 0
    recent_bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    recent_confs: List[float] = field(default_factory=list)
    sku_id: Optional[str] = None
    sku_name: Optional[str] = None
    sku_confidence: Optional[float] = None
    is_known_sku: bool = False
    state_reason: str = "Awaiting temporal confirmation"


@dataclass
class ProductTemporalSnapshot:
    """Frame-level summary of all active product tracks and temporal facing counts."""

    frame_index: int
    stable_facings_count: int
    uncertain_facings_count: int
    changing_facings_count: int
    total_active_tracks: int
    cumulative_unique_tracks: int
    active_tracks: List[ProductTrackRecord] = field(default_factory=list)
    transitions: List[Dict[str, Any]] = field(default_factory=list)


def _compute_bbox_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two bboxes (x1, y1, x2, y2)."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _compute_centroid_displacement_ratio(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    """Compute normalized centroid displacement relative to bbox diagonal."""
    cx1 = (b1[0] + b1[2]) / 2.0
    cy1 = (b1[1] + b1[3]) / 2.0
    cx2 = (b2[0] + b2[2]) / 2.0
    cy2 = (b2[1] + b2[3]) / 2.0

    w = max(1.0, float(b1[2] - b1[0]))
    h = max(1.0, float(b1[3] - b1[1]))
    diag = math.sqrt(w * w + h * h)
    dist = math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
    return dist / max(1.0, diag)


class ProductTrackStateTracker:
    """Per-product temporal state estimator for retail shelf monitoring.

    Tracks every product facing across video frames to:
    1. Avoid counting newly appearing tracks immediately (requires min_hits_for_stable frames).
    2. Promote persistent detections to STABLE.
    3. Mark disappearing/unstable tracks as UNCERTAIN before removing them (grace window).
    4. Detect significant position/bounding-box changes as POSSIBLY_CHANGING.
    5. Maintain camera-observable visible facings (strictly avoids claiming total stock).
    """

    def __init__(
        self,
        min_hits_for_stable: int = 3,
        max_misses_for_removal: int = 5,
        displacement_change_ratio: float = 0.25,
        iou_change_threshold: float = 0.60,
    ) -> None:
        self.min_hits_for_stable = min_hits_for_stable
        self.max_misses_for_removal = max_misses_for_removal
        self.displacement_change_ratio = displacement_change_ratio
        self.iou_change_threshold = iou_change_threshold

        self._tracks: Dict[int, ProductTrackRecord] = {}
        self._all_observed_track_ids: Set[int] = set()
        self._transitions_log: List[Dict[str, Any]] = []

    @property
    def cumulative_unique_tracks(self) -> int:
        return len(self._all_observed_track_ids)

    def update(
        self,
        batch: ShelfDetectionBatch,
        frame_index: int,
    ) -> ProductTemporalSnapshot:
        """Process a frame's product detections and update their temporal states."""
        frame_transitions: List[Dict[str, Any]] = []
        detected_track_ids: Set[int] = set()

        # 1. Update tracks present in the current frame
        for det in batch.product_detections:
            if det.track_id is None:
                det.temporal_state = ProductTemporalState.UNCERTAIN.value
                continue

            tid = det.track_id
            detected_track_ids.add(tid)
            self._all_observed_track_ids.add(tid)

            if tid not in self._tracks:
                # 1. Newly appearing track -> UNCERTAIN
                record = ProductTrackRecord(
                    track_id=tid,
                    state=ProductTemporalState.UNCERTAIN,
                    first_seen_frame=frame_index,
                    last_seen_frame=frame_index,
                    total_hits=1,
                    consecutive_hits=1,
                    consecutive_misses=0,
                    recent_bboxes=[det.bbox],
                    recent_confs=[det.confidence],
                    sku_id=det.sku_id,
                    sku_name=det.sku_name,
                    sku_confidence=det.sku_confidence,
                    is_known_sku=det.is_known_sku,
                    state_reason=f"New candidate track (1/{self.min_hits_for_stable} frames required)",
                )
                self._tracks[tid] = record
                trans = {
                    "frame_index": frame_index,
                    "track_id": tid,
                    "old_state": None,
                    "new_state": ProductTemporalState.UNCERTAIN.value,
                    "reason": record.state_reason,
                }
                frame_transitions.append(trans)
                self._transitions_log.append(trans)
            else:
                # Track update
                record = self._tracks[tid]
                record.total_hits += 1
                record.consecutive_hits += 1
                record.consecutive_misses = 0
                record.last_seen_frame = frame_index

                prev_bbox = record.recent_bboxes[-1] if record.recent_bboxes else det.bbox
                record.recent_bboxes.append(det.bbox)
                if len(record.recent_bboxes) > 5:
                    record.recent_bboxes.pop(0)

                record.recent_confs.append(det.confidence)
                if len(record.recent_confs) > 5:
                    record.recent_confs.pop(0)

                # Keep SKU memory updated or persistent
                if det.sku_name is not None:
                    record.sku_id = det.sku_id
                    record.sku_name = det.sku_name
                    record.sku_confidence = det.sku_confidence
                    record.is_known_sku = det.is_known_sku
                else:
                    det.sku_id = record.sku_id
                    det.sku_name = record.sku_name
                    det.sku_confidence = record.sku_confidence
                    det.is_known_sku = record.is_known_sku

                # Assess bounding-box stability & motion
                iou = _compute_bbox_iou(det.bbox, prev_bbox)
                disp = _compute_centroid_displacement_ratio(det.bbox, prev_bbox)
                old_state = record.state

                # Check for significant motion / shift
                if iou < self.iou_change_threshold or disp > self.displacement_change_ratio:
                    new_state = ProductTemporalState.POSSIBLY_CHANGING
                    reason = f"Bounding box displacement detected (disp={disp:.2f}, iou={iou:.2f})"
                elif record.consecutive_hits >= self.min_hits_for_stable:
                    new_state = ProductTemporalState.STABLE
                    reason = f"Confirmed stable facing across {record.consecutive_hits} frames"
                else:
                    new_state = ProductTemporalState.UNCERTAIN
                    reason = f"Awaiting stability confirmation ({record.consecutive_hits}/{self.min_hits_for_stable} frames)"

                if old_state != new_state:
                    trans = {
                        "frame_index": frame_index,
                        "track_id": tid,
                        "old_state": old_state.value,
                        "new_state": new_state.value,
                        "reason": reason,
                    }
                    frame_transitions.append(trans)
                    self._transitions_log.append(trans)

                record.state = new_state
                record.state_reason = reason

            det.temporal_state = self._tracks[tid].state.value

        # 2. Update missed tracks (grace period)
        for tid, record in list(self._tracks.items()):
            if tid not in detected_track_ids:
                record.consecutive_misses += 1
                record.consecutive_hits = 0

                if record.consecutive_misses > self.max_misses_for_removal:
                    # Remove track after grace window
                    trans = {
                        "frame_index": frame_index,
                        "track_id": tid,
                        "old_state": record.state.value,
                        "new_state": "REMOVED",
                        "reason": f"Track lost for >{self.max_misses_for_removal} consecutive frames",
                    }
                    frame_transitions.append(trans)
                    self._transitions_log.append(trans)
                    del self._tracks[tid]
                elif record.state != ProductTemporalState.UNCERTAIN:
                    # Mark disappearing track as UNCERTAIN
                    trans = {
                        "frame_index": frame_index,
                        "track_id": tid,
                        "old_state": record.state.value,
                        "new_state": ProductTemporalState.UNCERTAIN.value,
                        "reason": f"Track missed in frame; entering grace period ({record.consecutive_misses}/{self.max_misses_for_removal})",
                    }
                    frame_transitions.append(trans)
                    self._transitions_log.append(trans)
                    record.state = ProductTemporalState.UNCERTAIN
                    record.state_reason = trans["reason"]

        # 3. Calculate current state counts
        active_records = [self._tracks[tid] for tid in detected_track_ids if tid in self._tracks]
        stable_count = sum(1 for r in active_records if r.state == ProductTemporalState.STABLE)
        uncertain_count = sum(1 for r in active_records if r.state == ProductTemporalState.UNCERTAIN)
        changing_count = sum(1 for r in active_records if r.state == ProductTemporalState.POSSIBLY_CHANGING)

        return ProductTemporalSnapshot(
            frame_index=frame_index,
            stable_facings_count=stable_count,
            uncertain_facings_count=uncertain_count,
            changing_facings_count=changing_count,
            total_active_tracks=len(active_records),
            cumulative_unique_tracks=len(self._all_observed_track_ids),
            active_tracks=active_records,
            transitions=frame_transitions,
        )

    def get_full_transitions_log(self) -> List[Dict[str, Any]]:
        """Return the chronological history of all state transitions."""
        return list(self._transitions_log)
