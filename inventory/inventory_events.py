"""Inventory Change Event Detection Module for Retail Shelf Monitoring.

Builds on:
- Step 5 ProductTrackStateTracker (temporal facing states: STABLE, UNCERTAIN, POSSIBLY_CHANGING)
- Native ByteTrack track IDs
- Track-level SKU recognition cache

Detects and logs four core inventory events:
1. PRODUCT_APPEARED: New track becomes STABLE (confirmed visible facing).
2. PRODUCT_REMOVED: Existing STABLE track disappears after the grace period.
3. PRODUCT_MOVED: Track enters POSSIBLY_CHANGING and shows sustained bbox displacement.
4. SKU_CHANGED: Persistent track's recognized SKU changes with sufficient confidence.

Strictly preserves camera-observable visible-facing semantics:
Does NOT claim back-stock or total physical warehouse inventory.
"""

import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from inventory.shelf_detector import ShelfDetectionBatch
from inventory.shelf_state import (
    ProductTemporalSnapshot,
    ProductTemporalState,
    ProductTrackRecord,
    ProductTrackStateTracker,
    _compute_bbox_iou,
    _compute_centroid_displacement_ratio,
)
from inventory.sku_recognizer import BaseSKURecognizer


class InventoryEventType(str, Enum):
    """Categorized inventory change events."""

    PRODUCT_APPEARED = "PRODUCT_APPEARED"      # New track promoted to STABLE
    PRODUCT_REMOVED = "PRODUCT_REMOVED"        # Existing STABLE track removed after grace period
    PRODUCT_MOVED = "PRODUCT_MOVED"            # Sustained position displacement while in POSSIBLY_CHANGING
    SKU_CHANGED = "SKU_CHANGED"                # Persistent track SKU identity changed with high confidence
    OUT_OF_VIEW_PAN_EXIT = "OUT_OF_VIEW_PAN_EXIT"  # Product panned out of view across camera frame boundary


def _compute_centroid_distance_px(
    b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]
) -> float:
    """Compute Euclidean pixel distance between two bbox centroids."""
    cx1 = (b1[0] + b1[2]) / 2.0
    cy1 = (b1[1] + b1[3]) / 2.0
    cx2 = (b2[0] + b2[2]) / 2.0
    cy2 = (b2[1] + b2[3]) / 2.0
    return math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)


def _compute_majority_sku(
    history: Deque[Tuple[Optional[str], Optional[str], float, bool]]
) -> Tuple[Optional[str], Optional[str], float, bool, int]:
    """Compute consensus SKU prediction from rolling prediction history using majority voting.

    Returns:
        (best_sku_id, best_sku_name, avg_confidence, is_known, vote_count)
    """
    if not history:
        return None, None, 0.0, False, 0

    counts: Dict[str, int] = defaultdict(int)
    confs: Dict[str, List[float]] = defaultdict(list)
    names: Dict[str, str] = {}
    known_flags: Dict[str, bool] = {}

    for sid, sname, sconf, is_k in history:
        key = sid or "UNKNOWN"
        counts[key] += 1
        confs[key].append(sconf)
        if sname and key not in names:
            names[key] = sname
        if key not in known_flags:
            known_flags[key] = is_k

    best_key = max(counts.keys(), key=lambda k: (counts[k], float(np.mean(confs[k])) if confs[k] else 0.0))
    best_sku_id = None if best_key == "UNKNOWN" else best_key
    best_sku_name = names.get(best_key)
    best_conf = float(np.mean(confs[best_key])) if confs[best_key] else 0.0
    best_is_known = known_flags.get(best_key, False)
    vote_count = counts[best_key]

    return best_sku_id, best_sku_name, best_conf, best_is_known, vote_count


@dataclass
class InventoryChangeEvent:
    """Structured record of a detected inventory change event."""

    event_id: int
    frame_index: int
    timestamp_sec: float
    event_type: InventoryEventType
    track_id: int
    sku_id: Optional[str] = None
    sku_name: Optional[str] = None
    confidence: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None
    previous_bbox: Optional[Tuple[int, int, int, int]] = None
    displacement_px: float = 0.0
    displacement_ratio: float = 0.0
    previous_sku_id: Optional[str] = None
    previous_sku_name: Optional[str] = None
    previous_sku_confidence: float = 0.0
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to a JSON-compatible dictionary."""
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d


@dataclass
class TrackEventState:
    """Internal event lifecycle state maintained per track ID."""

    track_id: int
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    has_been_stable: bool = False
    has_emitted_appeared: bool = False
    is_currently_moving: bool = False
    moving_frames_count: int = 0
    anchor_stable_bbox: Optional[Tuple[int, int, int, int]] = None
    last_known_bbox: Optional[Tuple[int, int, int, int]] = None

    # Persistent SKU state
    confirmed_sku_id: Optional[str] = None
    confirmed_sku_name: Optional[str] = None
    confirmed_sku_confidence: float = 0.0
    confirmed_is_known: bool = False

    # Rolling prediction buffer for majority voting stabilization (last 5 predictions)
    sku_prediction_history: Deque[Tuple[Optional[str], Optional[str], float, bool]] = field(
        default_factory=lambda: deque(maxlen=5)
    )


@dataclass
class InventoryEventConfig:
    """Configuration parameters for Inventory Event Detection."""

    min_hits_for_stable: int = 3
    max_misses_for_removal: int = 45
    removal_grace_seconds: float = 1.5
    min_move_displacement_ratio: float = 0.22
    sustained_move_frames: int = 2
    sku_change_min_confidence: float = 0.70
    sku_recheck_interval: int = 5
    pan_boundary_margin_px: int = 120
    sku_window_size: int = 5
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None


class InventoryEventDetector:
    """Detects and logs inventory events based on temporal tracks and SKU matches.

    Guarantees:
    - Avoids counting newly appearing tracks immediately (requires STABLE state).
    - PRODUCT_REMOVED is emitted only for tracks that previously reached STABLE state.
    - PRODUCT_MOVED requires sustained displacement across consecutive frames to avoid jitter.
    - SKU_CHANGED is emitted only when a persistent track is re-identified as a different
      known SKU with confidence >= sku_change_min_confidence.
    """

    def __init__(
        self,
        config: Optional[InventoryEventConfig] = None,
        recognizer: Optional[BaseSKURecognizer] = None,
    ) -> None:
        self.config = config or InventoryEventConfig()
        self.recognizer = recognizer

        self._track_states: Dict[int, TrackEventState] = {}
        self._all_events: List[InventoryChangeEvent] = []
        self._event_counter: int = 0
        self._frame_width: Optional[int] = self.config.frame_width
        self._frame_height: Optional[int] = self.config.frame_height

    @property
    def all_events(self) -> List[InventoryChangeEvent]:
        return list(self._all_events)

    def _is_near_boundary(
        self,
        bbox: Optional[Tuple[int, int, int, int]],
        width: Optional[int],
        height: Optional[int],
        margin_px: int,
    ) -> bool:
        """Check whether a bounding box is located near any camera frame boundary."""
        if bbox is None:
            return False
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if width is not None:
            if x1 <= margin_px or x2 >= width - margin_px or cx <= margin_px or cx >= width - margin_px:
                return True
        if height is not None:
            if y1 <= margin_px or y2 >= height - margin_px or cy <= margin_px or cy >= height - margin_px:
                return True
        return False

    def process_frame(
        self,
        batch: ShelfDetectionBatch,
        snapshot: ProductTemporalSnapshot,
        frame_index: int,
        timestamp_sec: float = 0.0,
        frame_bgr: Optional[np.ndarray] = None,
    ) -> List[InventoryChangeEvent]:
        """Analyze current frame state transitions and detections to identify events."""
        frame_events: List[InventoryChangeEvent] = []
        detected_track_ids = set()

        if frame_bgr is not None:
            self._frame_height, self._frame_width = frame_bgr.shape[:2]

        # Map active detections by track_id
        det_by_track: Dict[int, Any] = {}
        for det in batch.product_detections:
            if det.track_id is not None:
                det_by_track[det.track_id] = det
                detected_track_ids.add(det.track_id)

        # ----------------------------------------------------------------------
        # Step 1: Manage active tracks, SKU updates, and STABLE / MOVED events
        # ----------------------------------------------------------------------
        for record in snapshot.active_tracks:
            tid = record.track_id
            det = det_by_track.get(tid)
            if det is None:
                continue

            # Initialize track event state if new
            if tid not in self._track_states:
                t_state = TrackEventState(
                    track_id=tid,
                    first_seen_frame=frame_index,
                    last_seen_frame=frame_index,
                    last_known_bbox=det.bbox,
                )
                # Initialize SKU if recognizer available
                if self.recognizer is not None and frame_bgr is not None:
                    crop = det.crop_from_frame(frame_bgr)
                    rec = self.recognizer.recognize_crop(crop)
                    t_state.sku_prediction_history.append(
                        (rec.sku_id, rec.name, rec.confidence, rec.is_known)
                    )
                elif det.sku_name or det.sku_id:
                    t_state.sku_prediction_history.append(
                        (det.sku_id, det.sku_name, det.sku_confidence or 0.0, det.is_known_sku)
                    )

                maj_sku_id, maj_sku_name, maj_conf, maj_is_k, _ = _compute_majority_sku(
                    t_state.sku_prediction_history
                )
                t_state.confirmed_sku_id = maj_sku_id
                t_state.confirmed_sku_name = maj_sku_name
                t_state.confirmed_sku_confidence = maj_conf
                t_state.confirmed_is_known = maj_is_k

                self._track_states[tid] = t_state
            else:
                t_state = self._track_states[tid]
                t_state.last_seen_frame = frame_index
                t_state.last_known_bbox = det.bbox

            # Always populate detection and track record with persistent SKU memory
            det.sku_id = t_state.confirmed_sku_id
            det.sku_name = t_state.confirmed_sku_name
            det.sku_confidence = t_state.confirmed_sku_confidence
            det.is_known_sku = t_state.confirmed_is_known

            record.sku_id = t_state.confirmed_sku_id
            record.sku_name = t_state.confirmed_sku_name
            record.sku_confidence = t_state.confirmed_sku_confidence
            record.is_known_sku = t_state.confirmed_is_known

            # ------------------------------------------------------------------
            # Check SKU_CHANGED: Uses rolling majority voting across last 5 predictions
            # ------------------------------------------------------------------
            if (
                self.recognizer is not None
                and frame_bgr is not None
                and (
                    frame_index % self.config.sku_recheck_interval == 0
                    or record.state == ProductTemporalState.POSSIBLY_CHANGING
                )
            ):
                crop = det.crop_from_frame(frame_bgr)
                rec = self.recognizer.recognize_crop(crop)
                t_state.sku_prediction_history.append((rec.sku_id, rec.name, rec.confidence, rec.is_known))

                maj_sku_id, maj_sku_name, maj_conf, maj_is_k, maj_count = _compute_majority_sku(
                    t_state.sku_prediction_history
                )

                # Strict majority: at least 3 out of 5 (or > len/2)
                min_votes = max(2, math.ceil(len(t_state.sku_prediction_history) / 2.0))
                if (
                    maj_is_k
                    and maj_conf >= self.config.sku_change_min_confidence
                    and t_state.confirmed_is_known
                    and maj_sku_id != t_state.confirmed_sku_id
                    and maj_count >= min_votes
                ):
                    # Sustained consensus transition verified
                    self._event_counter += 1
                    sku_event = InventoryChangeEvent(
                        event_id=self._event_counter,
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        event_type=InventoryEventType.SKU_CHANGED,
                        track_id=tid,
                        sku_id=maj_sku_id,
                        sku_name=maj_sku_name,
                        confidence=maj_conf,
                        bbox=det.bbox,
                        previous_sku_id=t_state.confirmed_sku_id,
                        previous_sku_name=t_state.confirmed_sku_name,
                        previous_sku_confidence=t_state.confirmed_sku_confidence,
                        details=(
                            f"Track #{tid} SKU changed from '{t_state.confirmed_sku_name}' "
                            f"({t_state.confirmed_sku_confidence * 100:.1f}%) to '{maj_sku_name}' "
                            f"({maj_conf * 100:.1f}%) via {maj_count}/{len(t_state.sku_prediction_history)} "
                            f"majority voting"
                        ),
                    )
                    frame_events.append(sku_event)
                    self._all_events.append(sku_event)

                    t_state.confirmed_sku_id = maj_sku_id
                    t_state.confirmed_sku_name = maj_sku_name
                    t_state.confirmed_sku_confidence = maj_conf
                    t_state.confirmed_is_known = True

                    det.sku_id = maj_sku_id
                    det.sku_name = maj_sku_name
                    det.sku_confidence = maj_conf
                    det.is_known_sku = True

                    record.sku_id = maj_sku_id
                    record.sku_name = maj_sku_name
                    record.sku_confidence = maj_conf
                    record.is_known_sku = True

            # ------------------------------------------------------------------
            # Check PRODUCT_APPEARED: Track becomes STABLE for the first time
            # ------------------------------------------------------------------
            if record.state == ProductTemporalState.STABLE:
                if not t_state.has_emitted_appeared:
                    t_state.has_emitted_appeared = True
                    t_state.has_been_stable = True
                    t_state.anchor_stable_bbox = det.bbox

                    self._event_counter += 1
                    app_event = InventoryChangeEvent(
                        event_id=self._event_counter,
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        event_type=InventoryEventType.PRODUCT_APPEARED,
                        track_id=tid,
                        sku_id=t_state.confirmed_sku_id,
                        sku_name=t_state.confirmed_sku_name,
                        confidence=det.confidence,
                        bbox=det.bbox,
                        details=(
                            f"Track #{tid} ({t_state.confirmed_sku_name or 'Product'}) "
                            f"confirmed STABLE after {record.consecutive_hits} consecutive frames "
                            f"(visible facing registered)"
                        ),
                    )
                    frame_events.append(app_event)
                    self._all_events.append(app_event)

                # Re-stabilized after movement
                if t_state.is_currently_moving:
                    t_state.is_currently_moving = False
                    t_state.moving_frames_count = 0
                    t_state.anchor_stable_bbox = det.bbox

            # ------------------------------------------------------------------
            # Check PRODUCT_MOVED: In POSSIBLY_CHANGING with sustained displacement
            # ------------------------------------------------------------------
            elif record.state == ProductTemporalState.POSSIBLY_CHANGING and t_state.has_been_stable:
                anchor = t_state.anchor_stable_bbox or det.bbox
                disp_ratio = _compute_centroid_displacement_ratio(det.bbox, anchor)
                disp_px = _compute_centroid_distance_px(det.bbox, anchor)

                if disp_ratio >= self.config.min_move_displacement_ratio:
                    t_state.moving_frames_count += 1
                    if (
                        t_state.moving_frames_count >= self.config.sustained_move_frames
                        and not t_state.is_currently_moving
                    ):
                        t_state.is_currently_moving = True
                        self._event_counter += 1
                        move_event = InventoryChangeEvent(
                            event_id=self._event_counter,
                            frame_index=frame_index,
                            timestamp_sec=timestamp_sec,
                            event_type=InventoryEventType.PRODUCT_MOVED,
                            track_id=tid,
                            sku_id=t_state.confirmed_sku_id,
                            sku_name=t_state.confirmed_sku_name,
                            confidence=det.confidence,
                            bbox=det.bbox,
                            previous_bbox=anchor,
                            displacement_px=round(disp_px, 1),
                            displacement_ratio=round(disp_ratio, 3),
                            details=(
                                f"Track #{tid} moved by {disp_px:.1f}px (ratio {disp_ratio:.2f}) "
                                f"sustained across {t_state.moving_frames_count} frames"
                            ),
                        )
                        frame_events.append(move_event)
                        self._all_events.append(move_event)
                else:
                    t_state.moving_frames_count = max(0, t_state.moving_frames_count - 1)

        # ----------------------------------------------------------------------
        # Step 2: Handle REMOVED tracks (exceeded grace period)
        # ----------------------------------------------------------------------
        for trans in snapshot.transitions:
            if trans.get("new_state") == "REMOVED":
                tid = trans.get("track_id")
                if tid in self._track_states:
                    t_state = self._track_states[tid]
                    # Only emit if the track was genuinely STABLE before
                    if t_state.has_been_stable:
                        last_box = t_state.last_known_bbox or t_state.anchor_stable_bbox
                        is_boundary = self._is_near_boundary(
                            last_box,
                            width=self._frame_width,
                            height=self._frame_height,
                            margin_px=self.config.pan_boundary_margin_px,
                        )
                        if is_boundary:
                            self._event_counter += 1
                            exit_event = InventoryChangeEvent(
                                event_id=self._event_counter,
                                frame_index=frame_index,
                                timestamp_sec=timestamp_sec,
                                event_type=InventoryEventType.OUT_OF_VIEW_PAN_EXIT,
                                track_id=tid,
                                sku_id=t_state.confirmed_sku_id,
                                sku_name=t_state.confirmed_sku_name,
                                confidence=0.0,
                                bbox=last_box,
                                previous_bbox=t_state.anchor_stable_bbox,
                                details=(
                                    f"Track #{tid} ({t_state.confirmed_sku_name or 'Product'}) "
                                    f"panned out of camera view near frame boundary (bbox={last_box}). "
                                    f"Suppressed PRODUCT_REMOVED."
                                ),
                            )
                            frame_events.append(exit_event)
                            self._all_events.append(exit_event)
                        else:
                            self._event_counter += 1
                            rem_event = InventoryChangeEvent(
                                event_id=self._event_counter,
                                frame_index=frame_index,
                                timestamp_sec=timestamp_sec,
                                event_type=InventoryEventType.PRODUCT_REMOVED,
                                track_id=tid,
                                sku_id=t_state.confirmed_sku_id,
                                sku_name=t_state.confirmed_sku_name,
                                confidence=0.0,
                                bbox=last_box,
                                previous_bbox=t_state.anchor_stable_bbox,
                                details=(
                                    f"Track #{tid} ({t_state.confirmed_sku_name or 'Product'}) "
                                    f"removed: previously STABLE facing disappeared in interior shelf and exceeded "
                                    f"{self.config.max_misses_for_removal}-frame grace period"
                                ),
                            )
                            frame_events.append(rem_event)
                            self._all_events.append(rem_event)

                    del self._track_states[tid]

        return frame_events

    def get_summary(self) -> Dict[str, Any]:
        """Return event count breakdown and operational metrics."""
        counts = {
            InventoryEventType.PRODUCT_APPEARED.value: 0,
            InventoryEventType.PRODUCT_REMOVED.value: 0,
            InventoryEventType.PRODUCT_MOVED.value: 0,
            InventoryEventType.SKU_CHANGED.value: 0,
            InventoryEventType.OUT_OF_VIEW_PAN_EXIT.value: 0,
        }
        for ev in self._all_events:
            counts[ev.event_type.value] = counts.get(ev.event_type.value, 0) + 1

        return {
            "total_events": len(self._all_events),
            "event_counts": counts,
            "interpretation_notice": (
                "Visible-facing events represent camera-observable changes on the shelf front row. "
                "They do NOT claim total back-stock or physical warehouse inventory."
            ),
        }
