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

    PRODUCT_APPEARED = "PRODUCT_APPEARED"  # New track promoted to STABLE
    PRODUCT_REMOVED = "PRODUCT_REMOVED"    # Existing STABLE track removed after grace period
    PRODUCT_MOVED = "PRODUCT_MOVED"        # Sustained position displacement while in POSSIBLY_CHANGING
    SKU_CHANGED = "SKU_CHANGED"            # Persistent track SKU identity changed with high confidence


def _compute_centroid_distance_px(
    b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]
) -> float:
    """Compute Euclidean pixel distance between two bbox centroids."""
    cx1 = (b1[0] + b1[2]) / 2.0
    cy1 = (b1[1] + b1[3]) / 2.0
    cx2 = (b2[0] + b2[2]) / 2.0
    cy2 = (b2[1] + b2[3]) / 2.0
    return math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)


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


@dataclass
class InventoryEventConfig:
    """Configuration parameters for Inventory Event Detection."""

    min_hits_for_stable: int = 3
    max_misses_for_removal: int = 5
    min_move_displacement_ratio: float = 0.22
    sustained_move_frames: int = 2
    sku_change_min_confidence: float = 0.70
    sku_recheck_interval: int = 5


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

    @property
    def all_events(self) -> List[InventoryChangeEvent]:
        return list(self._all_events)

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
                    t_state.confirmed_sku_id = rec.sku_id
                    t_state.confirmed_sku_name = rec.name
                    t_state.confirmed_sku_confidence = rec.confidence
                    t_state.confirmed_is_known = rec.is_known
                else:
                    t_state.confirmed_sku_id = det.sku_id
                    t_state.confirmed_sku_name = det.sku_name
                    t_state.confirmed_sku_confidence = det.sku_confidence or 0.0
                    t_state.confirmed_is_known = det.is_known_sku

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
            # Check SKU_CHANGED: Only if re-identified as different with high conf
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
                if (
                    rec.is_known
                    and rec.confidence >= self.config.sku_change_min_confidence
                    and t_state.confirmed_is_known
                    and rec.sku_id != t_state.confirmed_sku_id
                ):
                    # High-confidence SKU transition
                    self._event_counter += 1
                    sku_event = InventoryChangeEvent(
                        event_id=self._event_counter,
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        event_type=InventoryEventType.SKU_CHANGED,
                        track_id=tid,
                        sku_id=rec.sku_id,
                        sku_name=rec.name,
                        confidence=rec.confidence,
                        bbox=det.bbox,
                        previous_sku_id=t_state.confirmed_sku_id,
                        previous_sku_name=t_state.confirmed_sku_name,
                        previous_sku_confidence=t_state.confirmed_sku_confidence,
                        details=(
                            f"Track #{tid} SKU changed from '{t_state.confirmed_sku_name}' "
                            f"({t_state.confirmed_sku_confidence * 100:.1f}%) to '{rec.name}' "
                            f"({rec.confidence * 100:.1f}%)"
                        ),
                    )
                    frame_events.append(sku_event)
                    self._all_events.append(sku_event)

                    t_state.confirmed_sku_id = rec.sku_id
                    t_state.confirmed_sku_name = rec.name
                    t_state.confirmed_sku_confidence = rec.confidence
                    t_state.confirmed_is_known = True

                    det.sku_id = rec.sku_id
                    det.sku_name = rec.name
                    det.sku_confidence = rec.confidence
                    det.is_known_sku = True

                    record.sku_id = rec.sku_id
                    record.sku_name = rec.name
                    record.sku_confidence = rec.confidence
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
                    # Only emit PRODUCT_REMOVED if the track was genuinely STABLE before
                    if t_state.has_been_stable:
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
                            bbox=t_state.last_known_bbox,
                            previous_bbox=t_state.anchor_stable_bbox,
                            details=(
                                f"Track #{tid} ({t_state.confirmed_sku_name or 'Product'}) "
                                f"removed: previously STABLE facing disappeared and exceeded "
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
        }
        for ev in self._all_events:
            counts[ev.event_type.value] += 1

        return {
            "total_events": len(self._all_events),
            "event_counts": counts,
            "interpretation_notice": (
                "Visible-facing events represent camera-observable changes on the shelf front row. "
                "They do NOT claim total back-stock or physical warehouse inventory."
            ),
        }
