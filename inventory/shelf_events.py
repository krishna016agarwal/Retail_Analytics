"""Inventory Event & Action Lifecycle Module (Step 31).

Provides an end-to-end, persistent retail inventory event/action system:
- Lifecycle: VACANCY_DETECTED -> VACANCY_CONFIRMED -> REPLENISHMENT_RECOMMENDED -> SHELF_RESTORED
- Status: ACTIVE -> ACKNOWLEDGED -> RESOLVED
- Lightweight JSON persistence across runs and dashboard reloads
- Preserves all Step 30 CV foundations, zero SKU/brand claims, strictly generic vacancy semantics.
"""

from __future__ import annotations

import json
import os
import pathlib
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ShelfEventType(str, Enum):
    """Categorized shelf vacancy lifecycle events."""

    VACANCY_DETECTED = "VACANCY_DETECTED"
    VACANCY_CONFIRMED = "VACANCY_CONFIRMED"
    REPLENISHMENT_RECOMMENDED = "REPLENISHMENT_RECOMMENDED"
    SHELF_RESTORED = "SHELF_RESTORED"


class ShelfEventStatus(str, Enum):
    """Operational status for replenishment action tracking."""

    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


@dataclass
class ShelfInventoryEvent:
    """Structured record of an inventory vacancy / replenishment event."""

    event_id: str
    run_id: str
    timestamp_iso: str
    frame_index: int
    timestamp_sec: float
    camera_id: str
    shelf_id: str
    tier_id: str
    event_type: str  # ShelfEventType value
    status: str      # ShelfEventStatus value
    gap_coordinates: Dict[str, Any]
    visual_vacancy_confidence: float
    duration_frames: int
    duration_sec: float
    action_required: bool = False
    message: str = ""
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "timestamp_iso": self.timestamp_iso,
            "frame_index": self.frame_index,
            "timestamp_sec": round(self.timestamp_sec, 2),
            "camera_id": self.camera_id,
            "shelf_id": self.shelf_id,
            "tier_id": self.tier_id,
            "event_type": self.event_type,
            "status": self.status,
            "gap_coordinates": self.gap_coordinates,
            "visual_vacancy_confidence": round(self.visual_vacancy_confidence, 3),
            "duration_frames": self.duration_frames,
            "duration_sec": round(self.duration_sec, 2),
            "action_required": self.action_required,
            "message": self.message,
            "acknowledged_at": self.acknowledged_at,
            "resolved_at": self.resolved_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShelfInventoryEvent":
        return cls(
            event_id=data["event_id"],
            run_id=data.get("run_id", "RUN-001"),
            timestamp_iso=data.get("timestamp_iso", datetime.now(timezone.utc).isoformat()),
            frame_index=data.get("frame_index", 0),
            timestamp_sec=data.get("timestamp_sec", 0.0),
            camera_id=data.get("camera_id", "CAM-01"),
            shelf_id=data.get("shelf_id", "SHELF-01"),
            tier_id=data.get("tier_id", "SHELF-01-TIER-01"),
            event_type=data.get("event_type", ShelfEventType.VACANCY_CONFIRMED.value),
            status=data.get("status", ShelfEventStatus.ACTIVE.value),
            gap_coordinates=data.get("gap_coordinates", {}),
            visual_vacancy_confidence=data.get("visual_vacancy_confidence", 0.0),
            duration_frames=data.get("duration_frames", 1),
            duration_sec=data.get("duration_sec", 0.0),
            action_required=data.get("action_required", False),
            message=data.get("message", ""),
            acknowledged_at=data.get("acknowledged_at"),
            resolved_at=data.get("resolved_at"),
        )


class ShelfEventManager:
    """Thread-safe Lifecycle & Storage Manager for Shelf Vacancy Events."""

    def __init__(
        self,
        run_id: str = "RUN-001",
        camera_id: str = "CAM-01",
        storage_path: Optional[Path] = None,
        dashboard_sync_path: Optional[Path] = None,
    ) -> None:
        self.run_id = run_id
        self.camera_id = camera_id
        root_dir = Path(__file__).resolve().parent.parent
        self.storage_path = storage_path or (root_dir / "output" / "inventory_events" / "inventory_events.json")
        self.dashboard_sync_path = dashboard_sync_path or (root_dir / "dashboard" / "public" / "inventory_events.json")

        self._lock = threading.RLock()
        self._events: List[ShelfInventoryEvent] = []
        self._active_events: Dict[str, ShelfInventoryEvent] = {}  # track_key -> event
        self._event_counter = 0

        # Load persisted history on startup
        self._load_persisted_events()

    def _load_persisted_events(self) -> None:
        """Load prior events from disk if available."""
        target = self.storage_path if self.storage_path.is_file() else self.dashboard_sync_path
        if target and target.is_file():
            try:
                with open(target, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("events", []):
                        if not isinstance(item, dict):
                            continue
                        try:
                            ev = ShelfInventoryEvent.from_dict(item)
                            self._events.append(ev)
                            if ev.status in (ShelfEventStatus.ACTIVE.value, ShelfEventStatus.ACKNOWLEDGED.value):
                                t_key = f"{ev.run_id}:{ev.tier_id}:{ev.gap_coordinates.get('region_id', ev.event_id)}"
                                self._active_events[t_key] = ev
                        except Exception:
                            continue
                    self._event_counter = len(self._events)
            except Exception as e:
                print(f"[ShelfEventManager] Warning loading events: {e}", flush=True)

    def save(self) -> None:
        """Persist events to output and sync to dashboard public folder."""
        data = {
            "last_updated_iso": datetime.now(timezone.utc).isoformat(),
            "active_run_id": self.run_id,
            "total_events": len(self._events),
            "active_events_count": len([e for e in self._events if e.status == ShelfEventStatus.ACTIVE.value]),
            "events": [e.to_dict() for e in self._events],
        }

        with self._lock:
            for p in (self.storage_path, self.dashboard_sync_path):
                if p:
                    try:
                        p.parent.mkdir(parents=True, exist_ok=True)
                        with open(p, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2)
                    except Exception as e:
                        print(f"[ShelfEventManager] Error saving to {p}: {e}", flush=True)

    def process_frame(
        self,
        snapshot: Any,  # ShelfVacancySnapshot
        frame_index: int,
        timestamp_sec: float,
        fps: float = 25.0,
    ) -> List[ShelfInventoryEvent]:
        """Update event lifecycles based on frame vacancy snapshot."""
        new_or_updated_events: List[ShelfInventoryEvent] = []

        # 1. Occlusion / Camera motion freeze: do not create or confirm new alerts
        if snapshot.is_occluded or snapshot.is_camera_moving:
            return new_or_updated_events

        current_active_keys = set()

        for tier in getattr(snapshot, "tiers", []):
            t_id = tier.tier_id
            shelf_id = snapshot.shelf_id

            for gap in tier.gaps:
                if gap.is_occluded_by_person:
                    continue

                track_key = f"{self.run_id}:{t_id}:{gap.region_id}"
                current_active_keys.add(track_key)

                # Event admission: persistent meaningful gap (>= 10 frames)
                if gap.persistence_frames >= 10:
                    ev_type = (
                        ShelfEventType.REPLENISHMENT_RECOMMENDED.value
                        if gap.replenishment_recommended
                        else ShelfEventType.VACANCY_CONFIRMED.value
                    )

                    with self._lock:
                        if track_key not in self._active_events:
                            # CREATE NEW CONFIRMED EVENT
                            self._event_counter += 1
                            event_id = f"EVT-{self.run_id}-{t_id}-{self._event_counter:03d}"
                            coords = {
                                "region_id": gap.region_id,
                                "x1": gap.x1,
                                "y1": gap.y1,
                                "x2": gap.x2,
                                "y2": gap.y2,
                                "width_px": round(gap.width_px, 1),
                                "width_multiple": round(gap.width_multiple, 2),
                            }
                            msg = (
                                f"Persistent empty shelf space detected on {t_id}. "
                                f"Replenishment verification recommended ({round(gap.width_multiple, 1)}x facing width)."
                            )
                            event = ShelfInventoryEvent(
                                event_id=event_id,
                                run_id=self.run_id,
                                timestamp_iso=datetime.now(timezone.utc).isoformat(),
                                frame_index=frame_index,
                                timestamp_sec=timestamp_sec,
                                camera_id=self.camera_id,
                                shelf_id=shelf_id,
                                tier_id=t_id,
                                event_type=ev_type,
                                status=ShelfEventStatus.ACTIVE.value,
                                gap_coordinates=coords,
                                visual_vacancy_confidence=gap.vacancy_confidence,
                                duration_frames=gap.persistence_frames,
                                duration_sec=round(gap.persistence_frames / max(1.0, fps), 2),
                                action_required=gap.replenishment_recommended,
                                message=msg,
                            )
                            self._events.append(event)
                            self._active_events[track_key] = event
                            new_or_updated_events.append(event)
                        else:
                            # UPDATE EXISTING ACTIVE EVENT (Deduplication across frames)
                            event = self._active_events[track_key]
                            event.duration_frames = gap.persistence_frames
                            event.duration_sec = round(gap.persistence_frames / max(1.0, fps), 2)
                            event.visual_vacancy_confidence = gap.vacancy_confidence
                            if gap.replenishment_recommended:
                                event.event_type = ShelfEventType.REPLENISHMENT_RECOMMENDED.value
                                event.action_required = True
                            event.gap_coordinates.update({
                                "x1": gap.x1,
                                "x2": gap.x2,
                                "width_px": round(gap.width_px, 1),
                                "width_multiple": round(gap.width_multiple, 2),
                            })
                            new_or_updated_events.append(event)

        # 2. Check for shelf restoration (previously active gap has been refilled)
        keys_to_resolve = []
        for track_key, active_event in list(self._active_events.items()):
            # If active event belongs to this run and is no longer observed as an active unoccluded gap
            if active_event.run_id == self.run_id and track_key not in current_active_keys:
                # Mark as resolved
                active_event.status = ShelfEventStatus.RESOLVED.value
                active_event.resolved_at = datetime.now(timezone.utc).isoformat()
                active_event.action_required = False

                # Append a SHELF_RESTORED lifecycle event
                self._event_counter += 1
                restored_id = f"EVT-{self.run_id}-{active_event.tier_id}-R{self._event_counter:03d}"
                restored_event = ShelfInventoryEvent(
                    event_id=restored_id,
                    run_id=self.run_id,
                    timestamp_iso=datetime.now(timezone.utc).isoformat(),
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    camera_id=self.camera_id,
                    shelf_id=active_event.shelf_id,
                    tier_id=active_event.tier_id,
                    event_type=ShelfEventType.SHELF_RESTORED.value,
                    status=ShelfEventStatus.RESOLVED.value,
                    gap_coordinates=active_event.gap_coordinates,
                    visual_vacancy_confidence=0.0,
                    duration_frames=active_event.duration_frames,
                    duration_sec=active_event.duration_sec,
                    action_required=False,
                    message=f"Shelf space refilled/restored on {active_event.tier_id}. Vacancy closed.",
                    resolved_at=datetime.now(timezone.utc).isoformat(),
                )
                self._events.append(restored_event)
                keys_to_resolve.append(track_key)
                new_or_updated_events.append(restored_event)

        for k in keys_to_resolve:
            self._active_events.pop(k, None)

        if new_or_updated_events:
            self.save()

        return new_or_updated_events

    def acknowledge_event(self, event_id: str) -> Optional[ShelfInventoryEvent]:
        """Acknowledge an active replenishment alert."""
        with self._lock:
            for ev in self._events:
                if ev.event_id == event_id:
                    ev.status = ShelfEventStatus.ACKNOWLEDGED.value
                    ev.acknowledged_at = datetime.now(timezone.utc).isoformat()
                    self.save()
                    return ev
        return None

    def resolve_event(self, event_id: str) -> Optional[ShelfInventoryEvent]:
        """Manually resolve/close an event."""
        with self._lock:
            for ev in self._events:
                if ev.event_id == event_id:
                    ev.status = ShelfEventStatus.RESOLVED.value
                    ev.resolved_at = datetime.now(timezone.utc).isoformat()
                    ev.action_required = False
                    # Remove from active tracking
                    for k, v in list(self._active_events.items()):
                        if v.event_id == event_id:
                            self._active_events.pop(k, None)
                    self.save()
                    return ev
        return None

    def get_active_events(self, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return all currently active replenishment alerts and confirmed vacancies, optionally filtered by run_id."""
        with self._lock:
            events = [
                e
                for e in self._events
                if e.status in (ShelfEventStatus.ACTIVE.value, ShelfEventStatus.ACKNOWLEDGED.value)
            ]
            if run_id:
                events = [e for e in events if e.run_id == run_id]
            return [e.to_dict() for e in events]

    def reset_active_for_new_run(self, new_run_id: str) -> None:
        """Configure manager for a new analysis run, scoping active tracking to new_run_id."""
        with self._lock:
            self.run_id = new_run_id
            self._active_events.clear()
            self.save()

    def get_history(self, run_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Return chronological event history, optionally filtered by run_id."""
        with self._lock:
            events = self._events
            if run_id:
                events = [e for e in events if e.run_id == run_id]
            # Return newest first
            return [e.to_dict() for e in reversed(events)][:limit]

    def get_analytics_summary(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Compute lightweight real event statistics from persisted events."""
        with self._lock:
            events = self._events
            if run_id:
                events = [e for e in events if e.run_id == run_id]

            total_events = len(events)
            replenishment_recommended = sum(
                1 for e in events if e.event_type == ShelfEventType.REPLENISHMENT_RECOMMENDED.value
            )
            resolved_events = [e for e in events if e.status == ShelfEventStatus.RESOLVED.value]
            resolved_count = len(resolved_events)
            active_count = sum(
                1 for e in events if e.status in (ShelfEventStatus.ACTIVE.value, ShelfEventStatus.ACKNOWLEDGED.value)
            )

            # Compute resolution times ONLY if >= 2 resolved events exist
            resolution_durations: List[float] = []
            for e in resolved_events:
                if e.resolved_at and e.timestamp_iso:
                    try:
                        t_start = datetime.fromisoformat(e.timestamp_iso.replace("Z", "+00:00"))
                        t_end = datetime.fromisoformat(e.resolved_at.replace("Z", "+00:00"))
                        diff = (t_end - t_start).total_seconds()
                        if diff >= 0:
                            resolution_durations.append(diff)
                    except Exception:
                        if e.duration_sec > 0:
                            resolution_durations.append(e.duration_sec)
                elif e.duration_sec > 0:
                    resolution_durations.append(e.duration_sec)

            avg_resolution_sec = None
            has_sufficient_data = len(resolution_durations) >= 2
            if has_sufficient_data:
                avg_resolution_sec = round(sum(resolution_durations) / len(resolution_durations), 1)

            return {
                "total_events": total_events,
                "replenishment_recommendations": replenishment_recommended,
                "resolved_events": resolved_count,
                "active_alerts": active_count,
                "avg_resolution_time_sec": avg_resolution_sec,
                "has_sufficient_resolution_data": has_sufficient_data,
                "resolution_data_samples": len(resolution_durations),
            }

    def clear(self) -> None:
        """Clear in-memory and persisted events."""
        with self._lock:
            self._events.clear()
            self._active_events.clear()
            self._event_counter = 0
            self.save()
