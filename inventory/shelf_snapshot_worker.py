"""Periodic Shelf Snapshot Background Worker (SIH PS 179).

Manages continuous edge shelf monitoring at configurable intervals (default: 10s for demo),
advancing through the shelf camera stream, updating Edge DB/cache, and exposing live telemetry.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from inventory.shelf_snapshot_service import ShelfSnapshotService

ROOT_DIR = Path(__file__).resolve().parent.parent


class ShelfSnapshotWorker:
    """Thread-safe background scheduler for shelf snapshots."""

    _instance: Optional[ShelfSnapshotWorker] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> ShelfSnapshotWorker:
        with cls._lock:
            if cls._instance is None:
                cls._instance = ShelfSnapshotWorker()
            return cls._instance

    def __init__(self, interval_seconds: int = 10) -> None:
        self.interval_seconds = interval_seconds
        self.service = ShelfSnapshotService()
        self.is_running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._data_lock = threading.Lock()

        self.scan_count = 0
        self.frame_cursor = 0
        self.max_frames = 75

        self.latest_scan: Optional[Dict[str, Any]] = None
        self.scan_history: List[Dict[str, Any]] = []
        self.next_scan_time: float = time.time() + self.interval_seconds

    def start(self) -> None:
        """Start the background scheduler thread if not already running."""
        with self._data_lock:
            if self.is_running:
                return
            self.is_running = True
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ShelfSnapshotWorker")
            self._thread.start()
            print(f"[ShelfSnapshotWorker] Started background monitoring (interval: {self.interval_seconds}s)")

    def stop(self) -> None:
        """Stop the background worker."""
        with self._data_lock:
            self.is_running = False
            self._stop_event.set()

    def set_interval(self, seconds: int) -> int:
        """Update scan interval in seconds."""
        seconds = max(3, min(3600, int(seconds)))
        with self._data_lock:
            self.interval_seconds = seconds
            self.next_scan_time = time.time() + seconds
        print(f"[ShelfSnapshotWorker] Interval updated to {seconds}s")
        return seconds

    def trigger_scan_now(self) -> Dict[str, Any]:
        """Manually trigger an instant scan and reset the next countdown timer."""
        return self._execute_scan()

    def _execute_scan(self) -> Dict[str, Any]:
        """Execute one snapshot analysis."""
        with self._data_lock:
            self.scan_count += 1
            # Advance frame cursor through the video (steps of 15 frames)
            curr_frame = self.frame_cursor
            self.frame_cursor = (self.frame_cursor + 15) % max(self.max_frames, 1)

        try:
            result = self.service.capture_and_analyze_from_video(frame_offset=curr_frame)
            result["scan_number"] = self.scan_count
        except Exception as exc:
            print(f"[ShelfSnapshotWorker] Scan error: {exc}")
            result = {
                "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "scan_number": self.scan_count,
            }

        with self._data_lock:
            self.latest_scan = result
            self.next_scan_time = time.time() + self.interval_seconds
            # Maintain last 15 scans
            self.scan_history.insert(0, result)
            if len(self.scan_history) > 15:
                self.scan_history.pop()

        return result

    def _run_loop(self) -> None:
        """Main periodic loop."""
        # Initial scan on startup
        self._execute_scan()

        while not self._stop_event.is_set():
            time_to_wait = max(0.1, self.next_scan_time - time.time())
            if self._stop_event.wait(timeout=time_to_wait):
                break
            if time.time() >= self.next_scan_time:
                self._execute_scan()

    def get_telemetry(self) -> Dict[str, Any]:
        """Return current live state for REST API / dashboard."""
        with self._data_lock:
            now = time.time()
            rem_sec = max(0.0, round(self.next_scan_time - now, 1))
            return {
                "is_running": self.is_running,
                "interval_seconds": self.interval_seconds,
                "seconds_until_next_scan": rem_sec,
                "total_scans_completed": self.scan_count,
                "current_video_frame": self.frame_cursor,
                "latest_scan": self.latest_scan,
                "recent_scans_count": len(self.scan_history),
            }
