"""Video processing pipeline for Retail Analytics.

Coordinates video input, detection inference, ByteTrack multi-object tracking,
metric calculation, and display rendering.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

import cv2

from src.database import EdgeDatabase
from src.detector import PersonDetector
from src.heatmap import MovementHeatmap
from src.queue_analytics import QueueAnalytics, QueueMetrics
from src.retail_intelligence import RetailIntelligenceEngine
from src.shopper_analytics import EntryExitCounter
from src.tracker import PersonTracker
from src.visualizer import Visualizer
from src.zone_analytics import ZoneAnalyticsManager, ZoneMetrics


@dataclass
class PipelineMetrics:
    """Summary metrics of a pipeline run."""

    total_frames_processed: int = 0
    average_fps: float = 0.0
    max_simultaneous_tracked: int = 0
    unique_track_ids_observed: int = 0
    duration_seconds: float = 0.0
    total_entries: int = 0
    total_exits: int = 0
    current_occupancy: int = 0
    peak_occupancy: int = 0
    completed_visits: int = 0
    avg_dwell_time: float = 0.0
    max_dwell_time: float = 0.0
    min_dwell_time: float = 0.0
    # Phase 5 Queue Intelligence metrics
    current_queue_length: int = 0
    peak_queue_length: int = 0
    average_wait_time: float = 0.0
    maximum_wait_time: float = 0.0
    congestion_level: str = "LOW"
    recommendation: str = "QUEUE NORMAL"
    # Phase 6A Edge Database metrics
    database_snapshots_stored: int = 0
    database_pending_sync: int = 0
    # Phase 8 Retail Intelligence metrics
    total_alerts_generated: int = 0
    active_alerts_count: int = 0
    zone_metrics: dict = None



class VideoPipeline:
    """Orchestrates video reading, person detection, tracking, and visualization."""

    def __init__(
        self,
        detector: PersonDetector,
        visualizer: Visualizer,
        video_source: str,
        tracker: Optional[PersonTracker] = None,
        analytics_counter: Optional[EntryExitCounter] = None,
        entrance_line_y: Optional[int] = None,
        entry_direction: str = "down",
        queue_analytics: Optional[QueueAnalytics] = None,
        edge_db: Optional[EdgeDatabase] = None,
        store_id: str = "store_001",
        device_id: str = "edge_device_01",
        db_interval_seconds: float = 5.0,
        enable_tracking: bool = True,
        draw_trajectories: bool = True,
        enable_heatmap: bool = False,
        heatmap_alpha: float = 0.50,
        heatmap_blur: int = 31,
        heatmap: Optional[MovementHeatmap] = None,
        show_display: bool = True,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        window_title: str = "Intelligent Retail Analytics - Footfall & Tracking (SIH 179)",
        zone_analytics: Optional[ZoneAnalyticsManager] = None,
        intelligence_engine: Optional[RetailIntelligenceEngine] = None,
        enable_intelligence: bool = False,
        camera_id: str = "CAM_01",
    ):
        """Initialize pipeline with components and runtime flags."""
        self.detector = detector
        self.tracker = tracker
        self.analytics_counter = analytics_counter
        self.entrance_line_y = entrance_line_y
        self.entry_direction = entry_direction
        self.visualizer = visualizer
        self.video_source = str(video_source)
        self.queue_analytics = queue_analytics
        self.edge_db = edge_db
        self.store_id = store_id
        self.device_id = device_id
        self.db_interval_seconds = max(0.5, float(db_interval_seconds))
        self.enable_tracking = enable_tracking
        self.draw_trajectories = draw_trajectories
        self.enable_heatmap = enable_heatmap
        self.heatmap_alpha = heatmap_alpha
        self.heatmap_blur = heatmap_blur
        self.heatmap = heatmap
        self.show_display = show_display
        self.output_path = output_path
        self.max_frames = max_frames
        self.window_title = window_title
        self.zone_analytics = zone_analytics
        self.intelligence_engine = intelligence_engine
        self.enable_intelligence = enable_intelligence
        self.camera_id = camera_id


    def _record_db_snapshot(self, video_time: float) -> None:
        """Record an aggregated telemetry snapshot into SQLite edge database."""
        if self.edge_db is None:
            return
        entries = self.analytics_counter.total_entries if self.analytics_counter else 0
        exits = self.analytics_counter.total_exits if self.analytics_counter else 0
        occupancy = self.analytics_counter.current_occupancy if self.analytics_counter else 0
        peak_occ = self.analytics_counter.peak_occupancy if self.analytics_counter else 0
        queue_len = self.queue_analytics.current_queue_length if self.queue_analytics else 0
        peak_q = self.queue_analytics.peak_queue_length if self.queue_analytics else 0
        avg_dwell = self.analytics_counter.avg_dwell_time if self.analytics_counter else 0.0
        max_dwell = self.analytics_counter.max_dwell_time if self.analytics_counter else 0.0
        avg_wait = self.queue_analytics.average_wait_time if self.queue_analytics else 0.0

        timeline_ts = f"T+{video_time:.2f}s"
        self.edge_db.insert_snapshot(
            store_id=self.store_id,
            device_id=self.device_id,
            timestamp=timeline_ts,
            entries=entries,
            exits=exits,
            occupancy=occupancy,
            peak_occupancy=peak_occ,
            queue_length=queue_len,
            peak_queue=peak_q,
            avg_dwell=avg_dwell,
            max_dwell=max_dwell,
            avg_wait=avg_wait,
            sync_status="PENDING",
        )

        # Phase 8: Record zone snapshots if zone analytics is active
        if self.zone_analytics is not None:
            zone_records = self.zone_analytics.get_zone_snapshot_records(
                timestamp=timeline_ts,
                store_id=self.store_id,
                device_id=self.device_id,
                camera_id=self.camera_id,
            )
            for zr in zone_records:
                self.edge_db.insert_zone_snapshot(
                    snapshot_id=zr["snapshot_id"],
                    store_id=zr["store_id"],
                    device_id=zr["device_id"],
                    camera_id=zr["camera_id"],
                    zone_id=zr["zone_id"],
                    zone_name=zr["zone_name"],
                    timestamp=zr["timestamp"],
                    current_shoppers=zr["current_shoppers"],
                    peak_shoppers=zr["peak_shoppers"],
                    avg_dwell=zr["avg_dwell"],
                    traffic_level=zr["traffic_level"],
                    expected_staff=zr["expected_staff"],
                    sync_status="PENDING",
                )


    def run(self) -> PipelineMetrics:
        """Run detection and tracking pipeline on the configured video source.

        Returns:
            PipelineMetrics containing performance and tracking summary.
        """
        # Validate video file exists
        video_path = Path(self.video_source)
        if not video_path.exists():
            raise FileNotFoundError(
                f"Input video file not found at: {video_path.resolve()}"
            )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(
                f"Failed to open video source with OpenCV: {video_path.resolve()}"
            )

        # Video metadata
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        # Synchronize video FPS with analytics counter and queue analytics
        if self.analytics_counter is not None:
            self.analytics_counter.set_fps(source_fps)
        if self.queue_analytics is not None:
            self.queue_analytics.set_fps(source_fps)

        # Initialize heatmap if requested
        if self.enable_heatmap and self.heatmap is None:
            self.heatmap = MovementHeatmap(
                frame_shape=(frame_height, frame_width),
                blur_kernel=self.heatmap_blur,
                alpha=self.heatmap_alpha,
            )

        # Initialize optional video writer
        writer = None
        if self.output_path:
            out_p = Path(self.output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(out_p),
                fourcc,
                source_fps,
                (frame_width, frame_height),
            )

        # Performance tracking
        fps_window = deque(maxlen=20)
        total_frames = 0
        max_simultaneous = 0
        model_name = Path(self.detector.model_path).stem
        pipeline_start = time.perf_counter()
        prev_frame_time = pipeline_start
        last_db_snapshot_time = -self.db_interval_seconds
        total_alerts_count = 0
        last_zone_metrics = {}

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                total_frames += 1

                # Calculate smoothed FPS
                now = time.perf_counter()
                frame_dt = now - prev_frame_time
                prev_frame_time = now
                if frame_dt > 0:
                    fps_window.append(1.0 / frame_dt)
                current_fps = sum(fps_window) / len(fps_window) if fps_window else 0.0

                # 1. Detection Stage (YOLO inference on CPU)
                det_batch = self.detector.detect(frame)

                # 2. Tracking / Annotation Stage
                if self.enable_tracking and self.tracker is not None:
                    track_batch = self.tracker.update(det_batch, frame)
                    if track_batch.count > max_simultaneous:
                        max_simultaneous = track_batch.count

                    # Update Footfall Analytics if active
                    footfall_metrics = None
                    if self.analytics_counter is not None:
                        footfall_metrics = self.analytics_counter.update(
                            track_batch, self.tracker.trajectory_manager
                        )

                    # Update Queue Intelligence if active (Phase 5)
                    queue_metrics = None
                    if self.queue_analytics is not None:
                        queue_metrics = self.queue_analytics.update(
                            track_batch, self.tracker.trajectory_manager
                        )

                    # Update Zone Analytics if active (Phase 8)
                    zone_metrics = None
                    if self.zone_analytics is not None:
                        zone_metrics = self.zone_analytics.update(
                            track_batch, self.tracker.trajectory_manager
                        )
                        last_zone_metrics = zone_metrics

                    # Evaluate Retail Intelligence Engine (Phase 8)
                    if self.enable_intelligence and self.intelligence_engine is not None:
                        current_video_time = total_frames / source_fps
                        intel_res = self.intelligence_engine.process_analytics(
                            queue_metrics=queue_metrics,
                            zone_metrics_dict=zone_metrics,
                            footfall_metrics=footfall_metrics,
                            current_time=current_video_time,
                        )
                        new_alerts = intel_res.get("new_alerts", [])
                        if new_alerts:
                            total_alerts_count += len(new_alerts)
                            for a in new_alerts:
                                if self.edge_db is not None:
                                    self.edge_db.insert_alert(
                                        alert_id=a["alert_id"],
                                        store_id=self.store_id,
                                        device_id=self.device_id,
                                        camera_id=self.camera_id,
                                        zone_id=a.get("zone_id", "store"),
                                        type=a["type"],
                                        severity=a["severity"],
                                        title=a["title"],
                                        message=a["message"],
                                        current_value=a["current_value"],
                                        predicted_value=a.get("predicted_value"),
                                        threshold=a["threshold"],
                                        recommendation=a["recommendation"],
                                        status=a.get("status", "ACTIVE"),
                                        created_at=a.get("timestamp"),
                                    )
                                print(f"\n[!] RETAIL INTELLIGENCE ALERT ({a['severity']}): {a['title']} -> {a['recommendation']}")

                        resolved_alerts = intel_res.get("resolved_alerts", [])
                        if resolved_alerts and self.edge_db is not None:
                            for ra in resolved_alerts:
                                self.edge_db.resolve_alert(ra["alert_id"])

                    # Update and overlay Movement Heatmap (Phase 4)

                    render_frame = frame
                    if self.enable_heatmap and self.heatmap is not None:
                        self.heatmap.update(track_batch)
                        render_frame = self.heatmap.overlay(frame)

                    tracker_label = (
                        "BOTSORT"
                        if "botsort" in self.tracker.__class__.__name__.lower()
                        else "BYTETRACK"
                    )

                    annotated = self.visualizer.draw_tracks(
                        render_frame,
                        batch=track_batch,
                        trajectory_manager=self.tracker.trajectory_manager,
                        draw_trajectories=self.draw_trajectories,
                        fps=current_fps,
                        model_name=model_name,
                        tracker_label=tracker_label,
                        footfall_metrics=footfall_metrics,
                        entrance_line_y=self.entrance_line_y,
                        entry_direction=self.entry_direction,
                        queue_metrics=queue_metrics,
                    )

                    # Periodic Edge Database Telemetry Snapshot (Phase 6A)
                    if self.edge_db is not None:
                        current_video_time = total_frames / source_fps
                        if (current_video_time - last_db_snapshot_time) >= self.db_interval_seconds:
                            self._record_db_snapshot(current_video_time)
                            last_db_snapshot_time = current_video_time
                else:
                    # Detection-only fallback (Phase 1)
                    if det_batch.count > max_simultaneous:
                        max_simultaneous = det_batch.count

                    annotated = self.visualizer.draw_detections(
                        frame,
                        batch=det_batch,
                        fps=current_fps,
                        model_name=model_name,
                    )

                # 3. Write output if requested
                if writer is not None:
                    writer.write(annotated)

                # 4. Display if enabled
                if self.show_display:
                    cv2.imshow(self.window_title, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    # Press 'q' or 'ESC' to cleanly quit
                    if key == ord("q") or key == 27:
                        break

                # Frame limit check
                if self.max_frames and total_frames >= self.max_frames:
                    break

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if self.show_display:
                cv2.destroyAllWindows()

        duration = time.perf_counter() - pipeline_start
        avg_fps = total_frames / duration if duration > 0 else 0.0
        unique_ids = (
            self.tracker.trajectory_manager.unique_track_count
            if (self.enable_tracking and self.tracker is not None)
            else 0
        )
        entries = self.analytics_counter.total_entries if self.analytics_counter else 0
        exits = self.analytics_counter.total_exits if self.analytics_counter else 0
        occupancy = self.analytics_counter.current_occupancy if self.analytics_counter else 0
        peak_occ = self.analytics_counter.peak_occupancy if self.analytics_counter else 0
        completed_visits = (
            self.analytics_counter.completed_visits
            if self.analytics_counter
            else 0
        )
        avg_dwell = (
            self.analytics_counter.avg_dwell_time
            if self.analytics_counter
            else 0.0
        )
        max_dwell = (
            self.analytics_counter.max_dwell_time
            if self.analytics_counter
            else 0.0
        )
        min_dwell = (
            self.analytics_counter.min_dwell_time
            if self.analytics_counter
            else 0.0
        )

        # Finalize queue sessions
        if self.queue_analytics is not None:
            self.queue_analytics.finalize(total_frames)

        # Record final edge database snapshot (avoiding duplicate of recent interval snapshot)
        if self.edge_db is not None and total_frames > 0:
            final_video_time = total_frames / source_fps
            if (final_video_time - last_db_snapshot_time) >= 1.0:
                self._record_db_snapshot(final_video_time)
                last_db_snapshot_time = final_video_time

        db_stored = self.edge_db.get_total_count() if self.edge_db else 0
        db_pending = self.edge_db.get_unsynced_count() if self.edge_db else 0

        curr_queue = (
            self.queue_analytics.current_queue_length if self.queue_analytics else 0
        )
        peak_queue = (
            self.queue_analytics.peak_queue_length if self.queue_analytics else 0
        )
        avg_wait = (
            self.queue_analytics.average_wait_time if self.queue_analytics else 0.0
        )
        max_wait = (
            self.queue_analytics.maximum_wait_time if self.queue_analytics else 0.0
        )
        cong_level, rec = (
            self.queue_analytics._determine_congestion_and_recommendation(curr_queue)
            if self.queue_analytics
            else ("LOW", "QUEUE NORMAL")
        )

        active_alerts_cnt = (
            len(self.intelligence_engine.alert_manager.get_active_alerts())
            if (self.intelligence_engine and self.intelligence_engine.alert_manager)
            else 0
        )
        zone_metrics_dict = (
            {z_id: zm.to_dict() for z_id, zm in last_zone_metrics.items()}
            if last_zone_metrics
            else {}
        )

        return PipelineMetrics(
            total_frames_processed=total_frames,
            average_fps=avg_fps,
            max_simultaneous_tracked=max_simultaneous,
            unique_track_ids_observed=unique_ids,
            duration_seconds=duration,
            total_entries=entries,
            total_exits=exits,
            current_occupancy=occupancy,
            peak_occupancy=peak_occ,
            completed_visits=completed_visits,
            avg_dwell_time=avg_dwell,
            max_dwell_time=max_dwell,
            min_dwell_time=min_dwell,
            current_queue_length=curr_queue,
            peak_queue_length=peak_queue,
            average_wait_time=avg_wait,
            maximum_wait_time=max_wait,
            congestion_level=cong_level,
            recommendation=rec,
            database_snapshots_stored=db_stored,
            database_pending_sync=db_pending,
            total_alerts_generated=total_alerts_count,
            active_alerts_count=active_alerts_cnt,
            zone_metrics=zone_metrics_dict,
        )

