"""Intelligent Retail Analytics - Person Tracking Foundation (SIH 179).

Main entrypoint for video inference and ByteTrack person tracking with CPU optimization.
"""

import argparse
import sys
from pathlib import Path

from configs.config import (
    AnalyticsConfig,
    DatabaseConfig,
    DetectorConfig,
    PipelineConfig,
    QueueConfig,
    TrackerConfig,
    VisualizerConfig,
)
from src.database import EdgeDatabase
from src.detector import PersonDetector
from src.pipeline import VideoPipeline
from src.queue_analytics import QueueAnalytics
from src.shopper_analytics import EntryExitCounter
from src.tracker import PersonTracker
from src.tracker_botsort import BotSortTracker, BotSortTrackerConfig
from src.visualizer import Visualizer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SIH 179 - Intelligent Retail Analytics: Person Tracking (ByteTrack)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Accepts either --video or --source
    parser.add_argument(
        "--video",
        "--source",
        dest="video",
        type=str,
        default="videos/test.mp4",
        help="Path to input video file (configurable)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Ultralytics YOLO model weight name or path",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Confidence threshold for person detection",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference frame resolution (e.g., 640, 480, 320 for CPU speed)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Hardware execution device ('cpu' explicitly configured for Intel Iris Xe laptops)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save annotated output video (.mp4)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process (useful for automated testing)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Run in headless mode without opening an OpenCV GUI window",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack",
        choices=["bytetrack", "botsort"],
        help="Multi-object tracker algorithm: 'bytetrack' or 'botsort'",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable tracking and run in Phase-1 detection-only mode",
    )
    parser.add_argument(
        "--no-trail",
        action="store_true",
        help="Disable drawing trajectory motion trails on active tracks",
    )
    parser.add_argument(
        "--track-high-thresh",
        type=float,
        default=0.35,
        help="High-confidence threshold for 1st-stage association",
    )
    parser.add_argument(
        "--track-low-thresh",
        type=float,
        default=0.05,
        help="Low-confidence threshold for 2nd-stage association (occlusion handling)",
    )
    parser.add_argument(
        "--new-track-thresh",
        type=float,
        default=0.35,
        help="Minimum score to initiate a new tracklet",
    )
    parser.add_argument(
        "--track-buffer",
        type=int,
        default=60,
        help="Frame buffer to retain lost tracks before deletion",
    )
    parser.add_argument(
        "--match-thresh",
        type=float,
        default=0.80,
        help="IoU matching threshold for track association",
    )
    parser.add_argument(
        "--gmc-method",
        type=str,
        default="sparseOptFlow",
        choices=["sparseOptFlow", "none", "orb", "sift", "ecc"],
        help="Global motion compensation method for BoT-SORT",
    )
    parser.add_argument(
        "--with-reid",
        action="store_true",
        help="Enable appearance ReID feature extraction in BoT-SORT",
    )
    parser.add_argument(
        "--proximity-thresh",
        type=float,
        default=0.50,
        help="BoT-SORT minimum IoU to consider tracks proximate for ReID",
    )
    parser.add_argument(
        "--appearance-thresh",
        type=float,
        default=0.80,
        help="BoT-SORT minimum appearance cosine similarity for ReID match",
    )
    parser.add_argument(
        "--line-y",
        "--entrance-line-y",
        dest="line_y",
        type=int,
        default=300,
        help="Y-coordinate (pixel) for virtual entrance boundary line",
    )
    parser.add_argument(
        "--entry-direction",
        type=str,
        default="down",
        choices=["down", "up"],
        help="Direction representing store entry: 'down' (top-to-bottom) or 'up' (bottom-to-top)",
    )
    parser.add_argument(
        "--no-line",
        action="store_true",
        help="Disable virtual entrance line and footfall counting",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Enable spatial movement heatmap accumulation and overlay (Phase 4)",
    )
    parser.add_argument(
        "--heatmap-alpha",
        type=float,
        default=0.50,
        help="Alpha transparency factor for heatmap overlay (0.0 - 1.0)",
    )
    parser.add_argument(
        "--heatmap-blur",
        type=int,
        default=31,
        help="Gaussian blur kernel size for heatmap smoothing (odd integer)",
    )
    parser.add_argument(
        "--queue",
        action="store_true",
        help="Enable queue intelligence and waiting time tracking (Phase 5)",
    )
    parser.add_argument(
        "--queue-x1",
        type=int,
        default=300,
        help="X1 (left) pixel coordinate for queue zone",
    )
    parser.add_argument(
        "--queue-y1",
        type=int,
        default=200,
        help="Y1 (top) pixel coordinate for queue zone",
    )
    parser.add_argument(
        "--queue-x2",
        type=int,
        default=600,
        help="X2 (right) pixel coordinate for queue zone",
    )
    parser.add_argument(
        "--queue-y2",
        type=int,
        default=400,
        help="Y2 (bottom) pixel coordinate for queue zone",
    )
    parser.add_argument(
        "--queue-medium",
        type=int,
        default=3,
        help="Queue length threshold for MEDIUM congestion level",
    )
    parser.add_argument(
        "--queue-high",
        type=int,
        default=6,
        help="Queue length threshold for HIGH congestion level",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Disable recording analytics snapshots to local SQLite database",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/retail_edge.db",
        help="Path to local SQLite database file (defaults to data/retail_edge.db)",
    )
    parser.add_argument(
        "--store-id",
        type=str,
        default="store_001",
        help="Store branch identifier for edge telemetry snapshots",
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default="edge_device_01",
        help="Device node identifier for edge telemetry snapshots",
    )
    parser.add_argument(
        "--db-interval",
        type=float,
        default=5.0,
        help="Periodic interval in seconds of video time to record SQLite telemetry snapshots",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Start local FastAPI Edge REST API server (offline telemetry)",
    )
    parser.add_argument(
        "--api-host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind the FastAPI Edge API server",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="Port number to bind the FastAPI Edge API server",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Trigger manual sync of pending SQLite snapshots to Central Render API and exit",
    )
    parser.add_argument(
        "--auto-sync",
        action="store_true",
        help="Enable non-blocking periodic background sync during video analytics",
    )
    parser.add_argument(
        "--sync-interval",
        type=float,
        default=30.0,
        help="Polling interval in seconds for periodic background sync (default: 30.0)",
    )
    parser.add_argument(
        "--central-url",
        type=str,
        default=None,
        help="Target Central Cloud API URL (defaults to CENTRAL_API_URL environment variable)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Maximum snapshots per HTTPS batch request (default: 100)",
    )

    return parser.parse_args()



def main() -> int:
    """Main execution function."""
    args = parse_args()

    if args.sync:
        from src.sync import run_sync_cli

        return run_sync_cli(
            db_path=args.db_path,
            central_url=args.central_url,
            batch_size=args.batch_size,
        )

    if args.api:
        import uvicorn
        from src.api import app, set_db

        print("=" * 65)
        print("  Intelligent Retail Analytics - Edge REST API (SIH 179)")
        print("=" * 65)
        print(f"[*] Binding Host  : {args.api_host}")
        print(f"[*] Binding Port  : {args.api_port}")
        print(f"[*] SQLite DB     : {args.db_path}")
        print(f"[*] API Docs (UI) : http://{args.api_host}:{args.api_port}/docs")
        print("=" * 65)

        set_db(EdgeDatabase(db_path=args.db_path))
        uvicorn.run(app, host=args.api_host, port=args.api_port, log_level="info")
        return 0

    auto_sync_worker = None
    if args.auto_sync:
        from src.sync import BackgroundSyncThread, SyncClient

        sync_client = SyncClient(
            central_api_url=args.central_url,
            db_path=args.db_path,
        )
        auto_sync_worker = BackgroundSyncThread(
            sync_client=sync_client,
            interval_seconds=args.sync_interval,
            batch_size=args.batch_size,
        )
        auto_sync_worker.start()
        print(f"[*] Auto-Sync Worker   : Enabled (every {args.sync_interval}s -> {sync_client.central_api_url})")

    tracking_enabled = not args.no_track

    tracker_name = args.tracker.upper() if tracking_enabled else "NONE"

    print("=" * 65)
    print("  Intelligent Retail Analytics - Computer Vision & Tracking")
    print(f"  SIH Problem Statement 179 | Tracking: {tracker_name} (CPU)")
    print("=" * 65)
    print(f"[*] Input Video    : {args.video}")
    print(f"[*] Model          : {args.model}")
    print(f"[*] Device         : {args.device.upper()}")
    print(f"[*] Confidence     : {args.conf}")
    print(f"[*] Inference Size : {args.imgsz}")
    print(f"[*] Tracker        : {tracker_name if tracking_enabled else 'Disabled (Detection-only)'}")
    if tracking_enabled:
        print(f"[*] Association Cfg: high={args.track_high_thresh}, low={args.track_low_thresh}, new={args.new_track_thresh}, buffer={args.track_buffer}, match={args.match_thresh}")
        if args.tracker == "botsort":
            print(f"[*] BoT-SORT Extras: gmc={args.gmc_method}, with_reid={args.with_reid}, prox={args.proximity_thresh}, app={args.appearance_thresh}")
    print(f"[*] Trails Enabled : {not args.no_trail}")
    print(f"[*] Heatmap Overlay: {'Enabled' if args.heatmap else 'Disabled'} (alpha={args.heatmap_alpha}, blur={args.heatmap_blur})")
    print(f"[*] Headless Mode  : {args.no_show}")
    if args.max_frames:
        print(f"[*] Max Frames     : {args.max_frames}")
    if args.output:
        print(f"[*] Save Output To : {args.output}")
    print("-" * 65)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"\n[ERROR] Input video not found: {video_path.resolve()}")
        print("Please verify the video path or place a test video at 'videos/test.mp4'.")
        return 1

    # 1. Initialize Detector
    print(f"[*] Initializing PersonDetector (detector conf={args.conf}, imgsz={args.imgsz})...")
    detector_cfg = DetectorConfig(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        target_classes=[0],  # Strictly COCO class 0: person
    )
    try:
        detector = PersonDetector(config=detector_cfg)
    except Exception as e:
        print(f"\n[ERROR] Failed to load YOLO detector: {e}")
        return 1
    print("[✓] PersonDetector initialized successfully.")

    # 2. Initialize Tracker (if enabled)
    tracker = None
    if tracking_enabled:
        if args.tracker == "botsort":
            print("[*] Initializing BotSortTracker (BoT-SORT on CPU)...")
            tracker_cfg = BotSortTrackerConfig(
                track_high_thresh=args.track_high_thresh,
                track_low_thresh=args.track_low_thresh,
                new_track_thresh=args.new_track_thresh,
                track_buffer=args.track_buffer,
                match_thresh=args.match_thresh,
                gmc_method=args.gmc_method,
                with_reid=args.with_reid,
                proximity_thresh=args.proximity_thresh,
                appearance_thresh=args.appearance_thresh,
                device=args.device,
                draw_trajectories=not args.no_trail,
            )
            try:
                tracker = BotSortTracker(config=tracker_cfg)
            except Exception as e:
                print(f"\n[ERROR] Failed to initialize BoT-SORT tracker: {e}")
                return 1
            print("[✓] BotSortTracker initialized successfully.")
        else:
            print("[*] Initializing PersonTracker (ByteTrack on CPU)...")
            tracker_cfg = TrackerConfig(
                track_high_thresh=args.track_high_thresh,
                track_low_thresh=args.track_low_thresh,
                new_track_thresh=args.new_track_thresh,
                track_buffer=args.track_buffer,
                match_thresh=args.match_thresh,
                draw_trajectories=not args.no_trail,
            )
            try:
                tracker = PersonTracker(config=tracker_cfg)
            except Exception as e:
                print(f"\n[ERROR] Failed to initialize ByteTrack tracker: {e}")
                return 1
            print("[✓] PersonTracker initialized successfully.")

    # 3. Initialize Shopper Analytics (Footfall & Occupancy Counter)
    analytics_counter = None
    entrance_line_y = None
    if tracking_enabled and not args.no_line:
        entrance_line_y = args.line_y
        analytics_cfg = AnalyticsConfig(
            entrance_line_y=args.line_y,
            entry_direction=args.entry_direction,
        )
        analytics_counter = EntryExitCounter(config=analytics_cfg)
        print(f"[*] Shopper Analytics: Line Y={args.line_y}px, Direction={args.entry_direction.upper()}")

    # 3b. Initialize Queue Analytics (if enabled)
    queue_analytics = None
    if tracking_enabled and args.queue:
        queue_cfg = QueueConfig(
            enabled=True,
            zone_bbox=(args.queue_x1, args.queue_y1, args.queue_x2, args.queue_y2),
            medium_threshold=args.queue_medium,
            high_threshold=args.queue_high,
        )
        queue_analytics = QueueAnalytics(config=queue_cfg)
        print(f"[*] Queue Intelligence : Enabled zone=({args.queue_x1}, {args.queue_y1}, {args.queue_x2}, {args.queue_y2}), med={args.queue_medium}, high={args.queue_high}")

    # 3c. Initialize Edge Database (Phase 6A)
    edge_db = None
    if not args.no_db:
        edge_db = EdgeDatabase(db_path=args.db_path)
        print(f"[*] Edge Database      : Active at '{args.db_path}' (interval={args.db_interval}s)")

    # 4. Initialize Visualizer
    visualizer_cfg = VisualizerConfig()
    visualizer = Visualizer(config=visualizer_cfg)

    # 5. Setup and Run Video Pipeline
    pipeline_cfg = PipelineConfig()
    pipeline = VideoPipeline(
        detector=detector,
        tracker=tracker,
        analytics_counter=analytics_counter,
        entrance_line_y=entrance_line_y,
        entry_direction=args.entry_direction,
        queue_analytics=queue_analytics,
        edge_db=edge_db,
        store_id=args.store_id,
        device_id=args.device_id,
        db_interval_seconds=args.db_interval,
        visualizer=visualizer,
        video_source=args.video,
        enable_tracking=tracking_enabled,
        draw_trajectories=not args.no_trail,
        enable_heatmap=args.heatmap,
        heatmap_alpha=args.heatmap_alpha,
        heatmap_blur=args.heatmap_blur,
        show_display=not args.no_show,
        output_path=args.output,
        max_frames=args.max_frames,
        window_title=pipeline_cfg.window_title,
    )

    print("\n[*] Starting video processing pipeline...")
    if not args.no_show:
        print("[i] Press 'q' or 'ESC' in the video window to stop.")

    try:
        metrics = pipeline.run()
    except Exception as e:
        if auto_sync_worker:
            auto_sync_worker.stop()
        print(f"\n[ERROR] Pipeline runtime error: {e}")
        return 1

    # 6. Report Performance Summary
    print("\n" + "=" * 65)
    print("  Pipeline Execution Summary")
    print("=" * 65)
    print(f"[✓] Total Frames Processed : {metrics.total_frames_processed}")
    print(f"[✓] Total Processing Time  : {metrics.duration_seconds:.2f} s")
    print(f"[✓] Average CPU FPS        : {metrics.average_fps:.2f}")
    if tracking_enabled:
        print(f"[✓] Max Simultaneous Tracks: {metrics.max_simultaneous_tracked}")
        print(f"[✓] Total Unique Track IDs : {metrics.unique_track_ids_observed}")
    else:
        print(f"[✓] Peak People in a Frame : {metrics.max_simultaneous_tracked}")

    if analytics_counter:
        print("-" * 65)
        print("  Footfall, Occupancy & Dwell Summary")
        print("-" * 65)
        print(f"[✓] Entrance Line Y-Coord  : {args.line_y} px")
        print(f"[✓] Entry Direction (IN)   : {args.entry_direction.upper()}")
        print(f"[✓] Total Entries          : {metrics.total_entries}")
        print(f"[✓] Total Exits            : {metrics.total_exits}")
        print(f"[✓] Current Occupancy      : {metrics.current_occupancy}")
        print(f"[✓] Peak Occupancy         : {metrics.peak_occupancy}")
        print(f"[✓] Completed Visits       : {metrics.completed_visits}")
        print(f"[✓] Average Dwell Time     : {metrics.avg_dwell_time:.1f} s")
        print(f"[✓] Maximum Dwell Time     : {metrics.max_dwell_time:.1f} s")

    if queue_analytics:
        print("-" * 65)
        print("  Queue Intelligence Summary")
        print("-" * 65)
        print(f"[✓] Current Queue Length   : {metrics.current_queue_length}")
        print(f"[✓] Peak Queue Length      : {metrics.peak_queue_length}")
        print(f"[✓] Average Wait Time      : {metrics.average_wait_time:.1f} s")
        print(f"[✓] Maximum Wait Time      : {metrics.maximum_wait_time:.1f} s")
        print(f"[✓] Final Congestion Level : {metrics.congestion_level}")
        print(f"[✓] Counter Recommendation : {metrics.recommendation}")

    if edge_db:
        print("-" * 65)
        print("  Edge Database Summary")
        print("-" * 65)
        print(f"[✓] Edge Database          : {args.db_path}")
        print(f"[✓] Snapshots Stored       : {metrics.database_snapshots_stored}")
        print(f"[✓] Pending Sync           : {metrics.database_pending_sync}")
        edge_db.close()

    if auto_sync_worker:
        auto_sync_worker.stop()

    print("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())

