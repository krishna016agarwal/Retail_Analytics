"""Intelligent Retail Analytics - Person Tracking Foundation (SIH 179).

Main entrypoint for video inference and ByteTrack person tracking with CPU optimization.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2

from dotenv import load_dotenv

# Load environment variables from .env at application startup
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
load_dotenv()

from configs.config import (
    AnalyticsConfig,
    DatabaseConfig,
    DetectorConfig,
    PipelineConfig,
    QueueConfig,
    RetailIntelligenceConfig,
    TrackerConfig,
    VisualizerConfig,
    ZoneConfig,
)
from src.camera_manager import MultiCameraManager
from src.database import EdgeDatabase
from src.detector import PersonDetector
from src.pipeline import VideoPipeline
from src.queue_analytics import QueueAnalytics
from src.retail_intelligence import RetailIntelligenceEngine
from src.shopper_analytics import EntryExitCounter
from src.tracker import PersonTracker
from src.tracker_botsort import BotSortTracker, BotSortTrackerConfig
from src.visualizer import Visualizer
from src.zone_analytics import ZoneAnalyticsManager



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
        "--loop",
        action="store_true",
        help="Continuously loop video playback when it reaches the end",
    )
    parser.add_argument(
        "--calibrate-queue",
        action="store_true",
        help="Pause immediately on frame 1 in interactive mouse calibration mode to draw queue zone",
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
        help="Start local FastAPI Edge REST API server in background thread for live dashboard telemetry",
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Run only the FastAPI Edge REST API server in the foreground without video processing",
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
    # Phase 8 Retail Intelligence Engine arguments
    parser.add_argument(
        "--intelligence",
        action="store_true",
        help="Enable Phase 8 Retail Intelligence Engine (queue prediction, staffing alerts, crowd anomaly detection)",
    )
    parser.add_argument(
        "--zones-config",
        type=str,
        default=None,
        help="Optional path to JSON configuration defining store zones and expected staff",
    )
    parser.add_argument(
        "--four-cameras",
        "--multi-camera",
        dest="four_cameras",
        action="store_true",
        help="Run 4-camera concurrent video processing pipeline (Food, Electronics, Grocery, Checkout)",
    )
    parser.add_argument(
        "--inventory",
        action="store_true",
        help="Run Retail Inventory Shelf Stock Monitoring & Dual Video Stream Pipeline (SIH 179)",
    )
    parser.add_argument(
        "--demo-start-time",
        type=str,
        default="now",
        help="Simulated store start time (HH:MM:SS) or 'now' for current local IST time (default: 'now')",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Simulation time acceleration scale (1.0 = real-time, 60.0 = 1 sec is 1 min)",
    )
    parser.add_argument(
        "--historical-replay",
        action="store_true",
        help="Run multi-hour historical simulation replay to accumulate observations across multiple simulated hours",
    )
    parser.add_argument(
        "--replay-hours",
        type=int,
        default=4,
        help="Number of simulated hours to span during historical replay (default: 4 hours)",
    )
    parser.add_argument(
        "--multi-cam-test",
        action="store_true",
        help="Run multi-camera architecture simulation to test stream concurrency and edge aggregation",
    )
    parser.add_argument(
        "--multi-cam-frames",
        type=int,
        default=50,
        help="Number of frames to process in multi-camera simulation test (default: 50)",
    )

    return parser.parse_args()


def run_multi_camera_simulation(args) -> int:
    """Run multi-camera architecture simulation to test stream concurrency and edge aggregation."""
    import cv2

    print("=" * 65)
    print("  Intelligent Retail Analytics - Multi-Camera Architecture")
    print("  SIMULATION MODE | Testing Concurrency & Edge Aggregation")
    print("=" * 65)
    print(f"[*] Shared Video Source : {args.video}")
    print(f"[*] Number of Cameras   : 5 (CAM_01 to CAM_05)")
    print(f"[*] Simulation Frames   : {args.multi_cam_frames}")
    print("[*] Note: Real YOLO detections run across concurrent camera workers.")
    print("          Clearly tagged with is_simulation: True.\n")

    mgr = MultiCameraManager.create_simulated_setup(
        video_path=args.video,
        store_id=args.store_id,
        device_id=args.device_id,
    )

    det_cfg = DetectorConfig(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.conf,
        imgsz=args.imgsz,
    )
    detector = PersonDetector(config=det_cfg)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {args.video}")
        return 1

    frames = []
    for _ in range(max(1, args.multi_cam_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"[✓] Read {len(frames)} frames from video.")
    print("[*] Processing multi-camera frames through CameraWorkers...")

    for w in mgr.get_workers():
        w.detector = detector
        w.tracker = PersonTracker()

    last_agg = None
    for f in frames:
        analytics_list = []
        for w in mgr.get_workers():
            a = w.process_frame(f)
            analytics_list.append(a)
        last_agg = mgr.aggregate_and_evaluate(analytics_list)

    mgr.close()

    print("\n" + "=" * 65)
    print("  Multi-Camera Edge Aggregation Results")
    print("=" * 65)
    print(f"[✓] Logical Edge Device : {last_agg['edge_device']['device_id']} (Store: {last_agg['edge_device']['store_id']})")
    print(f"[✓] Active Camera Feeds : {last_agg['edge_device']['monitored_cameras']}")
    for c in last_agg.get("camera_analytics", []):
        print(f"    - {c['camera_id']} ({c['name']}) | Role: {c['role']} | Detections: {c['active_persons_detected']} | Sim: {c['is_simulation']}")
    q_pred = last_agg.get("queue_intelligence", {}).get("predicted_queue_3min", "N/A")
    print(f"[✓] Queue Prediction    : {q_pred} people in 3m")
    print(f"[✓] Total Active Alerts : {len(last_agg.get('active_alerts', []))}")
    print("=" * 65)
    return 0


def run_four_camera_pipeline(args) -> int:
    """Run concurrent 4-camera recorded video store processing pipeline."""
    import threading
    import time
    from configs.config import DetectorConfig
    from src.camera_manager import MultiCameraManager
    from src.database import EdgeDatabase

    print("=" * 70)
    print("  Intelligent Retail Analytics - 4-Camera Concurrent Processing")
    print("  DEMO SIMULATION MODE | Real Computer Vision Analytics")
    print("=" * 70)
    print("  CAM_01 -> videos/food/food.mp4        (Food Department)")
    print("  CAM_02 -> videos/electronics/electronics.mp4 (Electronics Department)")
    print("  CAM_03 -> videos/grocery/grocery.mp4    (Grocery Department)")
    print("  CAM_04 -> videos/checkout/checkout.mp4   (Checkout Queue)")
    print(f"[*] Simulated Store Start Time : {args.demo_start_time}")
    print(f"[*] Time Acceleration Scale    : {args.time_scale}x")
    print(f"[*] Mode                       : {'Historical Replay' if args.historical_replay else 'Live Demo Replay'}")
    print("=" * 70)

    # 1. Initialize local SQLite edge database
    edge_db = EdgeDatabase(db_path=args.db_path)
    print(f"[✓] Edge Database initialized at '{args.db_path}'")

    # 2. Configure time scale: if historical replay, accelerate to accumulate hours of observations
    effective_time_scale = args.time_scale
    if args.historical_replay and args.time_scale == 1.0:
        effective_time_scale = 30.0
        print(f"[*] Historical replay accelerated to {effective_time_scale}x time scale")

    # 3. Initialize MultiCameraManager with 4 cameras
    det_cfg = DetectorConfig(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.conf,
        imgsz=args.imgsz,
    )
    print(f"[*] Initializing YOLO11 detector ({args.model} on {args.device.upper()})...")

    # Load calibrated queue zone for CAM_04 (Checkout) from configs/queue_config.json or CLI override
    saved_queue_bbox = QueueAnalytics.load_zone_config("configs/queue_config.json")
    effective_queue_bbox = saved_queue_bbox
    if args.queue_x1 != 300 or args.queue_y1 != 200 or args.queue_x2 != 600 or args.queue_y2 != 400:
        effective_queue_bbox = (args.queue_x1, args.queue_y1, args.queue_x2, args.queue_y2)

    if effective_queue_bbox:
        print(f"[*] CAM_04 (Checkout): Loaded queue zone {effective_queue_bbox} (from configs/queue_config.json)")

    mgr = MultiCameraManager.create_four_camera_setup(
        video_dir="videos",
        store_id=args.store_id,
        device_id=args.device_id,
        detector_cfg=det_cfg,
        demo_start_time=args.demo_start_time,
        time_scale=effective_time_scale,
        queue_bbox=effective_queue_bbox,
    )
    print(f"[✓] Store Clock Initialized at : {mgr.clock.demo_start_time_str} (IST)")
    print("[✓] Initialized 4 concurrent camera workers with dedicated detectors & trackers.")

    # 4. If --api is requested, start Edge REST API server in background thread
    api_thread = None
    if args.api:
        import uvicorn
        from src.api import app, set_db, set_camera_manager

        set_db(edge_db)
        set_camera_manager(mgr)

        def start_api():
            uvicorn.run(app, host=args.api_host, port=args.api_port, log_level="warning")

        api_thread = threading.Thread(target=start_api, daemon=True, name="EdgeApiThread")
        api_thread.start()
        print(f"[✓] Edge REST API live at http://{args.api_host}:{args.api_port}/docs")

    # 5. If --auto-sync is requested, start background sync worker
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
        print(f"[✓] Auto-Sync Worker active (polling every {args.sync_interval}s -> {sync_client.central_api_url})")

    # 6. Main processing loop
    print("\n[*] Processing concurrent camera frames... Press Ctrl+C to stop.\n")
    step_count = 0
    max_steps = args.max_frames if args.max_frames is not None else (1000 if args.historical_replay else 1000000)
    last_db_save_time = time.time()
    db_save_interval = args.db_interval

    try:
        while step_count < max_steps:
            step_count += 1

            if args.historical_replay:
                sim_step_sec = (args.replay_hours * 3600.0) / max(1, max_steps)
                mgr.clock.update_elapsed(step_count * sim_step_sec)

            analytics_list, intel_res = mgr.process_concurrent_step()

            # Periodically write to SQLite database
            now = time.time()
            if now - last_db_save_time >= db_save_interval:
                mgr.save_snapshots_to_db(edge_db, analytics_list, intel_res)
                last_db_save_time = now

            # Console telemetry HUD every 10 steps
            if step_count % 10 == 0 or step_count == 1:
                clock_info = intel_res.get("simulation_clock", {})
                sim_time = clock_info.get("simulated_store_time", "17:00:00")
                totals = intel_res.get("store_totals", {})
                live_shoppers = totals.get("total_live_shoppers", 0)
                tot_footfall = totals.get("total_footfall", 0)
                q_intel = intel_res.get("queue_intelligence", {})
                current_q = q_intel.get("current_queue", 0)
                pred_q = q_intel.get("predicted_queue_3min", 0)
                rate = q_intel.get("growth_rate_per_min", 0.0)
                alerts_count = len(intel_res.get("active_alerts", []))

                line = (
                    f"[{sim_time} (Sim)] Live: {live_shoppers:2d} | "
                    f"Footfall: {tot_footfall:3d} | "
                    f"Queue: {current_q:2d} (Pred 3m: {pred_q:2d}, rate: {rate:+.1f}/m) | "
                    f"Alerts: {alerts_count} | "
                )
                dept_parts = []
                for a in analytics_list:
                    if a.role != "checkout":
                        dept_parts.append(f"{a.name[:4]}:{a.current_shoppers}")
                if dept_parts:
                    line += " ".join(dept_parts)
                print(line)
            if not args.no_show:
                grid = mgr.get_grid_frame(target_size=(1280, 720))
                if grid is not None:
                    cv2.imshow("Intelligent Retail Analytics - 4-Camera Live Stream", grid)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break

            if not args.historical_replay:
                time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n\n[*] Pipeline stopped by user.")
    finally:
        if not args.no_show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        try:
            if 'analytics_list' in locals() and 'intel_res' in locals():
                mgr.save_snapshots_to_db(edge_db, analytics_list, intel_res)
        except Exception:
            pass

        if auto_sync_worker:
            auto_sync_worker.stop()
        mgr.close()
        edge_db.close()

    print("\n" + "=" * 70)
    print("  4-Camera Processing Summary")
    print("=" * 70)
    print(f"[✓] Total Steps Processed   : {step_count}")
    print(f"[✓] SQLite Database Path    : {args.db_path}")
    print(f"[✓] Simulation Tag          : Recorded Video / Multi-Camera Demo Simulation")
    print("=" * 70)
    return 0


def main() -> int:
    """Main execution function."""
    args = parse_args()

    if args.four_cameras:
        return run_four_camera_pipeline(args)

    if args.inventory:
        from inventory.run_inventory_mode import run_inventory_pipeline
        return run_inventory_pipeline(args)

    if args.multi_cam_test:
        return run_multi_camera_simulation(args)

    if args.sync:

        from src.sync import run_sync_cli

        return run_sync_cli(
            db_path=args.db_path,
            central_url=args.central_url,
            batch_size=args.batch_size,
        )

    if args.api_only:
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

    api_thread = None
    if args.api:
        import threading
        import uvicorn
        from src.api import app, set_db

        set_db(EdgeDatabase(db_path=args.db_path))

        def start_api():
            uvicorn.run(app, host=args.api_host, port=args.api_port, log_level="warning")

        api_thread = threading.Thread(target=start_api, daemon=True, name="EdgeApiThread")
        api_thread.start()
        print(f"[✓] Edge REST API live at http://{args.api_host}:{args.api_port}/docs (Dashboard Connected)")

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

    # 3b. Initialize Queue Analytics (if enabled, intelligence requested, or calibration requested)
    queue_analytics = None
    if tracking_enabled and (args.queue or args.intelligence or args.calibrate_queue):
        # Check if saved queue config exists in configs/queue_config.json
        saved_bbox = QueueAnalytics.load_zone_config("configs/queue_config.json")
        if saved_bbox and args.queue_x1 == 300 and args.queue_y1 == 200 and args.queue_x2 == 600 and args.queue_y2 == 400:
            effective_bbox = saved_bbox
            print(f"[*] Loaded saved queue zone: {effective_bbox} (from configs/queue_config.json)")
        else:
            effective_bbox = (args.queue_x1, args.queue_y1, args.queue_x2, args.queue_y2)

        queue_cfg = QueueConfig(
            enabled=True,
            zone_bbox=effective_bbox,
            medium_threshold=args.queue_medium,
            high_threshold=args.queue_high,
        )
        queue_analytics = QueueAnalytics(config=queue_cfg)
        print(f"[*] Queue Intelligence : Enabled zone={effective_bbox}, med={args.queue_medium}, high={args.queue_high}")
        if not args.no_show:
            print("[i] Interactive Calibration: Click and drag mouse on video to adjust Queue Zone, or press 'C'.")

    # 3c. Initialize Edge Database (Phase 6A)
    edge_db = None
    if not args.no_db:
        edge_db = EdgeDatabase(db_path=args.db_path)
        print(f"[*] Edge Database      : Active at '{args.db_path}' (interval={args.db_interval}s)")

    # 3d. Initialize Retail Intelligence & Zone Analytics (Phase 8)
    zone_analytics = None
    intelligence_engine = None
    if tracking_enabled and args.intelligence:
        zone_configs = []
        if args.zones_config and Path(args.zones_config).exists():
            import json
            try:
                with open(args.zones_config, "r", encoding="utf-8") as f:
                    z_data = json.load(f)
                    for item in z_data.get("zones", []):
                        zone_configs.append(
                            ZoneConfig(
                                id=item["id"],
                                name=item["name"],
                                x1=int(item["x1"]),
                                y1=int(item["y1"]),
                                x2=int(item["x2"]),
                                y2=int(item["y2"]),
                                expected_staff=int(item.get("expected_staff", 1)),
                            )
                        )
            except Exception as e:
                print(f"[!] Warning: Could not parse zones configuration '{args.zones_config}': {e}")

        if not zone_configs:
            # Default store zones (coordinates configured for standard 640x480 resolution)
            zone_configs = [
                ZoneConfig(id="food", name="Food Section", x1=40, y1=80, x2=320, y2=450, expected_staff=2),
                ZoneConfig(id="electronics", name="Electronics", x1=320, y1=80, x2=620, y2=450, expected_staff=2),
            ]

        zone_analytics = ZoneAnalyticsManager(zones=zone_configs)
        intel_cfg = RetailIntelligenceConfig(
            enabled=True,
            queue_high_threshold=args.queue_high,
            zones=zone_configs,
        )
        intelligence_engine = RetailIntelligenceEngine(
            config=intel_cfg,
            store_id=args.store_id,
            device_id=args.device_id,
        )
        print(f"[*] Retail Intelligence: Enabled ({len(zone_configs)} zones monitored, 3-min queue forward horizon)")
        for z in zone_configs:
            print(f"    - Zone '{z.name}' ({z.id}): bbox=({z.x1},{z.y1},{z.x2},{z.y2}), expected_staff={z.expected_staff}")

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
        zone_analytics=zone_analytics,
        intelligence_engine=intelligence_engine,
        enable_intelligence=args.intelligence,
        loop=args.loop,
        calibrate_queue=args.calibrate_queue,
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

    if args.intelligence:
        print("-" * 65)
        print("  Phase 8: Retail Intelligence & Alert Summary")
        print("-" * 65)
        print(f"[✓] Total Alerts Generated : {metrics.total_alerts_generated}")
        print(f"[✓] Active Alerts Remaining: {metrics.active_alerts_count}")
        if metrics.zone_metrics:
            print("[*] Zone Analytics:")
            for z_id, zm in metrics.zone_metrics.items():
                print(f"    - {zm['name']} ({z_id}): {zm['current_shoppers']} shoppers (peak: {zm['peak_shoppers']}), load: {zm['shopper_load_per_staff']:.1f}/staff, status: {zm['zone_status']}")

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

