"""Step 8 Demo: Store-Level Inventory Reporting & Operational Snapshot.

Reuses the completed modular inventory pipeline:
- retail_detector_exp2.pt (conf=0.30)
- ByteTrack native tracking
- SpatialColorTextureRecognizer with persistent SKU cache
- ProductTemporalState tracker (STABLE, UNCERTAIN, POSSIBLY_CHANGING)
- InventoryEventDetector (APPEARED, REMOVED, MOVED, SKU_CHANGED)
- SKUInventoryAggregator (per-SKU facings and shelf status)
- InventoryAlertDetector (actionable alert detection)
- StoreInventoryReportBuilder (consolidated operational store snapshot)

Strictly adheres to visible-facing semantics:
- Camera-observable front-row visible facings only.
- Does NOT claim total physical store inventory or back-stock quantity.
- Completely isolated from crowd/queue pipeline (src/, configs/, main.py).
"""

import argparse
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2

from inventory.catalog import SKUCatalog
from inventory.config import InventoryModelConfig
from inventory.inventory_aggregator import (
    InventoryAggregatorConfig,
    SKUInventoryAggregator,
)
from inventory.inventory_alerts import (
    InventoryAlertConfig,
    InventoryAlertDetector,
)
from inventory.inventory_events import (
    InventoryEventConfig,
    InventoryEventDetector,
)
from inventory.inventory_report import StoreInventoryReportBuilder
from inventory.shelf_detector import ShelfProductDetector
from inventory.shelf_state import ProductTrackStateTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Store-Level Inventory Reporting Demo")
    parser.add_argument(
        "--source",
        type=str,
        default="inventory_data/demo_videos/shelf_pan_demo.mp4",
        help="Path to video file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="inventory_data/custom_model/retail_detector_exp2.pt",
        help="Path to trained retail detector",
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default="inventory_data/catalogs/demo_store_catalog.json",
        help="Path to store product catalog JSON",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="output/inventory_report/inventory_report.json",
        help="Path to output JSON report",
    )
    parser.add_argument(
        "--output-txt",
        type=str,
        default="output/inventory_report/inventory_report.txt",
        help="Path to output manager text report",
    )
    parser.add_argument("--conf", type=float, default=0.30, help="Detector confidence threshold")
    parser.add_argument("--match-thresh", type=float, default=0.65, help="SKU similarity threshold")
    parser.add_argument("--min-hits", type=int, default=3, help="Consecutive frames for STABLE")
    parser.add_argument("--max-misses", type=int, default=5, help="Grace period missed frames")
    parser.add_argument("--in-stock-thresh", type=int, default=3, help="Min stable facings for IN_STOCK")
    parser.add_argument("--low-stock-thresh", type=int, default=2, help="Max stable facings for LOW_STOCK")
    parser.add_argument("--max-frames", type=int, default=75, help="Max frames to process")
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu')")
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = pathlib.Path(args.source)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    out_json_path = pathlib.Path(args.output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_txt_path = pathlib.Path(args.output_txt)
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("STEP 8: STORE-LEVEL INVENTORY REPORTING & OPERATIONAL SNAPSHOT")
    print("================================================================================")
    print(f"Video Source:           {video_path.name}")
    print(f"Detector Model:         {args.model} (conf={args.conf})")
    print(f"Tracking Backend:       ByteTrack (native)")
    print("--------------------------------------------------------------------------------")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_path}")

    total_frames = min(args.max_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Dimensions:       {width}x{height} @ {video_fps:.1f} FPS, analyzing up to {total_frames} frames.")

    # 1. Initialize Pipeline Modules
    print("\n[1/3] Initializing Pipeline Modules...")
    det_cfg = InventoryModelConfig(
        model_path=args.model,
        model_tier="retail_specific",
        device=args.device,
        confidence_threshold=args.conf,
        target_classes=None,
    )
    detector = ShelfProductDetector(det_cfg)

    catalog = None
    recognizer = None
    if pathlib.Path(args.catalog).is_file():
        from inventory.sku_recognizer import SpatialColorTextureRecognizer

        catalog = SKUCatalog.load_json(args.catalog)
        recognizer = SpatialColorTextureRecognizer(catalog=catalog, match_threshold=args.match_thresh)
        recognizer.build_index()
        print(f"      Loaded Store Catalog '{catalog.name}' ({len(catalog)} SKUs).")

    tracker = ProductTrackStateTracker(
        min_hits_for_stable=args.min_hits,
        max_misses_for_removal=args.max_misses,
        displacement_change_ratio=0.25,
        iou_change_threshold=0.60,
    )

    event_cfg = InventoryEventConfig(
        min_hits_for_stable=args.min_hits,
        max_misses_for_removal=args.max_misses,
        min_move_displacement_ratio=0.22,
        sustained_move_frames=2,
        sku_change_min_confidence=0.70,
        sku_recheck_interval=5,
    )
    event_detector = InventoryEventDetector(config=event_cfg, recognizer=recognizer)

    agg_cfg = InventoryAggregatorConfig(
        in_stock_threshold=args.in_stock_thresh,
        low_stock_threshold=args.low_stock_thresh,
        snapshot_interval=15,
    )
    aggregator = SKUInventoryAggregator(catalog=catalog, config=agg_cfg)

    alert_cfg = InventoryAlertConfig(
        low_stock_threshold=args.low_stock_thresh,
        stockout_consecutive_frames=10,
        rapid_removal_count=3,
        rapid_removal_window_frames=20,
        movement_alert_count=3,
        movement_window_frames=20,
        uncertain_sku_ratio_threshold=0.30,
        uncertain_sku_min_count=15,
        alert_cooldown_frames=15,
    )
    alert_detector = InventoryAlertDetector(config=alert_cfg, catalog=catalog)

    frame_idx = 0
    start_time = time.perf_counter()
    all_events_history = []
    latest_timestamp_sec = 0.0

    print("\n[2/3] Processing Video Sequence & Generating Store Inventory Snapshot...")
    while cap.isOpened() and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        timestamp_sec = frame_idx / video_fps
        latest_timestamp_sec = timestamp_sec

        # Step A: Run detector with ByteTrack
        batch = detector.detect(
            frame=frame,
            frame_index=frame_idx,
            track=True,
            persist=True,
            tracker="bytetrack.yaml",
        )

        # Step B: Update per-product temporal state tracker
        snapshot = tracker.update(batch=batch, frame_index=frame_idx)

        # Step C: Detect Inventory Events
        frame_events = event_detector.process_frame(
            batch=batch,
            snapshot=snapshot,
            frame_index=frame_idx,
            timestamp_sec=timestamp_sec,
            frame_bgr=frame,
        )
        all_events_history.extend(frame_events)

        # Step D: Aggregate SKU Inventory
        sku_stats = aggregator.update(
            snapshot=snapshot,
            events=frame_events,
            frame_index=frame_idx,
            timestamp_sec=timestamp_sec,
        )

        # Step E: Detect Actionable Inventory Alerts
        alert_detector.process_frame(
            sku_stats=sku_stats,
            frame_events=frame_events,
            frame_index=frame_idx,
            timestamp_sec=timestamp_sec,
        )

        frame_idx += 1

    cap.release()
    elapsed_sec = time.perf_counter() - start_time
    fps_avg = frame_idx / max(1e-4, elapsed_sec)

    # 3. Build Store-Level Report
    print("\n[3/3] Compiling Store Inventory Report & Serializing Snapshots...")
    report = StoreInventoryReportBuilder.build_report(
        sku_stats=aggregator.current_stats,
        all_alerts=alert_detector.all_alerts,
        recent_events=all_events_history,
        frame_index=frame_idx - 1,
        timestamp_sec=latest_timestamp_sec,
        total_frames=frame_idx,
        video_source=str(video_path),
        catalog_skus_registered=len(catalog) if catalog else 0,
    )

    # Save outputs
    report.save_json(out_json_path)
    print(f"[Saved JSON Report]     -> {out_json_path}")
    report.save_text(out_txt_path)
    print(f"[Saved Text Report]     -> {out_txt_path}")

    # 4. Print Structured Terminal Summary
    print("\n================================================================================")
    print("STORE INVENTORY SNAPSHOT")
    print("--------------------------------------------------------------------------------")
    print(f"Active Visible Facings: {report.total_active_visible_facings}")
    print(f"Stable Facings:         {report.stable_facings}")
    print(f"Uncertain:              {report.uncertain_facings}")
    print(f"Changing:               {report.possibly_changing_facings}")
    print(f"Unknown:                {report.total_unknown_facings}")
    print("")
    print("SKU Status:")
    print(f"IN_STOCK:               {report.in_stock_skus_count}")
    print(f"LOW_STOCK:              {report.low_stock_skus_count}")
    print(f"OUT_OF_VIEW:            {report.out_of_view_skus_count}")
    print("")
    print("Alerts:")
    print(f"HIGH:                   {report.alerts_by_severity.get('HIGH', 0)}")
    print(f"MEDIUM:                 {report.alerts_by_severity.get('MEDIUM', 0)}")
    print(f"Verification Required:  {report.verification_required_count}")
    print("")
    print(f"Shelf Health:           {report.shelf_health.value}")
    print(f"Reason:                 {report.health_reason}")
    print("--------------------------------------------------------------------------------")
    print(f"Processing Time:        {elapsed_sec:.2f}s ({fps_avg:.2f} FPS)")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
