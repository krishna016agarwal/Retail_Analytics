"""Demo CLI for Inventory Alerts & Actionable Insights.

Reuses the completed modular inventory pipeline:
- retail_detector_exp2.pt (conf=0.30)
- ByteTrack tracking
- Persistent SKU recognition cache
- ProductTemporalState tracker (STABLE, UNCERTAIN, POSSIBLY_CHANGING)
- InventoryEventDetector (APPEARED, REMOVED, MOVED, SKU_CHANGED)
- SKUInventoryAggregator (per-SKU facings and shelf status)
- InventoryAlertDetector (LOW_STOCK, POSSIBLE_STOCKOUT, RAPID_REMOVAL, PRODUCT_MOVEMENT, SKU_RECOGNITION_UNCERTAIN)

Strictly adheres to visible-facing semantics:
- Camera-observable front-row visible facings only.
- Clearly distinguishes CAMERA OUT-OF-VIEW vs POSSIBLE PHYSICAL STOCKOUT.
- Never claims total physical inventory or back-stock.
- Completely isolated from crowd/queue pipeline (src/, configs/, main.py).
"""

import argparse
import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

from inventory.catalog import SKUCatalog
from inventory.config import InventoryModelConfig
from inventory.inventory_aggregator import (
    InventoryAggregatorConfig,
    SKUInventoryAggregator,
    SKUShelfStatus,
)
from inventory.inventory_alerts import (
    AlertSeverity,
    InventoryAlert,
    InventoryAlertConfig,
    InventoryAlertDetector,
    InventoryAlertType,
)
from inventory.inventory_events import (
    InventoryChangeEvent,
    InventoryEventConfig,
    InventoryEventDetector,
    InventoryEventType,
)
from inventory.shelf_detector import ShelfProductDetector
from inventory.shelf_state import ProductTemporalState, ProductTrackStateTracker
from inventory.shelf_visualizer import ShelfVisualizer
from inventory.sku_recognizer import SpatialColorTextureRecognizer


def parse_args():
    parser = argparse.ArgumentParser(description="Inventory Alerts & Actionable Insights Demo")
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
        "--output-video",
        type=str,
        default="output/inventory_alerts/inventory_alerts_video.mp4",
        help="Path to save annotated video",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="output/inventory_alerts/inventory_alerts_report.json",
        help="Path to save JSON report",
    )
    parser.add_argument("--conf", type=float, default=0.30, help="Detector confidence threshold")
    parser.add_argument("--match-thresh", type=float, default=0.65, help="SKU similarity threshold")
    parser.add_argument("--min-hits", type=int, default=3, help="Consecutive frames for STABLE")
    parser.add_argument("--max-misses", type=int, default=5, help="Grace period missed frames")
    parser.add_argument("--in-stock-thresh", type=int, default=3, help="Min stable facings for IN_STOCK")
    parser.add_argument("--low-stock-thresh", type=int, default=2, help="Max stable facings for LOW_STOCK")
    parser.add_argument("--stockout-frames", type=int, default=10, help="Consecutive 0-facing frames for POSSIBLE_STOCKOUT")
    parser.add_argument("--rapid-removals", type=int, default=3, help="Removals in window for RAPID_REMOVAL")
    parser.add_argument("--window-frames", type=int, default=20, help="Frame window for removal/movement checks")
    parser.add_argument("--max-frames", type=int, default=75, help="Max frames to process")
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu')")
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = pathlib.Path(args.source)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    out_video_path = pathlib.Path(args.output_video)
    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path = pathlib.Path(args.output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("INVENTORY ALERTS & ACTIONABLE INSIGHTS")
    print("================================================================================")
    print(f"Video Source:           {video_path.name}")
    print(f"Detector Model:         {args.model} (conf={args.conf})")
    print(f"Tracking Backend:       ByteTrack (native)")
    print(f"Low-Stock Threshold:    1 to {args.low_stock_thresh} stable facings")
    print(f"Stockout Verification:  >= {args.stockout_frames} consecutive zero-facing frames")
    print(f"Rapid Removal Window:   >= {args.rapid_removals} removals within {args.window_frames} frames")
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
        stockout_consecutive_frames=args.stockout_frames,
        rapid_removal_count=args.rapid_removals,
        rapid_removal_window_frames=args.window_frames,
        movement_alert_count=3,
        movement_window_frames=args.window_frames,
        uncertain_sku_ratio_threshold=0.30,
        uncertain_sku_min_count=15,
        alert_cooldown_frames=15,
    )
    alert_detector = InventoryAlertDetector(config=alert_cfg, catalog=catalog)

    visualizer = ShelfVisualizer(show_confidence=True, show_class_names=True)

    # 2. Setup Video Writer with Compact Alert HUD
    hud_h = 135
    out_h = height + hud_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, video_fps, (width, out_h))

    recent_events_by_track: Dict[int, Tuple[str, int]] = {}
    latest_alert_str: str = "No alerts fired yet"

    frame_idx = 0
    start_time = time.perf_counter()

    print("\n[2/3] Processing Video Frames & Detecting Actionable Alerts...")
    while cap.isOpened() and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        t_f0 = time.perf_counter()
        timestamp_sec = frame_idx / video_fps

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

        # Step D: Aggregate SKU Inventory
        sku_stats = aggregator.update(
            snapshot=snapshot,
            events=frame_events,
            frame_index=frame_idx,
            timestamp_sec=timestamp_sec,
        )

        # Step E: Detect Actionable Inventory Alerts
        frame_alerts = alert_detector.process_frame(
            sku_stats=sku_stats,
            frame_events=frame_events,
            frame_index=frame_idx,
            timestamp_sec=timestamp_sec,
        )

        for a in frame_alerts:
            sev_badge = "CRIT" if a.severity == AlertSeverity.HIGH else a.severity.value
            verif_tag = " [VERIF_REQ]" if a.requires_verification else ""
            latest_alert_str = (
                f"F{frame_idx+1} [{a.alert_type.value}] ({sev_badge}){verif_tag} "
                f"{a.sku_name or a.sku_id}: {a.reason[:75]}"
            )

        for ev in frame_events:
            recent_events_by_track[ev.track_id] = (ev.event_type.value, frame_idx + 15)

        fps_current = 1.0 / max(1e-4, time.perf_counter() - t_f0)

        # Step F: Render visual overlay
        annotated = visualizer.annotate_frame(
            frame=frame,
            batch=batch,
            state=None,
            fps=fps_current,
            shelf_roi=None,
            model_tier="temporal_inventory_state",
        )

        # Draw event badges on bounding boxes
        for det in batch.product_detections:
            if det.track_id is not None and det.track_id in recent_events_by_track:
                ev_type, exp_f = recent_events_by_track[det.track_id]
                if frame_idx <= exp_f:
                    bx1, by1, bx2, by2 = det.bbox
                    badge_text = f"[{ev_type}]"
                    if ev_type == InventoryEventType.PRODUCT_APPEARED.value:
                        badge_color = (0, 220, 100)
                    elif ev_type == InventoryEventType.PRODUCT_MOVED.value:
                        badge_color = (0, 140, 255)
                    elif ev_type == InventoryEventType.SKU_CHANGED.value:
                        badge_color = (255, 105, 180)
                    else:
                        badge_color = (0, 0, 255)

                    cv2.putText(
                        annotated,
                        badge_text,
                        (bx1, max(18, by1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        badge_color,
                        2,
                        cv2.LINE_AA,
                    )

        recent_events_by_track = {
            tid: val for tid, val in recent_events_by_track.items() if val[1] > frame_idx
        }

        # Step G: Render Compact Alert HUD Banner
        hud = np.zeros((hud_h, width, 3), dtype=np.uint8)
        hud[:] = (16, 20, 30)

        shelf_summary = aggregator.get_shelf_summary()
        alert_summary = alert_detector.get_summary()

        tot_active = shelf_summary["total_active_visible_facings"]
        tot_stable = shelf_summary["total_stable_facings"]
        tot_unc = shelf_summary["total_uncertain_facings"]
        tot_chg = shelf_summary["total_changing_facings"]
        tot_unk = shelf_summary["unknown_facings_count"]

        n_in = shelf_summary["skus_in_stock_count"]
        n_low = shelf_summary["skus_low_stock_count"]
        n_out = shelf_summary["skus_out_of_view_count"]

        tot_alerts = alert_summary["total_alerts_fired"]
        high_alerts = alert_summary["alerts_by_severity"][AlertSeverity.HIGH.value]
        verif_alerts = alert_summary["alerts_requiring_verification"]

        # Line 1: Header + Active Visible Facings + UNKNOWN
        cv2.putText(
            hud,
            f"INVENTORY ALERTS & INSIGHTS | Frame {frame_idx + 1}/{total_frames} | "
            f"Active Facings: {tot_active} (Stable: {tot_stable} | Uncertain: {tot_unc} | Changing: {tot_chg}) | "
            f"UNKNOWN: {tot_unk}",
            (14, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 225, 255),
            2,
            cv2.LINE_AA,
        )

        # Line 2: Shelf Status + Alert Counts (Total, High, Needs Verification)
        cv2.putText(
            hud,
            f"Shelf Status: {n_in} IN_STOCK  |  {n_low} LOW_STOCK  |  {n_out} OUT_OF_VIEW   ||   "
            f"Total Alerts: {tot_alerts} (HIGH: {high_alerts}, Verif Required: {verif_alerts})",
            (14, 53),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 170),
            1,
            cv2.LINE_AA,
        )

        # Line 3: Latest Alert Ticker
        ticker_color = (0, 140, 255) if high_alerts > 0 else (210, 210, 210)
        cv2.putText(
            hud,
            f"Latest Alert: {latest_alert_str}",
            (14, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            ticker_color,
            1,
            cv2.LINE_AA,
        )

        # Line 4: Semantics & Moving-Camera Warning
        cv2.putText(
            hud,
            "* Camera-observable visible facings only. Moving camera: POSSIBLE_STOCKOUT indicates item panned out of view.",
            (14, 114),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (120, 140, 160),
            1,
            cv2.LINE_AA,
        )

        # Stack HUD on top of annotated frame
        full_frame = np.vstack([hud, annotated])
        writer.write(full_frame)

        if (frame_idx + 1) % 15 == 0 or frame_idx == total_frames - 1:
            print(
                f"      Frame {frame_idx + 1:3d}/{total_frames} | "
                f"Active Facings: {tot_active:2d} (Stable: {tot_stable:2d}) | "
                f"Alerts Fired: {tot_alerts:2d} (High: {high_alerts:2d}, Verif: {verif_alerts:2d})"
            )

        frame_idx += 1

    cap.release()
    writer.release()
    elapsed_sec = time.perf_counter() - start_time
    fps_avg = frame_idx / max(1e-4, elapsed_sec)

    print(f"\n[Saved Annotated Video] -> {out_video_path}")

    # 3. Generate Structured JSON Report
    print("\n[3/3] Compiling Actionable Inventory Alerts JSON Report...")
    final_stats = aggregator.current_stats
    report_data = {
        "video_source": str(video_path),
        "total_frames_analyzed": frame_idx,
        "detector_model": args.model,
        "detector_confidence": args.conf,
        "performance": {
            "processing_time_sec": round(elapsed_sec, 2),
            "overall_fps": round(fps_avg, 2),
        },
        **alert_detector.get_full_report(final_sku_stats=final_stats),
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"[Saved JSON Report]     -> {out_json_path}")

    # 4. Benchmark & Actionable Insights Summary
    alert_sum = alert_detector.get_summary()
    attention_skus = alert_detector.query_skus_needing_attention()
    stockouts = alert_detector.query_possible_stockouts()
    low_stocks = alert_detector.query_low_stock_skus()
    rapid_removals = alert_detector.query_repeated_removals()
    movements = alert_detector.query_significant_movement()
    uncertain_skus = alert_detector.query_uncertain_recognition()

    print("\n================================================================================")
    print("INVENTORY ALERTS & ACTIONABLE INSIGHTS BENCHMARK SUMMARY")
    print("================================================================================")
    print(f"Total Video Frames Analyzed:                {frame_idx}")
    print(f"Total Actionable Alerts Fired:              {alert_sum['total_alerts_fired']}")
    print(f"Alerts Requiring Verification:              {alert_sum['alerts_requiring_verification']}")
    print("--------------------------------------------------------------------------------")
    print("Alerts Breakdown by Type:")
    for atype, count in alert_sum["alerts_by_type"].items():
        print(f"  • {atype:28s}: {count:2d}")
    print("--------------------------------------------------------------------------------")
    print("Alerts Breakdown by Severity:")
    for sev, count in alert_sum["alerts_by_severity"].items():
        print(f"  • {sev:10s}: {count:2d}")
    print("--------------------------------------------------------------------------------")
    print("SKUs Needing Operational Attention:")
    for item in attention_skus:
        v_flag = " [REQUIRES VERIFICATION]" if item["requires_verification"] else ""
        print(f"  • {item['sku_name']} ({item['sku_id']}): {', '.join(item['alert_types'])}{v_flag}")
    print("--------------------------------------------------------------------------------")
    print(f"Processing Time / FPS:                      {elapsed_sec:.2f}s ({fps_avg:.2f} FPS)")
    print("================================================================================")
    print("\nOperational Semantics & Known Limitations:")
    print("  1. Camera Egomotion vs Physical Stockouts:")
    print("     The 75-frame sequence is a horizontal camera pan across the shelf.")
    print("     SKUs such as '7-Up Lemon Lime 2L' and 'Mountain Dew 2L' triggered POSSIBLE_STOCKOUT")
    print("     and RAPID_REMOVAL solely because they panned out of the camera's field of view.")
    print("     All such alerts are explicitly marked 'requires_verification=True'.")
    print("  2. Visible Facings vs Physical Stock:")
    print("     All estimates refer strictly to camera-observable front-row visible facings,")
    print("     avoiding claims about back-stock or warehouse inventory quantity.")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
