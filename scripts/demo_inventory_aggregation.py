"""Demo CLI for SKU-Level Inventory Aggregation & Shelf Status.

Reuses the existing modular inventory components:
- retail_detector_exp2.pt (conf=0.30)
- ByteTrack native tracking
- SpatialColorTextureRecognizer with persistent SKU cache
- ProductTrackStateTracker (STABLE, UNCERTAIN, POSSIBLY_CHANGING)
- InventoryEventDetector (APPEARED, REMOVED, MOVED, SKU_CHANGED)
- SKUInventoryAggregator (per-SKU facings, status, event aggregation)

Strictly maintains visible-facing semantics (camera-observable front row only;
does NOT claim total physical stock or back-stock quantity).
Completely isolated from crowd/queue pipeline (src/, configs/, main.py).
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
from inventory.inventory_events import (
    InventoryEventConfig,
    InventoryEventDetector,
    InventoryEventType,
)
from inventory.shelf_detector import ShelfProductDetector
from inventory.shelf_state import ProductTemporalState, ProductTrackStateTracker
from inventory.shelf_visualizer import ShelfVisualizer
from inventory.sku_recognizer import SpatialColorTextureRecognizer


def parse_args():
    parser = argparse.ArgumentParser(description="SKU-Level Inventory Aggregation Demo")
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
        default="output/inventory_aggregation/inventory_aggregation_video.mp4",
        help="Path to output annotated video",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="output/inventory_aggregation/inventory_aggregation_report.json",
        help="Path to output JSON report",
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

    out_video_path = pathlib.Path(args.output_video)
    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path = pathlib.Path(args.output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("SKU-LEVEL INVENTORY AGGREGATION & SHELF STATUS")
    print("================================================================================")
    print(f"Video Source:           {video_path.name}")
    print(f"Detector Model:         {args.model} (conf={args.conf})")
    print(f"Tracking Backend:       ByteTrack (native)")
    print(f"In-Stock Threshold:     >= {args.in_stock_thresh} stable facings")
    print(f"Low-Stock Threshold:    1 to {args.low_stock_thresh} stable facings")
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

    visualizer = ShelfVisualizer(show_confidence=True, show_class_names=True)

    # 2. Setup Video Writer with Top HUD
    hud_h = 125
    out_h = height + hud_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, video_fps, (width, out_h))

    recent_events_by_track: Dict[int, Tuple[str, int]] = {}

    frame_idx = 0
    start_time = time.perf_counter()

    print("\n[2/3] Processing Frames & Aggregating SKU Facings...")
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

        # Step C: Detect Inventory Change Events
        frame_events = event_detector.process_frame(
            batch=batch,
            snapshot=snapshot,
            frame_index=frame_idx,
            timestamp_sec=timestamp_sec,
            frame_bgr=frame,
        )

        # Step D: Aggregate into SKU-level inventory stats
        sku_stats = aggregator.update(
            snapshot=snapshot,
            events=frame_events,
            frame_index=frame_idx,
            timestamp_sec=timestamp_sec,
        )

        # Update event memory for visualizer
        for ev in frame_events:
            recent_events_by_track[ev.track_id] = (ev.event_type.value, frame_idx + 15)

        fps_current = 1.0 / max(1e-4, time.perf_counter() - t_f0)

        # Step E: Render annotated frame
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

        # Step F: Render Enhanced Top HUD Banner for SKU Inventory Status
        hud = np.zeros((hud_h, width, 3), dtype=np.uint8)
        hud[:] = (16, 20, 30)

        shelf_summary = aggregator.get_shelf_summary()
        tot_active = shelf_summary["total_active_visible_facings"]
        tot_stable = shelf_summary["total_stable_facings"]
        tot_unc = shelf_summary["total_uncertain_facings"]
        tot_chg = shelf_summary["total_changing_facings"]
        tot_unk = shelf_summary["unknown_facings_count"]
        ev_totals = shelf_summary["event_totals"]

        # Line 1: Header + Active Visible Facings + UNKNOWN count
        cv2.putText(
            hud,
            f"SKU-LEVEL INVENTORY AGGREGATION | Frame {frame_idx + 1}/{total_frames} | "
            f"Active Visible Facings: {tot_active} (Stable: {tot_stable} | Uncertain: {tot_unc} | Changing: {tot_chg}) | "
            f"UNKNOWN: {tot_unk}",
            (14, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 225, 255),
            2,
            cv2.LINE_AA,
        )

        # Line 2: SKU Facings Breakdown pills
        # Build concise SKU facing string for registered SKUs
        sku_segments = []
        for s in sku_stats.values():
            if s.sku_id == "UNKNOWN":
                continue
            # Short name
            sname = s.sku_name.split()[0] if s.sku_name else s.sku_id
            st_color = "IN" if s.status == SKUShelfStatus.IN_STOCK else ("LOW" if s.status == SKUShelfStatus.LOW_STOCK else "OUT")
            sku_segments.append(f"{sname}: {s.stable_facings}s/{s.current_visible_facings}v [{st_color}]")

        sku_line = "  |  ".join(sku_segments[:6])
        cv2.putText(
            hud,
            f"SKU Facings (Stable/Total): {sku_line}",
            (14, 53),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (215, 215, 215),
            1,
            cv2.LINE_AA,
        )

        # Line 3: Shelf Status Totals & Event Counters
        n_in = shelf_summary["skus_in_stock_count"]
        n_low = shelf_summary["skus_low_stock_count"]
        n_out = shelf_summary["skus_out_of_view_count"]
        ev_app = ev_totals["appeared"]
        ev_rem = ev_totals["removed"]
        ev_mov = ev_totals["moved"]

        cv2.putText(
            hud,
            f"Shelf Status: {n_in} IN_STOCK  |  {n_low} LOW_STOCK  |  {n_out} OUT_OF_VIEW   ||   "
            f"Events: Appeared: {ev_app}  |  Removed: {ev_rem}  |  Moved: {ev_mov}",
            (14, 81),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 160),
            1,
            cv2.LINE_AA,
        )

        # Line 4: Semantics Disclaimer
        cv2.putText(
            hud,
            "* Visible facings = camera-observable front-row only (NOT total physical stock or back-stock)",
            (14, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (120, 140, 155),
            1,
            cv2.LINE_AA,
        )

        # Stack HUD on top of annotated frame
        full_frame = np.vstack([hud, annotated])
        writer.write(full_frame)

        if (frame_idx + 1) % 15 == 0 or frame_idx == total_frames - 1:
            print(
                f"      Frame {frame_idx + 1:3d}/{total_frames} | "
                f"Active Facings: {tot_active:2d} (Stable: {tot_stable:2d}, Unk: {tot_unk:2d}) | "
                f"SKUs: {n_in} IN_STOCK, {n_low} LOW_STOCK, {n_out} OUT_OF_VIEW | "
                f"Events: App: {ev_app:2d}, Rem: {ev_rem:2d}, Mov: {ev_mov:2d}"
            )

        frame_idx += 1

    cap.release()
    writer.release()
    elapsed_sec = time.perf_counter() - start_time
    fps_avg = frame_idx / max(1e-4, elapsed_sec)

    print(f"\n[Saved Annotated Video] -> {out_video_path}")

    # 3. Generate Structured JSON Report
    print("\n[3/3] Compiling SKU Inventory Aggregation JSON Report...")
    report_data = {
        "video_source": str(video_path),
        "total_frames_analyzed": frame_idx,
        "detector_model": args.model,
        "detector_confidence": args.conf,
        "parameters": {
            "min_hits_for_stable": args.min_hits,
            "max_misses_for_removal": args.max_misses,
            "in_stock_threshold": args.in_stock_thresh,
            "low_stock_threshold": args.low_stock_thresh,
        },
        "performance": {
            "processing_time_sec": round(elapsed_sec, 2),
            "overall_fps": round(fps_avg, 2),
        },
        **aggregator.get_full_report(),
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"[Saved JSON Report]     -> {out_json_path}")

    # 4. Concise Benchmark Summary
    summary = aggregator.get_shelf_summary()
    per_sku = aggregator.current_stats
    unique_tracks = tracker.cumulative_unique_tracks

    print("\n================================================================================")
    print("SKU-LEVEL INVENTORY AGGREGATION BENCHMARK SUMMARY")
    print("================================================================================")
    print(f"Total Video Frames Processed:               {frame_idx}")
    print(f"Total Unique Physical Tracks (ByteTrack):   {unique_tracks}")
    print(f"Final Active Visible Facings:               {summary['total_active_visible_facings']}")
    print(f"  • STABLE Facings (Confirmed Inventory):   {summary['total_stable_facings']}")
    print(f"  • UNCERTAIN Facings (New/Unconfirmed):    {summary['total_uncertain_facings']}")
    print(f"  • POSSIBLY_CHANGING (Shifting/Moving):    {summary['total_changing_facings']}")
    print(f"  • UNKNOWN (Unclassified Detections):      {summary['unknown_facings_count']}")
    print("--------------------------------------------------------------------------------")
    print("Visible Facings by SKU (Final Frame):")
    for s in per_sku.values():
        if s.sku_id == "UNKNOWN":
            print(f"  • {s.sku_name:30s}: {s.current_visible_facings:2d} active facings ({s.status.value})")
        else:
            print(
                f"  • {s.sku_name:30s}: {s.stable_facings:2d} stable / {s.current_visible_facings:2d} active "
                f"[{s.status.value}] (Appeared: {s.products_appeared}, Removed: {s.products_removed}, Moved: {s.products_moved})"
            )
    print("--------------------------------------------------------------------------------")
    print(f"Catalog SKUs Currently Observed:            {summary['skus_currently_observed_count']} / {summary['catalog_skus_registered']}")
    print(f"  • IN_STOCK (>= {args.in_stock_thresh} stable facings):       {summary['skus_in_stock_count']}")
    print(f"  • LOW_STOCK (1-{args.low_stock_thresh} stable facings):       {summary['skus_low_stock_count']}")
    print(f"  • OUT_OF_VIEW (0 stable facings):         {summary['skus_out_of_view_count']}")
    print("--------------------------------------------------------------------------------")
    print(f"Cumulative Inventory Events Logged:         Total: {sum(summary['event_totals'].values())}")
    print(f"  • PRODUCT_APPEARED:                       {summary['event_totals']['appeared']}")
    print(f"  • PRODUCT_REMOVED:                        {summary['event_totals']['removed']}")
    print(f"  • PRODUCT_MOVED:                          {summary['event_totals']['moved']}")
    print(f"  • SKU_CHANGED:                            {summary['event_totals']['sku_changes']}")
    print("--------------------------------------------------------------------------------")
    print(f"Processing Time / FPS:                      {elapsed_sec:.2f}s ({fps_avg:.2f} FPS)")
    print("================================================================================")


if __name__ == "__main__":
    main()
