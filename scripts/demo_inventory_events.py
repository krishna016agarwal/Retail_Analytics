"""Step 6 Demo: Inventory Change Event Detection for Retail Shelves.

Builds on:
- Step 5 temporal state tracker (STABLE, UNCERTAIN, POSSIBLY_CHANGING)
- Native ByteTrack tracking
- Persistent track-level SKU recognition cache

Detects and logs:
- PRODUCT_APPEARED: New track becomes STABLE (confirmed visible facing).
- PRODUCT_REMOVED: Existing STABLE track disappears after grace period.
- PRODUCT_MOVED: Track enters POSSIBLY_CHANGING with sustained displacement.
- SKU_CHANGED: Persistent track's recognized SKU changes with high confidence.

Strictly adheres to visible-facing semantics (does NOT claim total back-stock).
Completely isolated from the crowd/queue pipeline (src/, configs/, main.py).
"""

import argparse
import json
import pathlib
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

from inventory.catalog import SKUCatalog
from inventory.config import InventoryModelConfig
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
    parser = argparse.ArgumentParser(description="Inventory Change Event Detection Demo")
    parser.add_argument(
        "--source",
        type=str,
        default="inventory_data/demo_videos/shelf_pan_demo.mp4",
        help="Path to video file (default: inventory_data/demo_videos/shelf_pan_demo.mp4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="inventory_data/custom_model/retail_detector_exp2.pt",
        help="Path to trained retail detector weights",
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
        default="output/inventory_events/inventory_events_video.mp4",
        help="Path to save annotated output video",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="output/inventory_events/inventory_events_report.json",
        help="Path to save event JSON report",
    )
    parser.add_argument("--conf", type=float, default=0.30, help="Detector confidence threshold")
    parser.add_argument("--match-thresh", type=float, default=0.65, help="SKU match similarity cutoff")
    parser.add_argument("--min-hits", type=int, default=3, help="Consecutive frames required to promote to STABLE")
    parser.add_argument("--max-misses", type=int, default=5, help="Missed frames in grace period before removal")
    parser.add_argument("--max-frames", type=int, default=75, help="Maximum video frames to process")
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
    print("STEP 6: INVENTORY CHANGE EVENT DETECTION")
    print("================================================================================")
    print(f"Video Source:           {video_path.name}")
    print(f"Detector Model:         {args.model} (conf={args.conf})")
    print(f"Tracking Backend:       ByteTrack (native)")
    print(f"Promotion to STABLE:    >= {args.min_hits} consecutive frames")
    print(f"Grace Period:           {args.max_misses} missed frames")
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

    visualizer = ShelfVisualizer(show_confidence=True, show_class_names=True)

    # 2. Setup Video Writer
    hud_h = 95
    out_h = height + hud_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, video_fps, (width, out_h))

    # Event tracking collections
    recent_events_by_track: Dict[int, Tuple[str, int]] = {}  # track_id -> (event_name, expiry_frame)
    latest_event_str: str = "No events detected yet"

    frame_idx = 0
    start_time = time.perf_counter()

    print("\n[2/3] Processing Video Frames and Detecting Inventory Events...")
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

        # Step B: Update per-product temporal state tracker (STABLE / UNCERTAIN / POSSIBLY_CHANGING)
        snapshot = tracker.update(batch=batch, frame_index=frame_idx)

        # Step C: Detect Inventory Change Events
        frame_events = event_detector.process_frame(
            batch=batch,
            snapshot=snapshot,
            frame_index=frame_idx,
            timestamp_sec=timestamp_sec,
            frame_bgr=frame,
        )

        # Update event memory for visualizer
        for ev in frame_events:
            recent_events_by_track[ev.track_id] = (ev.event_type.value, frame_idx + 15)
            latest_event_str = f"F{frame_idx+1} [{ev.event_type.value}] #{ev.track_id}: {ev.details[:80]}"

        fps_current = 1.0 / max(1e-4, time.perf_counter() - t_f0)

        # Step D: Render visual overlay
        annotated = visualizer.annotate_frame(
            frame=frame,
            batch=batch,
            state=None,
            fps=fps_current,
            shelf_roi=None,
            model_tier="temporal_inventory_state",
        )

        # Draw event badges on bounding boxes if an event recently occurred
        for det in batch.product_detections:
            if det.track_id is not None and det.track_id in recent_events_by_track:
                ev_type, exp_f = recent_events_by_track[det.track_id]
                if frame_idx <= exp_f:
                    bx1, by1, bx2, by2 = det.bbox
                    badge_text = f"[{ev_type}]"
                    if ev_type == InventoryEventType.PRODUCT_APPEARED.value:
                        badge_color = (0, 220, 100)  # Bright green
                    elif ev_type == InventoryEventType.PRODUCT_MOVED.value:
                        badge_color = (0, 140, 255)  # Orange
                    elif ev_type == InventoryEventType.SKU_CHANGED.value:
                        badge_color = (255, 105, 180) # Magenta/Pink
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

        # Clean up expired event badges
        recent_events_by_track = {
            tid: val for tid, val in recent_events_by_track.items() if val[1] > frame_idx
        }

        # Step E: Render Top HUD Banner with Event Breakdown & Interpretation Notice
        hud = np.zeros((hud_h, width, 3), dtype=np.uint8)
        hud[:] = (18, 22, 32)

        summary_so_far = event_detector.get_summary()["event_counts"]
        n_app = summary_so_far[InventoryEventType.PRODUCT_APPEARED.value]
        n_rem = summary_so_far[InventoryEventType.PRODUCT_REMOVED.value]
        n_mov = summary_so_far[InventoryEventType.PRODUCT_MOVED.value]
        n_sku = summary_so_far[InventoryEventType.SKU_CHANGED.value]

        # Line 1: Header + Frame Index + Event Counters
        cv2.putText(
            hud,
            f"INVENTORY CHANGE EVENT DETECTION | Frame {frame_idx + 1}/{total_frames} | "
            f"APPEARED: {n_app}  |  REMOVED: {n_rem}  |  MOVED: {n_mov}  |  SKU CHANGED: {n_sku}",
            (14, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

        # Line 2: Facing breakdown + Active tracks
        cv2.putText(
            hud,
            f"Facings: STABLE: {snapshot.stable_facings_count} | UNCERTAIN: {snapshot.uncertain_facings_count} | "
            f"CHANGING: {snapshot.changing_facings_count} | Active Tracks: {snapshot.total_active_tracks} | "
            f"Unique Tracks Observed: {snapshot.cumulative_unique_tracks}",
            (14, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        # Line 3: Latest Event Ticker
        ticker_color = (0, 255, 140) if n_app + n_rem + n_mov + n_sku > 0 else (160, 160, 160)
        cv2.putText(
            hud,
            f"Latest Event: {latest_event_str}",
            (14, 73),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            ticker_color,
            1,
            cv2.LINE_AA,
        )

        # Line 4: Semantics Disclaimer (Visible Facings only)
        cv2.putText(
            hud,
            "* Camera-observable front-row visible facings only -- NOT back-stock or physical inventory",
            (width - 760, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
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
                f"Appeared: {n_app:2d} | Removed: {n_rem:2d} | Moved: {n_mov:2d} | SKU Changed: {n_sku:2d} | "
                f"Stable Facings: {snapshot.stable_facings_count:2d}"
            )

        frame_idx += 1

    cap.release()
    writer.release()
    elapsed_sec = time.perf_counter() - start_time
    fps_avg = frame_idx / max(1e-4, elapsed_sec)

    print(f"\n[Saved Annotated Video] -> {out_video_path}")

    # 3. Compile and Save Event Report JSON
    print("\n[3/3] Compiling Event Audit Report & False Positive Analysis...")
    all_events = event_detector.all_events
    summary = event_detector.get_summary()

    # False positive and operational analysis
    analysis_notes = {
        "camera_panning_effects": (
            "Because the demo video is a camera pan across the shelf, items entering the field of "
            "view on the right edge are promoted to STABLE after 3 frames, generating PRODUCT_APPEARED events. "
            "Conversely, items exiting the field of view on the left edge disappear from view and exceed "
            "the 5-frame grace period, generating PRODUCT_REMOVED events."
        ),
        "false_positive_assessment": {
            "PRODUCT_APPEARED": (
                "New track promotion accurately confirms visible facings on shelf. In a static camera setup, "
                "these represent genuine restocks. In a panning sequence, edge-entry is a field-of-view artifact "
                "unless paired with camera egomotion / homography compensation."
            ),
            "PRODUCT_REMOVED": (
                "Tracks removed after the 5-frame grace period were confirmed STABLE. In a static setup, "
                "these represent customer picks/out-of-stock. In a panning sequence, edge-exit is an FOV artifact."
            ),
            "PRODUCT_MOVED": (
                "Identifies tracks experiencing significant sustained displacement (>=22% bbox diagonal) "
                "across consecutive frames. Can occasionally trigger on severe camera perspective shifts during "
                "rapid pan movements."
            ),
            "SKU_CHANGED": (
                "Requires high confidence (>=70%) to override an existing confirmed known SKU. Filters out "
                "specular reflection fluctuations and ensures persistent label stability."
            ),
        },
        "recommendations": [
            "For stationary shelf cameras: PRODUCT_APPEARED and PRODUCT_REMOVED directly correspond to customer picks and shelf replenishment.",
            "For mobile/panning cameras: Incorporate homography or shelf coordinate mapping to ignore edge-boundary entry/exit.",
            "Keep visible facings strictly separated from back-stock warehouse quantity.",
        ],
    }

    report_data = {
        "video_source": str(video_path),
        "total_frames_analyzed": frame_idx,
        "detector_model": args.model,
        "detector_confidence": args.conf,
        "parameters": {
            "min_hits_for_stable": args.min_hits,
            "max_misses_for_removal": args.max_misses,
            "min_move_displacement_ratio": event_cfg.min_move_displacement_ratio,
            "sustained_move_frames": event_cfg.sustained_move_frames,
            "sku_change_min_confidence": event_cfg.sku_change_min_confidence,
        },
        "event_summary": summary,
        "performance": {
            "processing_time_sec": round(elapsed_sec, 2),
            "overall_fps": round(fps_avg, 2),
        },
        "analysis_notes": analysis_notes,
        "events": [ev.to_dict() for ev in all_events],
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"[Saved JSON Event Report] -> {out_json_path}")

    print("\n================================================================================")
    print("STEP 6: INVENTORY CHANGE EVENT DETECTION SUMMARY")
    print("================================================================================")
    print(f"Total Video Frames Processed:               {frame_idx}")
    print(f"Total Inventory Events Detected:            {summary['total_events']}")
    print(f"  • PRODUCT_APPEARED (New Stable Facing):   {summary['event_counts']['PRODUCT_APPEARED']}")
    print(f"  • PRODUCT_REMOVED (Exceeded Grace Period):{summary['event_counts']['PRODUCT_REMOVED']}")
    print(f"  • PRODUCT_MOVED (Sustained Displacement): {summary['event_counts']['PRODUCT_MOVED']}")
    print(f"  • SKU_CHANGED (High-Confidence Shift):    {summary['event_counts']['SKU_CHANGED']}")
    print("--------------------------------------------------------------------------------")
    print(f"Processing Time:                            {elapsed_sec:.2f}s ({fps_avg:.2f} FPS)")
    print("================================================================================")


if __name__ == "__main__":
    main()
