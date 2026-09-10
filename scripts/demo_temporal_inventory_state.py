"""Step 5: Temporal Inventory / Facing State Estimation for Retail Shelves.

Maintains per-product temporal state:
- STABLE: Confirmed persistent facing across consecutive frames.
- UNCERTAIN: Newly appearing track (awaiting confirmation) or missed in grace window.
- POSSIBLY_CHANGING: Significant bounding-box shift or displacement detected.

Key features:
1. Prevents newly appearing tracks from being counted immediately.
2. Promotes persistent tracks to STABLE after min_hits consecutive frames.
3. Holds disappearing tracks in UNCERTAIN grace period before removal.
4. Detects significant box shifts/displacement as POSSIBLY_CHANGING.
5. Strictly maintains camera-observable visible facings (does NOT claim total physical stock).
6. Preserves SKU recognition identities per track.

Completely isolated from the crowd/queue pipeline (src/).
"""

import argparse
import json
import pathlib
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

from inventory.catalog import SKUCatalog
from inventory.config import InventoryModelConfig
from inventory.shelf_detector import ShelfProductDetector
from inventory.shelf_state import ProductTemporalState, ProductTrackStateTracker
from inventory.shelf_visualizer import ShelfVisualizer
from inventory.sku_recognizer import SpatialColorTextureRecognizer


def parse_args():
    parser = argparse.ArgumentParser(description="Temporal Inventory State Estimation Demo")
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
        default="output/temporal_inventory_state/temporal_state_video.mp4",
        help="Path to save annotated output video",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="output/temporal_inventory_state/temporal_state_metrics.json",
        help="Path to save metrics JSON report",
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
    print("STEP 5: TEMPORAL INVENTORY / FACING STATE ESTIMATION")
    print("================================================================================")
    print(f"Video Source:           {video_path.name}")
    print(f"Detector Model:         {args.model} (conf={args.conf})")
    print(f"Tracking Backend:       ByteTrack (Ultralytics native)")
    print(f"Promotion to STABLE:    >= {args.min_hits} consecutive frames")
    print(f"UNCERTAIN Grace Period: {args.max_misses} missed frames")
    print("--------------------------------------------------------------------------------")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_path}")

    total_frames = min(args.max_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Dimensions:       {width}x{height} @ {video_fps:.1f} FPS, analyzing up to {total_frames} frames.")

    # 1. Initialize Detector, Recognizer, and Temporal Tracker
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
    visualizer = ShelfVisualizer(show_confidence=True, show_class_names=True)

    # 2. Setup Video Writer
    hud_h = 75
    out_h = height + hud_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, video_fps, (width, out_h))

    # Metric collections
    per_frame_timeline: List[Dict[str, Any]] = []
    track_sku_memory: Dict[int, Tuple[str, str, float, bool]] = {}

    frame_idx = 0
    start_time = time.perf_counter()

    print("\n[2/3] Processing Video Frames and Estimating Temporal Facing States...")
    while cap.isOpened() and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        t_f0 = time.perf_counter()

        # Step A: Run detector with ByteTrack
        batch = detector.detect(
            frame=frame,
            frame_index=frame_idx,
            track=True,
            persist=True,
            tracker="bytetrack.yaml",
        )

        # Step B: Attach / recall persistent SKU identity per track
        if recognizer is not None:
            for det in batch.product_detections:
                if det.track_id is not None:
                    tid = det.track_id
                    if tid not in track_sku_memory:
                        crop = det.crop_from_frame(frame)
                        rec_res = recognizer.recognize_crop(crop)
                        track_sku_memory[tid] = (rec_res.sku_id, rec_res.name, rec_res.confidence, rec_res.is_known)

                    det.sku_id, det.sku_name, det.sku_confidence, det.is_known_sku = track_sku_memory[tid]

        # Step C: Update per-product temporal state tracker
        snapshot = tracker.update(batch=batch, frame_index=frame_idx)

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

        # Step E: Render Top HUD Banner with State Breakdown
        hud = np.zeros((hud_h, width, 3), dtype=np.uint8)
        hud[:] = (20, 24, 34)

        # Line 1: Header + Frame Index + State Counts
        cv2.putText(
            hud,
            f"Temporal Inventory State | Frame {frame_idx + 1}/{total_frames} | Stable Facings: {snapshot.stable_facings_count} | Uncertain: {snapshot.uncertain_facings_count} | Changing: {snapshot.changing_facings_count}",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 215, 255),
            2,
            cv2.LINE_AA,
        )

        # Line 2: Legend + Facings semantics disclaimer
        cv2.putText(
            hud,
            f"STABLE (Green/Cyan) | UNCERTAIN (Amber, <{args.min_hits} hits) | CHANGING (Orange/Red, shift) | Cumulative Unique: {snapshot.cumulative_unique_tracks} | FPS: {fps_current:.1f}",
            (14, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (210, 215, 225),
            1,
            cv2.LINE_AA,
        )

        combined = np.vstack([hud, annotated])
        writer.write(combined)

        # Record timeline
        per_frame_timeline.append({
            "frame_index": frame_idx,
            "raw_detections": len(batch.product_detections),
            "stable_facings": snapshot.stable_facings_count,
            "uncertain_facings": snapshot.uncertain_facings_count,
            "changing_facings": snapshot.changing_facings_count,
            "total_active_tracks": snapshot.total_active_tracks,
            "cumulative_unique_tracks": snapshot.cumulative_unique_tracks,
            "new_transitions_in_frame": len(snapshot.transitions),
        })

        frame_idx += 1
        if frame_idx % 15 == 0 or frame_idx == total_frames:
            print(
                f"      Frame {frame_idx:3d}/{total_frames} | "
                f"Stable: {snapshot.stable_facings_count:2d} | "
                f"Uncertain: {snapshot.uncertain_facings_count:2d} | "
                f"Changing: {snapshot.changing_facings_count:2d} | "
                f"Active Tracks: {snapshot.total_active_tracks:2d} | "
                f"Unique: {snapshot.cumulative_unique_tracks:2d}"
            )

    cap.release()
    writer.release()
    total_time = time.perf_counter() - start_time
    print(f"\n[Saved Annotated Video] -> {out_video_path}")

    # 3. Compile Structured JSON Report
    print("\n[3/3] Compiling State Transitions & Metrics Report...")
    transitions_log = tracker.get_full_transitions_log()
    transition_counter = Counter(t["new_state"] for t in transitions_log)

    avg_raw = float(np.mean([f["raw_detections"] for f in per_frame_timeline]))
    avg_stable = float(np.mean([f["stable_facings"] for f in per_frame_timeline]))
    avg_uncertain = float(np.mean([f["uncertain_facings"] for f in per_frame_timeline]))
    avg_changing = float(np.mean([f["changing_facings"] for f in per_frame_timeline]))

    total_naive = sum(f["raw_detections"] for f in per_frame_timeline)
    total_unique = tracker.cumulative_unique_tracks
    dup_reduction = (1.0 - (total_unique / max(1, total_naive))) * 100.0

    report_payload = {
        "video_source": str(video_path),
        "total_frames_analyzed": frame_idx,
        "detector_model": args.model,
        "detector_confidence": args.conf,
        "parameters": {
            "min_hits_for_stable": args.min_hits,
            "max_misses_for_removal": args.max_misses,
            "displacement_threshold_ratio": 0.25,
            "iou_stability_threshold": 0.60,
        },
        "overall_summary": {
            "avg_raw_detections_per_frame": round(avg_raw, 2),
            "avg_stable_facings_per_frame": round(avg_stable, 2),
            "avg_uncertain_facings_per_frame": round(avg_uncertain, 2),
            "avg_changing_facings_per_frame": round(avg_changing, 2),
            "total_cumulative_naive_detections": total_naive,
            "total_unique_physical_tracks": total_unique,
            "duplicate_counting_reduction_pct": round(dup_reduction, 2),
            "total_state_transitions_recorded": len(transitions_log),
            "state_promotion_counts": dict(transition_counter),
            "processing_time_sec": round(total_time, 2),
            "overall_fps": round(frame_idx / max(0.01, total_time), 2),
        },
        "timeline": per_frame_timeline,
        "state_transitions_log": transitions_log,
        "active_tracks_snapshot": [
            {
                "track_id": r.track_id,
                "state": r.state.value,
                "state_reason": r.state_reason,
                "consecutive_hits": r.consecutive_hits,
                "total_hits": r.total_hits,
                "sku_id": r.sku_id,
                "sku_name": r.sku_name,
                "sku_confidence": round(float(r.sku_confidence), 4) if r.sku_confidence else None,
            }
            for r in tracker._tracks.values()
        ],
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"[Saved JSON Report]     -> {out_json_path}")

    # Print Summary Table
    print("\n" + "=" * 80)
    print("STEP 5: TEMPORAL INVENTORY / FACING STATE ESTIMATION SUMMARY")
    print("=" * 80)
    print(f"Total Video Frames Processed:               {frame_idx}")
    print(f"Average Raw Detections Per Frame:           {avg_raw:.2f} facings")
    print(f"Average STABLE Facings Per Frame:           {avg_stable:.2f} facings (confirmed inventory)")
    print(f"Average UNCERTAIN Facings Per Frame:        {avg_uncertain:.2f} facings (new/unconfirmed)")
    print(f"Average POSSIBLY_CHANGING Per Frame:        {avg_changing:.2f} facings (box displacement)")
    print("-" * 80)
    print(f"Total Naive Frame-by-Frame Detections:      {total_naive:,}")
    print(f"Total Unique Physical Tracks (ByteTrack):   {total_unique:,}")
    print(f"Duplicate Counting Elimination:             {dup_reduction:.2f}%")
    print(f"Total State Transitions Recorded:           {len(transitions_log):,}")
    print(f"  • Promoted to STABLE:                     {transition_counter.get('STABLE', 0):,}")
    print(f"  • Initiated as / Shifted to UNCERTAIN:    {transition_counter.get('UNCERTAIN', 0):,}")
    print(f"  • Flagged as POSSIBLY_CHANGING:           {transition_counter.get('POSSIBLY_CHANGING', 0):,}")
    print(f"  • Retired / Removed after Grace Period:   {transition_counter.get('REMOVED', 0):,}")
    print("=" * 80)


if __name__ == "__main__":
    main()
