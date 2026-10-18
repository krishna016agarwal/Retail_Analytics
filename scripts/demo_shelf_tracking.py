"""Step 4: Temporal Tracking for the Retail Inventory Pipeline.

Integrates ByteTrack multi-object tracking downstream of retail_detector_exp2.pt (conf=0.30):
- Assigns persistent, stable track IDs to product bounding boxes across video frames.
- Reuses existing SKU recognition module to maintain consistent identity per track.
- Quantifies duplicate counting reduction vs naive frame-by-frame counting.
- Generates annotated video with track IDs and structured JSON metrics.

Completely isolated from the crowd/queue pipeline (src/).
"""

import argparse
import json
import math
import pathlib
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

from inventory.catalog import SKUCatalog
from inventory.config import InventoryModelConfig
from inventory.shelf_detector import ShelfDetection, ShelfProductDetector
from inventory.shelf_visualizer import ShelfVisualizer
from inventory.sku_recognizer import SpatialColorTextureRecognizer


def generate_synthetic_shelf_video(
    image_path: pathlib.Path,
    output_video_path: pathlib.Path,
    num_frames: int = 90,
    fps: int = 25,
) -> pathlib.Path:
    """Generate a realistic camera pan/jitter video from a high-res shelf image."""
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load source image: {image_path}")

    h, w = img.shape[:2]
    # Viewport dimensions (e.g. 75% width, 75% height)
    vp_w = int(w * 0.72)
    vp_h = int(h * 0.72)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (vp_w, vp_h))

    max_x_offset = w - vp_w
    max_y_offset = h - vp_h

    for i in range(num_frames):
        progress = i / max(1, num_frames - 1)
        # Smooth pan from left to right with subtle vertical breathing and jitter
        x_base = int(max_x_offset * progress)
        y_base = int(max_y_offset * 0.5 + math.sin(progress * math.pi * 3) * 12)
        # Subtle realistic camera jitter (+/- 2 pixels)
        jitter_x = int(math.sin(i * 1.7) * 2)
        jitter_y = int(math.cos(i * 1.3) * 2)

        x1 = max(0, min(w - vp_w, x_base + jitter_x))
        y1 = max(0, min(h - vp_h, y_base + jitter_y))
        x2 = x1 + vp_w
        y2 = y1 + vp_h

        frame = img[y1:y2, x1:x2]
        writer.write(frame)

    writer.release()
    return output_video_path


def parse_args():
    parser = argparse.ArgumentParser(description="Retail Inventory Temporal Tracking Demo")
    parser.add_argument(
        "--source",
        type=str,
        default="demo",
        help="Path to video file, camera index (0), or 'demo' to synthesize a shelf pan video",
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
        default="output/shelf_tracking/tracked_shelf_video.mp4",
        help="Path to save annotated tracking video",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="output/shelf_tracking/tracking_metrics.json",
        help="Path to save tracking metrics JSON",
    )
    parser.add_argument("--conf", type=float, default=0.30, help="Detector confidence cutoff (default: 0.30)")
    parser.add_argument("--match-thresh", type=float, default=0.65, help="SKU similarity threshold")
    parser.add_argument("--max-frames", type=int, default=75, help="Max video frames to process")
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu')")
    return parser.parse_args()


def main():
    args = parse_args()
    out_video_path = pathlib.Path(args.output_video)
    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path = pathlib.Path(args.output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("STEP 4: TEMPORAL TRACKING FOR RETAIL INVENTORY PIPELINE")
    print("================================================================================")
    print(f"Detector Model:   {args.model} (conf={args.conf})")
    print(f"Tracking Backend: ByteTrack (Ultralytics native)")
    print(f"Confidence:       {args.conf}")
    print("--------------------------------------------------------------------------------")

    # 1. Resolve Video Source
    if args.source.lower() == "demo" or not pathlib.Path(args.source).is_file():
        demo_src_img = ROOT_DIR / "inventory_data" / "demo_images" / "SKU110K_fixed" / "images" / "test_1095.jpg"
        synth_video = ROOT_DIR / "inventory_data" / "demo_videos" / "shelf_pan_demo.mp4"
        print(f"\n[1/4] Preparing video source (synthesizing smooth shelf pan from {demo_src_img.name})...")
        video_file = generate_synthetic_shelf_video(demo_src_img, synth_video, num_frames=args.max_frames)
        print(f"      Synthesized video: {video_file} ({args.max_frames} frames)")
    else:
        video_file = pathlib.Path(args.source)
        print(f"\n[1/4] Using provided video source: {video_file}")

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_file}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"      Video metadata: {width}x{height} @ {video_fps:.1f} FPS, {total_video_frames} frames total.")

    # 2. Initialize Detector & Optional Recognizer
    print("\n[2/4] Initializing Retail Detector & ByteTrack...")
    det_cfg = InventoryModelConfig(
        model_path=args.model,
        model_tier="retail_specific",
        device=args.device,
        confidence_threshold=args.conf,
        target_classes=None,
    )
    detector = ShelfProductDetector(det_cfg)

    # Load Catalog for integrated SKU identification
    recognizer = None
    if pathlib.Path(args.catalog).is_file():
        catalog = SKUCatalog.load_json(args.catalog)
        recognizer = SpatialColorTextureRecognizer(catalog=catalog, match_threshold=args.match_thresh)
        recognizer.build_index()
        print(f"      Integrated SKU Recognizer with catalog '{catalog.name}' ({len(catalog)} SKUs).")

    visualizer = ShelfVisualizer(show_confidence=True, show_class_names=True)

    # 3. Setup Video Output Writer
    # Video HUD adds 70px to the top
    hud_h = 70
    out_h = height + hud_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, video_fps, (width, out_h))

    # Tracking Statistics
    all_observed_track_ids: Set[int] = set()
    active_tracks_per_frame: List[int] = []
    raw_detections_per_frame: List[int] = []
    track_hit_counts: Dict[int, int] = defaultdict(int)
    track_first_seen: Dict[int, int] = {}
    track_last_seen: Dict[int, int] = {}
    track_sku_memory: Dict[int, Tuple[str, str, float, bool]] = {} # track_id -> (sku_id, name, conf, is_known)

    frame_idx = 0
    start_time = time.perf_counter()

    print("\n[3/4] Processing Video Frames with Multi-Object Tracking...")
    while cap.isOpened() and frame_idx < args.max_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        t_f0 = time.perf_counter()

        # Run detection with ByteTrack tracking enabled
        batch = detector.detect(
            frame=frame,
            frame_index=frame_idx,
            track=True,
            persist=True,
            tracker="bytetrack.yaml",
        )

        active_tracks_in_frame = 0
        for det in batch.product_detections:
            if det.track_id is not None:
                tid = det.track_id
                active_tracks_in_frame += 1
                all_observed_track_ids.add(tid)
                track_hit_counts[tid] += 1
                if tid not in track_first_seen:
                    track_first_seen[tid] = frame_idx
                track_last_seen[tid] = frame_idx

                # SKU recognition: compute on first observation or recall from memory
                if recognizer is not None:
                    if tid not in track_sku_memory:
                        crop = det.crop_from_frame(frame)
                        rec_res = recognizer.recognize_crop(crop)
                        track_sku_memory[tid] = (rec_res.sku_id, rec_res.name, rec_res.confidence, rec_res.is_known)

                    det.sku_id, det.sku_name, det.sku_confidence, det.is_known_sku = track_sku_memory[tid]

        active_tracks_per_frame.append(active_tracks_in_frame)
        raw_detections_per_frame.append(len(batch.product_detections))

        fps_current = 1.0 / max(1e-4, time.perf_counter() - t_f0)

        # Render visual overlay
        annotated = visualizer.annotate_frame(
            frame=frame,
            batch=batch,
            state=None,
            fps=fps_current,
            shelf_roi=None,
            model_tier="retail_bytetrack",
        )

        # Add Tracking HUD Banner
        hud = np.zeros((hud_h, width, 3), dtype=np.uint8)
        hud[:] = (20, 24, 34)

        cumulative_raw = sum(raw_detections_per_frame)
        unique_so_far = len(all_observed_track_ids)
        dup_reduction = (1.0 - (unique_so_far / max(1, cumulative_raw))) * 100.0

        cv2.putText(
            hud,
            f"ByteTrack Temporal Tracking | Frame {frame_idx + 1}/{args.max_frames} | Active Tracks: {active_tracks_in_frame}",
            (14, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 215, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            hud,
            f"Unique Products: {unique_so_far} | Naive Frame Sum: {cumulative_raw} | Duplicate Reduction: {dup_reduction:.1f}% | FPS: {fps_current:.1f}",
            (14, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (110, 230, 140),
            1,
            cv2.LINE_AA,
        )

        combined = np.vstack([hud, annotated])
        writer.write(combined)

        frame_idx += 1
        if frame_idx % 15 == 0 or frame_idx == args.max_frames:
            print(
                f"      Frame {frame_idx:3d}/{args.max_frames} | "
                f"Active Tracks: {active_tracks_in_frame:2d} | "
                f"Cumulative Unique Tracks: {unique_so_far:2d} | "
                f"Naive Detection Sum: {cumulative_raw:4d} | "
                f"Duplicate Reduction: {dup_reduction:.1f}%"
            )

    cap.release()
    writer.release()

    total_time = time.perf_counter() - start_time
    total_raw_detections = sum(raw_detections_per_frame)
    total_unique_tracks = len(all_observed_track_ids)
    avg_active_tracks = float(np.mean(active_tracks_per_frame)) if active_tracks_per_frame else 0.0
    avg_raw_detections = float(np.mean(raw_detections_per_frame)) if raw_detections_per_frame else 0.0
    duplicate_reduction_pct = (1.0 - (total_unique_tracks / max(1, total_raw_detections))) * 100.0

    print(f"\n[Saved Tracking Video] -> {out_video_path}")

    # 4. Generate Structured JSON Report
    print("\n[4/4] Compiling Tracking Metrics Report...")
    report_data = {
        "video_source": str(video_file),
        "total_frames_analyzed": frame_idx,
        "detector_model": args.model,
        "detector_confidence": args.conf,
        "tracker": "ByteTrack",
        "metrics": {
            "total_raw_detections": total_raw_detections,
            "total_unique_product_tracks": total_unique_tracks,
            "avg_active_tracks_per_frame": round(avg_active_tracks, 2),
            "avg_raw_detections_per_frame": round(avg_raw_detections, 2),
            "duplicate_count_reduction_pct": round(duplicate_reduction_pct, 2),
            "total_processing_time_sec": round(total_time, 2),
            "overall_fps": round(frame_idx / max(0.01, total_time), 2),
        },
        "per_track_summary": [
            {
                "track_id": tid,
                "first_frame": track_first_seen[tid],
                "last_frame": track_last_seen[tid],
                "duration_frames": track_last_seen[tid] - track_first_seen[tid] + 1,
                "total_detections": track_hit_counts[tid],
                "sku_id": track_sku_memory.get(tid, ("UNKNOWN", "UNKNOWN", 0.0, False))[0],
                "sku_name": track_sku_memory.get(tid, ("UNKNOWN", "UNKNOWN", 0.0, False))[1],
                "sku_confidence": round(track_sku_memory.get(tid, ("UNKNOWN", "UNKNOWN", 0.0, False))[2], 4),
            }
            for tid in sorted(all_observed_track_ids)
        ],
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[Saved JSON Report]    -> {out_json_path}")

    # Print Summary Table
    print("\n" + "=" * 80)
    print("STEP 4: TEMPORAL TRACKING RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Video Frames Analyzed:             {frame_idx}")
    print(f"Average Detections Per Frame:            {avg_raw_detections:.2f}")
    print(f"Average Active Tracks Per Frame:         {avg_active_tracks:.2f}")
    print(f"Total Naive Cumulative Detections:       {total_raw_detections:,}")
    print(f"Total Unique Product Tracks (ByteTrack): {total_unique_tracks:,}")
    print(f"Duplicate Counting Reduction:            {duplicate_reduction_pct:.2f}%")
    print(f"Overall Processing Speed:                {frame_idx / max(0.01, total_time):.1f} FPS on CPU")
    print("=" * 80)


if __name__ == "__main__":
    main()
