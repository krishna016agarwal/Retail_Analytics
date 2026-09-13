"""Step 20: Camera Motion Tracking Fix Benchmark.

Evaluates isolated tracker association tuning on shelf_pan_demo.mp4 without GMC/CMC.
Compares:
1. Current ByteTrack configuration (inventory/bytetrack_shelf.yaml)
2. Relaxed association configuration(s) (lower match threshold / centroid-distance tolerance)

Strictly isolated: does NOT modify production code, weights, or crowd/queue logic.
"""

import json
import math
import pathlib
import sys
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import yaml

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inventory.catalog import SKUCatalog
from inventory.config import InventoryModelConfig
from inventory.inventory_events import InventoryEventConfig, InventoryEventDetector, InventoryEventType
from inventory.shelf_detector import ShelfProductDetector
from inventory.shelf_state import ProductTrackStateTracker
from inventory.sku_recognizer import SpatialColorTextureRecognizer

OUTPUT_DIR = ROOT_DIR / "output" / "tracking_fix_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_bbox_distance_and_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]):
    cx1 = (b1[0] + b1[2]) / 2.0
    cy1 = (b1[1] + b1[3]) / 2.0
    cx2 = (b2[0] + b2[2]) / 2.0
    cy2 = (b2[1] + b2[3]) / 2.0
    dist = math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)

    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    iou = inter / union if union > 0 else 0.0
    return dist, iou


def evaluate_tracker_configuration(
    video_path: pathlib.Path,
    tracker_yaml_path: pathlib.Path,
    config_label: str,
    max_frames: int = 75,
) -> Dict[str, Any]:
    print("=" * 80)
    print(f"BENCHMARKING: {config_label}")
    print(f"Tracker Config: {tracker_yaml_path}")
    print("=" * 80)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_to_process = min(total_frames, max_frames)

    # 1. Initialize detector with specific tracker config
    det_cfg = InventoryModelConfig(
        model_path="inventory_data/custom_model/retail_detector_exp2.pt",
        model_tier="retail_specific",
        device="cpu",
        confidence_threshold=0.30,
        tracker_config_path=str(tracker_yaml_path),
    )
    detector = ShelfProductDetector(det_cfg)

    catalog_path = ROOT_DIR / "inventory_data" / "catalogs" / "demo_store_catalog.json"
    cat = SKUCatalog.load_json(str(catalog_path)) if catalog_path.exists() else None
    recognizer = SpatialColorTextureRecognizer(catalog=cat, match_threshold=0.65) if cat else None
    if recognizer:
        recognizer.build_index()

    tracker = ProductTrackStateTracker(
        min_hits_for_stable=3,
        removal_grace_seconds=1.5,
        occlusion_freeze_enabled=True,
    )
    event_detector = InventoryEventDetector(
        config=InventoryEventConfig(
            min_hits_for_stable=3,
            max_misses_for_removal=45,
            removal_grace_seconds=1.5,
            pan_boundary_margin_px=120,
            sku_window_size=5,
        ),
        recognizer=recognizer,
    )

    all_observed_track_ids: Set[int] = set()
    track_first_seen: Dict[int, int] = {}
    track_last_seen: Dict[int, int] = {}
    track_bboxes_by_frame: Dict[int, Dict[int, Tuple[int, int, int, int]]] = defaultdict(dict)
    track_sku_history: Dict[int, List[str]] = defaultdict(list)

    detections_per_frame: List[int] = []
    active_tracks_per_frame: List[int] = []
    all_events: List[Dict] = []

    frame_idx = 0
    t0 = time.perf_counter()

    while frame_idx < frames_to_process:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        ts = frame_idx / fps
        batch = detector.detect(
            frame=frame,
            frame_index=frame_idx,
            track=True,
            persist=True,
            tracker=str(tracker_yaml_path),
        )
        snapshot = tracker.update(batch=batch, frame_index=frame_idx, fps=fps)
        events = event_detector.process_frame(
            batch=batch, snapshot=snapshot, frame_index=frame_idx, timestamp_sec=ts, frame_bgr=frame
        )
        for ev in events:
            all_events.append(ev.to_dict())

        det_count = len(batch.product_detections)
        detections_per_frame.append(det_count)

        active_tids = set()
        for det in batch.product_detections:
            if det.track_id is not None:
                tid = det.track_id
                active_tids.add(tid)
                all_observed_track_ids.add(tid)
                if tid not in track_first_seen:
                    track_first_seen[tid] = frame_idx
                track_last_seen[tid] = frame_idx
                track_bboxes_by_frame[tid][frame_idx] = det.bbox
                if det.sku_name:
                    track_sku_history[tid].append(det.sku_name)

        active_tracks_per_frame.append(len(active_tids))
        frame_idx += 1

    cap.release()
    elapsed_time = time.perf_counter() - t0

    # --------------------------------------------------------------------------
    # Metrics Computation
    # --------------------------------------------------------------------------
    # 1. Apparent ID switches
    id_switch_candidates = []
    sorted_tids = sorted(all_observed_track_ids)
    for tid_old in sorted_tids:
        last_f = track_last_seen[tid_old]
        if last_f >= frames_to_process - 2:
            continue
        last_box = track_bboxes_by_frame[tid_old].get(last_f)
        if last_box is None:
            continue

        for tid_new in sorted_tids:
            if tid_new <= tid_old:
                continue
            first_f = track_first_seen[tid_new]
            if 0 <= (first_f - last_f) <= 3:
                first_box = track_bboxes_by_frame[tid_new].get(first_f)
                if first_box is not None:
                    dist, iou = compute_bbox_distance_and_iou(last_box, first_box)
                    if iou > 0.40 or dist < 45.0:
                        id_switch_candidates.append({
                            "old_track_id": tid_old,
                            "new_track_id": tid_new,
                            "lost_at_frame": last_f,
                            "born_at_frame": first_f,
                            "centroid_distance_px": round(dist, 1),
                            "overlap_iou": round(iou, 2),
                        })

    # 2. Track Lifespan & Continuity
    lifespans = [track_last_seen[tid] - track_first_seen[tid] + 1 for tid in all_observed_track_ids]
    avg_lifespan = float(np.mean(lifespans)) if lifespans else 0.0
    short_lived_tracks = sum(1 for L in lifespans if L <= 5)
    persistent_tracks = sum(1 for L in lifespans if L >= 20)
    track_continuity_score = round((persistent_tracks / max(1, len(all_observed_track_ids))) * 100.0, 1)

    # 3. Duplicate PRODUCT_APPEARED events (Interior vs Boundary)
    boundary_margin_px = 120
    edge_appearances = 0
    interior_appearances = 0  # These are duplicates / re-appearances of fractured tracks
    edge_removals = 0
    interior_removals = 0
    pan_exits = 0

    for ev in all_events:
        bbox = ev.get("bbox") or ev.get("previous_bbox")
        if bbox:
            cx = (bbox[0] + bbox[2]) / 2.0
            is_edge = (cx < boundary_margin_px) or (cx > width - boundary_margin_px)
            if ev["event_type"] == InventoryEventType.PRODUCT_APPEARED.value:
                if is_edge:
                    edge_appearances += 1
                else:
                    interior_appearances += 1
            elif ev["event_type"] == InventoryEventType.PRODUCT_REMOVED.value:
                if is_edge:
                    edge_removals += 1
                else:
                    interior_removals += 1
            elif ev["event_type"] == InventoryEventType.OUT_OF_VIEW_PAN_EXIT.value:
                pan_exits += 1

    # 4. SKU Recognition Continuity
    sku_instabilities = 0
    stable_skus = 0
    for tid, names in track_sku_history.items():
        unique_names = set(names)
        if len(unique_names) > 1:
            sku_instabilities += 1
        elif len(unique_names) == 1:
            stable_skus += 1

    sku_continuity_rate = (stable_skus / max(1, stable_skus + sku_instabilities)) * 100.0
    avg_active_facings = float(np.mean(detections_per_frame)) if detections_per_frame else 1.0
    inflation_ratio = round(len(all_observed_track_ids) / max(1.0, avg_active_facings), 2)

    results = {
        "config_label": config_label,
        "tracker_config_file": tracker_yaml_path.name,
        "elapsed_time_sec": round(elapsed_time, 2),
        "total_unique_tracks": len(all_observed_track_ids),
        "avg_active_facings_per_frame": round(avg_active_facings, 1),
        "track_inflation_ratio": inflation_ratio,
        "apparent_id_switches": len(id_switch_candidates),
        "track_continuity_score_pct": track_continuity_score,
        "avg_lifespan_frames": round(avg_lifespan, 1),
        "short_lived_tracks": short_lived_tracks,
        "persistent_tracks": persistent_tracks,
        "sku_continuity_rate_pct": round(sku_continuity_rate, 1),
        "duplicate_product_appeared_interior": interior_appearances,
        "edge_product_appeared": edge_appearances,
        "product_removed_events": edge_removals + interior_removals,
        "out_of_view_pan_exits": pan_exits,
        "sample_id_switches": id_switch_candidates[:5],
    }
    return results


def main():
    pan_video = ROOT_DIR / "inventory_data" / "demo_videos" / "shelf_pan_demo.mp4"
    if not pan_video.exists():
        print(f"[ERROR] Video not found: {pan_video}")
        return

    # Configuration 1: Current ByteTrack Configuration
    curr_yaml = ROOT_DIR / "inventory" / "bytetrack_shelf.yaml"

    # Configuration 2A: Relaxed Association (IoU 0.85 cost limit, fuse_score False, new_track 0.30)
    relaxed_a_yaml = OUTPUT_DIR / "bytetrack_relaxed_a.yaml"
    relaxed_a_content = {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.10,
        "new_track_thresh": 0.30,
        "track_buffer": 60,
        "match_thresh": 0.85,
        "fuse_score": False,
    }
    relaxed_a_yaml.write_text(yaml.dump(relaxed_a_content), encoding="utf-8")

    # Configuration 2B: High-Tolerance Association (IoU 0.90 cost limit, fuse_score False, new_track 0.35)
    relaxed_b_yaml = OUTPUT_DIR / "bytetrack_relaxed_b.yaml"
    relaxed_b_content = {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.10,
        "new_track_thresh": 0.35,
        "track_buffer": 60,
        "match_thresh": 0.90,
        "fuse_score": False,
    }
    relaxed_b_yaml.write_text(yaml.dump(relaxed_b_content), encoding="utf-8")

    # Run benchmarks
    res_current = evaluate_tracker_configuration(
        pan_video, curr_yaml, "Approach 1: Current ByteTrack (match=0.80, fuse=True)"
    )
    res_relaxed_a = evaluate_tracker_configuration(
        pan_video, relaxed_a_yaml, "Approach 2A: Relaxed Association (match=0.85, fuse=False)"
    )
    res_relaxed_b = evaluate_tracker_configuration(
        pan_video, relaxed_b_yaml, "Approach 2B: High-Tolerance Association (match=0.90, fuse=False)"
    )

    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "video": pan_video.name,
        "frames_evaluated": 75,
        "configurations": {
            "current_bytetrack": res_current,
            "relaxed_association_a": res_relaxed_a,
            "relaxed_association_b": res_relaxed_b,
        },
    }

    report_path = OUTPUT_DIR / "tracking_fix_comparison_report.json"
    report_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\n[OK] Benchmark completed! Saved to {report_path}")

    # Summary table
    print("\n" + "=" * 90)
    print("STEP 20 — CAMERA MOTION TRACKING FIX: ASSOCIATION TUNING COMPARISON")
    print("=" * 90)
    headers = [
        "Metric",
        "Current (Baseline)",
        "Relaxed A (0.85/False)",
        "Relaxed B (0.90/False)",
    ]
    fmt = "{:<35} | {:<16} | {:<20} | {:<20}"
    print(fmt.format(*headers))
    print("-" * 90)
    print(fmt.format(
        "Total Unique Tracks",
        str(res_current["total_unique_tracks"]),
        str(res_relaxed_a["total_unique_tracks"]),
        str(res_relaxed_b["total_unique_tracks"]),
    ))
    print(fmt.format(
        "Track Inflation Ratio",
        f"{res_current['track_inflation_ratio']}x",
        f"{res_relaxed_a['track_inflation_ratio']}x",
        f"{res_relaxed_b['track_inflation_ratio']}x",
    ))
    print(fmt.format(
        "Apparent ID Switches",
        str(res_current["apparent_id_switches"]),
        str(res_relaxed_a["apparent_id_switches"]),
        str(res_relaxed_b["apparent_id_switches"]),
    ))
    print(fmt.format(
        "Track Continuity Score (%)",
        f"{res_current['track_continuity_score_pct']}%",
        f"{res_relaxed_a['track_continuity_score_pct']}%",
        f"{res_relaxed_b['track_continuity_score_pct']}%",
    ))
    print(fmt.format(
        "Avg Track Lifespan (frames)",
        f"{res_current['avg_lifespan_frames']} f",
        f"{res_relaxed_a['avg_lifespan_frames']} f",
        f"{res_relaxed_b['avg_lifespan_frames']} f",
    ))
    print(fmt.format(
        "Duplicate APPEARED (Interior)",
        str(res_current["duplicate_product_appeared_interior"]),
        str(res_relaxed_a["duplicate_product_appeared_interior"]),
        str(res_relaxed_b["duplicate_product_appeared_interior"]),
    ))
    print(fmt.format(
        "SKU Continuity Rate (%)",
        f"{res_current['sku_continuity_rate_pct']}%",
        f"{res_relaxed_a['sku_continuity_rate_pct']}%",
        f"{res_relaxed_b['sku_continuity_rate_pct']}%",
    ))
    print("=" * 90)


if __name__ == "__main__":
    main()
