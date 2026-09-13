"""Step 21: Comprehensive Pre-Production Validation of Approach 2A.

Evaluates:
1. Current Production Config (bytetrack_shelf.yaml: match=0.80, fuse=True)
2. Ablation 1: Only fuse_score=False (match=0.80, fuse=False)
3. Ablation 2: Only match_thresh=0.85 (match=0.85, fuse=True)
4. Approach 2A: match_thresh=0.85, fuse_score=False, new_track_thresh=0.30

Tested Across Three Real-World Operational Scenarios:
- Scenario 1: Camera Motion Panning (shelf_pan_demo.mp4, 75 frames @ 25 FPS)
- Scenario 2: Shopper Shelf Occlusion (store-aisle-detection.mp4, frames 360-520 @ 60 FPS)
- Scenario 3: Genuine Product Removal (stable shelf item permanently picked/removed)

Strictly isolated: does NOT modify production configs, weights, or crowd/queue code.
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

OUTPUT_DIR = ROOT_DIR / "output" / "step21_validation"
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


# ==============================================================================
# Scenario 1: Camera Motion Panning Evaluation
# ==============================================================================
def evaluate_camera_panning(
    video_path: pathlib.Path,
    tracker_yaml_path: pathlib.Path,
    max_frames: int = 75,
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    all_observed_tids: Set[int] = set()
    track_first_seen: Dict[int, int] = {}
    track_last_seen: Dict[int, int] = {}
    track_bboxes_by_frame: Dict[int, Dict[int, Tuple[int, int, int, int]]] = defaultdict(dict)
    track_sku_history: Dict[int, List[str]] = defaultdict(list)
    detections_per_frame: List[int] = []
    all_events: List[Dict] = []

    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        ts = frame_idx / fps
        batch = detector.detect(
            frame=frame, frame_index=frame_idx, track=True, persist=True, tracker=str(tracker_yaml_path)
        )
        snapshot = tracker.update(batch=batch, frame_index=frame_idx, fps=fps)
        events = event_detector.process_frame(
            batch=batch, snapshot=snapshot, frame_index=frame_idx, timestamp_sec=ts, frame_bgr=frame
        )
        for ev in events:
            all_events.append(ev.to_dict())

        detections_per_frame.append(len(batch.product_detections))
        for det in batch.product_detections:
            if det.track_id is not None:
                tid = det.track_id
                all_observed_tids.add(tid)
                if tid not in track_first_seen:
                    track_first_seen[tid] = frame_idx
                track_last_seen[tid] = frame_idx
                track_bboxes_by_frame[tid][frame_idx] = det.bbox
                if det.sku_name:
                    track_sku_history[tid].append(det.sku_name)

        frame_idx += 1
    cap.release()

    # Metrics
    id_switches = 0
    sorted_tids = sorted(all_observed_tids)
    for tid_old in sorted_tids:
        last_f = track_last_seen[tid_old]
        if last_f >= max_frames - 2:
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
                        id_switches += 1

    lifespans = [track_last_seen[tid] - track_first_seen[tid] + 1 for tid in all_observed_tids]
    avg_lifespan = float(np.mean(lifespans)) if lifespans else 0.0
    persistent_tracks = sum(1 for L in lifespans if L >= 20)
    continuity_pct = round((persistent_tracks / max(1, len(all_observed_tids))) * 100.0, 1)

    stable_skus = sum(1 for tid, names in track_sku_history.items() if len(set(names)) == 1)
    sku_continuity_pct = round((stable_skus / max(1, len(track_sku_history))) * 100.0, 1)

    interior_app = 0
    edge_app = 0
    rem_events = 0
    pan_exits = 0
    for ev in all_events:
        etype = ev["event_type"]
        bbox = ev.get("bbox") or ev.get("previous_bbox")
        if bbox:
            cx = (bbox[0] + bbox[2]) / 2.0
            is_edge = (cx < 120) or (cx > width - 120)
            if etype == InventoryEventType.PRODUCT_APPEARED.value:
                if is_edge:
                    edge_app += 1
                else:
                    interior_app += 1
            elif etype == InventoryEventType.PRODUCT_REMOVED.value:
                rem_events += 1
            elif etype == InventoryEventType.OUT_OF_VIEW_PAN_EXIT.value:
                pan_exits += 1

    avg_facings = float(np.mean(detections_per_frame)) if detections_per_frame else 1.0
    return {
        "total_unique_tracks": len(all_observed_tids),
        "track_inflation_ratio": round(len(all_observed_tids) / max(1.0, avg_facings), 2),
        "apparent_id_switches": id_switches,
        "track_continuity_pct": continuity_pct,
        "avg_lifespan_frames": round(avg_lifespan, 1),
        "duplicate_appeared_interior": interior_app,
        "product_removed_events": rem_events,
        "out_of_view_pan_exits": pan_exits,
        "sku_continuity_pct": sku_continuity_pct,
    }


# ==============================================================================
# Scenario 2: Shopper Occlusion Evaluation
# ==============================================================================
def evaluate_shopper_occlusion(
    video_path: pathlib.Path,
    tracker_yaml_path: pathlib.Path,
    start_frame: int = 360,
    end_frame: int = 520,
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
    det_cfg = InventoryModelConfig(
        model_path="inventory_data/custom_model/retail_detector_exp2.pt",
        model_tier="retail_specific",
        device="cpu",
        confidence_threshold=0.30,
        tracker_config_path=str(tracker_yaml_path),
    )
    detector = ShelfProductDetector(det_cfg)

    # 1.5s grace window at 60 FPS = 90 frames
    tracker = ProductTrackStateTracker(
        min_hits_for_stable=3,
        removal_grace_seconds=1.5,
        occlusion_freeze_enabled=True,
    )
    event_detector = InventoryEventDetector(
        config=InventoryEventConfig(
            min_hits_for_stable=3,
            removal_grace_seconds=1.5,
            pan_boundary_margin_px=120,
            sku_window_size=5,
        )
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    phase_a_tracks: Set[int] = set()
    phase_b_tracks: Set[int] = set()
    phase_c_tracks: Set[int] = set()
    emitted_removed: List[Dict] = []
    all_observed_tids: Set[int] = set()

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        ts = frame_idx / fps
        batch = detector.detect(
            frame=frame, frame_index=frame_idx, track=True, persist=True, tracker=str(tracker_yaml_path)
        )
        snapshot = tracker.update(batch=batch, frame_index=frame_idx, fps=fps)
        events = event_detector.process_frame(
            batch=batch, snapshot=snapshot, frame_index=frame_idx, timestamp_sec=ts, frame_bgr=frame
        )

        for ev in events:
            if ev.event_type == InventoryEventType.PRODUCT_REMOVED:
                emitted_removed.append(ev.to_dict())

        active_tids = {d.track_id for d in batch.product_detections if d.track_id is not None}
        all_observed_tids.update(active_tids)

        if frame_idx <= 400:
            phase_a_tracks.update(active_tids)
        elif 401 <= frame_idx <= 480:
            phase_b_tracks.update(active_tids)
        else:
            phase_c_tracks.update(active_tids)

        frame_idx += 1
    cap.release()

    # Recovery metrics
    # Retained tracks: stable tracks seen in Phase A that are successfully re-identified in Phase C
    retained_in_phase_c = len(phase_a_tracks.intersection(phase_c_tracks))
    retention_rate = round((retained_in_phase_c / max(1, len(phase_a_tracks))) * 100.0, 1)
    new_tracks_post_occlusion = len(phase_c_tracks - phase_a_tracks)

    return {
        "phase_a_baseline_tracks": len(phase_a_tracks),
        "retained_tracks_post_occlusion": retained_in_phase_c,
        "track_retention_rate_pct": retention_rate,
        "false_product_removed_during_occlusion": len(emitted_removed),
        "duplicate_new_tracks_post_occlusion": new_tracks_post_occlusion,
        "total_unique_tracks_in_session": len(all_observed_tids),
    }


# ==============================================================================
# Scenario 3: Genuine Product Removal Scenario
# ==============================================================================
def evaluate_genuine_removal(
    video_path: pathlib.Path,
    tracker_yaml_path: pathlib.Path,
    start_frame: int = 50,
    pick_frame: int = 100,
    end_frame: int = 195,
) -> Dict[str, Any]:
    """Tests permanent physical removal of a confirmed stable product facing.

    At pick_frame, a central product facing is permanently masked out (simulating
    customer picking the item and walking away). Evaluates whether the system:
    1. Successfully emits PRODUCT_REMOVED
    2. Does so within the calibrated grace window (no premature or delayed removal)
    3. Leaves untouched neighbor items unaffected
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
    det_cfg = InventoryModelConfig(
        model_path="inventory_data/custom_model/retail_detector_exp2.pt",
        model_tier="retail_specific",
        device="cpu",
        confidence_threshold=0.30,
        tracker_config_path=str(tracker_yaml_path),
    )
    detector = ShelfProductDetector(det_cfg)
    tracker = ProductTrackStateTracker(
        min_hits_for_stable=3,
        removal_grace_seconds=1.5,
        occlusion_freeze_enabled=True,
    )
    event_detector = InventoryEventDetector(
        config=InventoryEventConfig(
            min_hits_for_stable=3,
            removal_grace_seconds=1.5,
            pan_boundary_margin_px=60,
            sku_window_size=5,
        )
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    target_id: Optional[int] = None
    target_removed = False
    removal_frame: Optional[int] = None
    false_removals = 0

    frame_idx = start_frame
    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Simulate customer picking and removing central facing
        if frame_idx >= pick_frame:
            frame[230:260, 355:380] = 120

        batch = detector.detect(
            frame=frame, frame_index=frame_idx, track=True, persist=True, tracker=str(tracker_yaml_path)
        )

        # Identify target track before removal
        if frame_idx == pick_frame - 1:
            cands = [d for d in batch.product_detections if abs(d.bbox[0] - 359) < 15 and abs(d.bbox[1] - 237) < 15]
            if cands:
                target_id = cands[0].track_id

        snapshot = tracker.update(batch=batch, frame_index=frame_idx, fps=fps)
        events = event_detector.process_frame(
            batch=batch, snapshot=snapshot, frame_index=frame_idx, timestamp_sec=frame_idx / fps, frame_bgr=frame
        )

        for ev in events:
            if ev.event_type == InventoryEventType.PRODUCT_REMOVED:
                if target_id is not None and ev.track_id == target_id:
                    target_removed = True
                    removal_frame = frame_idx
                else:
                    false_removals += 1

        frame_idx += 1
    cap.release()

    return {
        "target_track_id_picked": target_id,
        "genuine_removal_detected": target_removed,
        "removal_event_emitted": target_removed,
        "removal_frame": removal_frame,
        "grace_latency_seconds": round((removal_frame - pick_frame) / fps, 2) if removal_frame else None,
        "false_removals_of_other_items": false_removals,
    }


def main():
    pan_video = ROOT_DIR / "inventory_data" / "demo_videos" / "shelf_pan_demo.mp4"
    shopper_video = ROOT_DIR / "videos" / "store-aisle-detection.mp4"

    # Define the 4 configurations
    configs = {
        "current_production": {
            "label": "1. Current Production (match=0.80, fuse=True)",
            "yaml_data": {
                "tracker_type": "bytetrack",
                "track_high_thresh": 0.25,
                "track_low_thresh": 0.10,
                "new_track_thresh": 0.25,
                "track_buffer": 60,
                "match_thresh": 0.80,
                "fuse_score": True,
            },
        },
        "ablation_fuse_false_only": {
            "label": "2. Ablation: fuse_score=False Only (match=0.80)",
            "yaml_data": {
                "tracker_type": "bytetrack",
                "track_high_thresh": 0.25,
                "track_low_thresh": 0.10,
                "new_track_thresh": 0.25,
                "track_buffer": 60,
                "match_thresh": 0.80,
                "fuse_score": False,
            },
        },
        "ablation_match_85_only": {
            "label": "3. Ablation: match_thresh=0.85 Only (fuse=True)",
            "yaml_data": {
                "tracker_type": "bytetrack",
                "track_high_thresh": 0.25,
                "track_low_thresh": 0.10,
                "new_track_thresh": 0.25,
                "track_buffer": 60,
                "match_thresh": 0.85,
                "fuse_score": True,
            },
        },
        "approach_2a_combined": {
            "label": "4. Approach 2A: Combined (match=0.85, fuse=False)",
            "yaml_data": {
                "tracker_type": "bytetrack",
                "track_high_thresh": 0.25,
                "track_low_thresh": 0.10,
                "new_track_thresh": 0.30,
                "track_buffer": 60,
                "match_thresh": 0.85,
                "fuse_score": False,
            },
        },
    }

    full_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluations": {},
    }

    for cfg_id, cfg in configs.items():
        print("\n" + "#" * 80)
        print(f"EVALUATING CONFIGURATION: {cfg['label']}")
        print("#" * 80)

        cfg_yaml_path = OUTPUT_DIR / f"{cfg_id}.yaml"
        cfg_yaml_path.write_text(yaml.dump(cfg["yaml_data"]), encoding="utf-8")

        # 1. Camera Panning
        print("-> Running Scenario 1: Camera Panning (shelf_pan_demo.mp4)...")
        pan_res = evaluate_camera_panning(pan_video, cfg_yaml_path, max_frames=75)

        # 2. Shopper Occlusion
        print("-> Running Scenario 2: Shopper Occlusion (store-aisle-detection.mp4)...")
        occ_res = evaluate_shopper_occlusion(shopper_video, cfg_yaml_path, start_frame=360, end_frame=520)

        # 3. Genuine Product Removal
        print("-> Running Scenario 3: Genuine Product Removal...")
        rem_res = evaluate_genuine_removal(shopper_video, cfg_yaml_path, start_frame=50, pick_frame=100, end_frame=195)

        full_results["evaluations"][cfg_id] = {
            "label": cfg["label"],
            "camera_panning": pan_res,
            "shopper_occlusion": occ_res,
            "genuine_removal": rem_res,
        }

    out_file = OUTPUT_DIR / "step21_validation_report.json"
    out_file.write_text(json.dumps(full_results, indent=2), encoding="utf-8")
    print(f"\n[OK] Validation complete! Full report saved to: {out_file}")

    # Print Summary Table
    print("\n" + "=" * 105)
    print("STEP 21: FINAL PRE-PRODUCTION VALIDATION ACROSS ALL 3 OPERATIONAL SCENARIOS")
    print("=" * 105)
    fmt = "{:<32} | {:<16} | {:<16} | {:<16} | {:<16}"
    print(fmt.format("Metric", "1. Current Prod", "2. fuse=False", "3. match=0.85", "4. Approach 2A"))
    print("-" * 105)

    ev = full_results["evaluations"]
    c1 = ev["current_production"]
    c2 = ev["ablation_fuse_false_only"]
    c3 = ev["ablation_match_85_only"]
    c4 = ev["approach_2a_combined"]

    p1, p2, p3, p4 = c1["camera_panning"], c2["camera_panning"], c3["camera_panning"], c4["camera_panning"]
    o1, o2, o3, o4 = c1["shopper_occlusion"], c2["shopper_occlusion"], c3["shopper_occlusion"], c4["shopper_occlusion"]
    r1, r2, r3, r4 = c1["genuine_removal"], c2["genuine_removal"], c3["genuine_removal"], c4["genuine_removal"]

    print("[1. Camera Panning: shelf_pan_demo.mp4]")
    print(fmt.format("  Total Unique Tracks", str(p1["total_unique_tracks"]), str(p2["total_unique_tracks"]), str(p3["total_unique_tracks"]), str(p4["total_unique_tracks"])))
    print(fmt.format("  Track Inflation Ratio", f"{p1['track_inflation_ratio']}x", f"{p2['track_inflation_ratio']}x", f"{p3['track_inflation_ratio']}x", f"{p4['track_inflation_ratio']}x"))
    print(fmt.format("  Apparent ID Switches", str(p1["apparent_id_switches"]), str(p2["apparent_id_switches"]), str(p3["apparent_id_switches"]), str(p4["apparent_id_switches"])))
    print(fmt.format("  Track Continuity (%)", f"{p1['track_continuity_pct']}%", f"{p2['track_continuity_pct']}%", f"{p3['track_continuity_pct']}%", f"{p4['track_continuity_pct']}%"))
    print(fmt.format("  Avg Lifespan (frames)", f"{p1['avg_lifespan_frames']}f", f"{p2['avg_lifespan_frames']}f", f"{p3['avg_lifespan_frames']}f", f"{p4['avg_lifespan_frames']}f"))
    print(fmt.format("  Duplicate Interior APPEARED", str(p1["duplicate_appeared_interior"]), str(p2["duplicate_appeared_interior"]), str(p3["duplicate_appeared_interior"]), str(p4["duplicate_appeared_interior"])))
    print(fmt.format("  PRODUCT_REMOVED (false)", str(p1["product_removed_events"]), str(p2["product_removed_events"]), str(p3["product_removed_events"]), str(p4["product_removed_events"])))
    print(fmt.format("  OUT_OF_VIEW_PAN_EXIT", str(p1["out_of_view_pan_exits"]), str(p2["out_of_view_pan_exits"]), str(p3["out_of_view_pan_exits"]), str(p4["out_of_view_pan_exits"])))
    print(fmt.format("  SKU Continuity (%)", f"{p1['sku_continuity_pct']}%", f"{p2['sku_continuity_pct']}%", f"{p3['sku_continuity_pct']}%", f"{p4['sku_continuity_pct']}%"))

    print("\n[2. Shopper Occlusion: store-aisle-detection.mp4]")
    print(fmt.format("  Track Retention Post-Occ (%)", f"{o1['track_retention_rate_pct']}%", f"{o2['track_retention_rate_pct']}%", f"{o3['track_retention_rate_pct']}%", f"{o4['track_retention_rate_pct']}%"))
    print(fmt.format("  False REMOVED during Occ", str(o1["false_product_removed_during_occlusion"]), str(o2["false_product_removed_during_occlusion"]), str(o3["false_product_removed_during_occlusion"]), str(o4["false_product_removed_during_occlusion"])))
    print(fmt.format("  Duplicate Tracks Post-Occ", str(o1["duplicate_new_tracks_post_occlusion"]), str(o2["duplicate_new_tracks_post_occlusion"]), str(o3["duplicate_new_tracks_post_occlusion"]), str(o4["duplicate_new_tracks_post_occlusion"])))

    print("\n[3. Genuine Removal Scenario]")
    print(fmt.format("  Genuine Removal Detected?", str(r1["genuine_removal_detected"]), str(r2["genuine_removal_detected"]), str(r3["genuine_removal_detected"]), str(r4["genuine_removal_detected"])))
    print(fmt.format("  Removal Grace Latency", f"{r1['grace_latency_seconds']}s", f"{r2['grace_latency_seconds']}s", f"{r3['grace_latency_seconds']}s", f"{r4['grace_latency_seconds']}s"))
    print(fmt.format("  False REMOVED other items", str(r1["false_removals_of_other_items"]), str(r2["false_removals_of_other_items"]), str(r3["false_removals_of_other_items"]), str(r4["false_removals_of_other_items"])))
    print("=" * 105)


if __name__ == "__main__":
    main()
