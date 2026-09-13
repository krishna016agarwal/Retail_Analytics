"""Step 16 & 17: Occlusion & Person Robustness Benchmark.

Compares:
1. Baseline Configuration (Pre-fix: max_misses=5, no person freeze, track_buffer=30)
2. Implemented Fix (Post-fix: ~1.5s grace window, person-occlusion freeze, track_buffer=60, alert suppression)

Measures:
- Products temporarily hidden by shoppers
- False PRODUCT_REMOVED events
- False/duplicate PRODUCT_APPEARED events
- Track continuity rate (%) across occlusion
- Duplicate / ghost tracks spawned
- False POSSIBLE_STOCKOUT alerts
"""

import json
import pathlib
import sys
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inventory.catalog import SKUCatalog
from inventory.config import InventoryModelConfig
from inventory.inventory_aggregator import SKUInventoryAggregator
from inventory.inventory_alerts import InventoryAlertConfig, InventoryAlertDetector
from inventory.inventory_events import InventoryEventConfig, InventoryEventDetector, InventoryEventType
from inventory.shelf_detector import ShelfProductDetector
from inventory.shelf_state import ProductTemporalState, ProductTrackStateTracker
from inventory.sku_recognizer import SpatialColorTextureRecognizer

OUTPUT_DIR = ROOT_DIR / "output" / "occlusion_robustness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_pipeline_eval(
    frames_cache: List[Tuple[int, np.ndarray]],
    mode_name: str,
    tracker_config_yaml: str,
    removal_grace_sec: float,
    max_misses: int,
    occlusion_freeze: bool,
    fps: float,
):
    print(f"\n--- Running evaluation: {mode_name} ---")
    det_cfg = InventoryModelConfig(
        model_path="inventory_data/custom_model/retail_detector_exp2.pt",
        model_tier="retail_specific",
        device="cpu",
        confidence_threshold=0.30,
        tracker_config_path=tracker_config_yaml,
    )
    detector = ShelfProductDetector(det_cfg)

    catalog_path = ROOT_DIR / "inventory_data" / "catalogs" / "demo_store_catalog.json"
    cat = SKUCatalog.load_json(str(catalog_path)) if catalog_path.exists() else None
    recognizer = SpatialColorTextureRecognizer(catalog=cat, match_threshold=0.65) if cat else None
    if recognizer:
        recognizer.build_index()

    aggregator = SKUInventoryAggregator(catalog=cat)
    temporal_tracker = ProductTrackStateTracker(
        min_hits_for_stable=3,
        max_misses_for_removal=max_misses,
        removal_grace_seconds=removal_grace_sec,
        occlusion_freeze_enabled=occlusion_freeze,
        occlusion_bbox_margin_px=25,
    )
    event_detector = InventoryEventDetector(
        config=InventoryEventConfig(
            min_hits_for_stable=3,
            max_misses_for_removal=max_misses if max_misses != 5 else 45,
            removal_grace_seconds=removal_grace_sec,
        ),
        recognizer=recognizer,
    )
    alert_detector = InventoryAlertDetector(
        config=InventoryAlertConfig(stockout_consecutive_frames=10),
        catalog=cat,
    )

    phase_a_stable_tracks: Set[int] = set()
    phase_b_hidden_tracks: Set[int] = set()
    phase_c_tracks_seen: Set[int] = set()
    phase_c_new_tracks: Set[int] = set()

    all_emitted_events: List[Dict] = []
    all_emitted_alerts: List[Dict] = []

    for frame_idx, frame in frames_cache:
        ts = frame_idx / fps

        batch = detector.detect(
            frame=frame,
            frame_index=frame_idx,
            track=True,
            persist=True,
            tracker=tracker_config_yaml,
        )

        snapshot = temporal_tracker.update(batch, frame_index=frame_idx, fps=fps)
        events = event_detector.process_frame(
            batch=batch,
            snapshot=snapshot,
            frame_index=frame_idx,
            timestamp_sec=ts,
            frame_bgr=frame,
        )
        for ev in events:
            all_emitted_events.append(ev.to_dict())

        sku_stats = aggregator.update(
            snapshot=snapshot,
            events=events,
            frame_index=frame_idx,
            timestamp_sec=ts,
        )

        alerts = alert_detector.process_frame(
            sku_stats=sku_stats,
            frame_events=events,
            frame_index=frame_idx,
            timestamp_sec=ts,
            person_present=batch.person_present,
        )
        for al in alerts:
            all_emitted_alerts.append(al.to_dict())

        active_tids = {d.track_id for d in batch.product_detections if d.track_id is not None}
        stable_tids = {r.track_id for r in snapshot.active_tracks if r.state == ProductTemporalState.STABLE}

        if frame_idx <= 415:
            phase_a_stable_tracks.update(stable_tids)
        elif frame_idx <= 475:
            for tid in phase_a_stable_tracks:
                if tid not in active_tids:
                    phase_b_hidden_tracks.add(tid)
        else:
            for tid in active_tids:
                phase_c_tracks_seen.add(tid)
                if tid not in phase_a_stable_tracks:
                    phase_c_new_tracks.add(tid)

    removed_events = [e for e in all_emitted_events if e["event_type"] == InventoryEventType.PRODUCT_REMOVED.value]
    appeared_events = [e for e in all_emitted_events if e["event_type"] == InventoryEventType.PRODUCT_APPEARED.value]
    stockout_alerts = [a for a in all_emitted_alerts if a["alert_type"] == "POSSIBLE_STOCKOUT"]

    retained_tracks = phase_a_stable_tracks.intersection(phase_c_tracks_seen)
    lost_tracks = phase_a_stable_tracks - phase_c_tracks_seen
    continuity_rate = len(retained_tracks) / max(1, len(phase_a_stable_tracks))

    return {
        "mode": mode_name,
        "baseline_stable_tracks": len(phase_a_stable_tracks),
        "products_hidden_by_shopper": len(phase_b_hidden_tracks),
        "false_product_removed_events": len(removed_events),
        "duplicate_appeared_events": max(0, len(appeared_events) - len(phase_a_stable_tracks)),
        "tracks_retained_across_occlusion": len(retained_tracks),
        "tracks_lost": len(lost_tracks),
        "track_continuity_rate_pct": round(continuity_rate * 100.0, 1),
        "duplicate_new_tracks_in_recovery": len(phase_c_new_tracks),
        "false_stockout_alerts": len(stockout_alerts),
        "removed_events_log": removed_events,
    }


def main():
    video_path = ROOT_DIR / "videos" / "store-aisle-detection.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    start_frame = 360
    end_frame = 520
    fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print("=" * 80)
    print("STEP 17: OCCLUSION FIX VERIFICATION BENCHMARK")
    print(f"Loading frames {start_frame} -> {end_frame} into memory cache...")

    frames_cache = []
    f_idx = start_frame
    while f_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frames_cache.append((f_idx, frame))
        f_idx += 1
    cap.release()
    print(f"Cached {len(frames_cache)} video frames.")

    # 1. Evaluate Baseline (Pre-fix)
    res_baseline = run_pipeline_eval(
        frames_cache=frames_cache,
        mode_name="Baseline (Pre-Fix)",
        tracker_config_yaml="bytetrack.yaml",
        removal_grace_sec=0.10,
        max_misses=5,
        occlusion_freeze=False,
        fps=fps,
    )

    # 2. Evaluate Post-Fix (Implemented Occlusion Fixes)
    shelf_tracker_yaml = str(ROOT_DIR / "inventory" / "bytetrack_shelf.yaml")
    res_fixed = run_pipeline_eval(
        frames_cache=frames_cache,
        mode_name="Implemented Fix (Post-Fix)",
        tracker_config_yaml=shelf_tracker_yaml,
        removal_grace_sec=1.5,
        max_misses=None,  # will auto-calibrate to ~1.5s * FPS (~90 frames)
        occlusion_freeze=True,
        fps=fps,
    )

    summary = {
        "video": str(video_path.name),
        "frames_evaluated": len(frames_cache),
        "fps": round(fps, 2),
        "baseline_pre_fix": res_baseline,
        "implemented_post_fix": res_fixed,
    }

    out_file = OUTPUT_DIR / "step17_occlusion_verification_report.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("STEP 17 VERIFICATION SUMMARY: BEFORE vs. AFTER")
    print("=" * 80)
    print(f"{'Metric':<38} | {'Baseline (Pre-Fix)':<18} | {'Implemented (Post-Fix)':<22}")
    print("-" * 84)
    print(f"{'False PRODUCT_REMOVED events':<38} | {res_baseline['false_product_removed_events']:<18} | {res_fixed['false_product_removed_events']:<22}")
    print(f"{'Duplicate PRODUCT_APPEARED events':<38} | {res_baseline['duplicate_appeared_events']:<18} | {res_fixed['duplicate_appeared_events']:<22}")
    print(f"{'Track Continuity Rate':<38} | {res_baseline['track_continuity_rate_pct']}%{'':<13} | {res_fixed['track_continuity_rate_pct']}%{'':<17}")
    print(f"{'Tracks Retained / Total':<38} | {res_baseline['tracks_retained_across_occlusion']}/{res_baseline['baseline_stable_tracks']}{'':<14} | {res_fixed['tracks_retained_across_occlusion']}/{res_fixed['baseline_stable_tracks']}{'':<18}")
    print(f"{'Duplicate Tracks in Recovery':<38} | {res_baseline['duplicate_new_tracks_in_recovery']:<18} | {res_fixed['duplicate_new_tracks_in_recovery']:<22}")
    print(f"{'False Stockout Alerts':<38} | {res_baseline['false_stockout_alerts']:<18} | {res_fixed['false_stockout_alerts']:<22}")
    print("=" * 80)
    print(f"Results saved to: {out_file}\n")


if __name__ == "__main__":
    main()
