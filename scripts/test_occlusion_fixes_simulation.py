"""Simulation script comparing baseline occlusion handling vs. potential fixes.

Strictly isolated simulation. Tests:
1. Baseline: max_misses=5, no person-awareness
2. Config Fix: max_misses=30 (lengthened grace window to match ByteTrack buffer)
3. Person-Aware Gating: freeze miss counter when a person overlaps or is present

Evaluates false PRODUCT_REMOVED, false alerts, track continuity, and duplicate tracks.
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

from inventory.config import InventoryModelConfig
from inventory.inventory_events import InventoryChangeEvent, InventoryEventType
from inventory.shelf_detector import ShelfProductDetector
from inventory.shelf_state import (
    ProductTemporalSnapshot,
    ProductTemporalState,
    ProductTrackRecord,
    ProductTrackStateTracker,
)

OUTPUT_DIR = ROOT_DIR / "output" / "occlusion_robustness"


def bboxes_intersect(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int], margin: int = 20) -> bool:
    """Check if bounding box b1 intersects b2 (with margin)."""
    x1 = max(b1[0] - margin, b2[0])
    y1 = max(b1[1] - margin, b2[1])
    x2 = min(b1[2] + margin, b2[2])
    y2 = min(b1[3] + margin, b2[3])
    return (x2 > x1) and (y2 > y1)


def simulate_policy(
    frames_batches,
    policy_name: str,
    max_misses: int = 5,
    person_aware: bool = False,
):
    """Run tracking and event generation simulation on precomputed batches."""
    tracks: Dict[int, ProductTrackRecord] = {}
    track_states: Dict[int, Dict] = {}
    emitted_removed: List[Dict] = []
    emitted_appeared: List[Dict] = []
    all_observed_tids: Set[int] = set()

    for frame_idx, batch in frames_batches:
        detected_tids = set()
        person_boxes = [p.bbox for p in batch.person_detections]

        # 1. Update detected tracks
        for det in batch.product_detections:
            if det.track_id is None:
                continue
            tid = det.track_id
            detected_tids.add(tid)
            all_observed_tids.add(tid)

            if tid not in tracks:
                rec = ProductTrackRecord(
                    track_id=tid,
                    state=ProductTemporalState.UNCERTAIN,
                    first_seen_frame=frame_idx,
                    last_seen_frame=frame_idx,
                    total_hits=1,
                    consecutive_hits=1,
                    consecutive_misses=0,
                    recent_bboxes=[det.bbox],
                )
                tracks[tid] = rec
                track_states[tid] = {"has_been_stable": False, "last_bbox": det.bbox}
            else:
                rec = tracks[tid]
                rec.total_hits += 1
                rec.consecutive_hits += 1
                rec.consecutive_misses = 0
                rec.last_seen_frame = frame_idx
                rec.recent_bboxes.append(det.bbox)
                track_states[tid]["last_bbox"] = det.bbox

                if rec.consecutive_hits >= 3:
                    rec.state = ProductTemporalState.STABLE
                    if not track_states[tid]["has_been_stable"]:
                        track_states[tid]["has_been_stable"] = True
                        emitted_appeared.append({"frame": frame_idx, "track_id": tid})

        # 2. Update missed tracks
        for tid, rec in list(tracks.items()):
            if tid not in detected_tids:
                # Check if missed track is in the path of a person
                is_occluded_by_person = False
                last_bbox = track_states[tid]["last_bbox"]
                if person_aware and person_boxes and last_bbox:
                    for pb in person_boxes:
                        if bboxes_intersect(last_bbox, pb, margin=35):
                            is_occluded_by_person = True
                            break

                if person_aware and is_occluded_by_person:
                    # Freeze or slow down miss counter during person occlusion
                    rec.state = ProductTemporalState.UNCERTAIN
                    # Do not increment consecutive misses or only slowly
                    continue

                rec.consecutive_misses += 1
                rec.consecutive_hits = 0

                if rec.consecutive_misses > max_misses:
                    if track_states[tid]["has_been_stable"]:
                        emitted_removed.append({
                            "frame": frame_idx,
                            "track_id": tid,
                            "consecutive_misses": rec.consecutive_misses,
                        })
                    del tracks[tid]
                else:
                    rec.state = ProductTemporalState.UNCERTAIN

    return {
        "policy": policy_name,
        "max_misses_setting": max_misses,
        "person_aware": person_aware,
        "false_removed_events": len(emitted_removed),
        "total_appeared_events": len(emitted_appeared),
        "total_unique_tracks_registered": len(all_observed_tids),
    }


def main():
    print("Extracting precomputed detection batches from frames 360 - 520...")
    cap = cv2.VideoCapture(str(ROOT_DIR / "videos" / "store-aisle-detection.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 360)

    det_cfg = InventoryModelConfig(
        model_path="inventory_data/custom_model/retail_detector_exp2.pt",
        model_tier="retail_specific",
        device="cpu",
        confidence_threshold=0.30,
    )
    detector = ShelfProductDetector(det_cfg)

    frames_batches = []
    frame_idx = 360
    while frame_idx <= 520:
        ret, frame = cap.read()
        if not ret:
            break
        batch = detector.detect(frame, frame_index=frame_idx, track=True, persist=True)
        frames_batches.append((frame_idx, batch))
        frame_idx += 1
    cap.release()

    # Test Policy 1: Current Baseline (max_misses=5, person_aware=False)
    p1 = simulate_policy(frames_batches, "Baseline (Current: max_misses=5)", max_misses=5, person_aware=False)

    # Test Policy 2: Lengthened Grace Window (max_misses=30 ~ 1.2 sec, matches ByteTrack buffer)
    p2 = simulate_policy(frames_batches, "Lengthened Grace Window (max_misses=30)", max_misses=30, person_aware=False)

    # Test Policy 3: Person-Aware Gating (freeze misses when person bbox overlaps shelf track)
    p3 = simulate_policy(frames_batches, "Person-Aware Occlusion Gating", max_misses=5, person_aware=True)

    # Test Policy 4: Hybrid (max_misses=30 + Person-Aware)
    p4 = simulate_policy(frames_batches, "Hybrid (max_misses=30 + Person-Aware)", max_misses=30, person_aware=True)

    summary = [p1, p2, p3, p4]
    print("\n" + "=" * 78)
    print("POLICY COMPARISON SUMMARY")
    print("=" * 78)
    print(f"{'Policy':<40} | {'False REMOVED':<14} | {'Appeared Ev':<12} | {'Total Tracks':<12}")
    print("-" * 84)
    for p in summary:
        print(f"{p['policy']:<40} | {p['false_removed_events']:<14} | {p['total_appeared_events']:<12} | {p['total_unique_tracks_registered']:<12}")
    print("=" * 84)

    out_file = OUTPUT_DIR / "policy_comparison_results.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
