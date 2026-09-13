"""Step 22: Smoke Test for Updated Production ByteTrack Configuration.

Verifies:
1. Tracker loads 'inventory/bytetrack_shelf.yaml' cleanly
2. Detections and track IDs are successfully generated
3. ProductTrackStateTracker updates and temporal snapshots function
4. InventoryEventDetector processes frames and generates events
5. Zero runtime errors
"""

import pathlib
import sys
import cv2

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inventory.config import InventoryModelConfig
from inventory.shelf_detector import ShelfProductDetector
from inventory.shelf_state import ProductTrackStateTracker
from inventory.inventory_events import InventoryEventDetector, InventoryEventConfig, InventoryEventType


def run_smoke_test():
    print("=" * 70)
    print("STEP 22: INVENTORY TRACKER SMOKE TEST")
    print("=" * 70)

    yaml_path = ROOT_DIR / "inventory" / "bytetrack_shelf.yaml"
    video_path = ROOT_DIR / "inventory_data" / "demo_videos" / "shelf_pan_demo.mp4"

    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing tracker config: {yaml_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Missing test video: {video_path}")

    # 1. Tracker Initialization
    print(f"[1/4] Loading ShelfProductDetector with {yaml_path.name}...")
    det_cfg = InventoryModelConfig(
        model_path="inventory_data/custom_model/retail_detector_exp2.pt",
        model_tier="retail_specific",
        device="cpu",
        confidence_threshold=0.30,
        tracker_config_path=str(yaml_path),
    )
    detector = ShelfProductDetector(det_cfg)
    print("      -> Model and ByteTrack successfully initialized.")

    # 2. State & Event Trackers
    print("[2/4] Initializing ProductTrackStateTracker and InventoryEventDetector...")
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
    print("      -> State and Event detectors successfully initialized.")

    # 3. Process video frames
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames_to_test = 30

    print(f"[3/4] Processing {total_frames_to_test} test frames from {video_path.name}...")
    all_tracks = set()
    all_events = []
    frames_processed = 0

    for f_idx in range(total_frames_to_test):
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        batch = detector.detect(
            frame=frame,
            frame_index=f_idx,
            track=True,
            persist=True,
            tracker=str(yaml_path),
        )

        active_tids = [d.track_id for d in batch.product_detections if d.track_id is not None]
        all_tracks.update(active_tids)

        snapshot = tracker.update(batch=batch, frame_index=f_idx, fps=fps)
        events = event_detector.process_frame(
            batch=batch,
            snapshot=snapshot,
            frame_index=f_idx,
            timestamp_sec=f_idx / fps,
            frame_bgr=frame,
        )
        all_events.extend(events)
        frames_processed += 1

    cap.release()

    # 4. Assertions & Verification
    print(f"[4/4] Verifying outputs...")
    print(f"      - Frames processed: {frames_processed}/{total_frames_to_test}")
    print(f"      - Total unique tracks identified: {len(all_tracks)}")
    print(f"      - Stable facings at frame {frames_processed-1}: {snapshot.stable_facings_count}")
    print(f"      - Total inventory events emitted: {len(all_events)}")
    
    event_counts = {}
    for ev in all_events:
        event_counts[ev.event_type.value] = event_counts.get(ev.event_type.value, 0) + 1
    for etype, count in event_counts.items():
        print(f"        * {etype}: {count}")

    assert frames_processed == total_frames_to_test, "Did not process expected number of frames"
    assert len(all_tracks) > 0, "No track IDs were assigned by ByteTrack"
    assert snapshot.stable_facings_count > 0, "No facings reached STABLE state"
    assert len(all_events) > 0, "No inventory events were emitted"

    print("\n[SUCCESS] Smoke test PASSED with 0 errors!")
    return True


if __name__ == "__main__":
    run_smoke_test()
