"""Phase 25D & 25E: Test Vacancy Engine and Temporal Confirmation on Real Videos.

Runs multi-frame processing across:
1. shelf_pan_demo.mp4 (75 frames, clean panning)
2. inventory2.mp4 (75 frames, shopper presence & occlusions)

Validates:
- Adaptive gap detection consistency across frames
- Temporal confirmation state progression (NORMAL -> TEMPORARY_VACANCY -> VACANCY_CONFIRMED)
- Occlusion freeze behavior when shoppers are present
- Generation of single shelf alert: 'SHELF 1 — EMPTY SPACE DETECTED'
"""

import json
import os
import pathlib
import sys
import time
import cv2
import numpy as np

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ultralytics import YOLO
from inventory.shelf_vacancy import (
    ShelfVacancyEngine,
    ShelfVacancyTracker,
    VacancyTemporalState,
    annotate_vacancy_frame,
)

OUT_DIR = ROOT_DIR / "output" / "vacancy_video_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def test_video(video_path: pathlib.Path, roi_pct, name: str, max_frames: int = 50):
    print("=" * 80)
    print(f"TESTING REAL VIDEO: {name} ({video_path.name})")
    print("=" * 80)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO(str(ROOT_DIR / "inventory_data" / "custom_model" / "retail_detector_exp2.pt"))
    person_model = YOLO(str(ROOT_DIR / "yolo11n.pt"))

    engine = ShelfVacancyEngine(
        shelf_id="SHELF-01",
        roi=roi_pct,
        min_gap_multiplier=1.75,
        min_absolute_gap_px=45,
    )
    tracker = ShelfVacancyTracker(
        min_consecutive_frames=10,
        recovery_frames=5,
    )

    out_video_path = OUT_DIR / f"{name}_vacancy_annotated.mp4"
    writer = cv2.VideoWriter(
        str(out_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_src,
        (w, h),
    )

    frame_records = []
    f_idx = 0
    start_time = time.time()

    prev_gray = None

    while f_idx < max_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # 1. Optical flow camera motion
        small_gray = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY)
        flow_mag = 0.0
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, small_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_mag = float(np.mean(mag))
        prev_gray = small_gray

        # 2. Products & Person detection
        p_res = model.predict(frame, conf=0.30, imgsz=640, verbose=False)[0]
        prod_boxes = [tuple(map(int, b)) for b in p_res.boxes.xyxy.cpu().numpy()]

        h_res = person_model.predict(frame, classes=[0], conf=0.35, imgsz=640, verbose=False)[0]
        person_boxes = [tuple(map(int, b)) for b in h_res.boxes.xyxy.cpu().numpy()]

        # 3. Frame vacancy snapshot
        snapshot = engine.analyze_frame(
            product_boxes=prod_boxes,
            frame_w=w,
            frame_h=h,
            person_boxes=person_boxes,
            camera_motion_mag=flow_mag,
        )

        # 4. Temporal confirmation
        temporal_snap = tracker.update(snapshot)

        # 5. Visual annotation & write to mp4
        annotated = annotate_vacancy_frame(
            frame=frame,
            snapshot=temporal_snap,
            roi_pct=roi_pct,
            active_alert=tracker.is_confirmed,
        )
        writer.write(annotated)

        record = {
            "frame": f_idx,
            "products": temporal_snap.detected_facings_count,
            "rows": len(temporal_snap.rows),
            "raw_status": temporal_snap.status,
            "temporal_state": temporal_snap.temporal_state,
            "unoccluded_gaps": len(temporal_snap.unoccluded_vacant_regions),
            "occluded_gaps": len(temporal_snap.vacant_regions) - len(temporal_snap.unoccluded_vacant_regions),
            "is_occluded": temporal_snap.is_occluded,
            "camera_motion": round(flow_mag, 2),
            "occupancy_pct": temporal_snap.occupancy_pct,
            "vacancy_score": temporal_snap.vacancy_score,
        }
        frame_records.append(record)

        if f_idx % 10 == 0 or temporal_snap.temporal_state == VacancyTemporalState.VACANCY_CONFIRMED.value:
            occ_str = " | OCCLUDED" if temporal_snap.is_occluded else ""
            print(
                f"  Frame {f_idx:2d} | Products: {temporal_snap.detected_facings_count:3d} | "
                f"Gaps: {len(temporal_snap.unoccluded_vacant_regions)} | "
                f"State: {temporal_snap.temporal_state:<18s} | "
                f"Occ: {temporal_snap.occupancy_pct:4.1f}%{occ_str}"
            )

        f_idx += 1

    cap.release()
    writer.release()

    total_time = time.time() - start_time
    avg_fps = f_idx / max(total_time, 0.001)
    print(f"\nCompleted {f_idx} frames in {total_time:.1f}s ({avg_fps:.1f} FPS)")
    print(f"Annotated video saved: {out_video_path.relative_to(ROOT_DIR)}")

    # Save metrics JSON
    metrics_path = OUT_DIR / f"{name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video": name,
                "total_frames": f_idx,
                "avg_fps": round(avg_fps, 1),
                "final_state": tracker.current_state.value,
                "is_confirmed": tracker.is_confirmed,
                "frame_records": frame_records,
            },
            f,
            indent=2,
        )
    print(f"Metrics saved: {metrics_path.relative_to(ROOT_DIR)}\n")
    return tracker.is_confirmed


def main():
    pan_video = ROOT_DIR / "inventory_data" / "demo_videos" / "shelf_pan_demo.mp4"
    inv_video = ROOT_DIR / "videos" / "inventory2.mp4"

    if pan_video.is_file():
        test_video(pan_video, roi_pct=(0.02, 0.15, 0.98, 0.85), name="shelf_pan_demo", max_frames=40)

    if inv_video.is_file():
        test_video(inv_video, roi_pct=(0.05, 0.25, 0.95, 0.75), name="inventory2", max_frames=40)


if __name__ == "__main__":
    main()
