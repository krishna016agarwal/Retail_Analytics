"""Step 27: Real-Video Vacancy Validation, Gap Investigation & Threshold Calibration.

Analyzes real frames from:
1. inventory_data/demo_videos/shelf_pan_demo.mp4
2. videos/inventory2.mp4

Performs:
- Full frame-by-frame vacancy extraction and gap logging
- In-depth investigation of the ~618px / ~4.5x median gap in shelf_pan_demo.mp4
- Generation of color-coded visual evidence frames (GREEN=product, YELLOW=temp vacancy, RED=confirmed vacancy, ORANGE=occluded)
- Creation of a multi-panel visual contact sheet for easy inspection
- Threshold sensitivity analysis (1.50x, 1.60x, 1.75x, 1.90x, 2.00x, 2.25x)
"""

import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Tuple

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
    ShelfVacancySnapshot,
    VacantRegion,
)

EVIDENCE_DIR = ROOT_DIR / "output" / "step27_evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


def draw_color_coded_evidence(
    frame: np.ndarray,
    snapshot: ShelfVacancySnapshot,
    roi_pct: Tuple[float, float, float, float],
    is_confirmed: bool,
    frame_idx: int,
    video_name: str,
    flow_mag: float,
) -> np.ndarray:
    """Renders annotated evidence frame with precise Step 27 color coding:
    - GREEN: Normal product bounding boxes
    - YELLOW: Temporary vacancy candidate (< 10 frames)
    - RED: Confirmed vacancy (>= 10 frames)
    - ORANGE: Uncertain / Shopper-occluded gap
    - CYAN: Shelf ROI boundary
    """
    canvas = frame.copy()
    fh, fw = canvas.shape[:2]

    # Draw Shelf ROI
    rx1, ry1, rx2, ry2 = [
        int(roi_pct[0] * fw),
        int(roi_pct[1] * fh),
        int(roi_pct[2] * fw),
        int(roi_pct[3] * fh),
    ]
    cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (255, 200, 0), 3)
    cv2.putText(
        canvas,
        f"SHELF-01 ROI [{rx1},{ry1} - {rx2},{ry2}]",
        (rx1 + 10, ry1 + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 200, 0),
        2,
        cv2.LINE_AA,
    )

    # 1. Draw Products in GREEN
    for row in snapshot.rows:
        for p in row.products:
            bx1, by1, bx2, by2 = p["bbox"]
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 220, 0), 2)

    # 2. Draw Gaps
    overlay = canvas.copy()
    for row in snapshot.rows:
        for gap in row.gaps:
            gx1, gy1, gx2, gy2 = gap.x1, gap.y1, gap.x2, gap.y2

            if gap.is_occluded_by_person or snapshot.is_occluded:
                # ORANGE: Shopper occluded / uncertain
                color = (0, 140, 255)
                tag = "OCCLUDED / UNCERTAIN"
            elif is_confirmed:
                # RED: Confirmed vacancy
                color = (0, 0, 240)
                tag = "CONFIRMED VACANCY"
            else:
                # YELLOW: Temporary candidate
                color = (0, 230, 255)
                tag = "TEMP VACANCY"

            # Fill translucent rectangle
            cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), color, -1)
            cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), color, 3)

            # Label gap
            label_text = f"{tag}: {gap.width_px:.0f}px ({gap.width_multiple:.1f}x med)"
            cv2.putText(
                canvas,
                label_text,
                (gx1 + 5, gy1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

    # Blend translucent fill (30% opacity)
    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    # Telemetry Banner at top
    banner_h = 90
    cv2.rectangle(canvas, (0, 0), (fw, banner_h), (20, 20, 20), -1)

    state_color = (
        (0, 0, 240)
        if is_confirmed
        else (0, 230, 255)
        if snapshot.temporal_state == "TEMPORARY_VACANCY"
        else (0, 140, 255)
        if snapshot.is_occluded
        else (0, 220, 0)
    )

    t1 = f"[{video_name}] FRAME {frame_idx:03d} | SHELF-01: {snapshot.status} ({snapshot.temporal_state})"
    t2 = (
        f"Products: {snapshot.detected_facings_count} | Visible Occupancy: {snapshot.occupancy_pct:.1f}% | "
        f"Gaps: {len(snapshot.unoccluded_vacant_regions)} unoccluded, {len(snapshot.vacant_regions) - len(snapshot.unoccluded_vacant_regions)} occluded | "
        f"Motion: {flow_mag:.2f}px/f"
    )
    cv2.putText(canvas, t1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, t2, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, cv2.LINE_AA)

    if is_confirmed:
        alert_box_w = 520
        cv2.rectangle(canvas, (fw - alert_box_w - 20, 15), (fw - 20, 75), (0, 0, 200), -1)
        cv2.putText(
            canvas,
            "SHELF 1 - EMPTY SPACE DETECTED",
            (fw - alert_box_w, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas


def create_contact_sheet(
    image_paths: List[pathlib.Path],
    output_path: pathlib.Path,
    grid_cols: int = 3,
    thumb_w: int = 640,
    thumb_h: int = 480,
) -> None:
    """Creates a consolidated multi-panel contact sheet of key evidence frames."""
    if not image_paths:
        return

    images = []
    for p in image_paths:
        if p.is_file():
            img = cv2.imread(str(p))
            if img is not None:
                resized = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                images.append(resized)

    if not images:
        return

    num_images = len(images)
    grid_rows = int(np.ceil(num_images / grid_cols))
    sheet_h = grid_rows * thumb_h
    sheet_w = grid_cols * thumb_w

    contact_sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        r = idx // grid_cols
        c = idx % grid_cols
        y1 = r * thumb_h
        y2 = y1 + thumb_h
        x1 = c * thumb_w
        x2 = x1 + thumb_w
        contact_sheet[y1:y2, x1:x2] = img

    cv2.imwrite(str(output_path), contact_sheet)
    print(f"Contact sheet saved: {output_path.name} ({sheet_w}x{sheet_h})")


def analyze_video_stream(
    video_path: pathlib.Path,
    roi_pct: Tuple[float, float, float, float],
    video_name: str,
    model: YOLO,
    person_model: YOLO,
    min_gap_multiplier: float = 1.75,
    min_abs_gap: int = 45,
    temporal_confirm_frames: int = 10,
    temporal_recovery_frames: int = 5,
    max_frames: int = 75,
) -> Tuple[List[Dict[str, Any]], List[pathlib.Path], Dict[str, Any]]:
    """Runs frame-by-frame analysis, records all gaps, outputs evidence images."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    engine = ShelfVacancyEngine(
        shelf_id="SHELF-01",
        roi=roi_pct,
        min_gap_multiplier=min_gap_multiplier,
        min_absolute_gap_px=min_abs_gap,
    )
    tracker = ShelfVacancyTracker(
        min_consecutive_frames=temporal_confirm_frames,
        recovery_frames=temporal_recovery_frames,
    )

    gap_records: List[Dict[str, Any]] = []
    saved_evidence_frames: List[pathlib.Path] = []
    frame_metrics: List[Dict[str, Any]] = []

    f_idx = 0
    prev_gray = None

    # Step through frames
    while f_idx < min(max_frames, total_video_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Optical flow motion
        small_gray = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY)
        flow_mag = 0.0
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, small_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_mag = float(np.mean(mag))
        prev_gray = small_gray

        # Product detection
        p_res = model.predict(frame, conf=0.30, imgsz=640, verbose=False)[0]
        prod_boxes = [tuple(map(int, b)) for b in p_res.boxes.xyxy.cpu().numpy()]

        # Person detection
        h_res = person_model.predict(frame, classes=[0], conf=0.35, imgsz=640, verbose=False)[0]
        person_boxes = [tuple(map(int, b)) for b in h_res.boxes.xyxy.cpu().numpy()]

        # Vacancy Engine snapshot
        snapshot = engine.analyze_frame(
            product_boxes=prod_boxes,
            frame_w=w,
            frame_h=h,
            person_boxes=person_boxes,
            camera_motion_mag=flow_mag,
        )

        # Temporal state update
        snapshot = tracker.update(snapshot)

        # Record every detected gap
        timestamp_sec = round(f_idx / fps, 3)
        for row in snapshot.rows:
            for gap in row.gaps:
                gap_info = {
                    "video": video_name,
                    "frame": f_idx,
                    "timestamp_s": timestamp_sec,
                    "shelf_id": "SHELF-01",
                    "row_id": gap.row_id,
                    "region_id": gap.region_id,
                    "gap_x1": gap.x1,
                    "gap_y1": gap.y1,
                    "gap_x2": gap.x2,
                    "gap_y2": gap.y2,
                    "gap_width_px": gap.width_px,
                    "row_median_w": gap.width_px / gap.width_multiple if gap.width_multiple > 0 else 0,
                    "gap_ratio": gap.width_multiple,
                    "is_occluded_by_person": gap.is_occluded_by_person,
                    "camera_motion_mag": round(flow_mag, 2),
                    "is_camera_moving": snapshot.is_camera_moving,
                    "temporal_state": snapshot.temporal_state,
                    "alert_generated": tracker.is_confirmed,
                }
                gap_records.append(gap_info)

        # Save key representative frames:
        # 1. First frame (0)
        # 2. When temporary vacancy first appears
        # 3. When vacancy becomes confirmed
        # 4. When shopper occludes
        # 5. Middle and end frames
        save_this_frame = (
            f_idx in [0, 5, 10, 15, 25, 40, 60, total_video_frames - 1]
            or (snapshot.temporal_state == "TEMPORARY_VACANCY" and f_idx <= 2)
            or (snapshot.temporal_state == "VACANCY_CONFIRMED" and f_idx == 10)
            or (snapshot.is_occluded and f_idx % 20 == 0)
        )

        if save_this_frame:
            evidence_img = draw_color_coded_evidence(
                frame=frame,
                snapshot=snapshot,
                roi_pct=roi_pct,
                is_confirmed=tracker.is_confirmed,
                frame_idx=f_idx,
                video_name=video_name,
                flow_mag=flow_mag,
            )
            out_img_name = f"{video_name}_f{f_idx:03d}_{snapshot.temporal_state}.jpg"
            out_img_path = EVIDENCE_DIR / out_img_name
            cv2.imwrite(str(out_img_path), evidence_img)
            saved_evidence_frames.append(out_img_path)

        frame_metrics.append(
            {
                "frame": f_idx,
                "products": snapshot.detected_facings_count,
                "occupancy_pct": snapshot.occupancy_pct,
                "state": snapshot.temporal_state,
                "unoccluded_gaps": len(snapshot.unoccluded_vacant_regions),
                "is_occluded": snapshot.is_occluded,
            }
        )

        f_idx += 1

    cap.release()

    summary = {
        "video": video_name,
        "processed_frames": f_idx,
        "total_gaps_logged": len(gap_records),
        "final_state": tracker.current_state.value,
        "confirmed_alert": tracker.is_confirmed,
        "unoccluded_gap_frames": sum(1 for m in frame_metrics if m["unoccluded_gaps"] > 0),
        "occluded_frames": sum(1 for m in frame_metrics if m["is_occluded"]),
    }

    return gap_records, saved_evidence_frames, summary


def evaluate_threshold_sweep(
    video_path: pathlib.Path,
    roi_pct: Tuple[float, float, float, float],
    model: YOLO,
    person_model: YOLO,
    multipliers: List[float],
    frames_to_test: int = 40,
) -> Dict[float, Dict[str, Any]]:
    """Evaluates gap detection behavior across different min_gap_multipliers."""
    results = {}

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < frames_to_test:
        ret, f = cap.read()
        if not ret or f is None:
            break
        frames.append(f)
    cap.release()

    if not frames:
        return {}

    fh, fw = frames[0].shape[:2]

    # Pre-detect products and persons for all frames
    cached_dets = []
    for frame in frames:
        p_res = model.predict(frame, conf=0.30, imgsz=640, verbose=False)[0]
        p_boxes = [tuple(map(int, b)) for b in p_res.boxes.xyxy.cpu().numpy()]

        h_res = person_model.predict(frame, classes=[0], conf=0.35, imgsz=640, verbose=False)[0]
        person_boxes = [tuple(map(int, b)) for b in h_res.boxes.xyxy.cpu().numpy()]
        cached_dets.append((p_boxes, person_boxes))

    for mult in multipliers:
        engine = ShelfVacancyEngine(
            shelf_id="SHELF-01",
            roi=roi_pct,
            min_gap_multiplier=mult,
            min_absolute_gap_px=45,
        )
        tracker = ShelfVacancyTracker(min_consecutive_frames=10, recovery_frames=5)

        total_gaps = 0
        confirmed_count = 0
        gap_widths = []

        for idx, (p_boxes, pers_boxes) in enumerate(cached_dets):
            snap = engine.analyze_frame(
                product_boxes=p_boxes,
                frame_w=fw,
                frame_h=fh,
                person_boxes=pers_boxes,
                camera_motion_mag=0.0,
            )
            snap = tracker.update(snap)
            total_gaps += len(snap.unoccluded_vacant_regions)
            if tracker.is_confirmed:
                confirmed_count += 1
            for g in snap.unoccluded_vacant_regions:
                gap_widths.append(g.width_px)

        results[mult] = {
            "multiplier": mult,
            "total_unoccluded_gaps": total_gaps,
            "frames_with_alert": confirmed_count,
            "avg_gap_width_px": round(float(np.mean(gap_widths)), 1) if gap_widths else 0.0,
            "max_gap_width_px": round(float(np.max(gap_widths)), 1) if gap_widths else 0.0,
            "min_gap_width_px": round(float(np.min(gap_widths)), 1) if gap_widths else 0.0,
        }

    return results


def main():
    print("=" * 80)
    print("STEP 27: REAL-VIDEO VACANCY VALIDATION & THRESHOLD CALIBRATION")
    print("=" * 80)

    model_path = ROOT_DIR / "inventory_data" / "custom_model" / "retail_detector_exp2.pt"
    person_model_path = ROOT_DIR / "yolo11n.pt"
    pan_video_path = ROOT_DIR / "inventory_data" / "demo_videos" / "shelf_pan_demo.mp4"
    inv_video_path = ROOT_DIR / "videos" / "inventory2.mp4"

    model = YOLO(str(model_path))
    person_model = YOLO(str(person_model_path))

    pan_roi = (0.02, 0.15, 0.98, 0.85)
    inv_roi = (0.05, 0.25, 0.95, 0.75)

    # 1. Analyze shelf_pan_demo.mp4
    print("\n--- 1. Analyzing shelf_pan_demo.mp4 ---")
    pan_gaps, pan_evidence, pan_summary = analyze_video_stream(
        video_path=pan_video_path,
        roi_pct=pan_roi,
        video_name="shelf_pan_demo",
        model=model,
        person_model=person_model,
        max_frames=75,
    )
    print(f"shelf_pan_demo: Processed {pan_summary['processed_frames']} frames, {pan_summary['total_gaps_logged']} gaps logged.")
    print(f"Final State: {pan_summary['final_state']}, Confirmed: {pan_summary['confirmed_alert']}")

    # 2. Analyze inventory2.mp4
    print("\n--- 2. Analyzing inventory2.mp4 ---")
    inv_gaps, inv_evidence, inv_summary = analyze_video_stream(
        video_path=inv_video_path,
        roi_pct=inv_roi,
        video_name="inventory2",
        model=model,
        person_model=person_model,
        max_frames=75,
    )
    print(f"inventory2: Processed {inv_summary['processed_frames']} frames, {inv_summary['total_gaps_logged']} gaps logged.")
    print(f"Final State: {inv_summary['final_state']}, Occluded Frames: {inv_summary['occluded_frames']}")

    # 3. Create Contact Sheets
    print("\n--- 3. Generating Contact Sheets ---")
    all_evidence = pan_evidence + inv_evidence
    create_contact_sheet(
        image_paths=all_evidence[:12],
        output_path=EVIDENCE_DIR / "step27_contact_sheet.jpg",
        grid_cols=3,
        thumb_w=640,
        thumb_h=480,
    )

    # 4. Specific Investigation of the ~618px gap in shelf_pan_demo.mp4
    print("\n--- 4. Detailed Investigation of Gaps in shelf_pan_demo.mp4 ---")
    large_gaps = [g for g in pan_gaps if g["gap_width_px"] >= 300]
    print(f"Found {len(large_gaps)} gap instances with width >= 300px in shelf_pan_demo:")
    sample_gaps = large_gaps[::10] if len(large_gaps) > 10 else large_gaps
    for g in sample_gaps:
        print(
            f"  Frame {g['frame']:02d} ({g['timestamp_s']:.2f}s) | Row {g['row_id']} | "
            f"Span [{g['gap_x1']:.0f} - {g['gap_x2']:.0f}] | Width: {g['gap_width_px']:.1f}px "
            f"({g['gap_ratio']:.2f}x median) | State: {g['temporal_state']}"
        )

    # 5. Threshold Sensitivity Sweep
    print("\n--- 5. Threshold Sensitivity Sweep (1.50x to 2.25x) ---")
    multipliers = [1.50, 1.60, 1.75, 1.90, 2.00, 2.25]
    sweep_pan = evaluate_threshold_sweep(
        video_path=pan_video_path,
        roi_pct=pan_roi,
        model=model,
        person_model=person_model,
        multipliers=multipliers,
        frames_to_test=40,
    )
    print("Sweep Results for shelf_pan_demo.mp4:")
    print(f"{'Multiplier':<12} | {'Unoccluded Gaps':<16} | {'Alert Frames':<14} | {'Avg Width (px)':<14} | {'Max Width (px)'}")
    print("-" * 75)
    for m, d in sweep_pan.items():
        print(f"{m:<12.2f} | {d['total_unoccluded_gaps']:<16d} | {d['frames_with_alert']:<14d} | {d['avg_gap_width_px']:<14.1f} | {d['max_gap_width_px']}")

    # Save comprehensive Step 27 JSON dataset
    dataset_path = EVIDENCE_DIR / "step27_gap_dataset.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pan_summary": pan_summary,
                "inv_summary": inv_summary,
                "threshold_sweep_pan": sweep_pan,
                "pan_gap_sample_count": len(pan_gaps),
                "inv_gap_sample_count": len(inv_gaps),
                "pan_gaps_sample": pan_gaps[:50],
                "inv_gaps_sample": inv_gaps[:50],
            },
            f,
            indent=2,
        )
    print(f"\nStep 27 dataset saved to: {dataset_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
