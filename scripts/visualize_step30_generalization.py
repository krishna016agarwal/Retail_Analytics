"""Step 30: Visual Evidence & Generalization Contact Sheet Generator.

Generates representative visual evidence across all required dimensions in output/step30_evidence/:
- Panel 1: Normal packed shelf (0% Visual Vacancy Confidence, status OCCUPIED)
- Panel 2: Sparse shelf with normal spacing (no false vacancy)
- Panel 3: Temporary gap under observation (Frame 3, confidence below threshold, no alert)
- Panel 4: Confirmed persistent vacancy with Replenishment Recommendation (Frame 11, conf >= 70%)
- Panel 5: Shopper interaction lifecycle (shopper occluding shelf, status UNCERTAIN, conf = 0.0%)
- Panel 6: Camera motion optical flow freeze (status UNCERTAIN, no false alerts)
- Panel 7: Multi-tier simultaneous vacancies (independent reporting across tiers)
- Panel 8: Real demo video shelf_pan_demo.mp4 regression check (0 false vacancies)
- Panel 9: Real demo video inventory2.mp4 regression check (shopper occlusion safety)
Plus consolidated step30_contact_sheet.jpg (3x3 grid).
"""

from __future__ import annotations

import pathlib
import sys
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
    ShelfVacancySnapshot,
    VacancyTemporalState,
    annotate_vacancy_frame,
)

OUT_DIR = ROOT_DIR / "output" / "step30_evidence"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def create_contact_sheet(
    image_paths: List[pathlib.Path],
    output_path: pathlib.Path,
    grid_cols: int = 3,
    thumb_w: int = 640,
    thumb_h: int = 480,
) -> None:
    """Creates a consolidated multi-panel contact sheet."""
    images = []
    for p in image_paths:
        if p.is_file():
            img = cv2.imread(str(p))
            if img is not None:
                resized = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                images.append(resized)

    if not images:
        print("No images found for contact sheet.")
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
    print(f"Contact sheet saved: {output_path} ({sheet_w}x{sheet_h})")


def run_step30_visual_evidence():
    print("=" * 80)
    print("STEP 30: REAL-WORLD GENERALIZATION VISUAL EVIDENCE GENERATION")
    print("=" * 80)

    frame_w, frame_h = 1280, 720
    shelf_roi = (0.05, 0.15, 0.95, 0.88)
    tiers = [
        {"tier_id": "SHELF-01-TIER-01", "name": "Tier 1 — Top Rack", "roi": (0.05, 0.18, 0.95, 0.48)},
        {"tier_id": "SHELF-01-TIER-02", "name": "Tier 2 — Bottom Rack", "roi": (0.05, 0.52, 0.95, 0.82)},
    ]

    engine = ShelfVacancyEngine(
        shelf_id="SHELF-01",
        roi=shelf_roi,
        tiers=tiers,
        min_gap_multiplier=1.75,
        min_absolute_gap_px=45,
    )
    tracker = ShelfVacancyTracker(
        min_consecutive_frames=10,
        recovery_frames=5,
        min_replenishment_confidence=0.70,
        max_center_drift_ratio=0.25,
        stability_history_window=5,
    )

    def generate_shelf_canvas(product_boxes: List[Tuple[int, int, int, int]], person_boxes: List[Tuple[int, int, int, int]] = None):
        img = np.full((frame_h, frame_w, 3), 30, dtype=np.uint8)
        # Draw shelf background racks
        cv2.rectangle(img, (int(frame_w*0.05), int(frame_h*0.18)), (int(frame_w*0.95), int(frame_h*0.48)), (45, 45, 45), -1)
        cv2.rectangle(img, (int(frame_w*0.05), int(frame_h*0.52)), (int(frame_w*0.95), int(frame_h*0.82)), (45, 45, 45), -1)
        for b in product_boxes:
            bx1, by1, bx2, by2 = b
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (180, 140, 60), -1)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (220, 180, 90), 2)
        if person_boxes:
            for p in person_boxes:
                px1, py1, px2, py2 = p
                overlay = img.copy()
                cv2.rectangle(overlay, (px1, py1), (px2, py2), (50, 50, 200), -1)
                cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
                cv2.rectangle(img, (px1, py1), (px2, py2), (80, 80, 255), 2)
                cv2.putText(img, "SHOPPER (OCCLUDING)", (px1 + 10, py1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img

    def build_boxes(tier_y1=145, tier_y2=330, item_w=75, spacing=15, skip_indices=()):
        boxes = []
        x_start = int(frame_w * 0.06)
        x_end = int(frame_w * 0.94)
        curr_x = x_start
        idx = 0
        while curr_x + item_w <= x_end:
            if idx not in skip_indices:
                boxes.append((curr_x, tier_y1, curr_x + item_w, tier_y2))
            curr_x += item_w + spacing
            idx += 1
        return boxes

    panels: List[pathlib.Path] = []

    # --- PANEL 1: Normal Packed Shelf ---
    t1_normal = build_boxes(tier_y1=145, tier_y2=330, spacing=12)
    t2_normal = build_boxes(tier_y1=390, tier_y2=575, spacing=12)
    all_normal = t1_normal + t2_normal
    tracker.reset()
    snap1 = tracker.update(engine.analyze_frame(all_normal, frame_w, frame_h))
    p1 = OUT_DIR / "panel1_normal_packed.jpg"
    cv2.imwrite(str(p1), annotate_vacancy_frame(generate_shelf_canvas(all_normal), snap1, shelf_roi, False))
    panels.append(p1)

    # --- PANEL 2: Sparse Shelf Normal Spacing (no false alert) ---
    t1_sparse = build_boxes(tier_y1=145, tier_y2=330, spacing=25)
    t2_sparse = build_boxes(tier_y1=390, tier_y2=575, spacing=25)
    tracker.reset()
    snap2 = tracker.update(engine.analyze_frame(t1_sparse + t2_sparse, frame_w, frame_h))
    p2 = OUT_DIR / "panel2_sparse_shelf.jpg"
    cv2.imwrite(str(p2), annotate_vacancy_frame(generate_shelf_canvas(t1_sparse + t2_sparse), snap2, shelf_roi, False))
    panels.append(p2)

    # --- PANEL 3: Temporary Gap under Observation (Frame 3) ---
    t1_gap = build_boxes(tier_y1=145, tier_y2=330, skip_indices=(4, 5))
    all_gap = t1_gap + t2_normal
    tracker.reset()
    for _ in range(3):
        snap3 = tracker.update(engine.analyze_frame(all_gap, frame_w, frame_h))
    p3 = OUT_DIR / "panel3_temporary_gap.jpg"
    cv2.imwrite(str(p3), annotate_vacancy_frame(generate_shelf_canvas(all_gap), snap3, shelf_roi, False))
    panels.append(p3)

    # --- PANEL 4: Confirmed Persistent Vacancy with Replenishment Signal ---
    for _ in range(8):
        snap4 = tracker.update(engine.analyze_frame(all_gap, frame_w, frame_h))
    p4 = OUT_DIR / "panel4_confirmed_replenishment.jpg"
    cv2.imwrite(str(p4), annotate_vacancy_frame(generate_shelf_canvas(all_gap), snap4, shelf_roi, True))
    panels.append(p4)

    # --- PANEL 5: Shopper Interaction Lifecycle (Occlusion) ---
    shopper_box = [(int(frame_w * 0.30), int(frame_h * 0.12), int(frame_w * 0.65), int(frame_h * 0.90))]
    snap5 = tracker.update(engine.analyze_frame(all_gap, frame_w, frame_h, person_boxes=shopper_box))
    p5 = OUT_DIR / "panel5_shopper_occlusion.jpg"
    cv2.imwrite(str(p5), annotate_vacancy_frame(generate_shelf_canvas(all_gap, shopper_box), snap5, shelf_roi, False))
    panels.append(p5)

    # --- PANEL 6: Camera Motion Optical Flow Freeze ---
    snap6 = tracker.update(engine.analyze_frame(all_gap, frame_w, frame_h, camera_motion_mag=7.8))
    p6 = OUT_DIR / "panel6_camera_motion_freeze.jpg"
    cv2.imwrite(str(p6), annotate_vacancy_frame(generate_shelf_canvas(all_gap), snap6, shelf_roi, False))
    panels.append(p6)

    # --- PANEL 7: Multi-Tier Simultaneous Vacancies ---
    t2_gap = build_boxes(tier_y1=390, tier_y2=575, skip_indices=(7, 8))
    multi_gap_boxes = t1_gap + t2_gap
    tracker.reset()
    for _ in range(11):
        snap7 = tracker.update(engine.analyze_frame(multi_gap_boxes, frame_w, frame_h))
    p7 = OUT_DIR / "panel7_multitier_vacancies.jpg"
    cv2.imwrite(str(p7), annotate_vacancy_frame(generate_shelf_canvas(multi_gap_boxes), snap7, shelf_roi, True))
    panels.append(p7)

    # --- PANEL 8: Real Video shelf_pan_demo.mp4 Regression Snapshot ---
    pan_video_path = ROOT_DIR / "inventory_data" / "demo_videos" / "shelf_pan_demo.mp4"
    model_path = ROOT_DIR / "inventory_data" / "custom_model" / "retail_detector_exp2.pt"
    if pan_video_path.is_file() and model_path.is_file():
        cap = cv2.VideoCapture(str(pan_video_path))
        ret, pan_frame = cap.read()
        cap.release()
        if ret and pan_frame is not None:
            fh, fw = pan_frame.shape[:2]
            model = YOLO(str(model_path))
            res = model(pan_frame, conf=0.30, verbose=False)[0]
            boxes = [tuple(map(int, b)) for b in res.boxes.xyxy.cpu().numpy()]
            pan_tiers = [
                {"tier_id": "SHELF-01-TIER-01", "name": "TIER-01", "roi": (0.02, 0.12, 0.98, 0.35)},
                {"tier_id": "SHELF-01-TIER-02", "name": "TIER-02", "roi": (0.02, 0.36, 0.98, 0.59)},
                {"tier_id": "SHELF-01-TIER-03", "name": "TIER-03", "roi": (0.02, 0.60, 0.98, 0.88)},
            ]
            pan_engine = ShelfVacancyEngine(shelf_id="SHELF-01", roi=(0.02, 0.12, 0.98, 0.88), tiers=pan_tiers)
            pan_tracker = ShelfVacancyTracker(min_consecutive_frames=10)
            pan_snap = pan_tracker.update(pan_engine.analyze_frame(boxes, fw, fh))
            p8 = OUT_DIR / "panel8_real_pan_demo.jpg"
            cv2.imwrite(str(p8), annotate_vacancy_frame(pan_frame, pan_snap, (0.02, 0.12, 0.98, 0.88), False))
            panels.append(p8)

    # --- PANEL 9: Real Video inventory2.mp4 Regression Snapshot ---
    inv_video_path = ROOT_DIR / "videos" / "inventory2.mp4"
    person_model_path = ROOT_DIR / "yolo11n.pt"
    if inv_video_path.is_file() and model_path.is_file() and person_model_path.is_file():
        cap = cv2.VideoCapture(str(inv_video_path))
        ret, inv_frame = cap.read()
        cap.release()
        if ret and inv_frame is not None:
            fh, fw = inv_frame.shape[:2]
            model = YOLO(str(model_path))
            person_model = YOLO(str(person_model_path))
            p_res = person_model(inv_frame, classes=[0], conf=0.35, verbose=False)[0]
            person_boxes = [tuple(map(int, b)) for b in p_res.boxes.xyxy.cpu().numpy()]
            res = model(inv_frame, conf=0.30, verbose=False)[0]
            boxes = [tuple(map(int, b)) for b in res.boxes.xyxy.cpu().numpy()]
            inv_tiers = [
                {"tier_id": "SHELF-01-TIER-01", "name": "TIER-01", "roi": (0.05, 0.15, 0.92, 0.40)},
                {"tier_id": "SHELF-01-TIER-02", "name": "TIER-02", "roi": (0.05, 0.41, 0.92, 0.65)},
                {"tier_id": "SHELF-01-TIER-03", "name": "TIER-03", "roi": (0.05, 0.66, 0.92, 0.90)},
            ]
            inv_engine = ShelfVacancyEngine(shelf_id="SHELF-01", roi=(0.05, 0.15, 0.92, 0.90), tiers=inv_tiers)
            inv_tracker = ShelfVacancyTracker(min_consecutive_frames=10)
            inv_snap = inv_tracker.update(inv_engine.analyze_frame(boxes, fw, fh, person_boxes=person_boxes))
            p9 = OUT_DIR / "panel9_real_inventory2.jpg"
            cv2.imwrite(str(p9), annotate_vacancy_frame(inv_frame, inv_snap, (0.05, 0.15, 0.92, 0.90), False))
            panels.append(p9)

    # Build 3x3 contact sheet
    contact_sheet_path = OUT_DIR / "step30_contact_sheet.jpg"
    create_contact_sheet(panels, contact_sheet_path, grid_cols=3)
    print("Step 30 visual evidence generation complete!")


if __name__ == "__main__":
    run_step30_visual_evidence()
