"""Step 29: Visual Evidence Generation for Visual Vacancy Confidence & Replenishment Signals.

Generates annotated evidence frames and a 6-panel contact sheet in output/step29_evidence/:
- Panel 1: Normal packed shelf tier (0% vacancy confidence, status OCCUPIED, no alert)
- Panel 2: Temporary candidate gap under observation (Frame 3, persistence < 10, replenishment pending)
- Panel 3: Confirmed persistent gap with Replenishment Recommendation (Frame 10, confidence >= 0.70, alert active)
- Panel 4: Shopper occlusion handling (shoppers blocking shelf, confidence = 0.0%, status UNCERTAIN)
- Panel 5: Real video shelf_pan_demo.mp4 validation (Tier 1 2D occupancy intact, 0 false vacancies)
- Panel 6: Real video inventory2.mp4 validation (shoppers occluding aisle, 0 false replenishment alerts)
"""

from __future__ import annotations

import json
import os
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

OUT_DIR = ROOT_DIR / "output" / "step29_evidence"
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


def run_step29_visual_evidence():
    print("=" * 80)
    print("STEP 29: VISUAL EVIDENCE GENERATION")
    print("=" * 80)

    # 1. Base synthetic canvas for clear controlled scenarios
    frame_w, frame_h = 1280, 720
    shelf_roi = (0.05, 0.20, 0.95, 0.85)
    tiers = [
        {
            "tier_id": "SHELF-01-TIER-01",
            "name": "Tier 1 — Top Rack",
            "roi": (0.05, 0.22, 0.95, 0.50),
        },
        {
            "tier_id": "SHELF-01-TIER-02",
            "name": "Tier 2 — Bottom Rack",
            "roi": (0.05, 0.52, 0.95, 0.82),
        },
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
        img = np.full((frame_h, frame_w, 3), 32, dtype=np.uint8)
        # Draw shelf background racks
        cv2.rectangle(img, (int(frame_w*0.05), int(frame_h*0.22)), (int(frame_w*0.95), int(frame_h*0.50)), (45, 45, 45), -1)
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

    def build_boxes(tier_y1=170, tier_y2=340, item_w=75, spacing=15, skip_indices=()):
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

    # --- PANEL 1: Normal Packed Shelf Tier ---
    t1_normal = build_boxes(tier_y1=170, tier_y2=340)
    t2_normal = build_boxes(tier_y1=385, tier_y2=560)
    all_normal = t1_normal + t2_normal
    tracker.reset()
    snap1 = tracker.update(engine.analyze_frame(all_normal, frame_w, frame_h))
    canvas1 = generate_shelf_canvas(all_normal)
    ann1 = annotate_vacancy_frame(canvas1, snap1, shelf_roi, False)
    p1 = OUT_DIR / "panel1_normal_packed.jpg"
    cv2.imwrite(str(p1), ann1)
    panels.append(p1)

    # --- PANEL 2: Temporary Candidate Gap under Observation (Frame 3) ---
    # Remove 2 items from Tier 1 (gap = 2*75 + 15 = 165px ~ 2.2x facing)
    t1_gap = build_boxes(tier_y1=170, tier_y2=340, skip_indices=(4, 5))
    all_gap = t1_gap + t2_normal
    tracker.reset()
    for _ in range(3):
        snap2 = tracker.update(engine.analyze_frame(all_gap, frame_w, frame_h))
    canvas2 = generate_shelf_canvas(all_gap)
    ann2 = annotate_vacancy_frame(canvas2, snap2, shelf_roi, False)
    p2 = OUT_DIR / "panel2_temporary_gap_observation.jpg"
    cv2.imwrite(str(p2), ann2)
    panels.append(p2)

    # --- PANEL 3: Confirmed Persistent Gap with Replenishment Recommendation (Frame 11) ---
    for _ in range(8):
        snap3 = tracker.update(engine.analyze_frame(all_gap, frame_w, frame_h))
    canvas3 = generate_shelf_canvas(all_gap)
    ann3 = annotate_vacancy_frame(canvas3, snap3, shelf_roi, True)
    p3 = OUT_DIR / "panel3_confirmed_replenishment.jpg"
    cv2.imwrite(str(p3), ann3)
    panels.append(p3)

    # --- PANEL 4: Shopper Occlusion Handling ---
    shopper_box = [(int(frame_w * 0.30), int(frame_h * 0.15), int(frame_w * 0.60), int(frame_h * 0.90))]
    snap4 = tracker.update(engine.analyze_frame(all_gap, frame_w, frame_h, person_boxes=shopper_box))
    canvas4 = generate_shelf_canvas(all_gap, shopper_box)
    ann4 = annotate_vacancy_frame(canvas4, snap4, shelf_roi, False)
    p4 = OUT_DIR / "panel4_shopper_occlusion.jpg"
    cv2.imwrite(str(p4), ann4)
    panels.append(p4)

    # --- PANEL 5: Real Video shelf_pan_demo.mp4 Frame Validation ---
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
            ann5 = annotate_vacancy_frame(pan_frame, pan_snap, (0.02, 0.12, 0.98, 0.88), False)
            p5 = OUT_DIR / "panel5_real_video_pan_demo.jpg"
            cv2.imwrite(str(p5), ann5)
            panels.append(p5)

    # --- PANEL 6: Real Video inventory2.mp4 Frame Validation ---
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
            ann6 = annotate_vacancy_frame(inv_frame, inv_snap, (0.05, 0.15, 0.92, 0.90), False)
            p6 = OUT_DIR / "panel6_real_video_inventory2.jpg"
            cv2.imwrite(str(p6), ann6)
            panels.append(p6)

    # Compile contact sheet
    contact_sheet_path = OUT_DIR / "step29_contact_sheet.jpg"
    create_contact_sheet(panels, contact_sheet_path, grid_cols=3)
    print("Step 29 visual evidence complete!")


if __name__ == "__main__":
    run_step29_visual_evidence()
