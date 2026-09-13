"""Phase 25C: Visualize Detected Shelf Vacancy Regions.

Generates visual proof of generic shelf vacancy detection on real frames:
1. Filters detections to SHELF-01 ROI.
2. Performs product-row grouping.
3. Computes adaptive product geometry and gap detection per row.
4. Highlights vacant gaps in translucent amber/red with labels.
5. Saves annotated evidence images to output/vacancy_inspection/.
"""

import os
import pathlib
import sys
import cv2
import numpy as np

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ultralytics import YOLO
from inventory.shelf_vacancy import ShelfVacancyEngine, annotate_vacancy_frame

OUT_DIR = ROOT_DIR / "output" / "vacancy_inspection"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_visualization_demo():
    print("=" * 80)
    print("PHASE 25C: VISUALIZE DETECTED SHELF VACANCY REGIONS")
    print("=" * 80)

    model_path = ROOT_DIR / "inventory_data" / "custom_model" / "retail_detector_exp2.pt"
    person_model_path = ROOT_DIR / "yolo11n.pt"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model missing: {model_path}")

    model = YOLO(str(model_path))
    p_model = YOLO(str(person_model_path)) if person_model_path.is_file() else None

    test_cases = [
        {
            "name": "shelf_pan_demo",
            "frame_path": ROOT_DIR / "output" / "inspection" / "shelf_pan_demo_frame0.jpg",
            "roi": (0.02, 0.15, 0.98, 0.85),
            "min_gap_mult": 1.75,
            "min_abs_gap": 45,
        },
        {
            "name": "inventory2",
            "frame_path": ROOT_DIR / "output" / "inspection" / "inventory2_frame0.jpg",
            "roi": (0.05, 0.25, 0.95, 0.75),
            "min_gap_mult": 1.75,
            "min_abs_gap": 50,
        },
    ]

    for tc in test_cases:
        img_path = tc["frame_path"]
        if not img_path.is_file():
            print(f"[SKIP] Missing frame {img_path}")
            continue

        frame = cv2.imread(str(img_path))
        h, w = frame.shape[:2]
        print(f"\nProcessing {tc['name']} ({w}x{h})...")

        # 1. Product detection
        res = model.predict(frame, conf=0.30, imgsz=640, verbose=False)[0]
        prod_boxes = [tuple(map(int, b)) for b in res.boxes.xyxy.cpu().numpy()]
        print(f"  -> Detected {len(prod_boxes)} products")

        # 2. Person detection
        person_boxes = []
        if p_model:
            p_res = p_model.predict(frame, classes=[0], conf=0.35, imgsz=640, verbose=False)[0]
            person_boxes = [tuple(map(int, b)) for b in p_res.boxes.xyxy.cpu().numpy()]
            print(f"  -> Detected {len(person_boxes)} persons (shopper occlusion signal)")

        # 3. Vacancy engine analysis
        engine = ShelfVacancyEngine(
            shelf_id="SHELF-01",
            roi=tc["roi"],
            min_gap_multiplier=tc["min_gap_mult"],
            min_absolute_gap_px=tc["min_abs_gap"],
        )

        snapshot = engine.analyze_frame(
            product_boxes=prod_boxes,
            frame_w=w,
            frame_h=h,
            person_boxes=person_boxes,
        )

        print(f"  -> SHELF-01 Status: {snapshot.status}")
        print(f"  -> Rows identified: {len(snapshot.rows)}")
        print(f"  -> Total gaps: {len(snapshot.vacant_regions)} (Unoccluded: {len(snapshot.unoccluded_vacant_regions)})")
        print(f"  -> Occupancy: {snapshot.occupancy_pct:.1f}% | Vacancy Score: {snapshot.vacancy_score:.3f}")

        for gap in snapshot.vacant_regions:
            occ_tag = " [SHOPPER OCCLUDED]" if gap.is_occluded_by_person else " [VALID VACANCY]"
            print(f"     * {gap.region_id} ({gap.row_id}): width={gap.width_px:.0f}px ({gap.width_multiple:.1f}x median){occ_tag}")

        # 4. Render visual annotation
        annotated = annotate_vacancy_frame(
            frame=frame,
            snapshot=snapshot,
            roi_pct=tc["roi"],
            active_alert=snapshot.vacancy_detected,
        )

        out_path = OUT_DIR / f"{tc['name']}_vacancy_annotated.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"  -> Annotated frame saved: {out_path.relative_to(ROOT_DIR)}")

    print("\n" + "=" * 80)
    print("PHASE 25C COMPLETED: Evidence images generated successfully.")
    print("=" * 80)


if __name__ == "__main__":
    run_visualization_demo()
