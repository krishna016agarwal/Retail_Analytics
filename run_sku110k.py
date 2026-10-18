"""SKU-110K dataset runner for the Inventory Foundation pipeline.

Runs COCO-baseline YOLO detection on SKU-110K test images and overlays:
  - GREEN boxes  : SKU-110K ground-truth product regions (from CSV annotations)
  - CYAN boxes   : COCO YOLO detections (generic object classes)

This makes the gap between generic COCO detection and real retail dense
product localisation immediately visible — motivating a retail-specific model.

Usage:
    venv\\Scripts\\python.exe run_sku110k.py
    venv\\Scripts\\python.exe run_sku110k.py --n-images 20 --conf 0.15
    venv\\Scripts\\python.exe run_sku110k.py --single inventory_data/demo_images/SKU110K_fixed/images/test_0.jpg
"""

import argparse
import csv
import pathlib
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np

DATASET_ROOT = pathlib.Path("inventory_data/demo_images/SKU110K_fixed")
IMAGES_DIR   = DATASET_ROOT / "images"
ANN_CSV      = DATASET_ROOT / "annotations" / "annotations_test.csv"
OUTPUT_DIR   = pathlib.Path("output/sku110k_run")

# Visual constants
C_GT         = (0,  210,  80)    # Green  — ground-truth boxes
C_COCO       = (255, 200,   0)   # Cyan/yellow — COCO YOLO detections
C_GT_FILL    = (0,  210,  80)
C_HUD_BG     = (18,  22,  32)
C_ACCENT     = (0,  215, 255)
C_TEXT       = (240, 240, 240)
C_WARN       = (0,  165, 255)
FONT         = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Load GT annotations from CSV
# ---------------------------------------------------------------------------

def load_gt_annotations(csv_path: pathlib.Path) -> Dict[str, List[Tuple[int,int,int,int]]]:
    """Return dict: image_name -> list of (x1,y1,x2,y2) GT boxes."""
    gt: Dict[str, List] = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            name, x1, y1, x2, y2 = row[0], int(row[1]), int(row[2]), int(row[3]), int(row[4])
            gt[name].append((x1, y1, x2, y2))
    return dict(gt)


# ---------------------------------------------------------------------------
# Annotate a single frame
# ---------------------------------------------------------------------------

def annotate_frame(
    frame: np.ndarray,
    gt_boxes: List[Tuple[int,int,int,int]],
    coco_detections: list,          # list of (bbox, conf, cls_name)
    image_name: str,
    inference_ms: float,
    gt_count: int,
    coco_count: int,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # --- GT boxes (green, semi-filled) ---
    for x1, y1, x2, y2 in gt_boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        cv2.rectangle(out, (x1,y1), (x2,y2), C_GT, 1, cv2.LINE_AA)
        # Tiny corner ticks
        L = min(8, (x2-x1)//3, (y2-y1)//3)
        for ox, oy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(out,(ox,oy),(ox+dx*L,oy),C_GT,1,cv2.LINE_AA)
            cv2.line(out,(ox,oy),(ox,oy+dy*L),C_GT,1,cv2.LINE_AA)

    # --- COCO YOLO detections (cyan, thicker) ---
    for (x1,y1,x2,y2), conf, cls_name in coco_detections:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        cv2.rectangle(out, (x1,y1), (x2,y2), C_COCO, 2, cv2.LINE_AA)
        label = f"{cls_name} {conf*100:.0f}%"
        (tw, th), bl = cv2.getTextSize(label, FONT, 0.38, 1)
        pad = 2
        bg_y = max(0, y1 - th - pad*2 - bl)
        cv2.rectangle(out, (x1, bg_y), (x1+tw+pad*2, y1), (20,20,20), -1)
        cv2.putText(out, label, (x1+pad, y1-pad-bl), FONT, 0.38, C_COCO, 1, cv2.LINE_AA)

    # --- HUD ---
    hud_h = 148
    overlay = out.copy()
    cv2.rectangle(overlay, (0,0), (w, hud_h), C_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.80, out, 0.20, 0, out)
    cv2.line(out, (0, hud_h), (w, hud_h), C_ACCENT, 1, cv2.LINE_AA)

    cv2.putText(out, "RETAIL SHELF MONITOR  |  SKU-110K EXPERIMENT  |  INVENTORY FOUNDATION",
                (10, 22), FONT, 0.48, C_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(out, f"Image: {image_name}",
                (10, 44), FONT, 0.46, C_TEXT, 1, cv2.LINE_AA)

    # Legend
    cv2.rectangle(out, (10, 56), (22, 68), C_GT, -1)
    cv2.putText(out, f"GT Products (SKU-110K): {gt_count} boxes  [dense product regions, NO class labels]",
                (28, 67), FONT, 0.44, C_GT, 1, cv2.LINE_AA)

    cv2.rectangle(out, (10, 76), (22, 88), C_COCO, -1)
    cv2.putText(out, f"COCO YOLO Detections: {coco_count} boxes  [generic classes -- NOT retail SKU recognition]",
                (28, 87), FONT, 0.44, C_COCO, 1, cv2.LINE_AA)

    # Coverage gap metric
    coverage_note = "Gap: COCO model misses most dense retail products  ->  retail-specific model needed"
    cv2.putText(out, coverage_note, (10, 108), FONT, 0.42, C_WARN, 1, cv2.LINE_AA)

    cv2.putText(out, f"Inf: {inference_ms:.0f} ms",
                (w-120, 44), FONT, 0.44, C_TEXT, 1, cv2.LINE_AA)

    # Footer disclaimer
    cv2.putText(out,
        "* visible_facings = camera-observable product instances only  |  NOT physical inventory quantity",
        (10, hud_h-6), FONT, 0.36, (150,150,150), 1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Contact sheet  (grid of N annotated thumbnails)
# ---------------------------------------------------------------------------

def make_contact_sheet(
    annotated_frames: List[np.ndarray],
    cols: int = 2,
    thumb_w: int = 640,
) -> np.ndarray:
    """Tile annotated frames into a grid image."""
    rows = (len(annotated_frames) + cols - 1) // cols
    cells = []
    for f in annotated_frames:
        h, w = f.shape[:2]
        scale = thumb_w / w
        new_h = int(h * scale)
        cells.append(cv2.resize(f, (thumb_w, new_h), interpolation=cv2.INTER_AREA))

    # Pad to uniform height
    max_h = max(c.shape[0] for c in cells)
    padded = []
    for c in cells:
        dh = max_h - c.shape[0]
        pad = np.zeros((dh, thumb_w, 3), dtype=np.uint8)
        padded.append(np.vstack([c, pad]))

    # Fill last row if needed
    while len(padded) % cols != 0:
        padded.append(np.zeros((max_h, thumb_w, 3), dtype=np.uint8))

    rows_list = []
    for i in range(0, len(padded), cols):
        rows_list.append(np.hstack(padded[i:i+cols]))

    return np.vstack(rows_list)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run inventory pipeline on SKU-110K test images")
    p.add_argument("--n-images", type=int, default=10, help="Number of images to process (default: 10)")
    p.add_argument("--conf", type=float, default=0.20, help="COCO YOLO confidence threshold")
    p.add_argument("--iou", type=float, default=0.40, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO inference resolution")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--single", type=str, default=None, help="Run on a single image path")
    p.add_argument("--seed", type=int, default=42, help="Random seed for image selection")
    p.add_argument("--no-show", action="store_true", help="Skip display window")
    return p.parse_args()


def main():
    args = parse_args()

    if not IMAGES_DIR.exists():
        print(f"[ERROR] Dataset not found: {IMAGES_DIR}")
        print("        Expected: inventory_data/demo_images/SKU110K_fixed/images/")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  RETAIL SHELF MONITOR  |  SKU-110K EXPERIMENT")
    print("=" * 72)
    print(f"  Dataset        : {DATASET_ROOT}")
    print(f"  Annotation CSV : {ANN_CSV.name}  ({ANN_CSV.stat().st_size/1e6:.1f} MB)")
    print(f"  Output Dir     : {OUTPUT_DIR}")
    print()
    print("  IMPORTANT: COCO model detects GENERIC objects (bottle, cup, etc.)")
    print("             It does NOT recognise retail brands or SKUs.")
    print("             GREEN = ground-truth boxes (dense products, no class label)")
    print("             CYAN  = COCO YOLO detections (generic categories)")
    print("=" * 72 + "\n")

    # Load GT
    print("[1/4] Loading ground-truth annotations...")
    gt_all = load_gt_annotations(ANN_CSV)
    print(f"      {len(gt_all)} images annotated in test CSV.")

    # Select images
    all_imgs = sorted(IMAGES_DIR.glob("*.jpg"))
    if args.single:
        selected = [pathlib.Path(args.single)]
    else:
        import random
        rng = random.Random(args.seed)
        available = [p for p in all_imgs if p.name in gt_all]
        n = min(args.n_images, len(available))
        selected = rng.sample(available, n)
        selected.sort()
        print(f"[2/4] Selected {len(selected)} images from {len(available)} annotated test images.\n")

    # Load YOLO
    print("[3/4] Loading YOLO model (coco_baseline)...")
    from ultralytics import YOLO
    from inventory.config import COCO_RETAIL_CLASS_IDS
    model = YOLO("yolo11n.pt")
    model.to(args.device)

    # Warmup
    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    model.predict(source=dummy, device=args.device, conf=args.conf,
                  imgsz=args.imgsz, verbose=False)
    print("      Model ready.\n")

    # Process images
    print("[4/4] Processing images...\n")
    annotated_frames = []
    summary_rows = []

    for idx, img_path in enumerate(selected):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [SKIP] Cannot read {img_path.name}")
            continue

        h_orig, w_orig = frame.shape[:2]

        # GT boxes for this image
        gt_boxes = gt_all.get(img_path.name, [])

        # YOLO inference (COCO retail-plausible classes only)
        t0 = time.perf_counter()
        results = model.predict(
            source=frame,
            device=args.device,
            classes=list(COCO_RETAIL_CLASS_IDS),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )
        inf_ms = (time.perf_counter() - t0) * 1000.0

        # Parse COCO detections
        coco_dets = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            names = results[0].names
            for xyxy, conf, cls_id in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.conf.cpu().numpy(),
                boxes.cls.cpu().numpy().astype(int),
            ):
                x1, y1, x2, y2 = map(int, xyxy)
                cls_name = names.get(int(cls_id), f"cls{cls_id}")
                if int(cls_id) != 0:  # exclude person from product list
                    coco_dets.append(((x1,y1,x2,y2), float(conf), cls_name))

        # Annotate
        annotated = annotate_frame(
            frame, gt_boxes, coco_dets,
            img_path.name, inf_ms, len(gt_boxes), len(coco_dets),
        )

        # Save individual result
        out_file = OUTPUT_DIR / f"{img_path.stem}_analysis.jpg"
        cv2.imwrite(str(out_file), annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
        annotated_frames.append(annotated)

        detection_ratio = len(coco_dets) / max(len(gt_boxes), 1) * 100
        print(f"  [{idx+1:3d}/{len(selected)}] {img_path.name:<22s} | "
              f"GT: {len(gt_boxes):4d} boxes | "
              f"COCO: {len(coco_dets):3d} detected | "
              f"Coverage: {detection_ratio:5.1f}% | "
              f"Inf: {inf_ms:.0f}ms")

        summary_rows.append({
            "image": img_path.name,
            "gt_boxes": len(gt_boxes),
            "coco_detections": len(coco_dets),
            "detection_ratio_pct": round(detection_ratio, 1),
            "inference_ms": round(inf_ms, 1),
        })

    # Contact sheet
    if annotated_frames:
        print("\n[+] Generating contact sheet...")
        sheet = make_contact_sheet(annotated_frames, cols=2, thumb_w=720)
        sheet_path = OUTPUT_DIR / "sku110k_contact_sheet.jpg"
        cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"    Contact sheet  ->  {sheet_path}")

    # Summary table
    print()
    print("=" * 72)
    print("  SKU-110K EXPERIMENT SUMMARY")
    print("=" * 72)
    print(f"  {'Image':<22s} {'GT':>6s} {'COCO':>6s} {'Coverage':>10s} {'Inf(ms)':>8s}")
    print("  " + "-" * 60)
    total_gt = total_coco = 0
    for row in summary_rows:
        print(f"  {row['image']:<22s} {row['gt_boxes']:>6d} {row['coco_detections']:>6d} "
              f"{row['detection_ratio_pct']:>9.1f}% {row['inference_ms']:>7.0f}ms")
        total_gt   += row['gt_boxes']
        total_coco += row['coco_detections']

    overall_cov = total_coco / max(total_gt, 1) * 100
    print("  " + "-" * 60)
    print(f"  {'TOTAL/AVERAGE':<22s} {total_gt:>6d} {total_coco:>6d} {overall_cov:>9.1f}%")
    print()
    print("  KEY FINDING:")
    print(f"  COCO model detected {overall_cov:.1f}% of GT product boxes.")
    print("  Most dense retail products are NOT recognised by the generic COCO model.")
    print("  This confirms the need for a retail-specific or SKU-trained model.")
    print()
    print(f"  Output: {OUTPUT_DIR}/")
    print("=" * 72)

    # Show contact sheet
    if annotated_frames and not args.no_show:
        print("\n[Press any key to close the display window...]")
        cv2.namedWindow("SKU-110K Experiment — Contact Sheet", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("SKU-110K Experiment — Contact Sheet", 1440, 900)
        cv2.imshow("SKU-110K Experiment — Contact Sheet", sheet)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
