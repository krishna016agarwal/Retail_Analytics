"""Evaluate retail-specific product detector against COCO baseline on SKU-110K.

Computes:
- Official YOLO validation metrics on unseen test split (mAP@50, mAP@50-95, Precision, Recall)
- Standard IoU>=0.50 Precision, Recall, and F1 metrics for both Retail Detector and COCO Baseline
- Average inference latency (ms)
- Side-by-side visual comparison images & contact sheet

Strict compliance:
- True IoU-matched detection metrics are reported (NOT count / ground-truth ratio).
- Single class: product (instance localization only, no SKU/brand claims).
"""

import argparse
import csv
import json
import pathlib
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

# Ensure project root is on sys.path
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
from ultralytics import YOLO

# Visual constants (BGR)
C_GT = (0, 210, 80)          # Green: Ground Truth
C_COCO = (255, 200, 0)       # Cyan: COCO Baseline
C_RETAIL = (0, 140, 255)     # Orange: Retail-Specific Detector
C_HUD_BG = (20, 24, 34)
C_ACCENT = (0, 215, 255)
C_TEXT = (245, 245, 245)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Retail Detector vs COCO on SKU-110K test set")
    parser.add_argument(
        "--retail-model",
        type=str,
        default="inventory_data/custom_model/retail_detector_exp2.pt",
        help="Path to trained retail detector weights",
    )
    parser.add_argument(
        "--coco-model",
        type=str,
        default="yolo11n.pt",
        help="Path to COCO baseline weights",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="inventory_data/demo_images/SKU110K_fixed/sku110k_exp2.yaml",
        help="Dataset YAML config",
    )
    parser.add_argument(
        "--test-list",
        type=str,
        default="inventory_data/demo_images/SKU110K_fixed/subsets/exp2_test.txt",
        help="Path to file listing test image paths",
    )
    parser.add_argument(
        "--ann-csv",
        type=str,
        default="inventory_data/demo_images/SKU110K_fixed/annotations/annotations_test.csv",
        help="Test annotations CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/retail_detector_eval",
        help="Output directory for results and images",
    )
    parser.add_argument("--conf", type=float, default=0.10, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image resolution")
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu' or '0')")
    parser.add_argument("--n-visual", type=int, default=10, help="Number of visual comparison images to render")
    return parser.parse_args()


def load_gt_annotations(csv_path: pathlib.Path) -> Dict[str, List[Tuple[int, int, int, int]]]:
    """Load GT annotations: image_name -> list of (x1, y1, x2, y2)."""
    gt = defaultdict(list)
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            try:
                name = row[0]
                x1, y1, x2, y2 = int(float(row[1])), int(float(row[2])), int(float(row[3])), int(float(row[4]))
                gt[name].append((x1, y1, x2, y2))
            except ValueError:
                continue
    return dict(gt)


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union between two (x1, y1, x2, y2) boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = b1_area + b2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def compute_matched_metrics(
    gt_boxes: List[Tuple[int, int, int, int]],
    pred_boxes: List[Tuple[int, int, int, int]],
    iou_thresh: float = 0.50,
) -> Tuple[int, int, int, float, float, float]:
    """Compute TP, FP, FN, Precision, Recall, F1 at given IoU threshold."""
    if not pred_boxes and not gt_boxes:
        return 0, 0, 0, 1.0, 1.0, 1.0
    if not pred_boxes:
        return 0, 0, len(gt_boxes), 0.0, 0.0, 0.0
    if not gt_boxes:
        return 0, len(pred_boxes), 0, 0.0, 0.0, 0.0

    matched_gt = set()
    tp = 0
    fp = 0

    for p_box in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for g_idx, g_box in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue
            iou = compute_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_thresh and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return tp, fp, fn, precision, recall, f1


def annotate_comparison(
    frame: np.ndarray,
    gt_boxes: List[Tuple[int, int, int, int]],
    coco_dets: List[Tuple[int, int, int, int]],
    retail_dets: List[Tuple[int, int, int, int]],
    img_name: str,
    coco_p: float, coco_r: float, coco_lat: float,
    ret_p: float, ret_r: float, ret_lat: float,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Draw boxes
    for x1, y1, x2, y2 in gt_boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), C_GT, 1, cv2.LINE_AA)
    for x1, y1, x2, y2 in coco_dets:
        cv2.rectangle(out, (x1, y1), (x2, y2), C_COCO, 2, cv2.LINE_AA)
    for x1, y1, x2, y2 in retail_dets:
        cv2.rectangle(out, (x1, y1), (x2, y2), C_RETAIL, 2, cv2.LINE_AA)

    # HUD Banner
    hud_h = 162
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), C_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)
    cv2.line(out, (0, hud_h), (w, hud_h), C_ACCENT, 1, cv2.LINE_AA)

    cv2.putText(out, "RETAIL DETECTOR (EXP 2) vs COCO BASELINE EVALUATION", (12, 24), FONT, 0.48, C_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(out, f"Test Image: {img_name} | GT: {len(gt_boxes)} boxes", (12, 46), FONT, 0.44, C_TEXT, 1, cv2.LINE_AA)

    # Legends & IoU-matched metrics
    cv2.rectangle(out, (12, 58), (24, 70), C_GT, -1)
    cv2.putText(out, f"Ground Truth (SKU-110K): {len(gt_boxes)} product instances", (30, 69), FONT, 0.42, C_GT, 1, cv2.LINE_AA)

    cv2.rectangle(out, (12, 78), (24, 90), C_COCO, -1)
    cv2.putText(out, f"COCO Baseline: {len(coco_dets)} det | Prec: {coco_p*100:.1f}% | Rec: {coco_r*100:.1f}% | Latency: {coco_lat:.0f}ms", (30, 89), FONT, 0.42, C_COCO, 1, cv2.LINE_AA)

    cv2.rectangle(out, (12, 98), (24, 110), C_RETAIL, -1)
    cv2.putText(out, f"Retail Detector (Exp 2): {len(retail_dets)} det | Prec: {ret_p*100:.1f}% | Rec: {ret_r*100:.1f}% | Latency: {ret_lat:.0f}ms", (30, 109), FONT, 0.42, C_RETAIL, 1, cv2.LINE_AA)

    delta_r = (ret_r - coco_r) * 100
    sign = "+" if delta_r >= 0 else ""
    cv2.putText(out, f"Standard IoU>=0.50 Recall Delta: {sign}{delta_r:.1f}% (True matched detection coverage)", (12, 134), FONT, 0.40, C_TEXT, 1, cv2.LINE_AA)
    cv2.putText(out, "* Metrics matched at IoU>=0.50 | Single class: product | Camera visible facings only", (12, hud_h - 6), FONT, 0.35, (160, 160, 160), 1, cv2.LINE_AA)

    return out


def make_contact_sheet(frames: List[np.ndarray], cols: int = 2, thumb_w: int = 720) -> np.ndarray:
    cells = []
    for f in frames:
        h, w = f.shape[:2]
        scale = thumb_w / w
        new_h = int(h * scale)
        cells.append(cv2.resize(f, (thumb_w, new_h), interpolation=cv2.INTER_AREA))

    max_h = max(c.shape[0] for c in cells)
    padded = []
    for c in cells:
        dh = max_h - c.shape[0]
        pad = np.zeros((dh, thumb_w, 3), dtype=np.uint8)
        padded.append(np.vstack([c, pad]))

    while len(padded) % cols != 0:
        padded.append(np.zeros((max_h, thumb_w, 3), dtype=np.uint8))

    rows = []
    for i in range(0, len(padded), cols):
        rows.append(np.hstack(padded[i:i + cols]))
    return np.vstack(rows)


def main():
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_list_path = pathlib.Path(args.test_list)
    if not test_list_path.exists():
        print(f"[ERROR] Test image list not found: {test_list_path}")
        return 1

    test_image_paths = [pathlib.Path(p.strip()) for p in test_list_path.read_text(encoding="utf-8").splitlines() if p.strip()]
    print("=" * 72)
    print("  RETAIL PRODUCT DETECTOR (EXP 2) vs COCO BASELINE EVALUATION")
    print("=" * 72)
    print(f"  Retail Model       : {args.retail_model}")
    print(f"  COCO Baseline Model: {args.coco_model}")
    print(f"  Unseen Test Images : {len(test_image_paths)} images")
    print(f"  Confidence Cutoff  : {args.conf}")
    print(f"  IoU Eval Threshold : 0.50 (standard Pascal VOC / COCO metric)")
    print(f"  Image Size         : {args.imgsz}")
    print(f"  Output Directory   : {output_dir}")
    print("=" * 72 + "\n")

    # 1. Official YOLO validation on test split
    print("[1/3] Running official YOLO validation pass on unseen test split...")
    retail_model = YOLO(args.retail_model)
    val_results = retail_model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        batch=4,
        device=args.device,
        verbose=False,
    )

    official_metrics = {
        "precision": round(float(val_results.results_dict.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(val_results.results_dict.get("metrics/recall(B)", 0.0)), 4),
        "mAP50": round(float(val_results.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50-95": round(float(val_results.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
        "inference_ms": round(float(val_results.speed.get("inference", 0.0)), 2),
    }
    print(f"      Retail Detector Test mAP@50   : {official_metrics['mAP50']:.4f}")
    print(f"      Retail Detector Test mAP@50-95: {official_metrics['mAP50-95']:.4f}")
    print(f"      Retail Detector Test Precision: {official_metrics['precision']:.4f}")
    print(f"      Retail Detector Test Recall   : {official_metrics['recall']:.4f}")
    print(f"      Retail Detector Test Latency  : {official_metrics['inference_ms']:.2f} ms")

    # 2. Side-by-side benchmark across all unseen test images
    print(f"\n[2/3] Computing per-image IoU>=0.50 matched metrics across {len(test_image_paths)} test images...")
    gt_all = load_gt_annotations(pathlib.Path(args.ann_csv))
    from inventory.config import COCO_RETAIL_CLASS_IDS
    coco_model = YOLO(args.coco_model)

    coco_tp_tot = coco_fp_tot = coco_fn_tot = 0
    ret_tp_tot = ret_fp_tot = ret_fn_tot = 0
    coco_latencies = []
    ret_latencies = []

    visual_frames = []
    per_image_rows = []

    for idx, img_path in enumerate(test_image_paths):
        if not img_path.exists():
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        gt_boxes = gt_all.get(img_path.name, [])

        # COCO inference
        t0 = time.perf_counter()
        coco_res = coco_model.predict(
            source=frame,
            classes=list(COCO_RETAIL_CLASS_IDS),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        coco_lat = (time.perf_counter() - t0) * 1000.0
        coco_latencies.append(coco_lat)

        coco_boxes = []
        if coco_res and coco_res[0].boxes is not None:
            b = coco_res[0].boxes
            for xyxy, conf, cls_id in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)):
                if int(cls_id) != 0:  # exclude person
                    coco_boxes.append(tuple(map(int, xyxy)))

        # Retail Detector inference
        t1 = time.perf_counter()
        ret_res = retail_model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        ret_lat = (time.perf_counter() - t1) * 1000.0
        ret_latencies.append(ret_lat)

        ret_boxes = []
        if ret_res and ret_res[0].boxes is not None:
            b = ret_res[0].boxes
            for xyxy in b.xyxy.cpu().numpy():
                ret_boxes.append(tuple(map(int, xyxy)))

        # IoU matched metrics
        c_tp, c_fp, c_fn, c_p, c_r, c_f1 = compute_matched_metrics(gt_boxes, coco_boxes, iou_thresh=0.50)
        r_tp, r_fp, r_fn, r_p, r_r, r_f1 = compute_matched_metrics(gt_boxes, ret_boxes, iou_thresh=0.50)

        coco_tp_tot += c_tp
        coco_fp_tot += c_fp
        coco_fn_tot += c_fn

        ret_tp_tot += r_tp
        ret_fp_tot += r_fp
        ret_fn_tot += r_fn

        per_image_rows.append({
            "image": img_path.name,
            "gt_count": len(gt_boxes),
            "coco_detections": len(coco_boxes),
            "coco_tp": c_tp, "coco_fp": c_fp, "coco_fn": c_fn,
            "coco_precision": round(c_p, 4), "coco_recall": round(c_r, 4),
            "retail_detections": len(ret_boxes),
            "retail_tp": r_tp, "retail_fp": r_fp, "retail_fn": r_fn,
            "retail_precision": round(r_p, 4), "retail_recall": round(r_r, 4),
        })

        if idx < args.n_visual:
            annotated = annotate_comparison(
                frame, gt_boxes, coco_boxes, ret_boxes,
                img_path.name, c_p, c_r, coco_lat, r_p, r_r, ret_lat,
            )
            visual_frames.append(annotated)
            cv2.imwrite(str(output_dir / f"{img_path.stem}_exp2_eval.jpg"), annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])

        if (idx + 1) % 25 == 0 or (idx + 1) == len(test_image_paths):
            print(f"      Processed {idx + 1}/{len(test_image_paths)} test images...")

    # Aggregate IoU-matched metrics
    agg_coco_p = coco_tp_tot / (coco_tp_tot + coco_fp_tot) if (coco_tp_tot + coco_fp_tot) > 0 else 0.0
    agg_coco_r = coco_tp_tot / (coco_tp_tot + coco_fn_tot) if (coco_tp_tot + coco_fn_tot) > 0 else 0.0
    agg_ret_p = ret_tp_tot / (ret_tp_tot + ret_fp_tot) if (ret_tp_tot + ret_fp_tot) > 0 else 0.0
    agg_ret_r = ret_tp_tot / (ret_tp_tot + ret_fn_tot) if (ret_tp_tot + ret_fn_tot) > 0 else 0.0

    avg_coco_lat = np.mean(coco_latencies) if coco_latencies else 0.0
    avg_ret_lat = np.mean(ret_latencies) if ret_latencies else 0.0

    # 3. Save contact sheet & JSON report
    print("\n[3/3] Generating summary report and contact sheet...")
    if visual_frames:
        sheet = make_contact_sheet(visual_frames, cols=2, thumb_w=720)
        sheet_path = output_dir / "exp2_comparison_sheet.jpg"
        cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"      Contact sheet saved -> {sheet_path}")

    full_report = {
        "test_dataset_size": len(test_image_paths),
        "official_yolo_test_metrics": official_metrics,
        "matched_iou50_evaluation": {
            "coco_baseline": {
                "total_detections": coco_tp_tot + coco_fp_tot,
                "true_positives": coco_tp_tot,
                "false_positives": coco_fp_tot,
                "false_negatives": coco_fn_tot,
                "precision": round(agg_coco_p, 4),
                "recall": round(agg_coco_r, 4),
                "avg_latency_ms": round(float(avg_coco_lat), 2),
            },
            "retail_detector_exp2": {
                "total_detections": ret_tp_tot + ret_fp_tot,
                "true_positives": ret_tp_tot,
                "false_positives": ret_fp_tot,
                "false_negatives": ret_fn_tot,
                "precision": round(agg_ret_p, 4),
                "recall": round(agg_ret_r, 4),
                "avg_latency_ms": round(float(avg_ret_lat), 2),
            },
            "recall_improvement_delta": round(agg_ret_r - agg_coco_r, 4),
        },
        "sample_per_image": per_image_rows[:15],
    }

    report_file = output_dir / "exp2_evaluation_report.json"
    report_file.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"      JSON report saved   -> {report_file}")

    print("\n" + "=" * 72)
    print("  EXPERIMENT 2 FINAL TEST BENCHMARK RESULTS")
    print("=" * 72)
    print(f"  Test Images Evaluated      : {len(test_image_paths)}")
    print(f"  Retail Detector mAP@50     : {official_metrics['mAP50']:.4f}")
    print(f"  Retail Detector mAP@50-95  : {official_metrics['mAP50-95']:.4f}")
    print(f"  Retail Detector Precision  : {agg_ret_p:.4f}  (Official: {official_metrics['precision']:.4f})")
    print(f"  Retail Detector Recall     : {agg_ret_r:.4f}  (Official: {official_metrics['recall']:.4f})")
    print(f"  Retail Detector Latency    : {avg_ret_lat:.1f} ms / image")
    print("-" * 72)
    print(f"  COCO Baseline Precision    : {agg_coco_p:.4f}")
    print(f"  COCO Baseline Recall       : {agg_coco_r:.4f}")
    print(f"  COCO Baseline Latency      : {avg_coco_lat:.1f} ms / image")
    print("-" * 72)
    print(f"  IoU>=0.50 Recall Delta     : {(agg_ret_r - agg_coco_r)*100:+.1f}%")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
