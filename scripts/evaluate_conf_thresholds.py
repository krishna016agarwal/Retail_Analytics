"""Step 3: Confidence Threshold Evaluation for Retail Inventory Detector.

Evaluates trained retail detector (retail_detector_exp2.pt) across multiple confidence thresholds:
[0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

Computes:
- Total detections
- True Positives (IoU >= 0.50)
- False Positives
- False Negatives
- Precision
- Recall
- F1 score
- Detection Coverage (total detections / total ground-truth instances)
- Average Inference Latency (ms)

Generates:
1. CSV and JSON reports
2. Precision/Recall/F1 vs Confidence curve plot
3. Visual comparison panels for 3 representative test images (Low, Recommended, High thresholds + Ground Truth)
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
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# Visualization styling
C_GT = (0, 220, 70)       # Green
C_LOW = (0, 215, 255)     # Yellow/Gold
C_REC = (0, 140, 255)     # Vibrant Orange
C_HIGH = (255, 60, 180)   # Magenta/Violet
C_HUD_BG = (20, 24, 34)
C_TEXT = (245, 245, 245)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args():
    parser = argparse.ArgumentParser(description="Confidence Threshold Evaluation for Retail Detector")
    parser.add_argument(
        "--model",
        type=str,
        default="inventory_data/custom_model/retail_detector_exp2.pt",
        help="Path to retail detector weights",
    )
    parser.add_argument(
        "--test-list",
        type=str,
        default="inventory_data/demo_images/SKU110K_fixed/subsets/exp2_test.txt",
        help="List of test images",
    )
    parser.add_argument(
        "--ann-csv",
        type=str,
        default="inventory_data/demo_images/SKU110K_fixed/annotations/annotations_test.csv",
        help="Annotations CSV file for test set",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/conf_eval",
        help="Directory to save evaluation reports and visualizations",
    )
    parser.add_argument("--imgsz", type=int, default=416, help="Inference resolution")
    parser.add_argument("--iou-match", type=float, default=0.50, help="IoU matching threshold for TP")
    parser.add_argument("--nms-iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu)")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
        help="Confidence thresholds to evaluate",
    )
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
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def match_boxes_greedy(
    pred_boxes: List[Tuple[int, int, int, int, float]],
    gt_boxes: List[Tuple[int, int, int, int]],
    iou_thresh: float = 0.50,
) -> Tuple[int, int, int, List[int], List[int]]:
    """Greedy bipartite matching sorted by prediction confidence descending.

    Returns:
        tp, fp, fn, matched_pred_indices, false_positive_indices
    """
    if not pred_boxes and not gt_boxes:
        return 0, 0, 0, [], []
    if not pred_boxes:
        return 0, 0, len(gt_boxes), [], []
    if not gt_boxes:
        return 0, len(pred_boxes), 0, [], list(range(len(pred_boxes)))

    # Sort predicted boxes by score descending
    sorted_pred_indices = sorted(range(len(pred_boxes)), key=lambda i: pred_boxes[i][4], reverse=True)

    matched_gt = set()
    tp_indices = []
    fp_indices = []

    for p_idx in sorted_pred_indices:
        p_box = pred_boxes[p_idx][:4]
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
            tp_indices.append(p_idx)
            matched_gt.add(best_gt_idx)
        else:
            fp_indices.append(p_idx)

    tp = len(tp_indices)
    fp = len(fp_indices)
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, tp_indices, fp_indices


def render_panel(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int],
    title: str,
    subtitle: str,
) -> np.ndarray:
    """Render an annotated panel with a HUD header."""
    panel = image.copy()
    h, w = panel.shape[:2]

    # Draw boxes
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)

    # Top HUD banner
    hud_h = 70
    hud = np.zeros((hud_h, w, 3), dtype=np.uint8)
    hud[:] = C_HUD_BG

    cv2.putText(hud, title, (14, 28), FONT, 0.75, color, 2, cv2.LINE_AA)
    cv2.putText(hud, subtitle, (14, 55), FONT, 0.52, (200, 205, 215), 1, cv2.LINE_AA)

    return np.vstack([hud, panel])


def create_visual_comparison_sheet(
    image_path: pathlib.Path,
    gt_boxes: List[Tuple[int, int, int, int]],
    model: YOLO,
    low_conf: float,
    rec_conf: float,
    high_conf: float,
    iou_match: float,
    nms_iou: float,
    device: str,
    imgsz: int,
    output_path: pathlib.Path,
):
    """Generate a 4-panel visual comparison (GT, Low Conf, Recommended Conf, High Conf)."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return

    # Run predictions for each threshold
    preds_low = run_single_image_pred(model, img_bgr, low_conf, nms_iou, device, imgsz)
    preds_rec = run_single_image_pred(model, img_bgr, rec_conf, nms_iou, device, imgsz)
    preds_high = run_single_image_pred(model, img_bgr, high_conf, nms_iou, device, imgsz)

    # Compute matches
    tp_l, fp_l, fn_l, _, _ = match_boxes_greedy(preds_low, gt_boxes, iou_match)
    p_l = tp_l / (tp_l + fp_l) if (tp_l + fp_l) > 0 else 0.0
    r_l = tp_l / len(gt_boxes) if gt_boxes else 0.0

    tp_m, fp_m, fn_m, _, _ = match_boxes_greedy(preds_rec, gt_boxes, iou_match)
    p_m = tp_m / (tp_m + fp_m) if (tp_m + fp_m) > 0 else 0.0
    r_m = tp_m / len(gt_boxes) if gt_boxes else 0.0

    tp_h, fp_h, fn_h, _, _ = match_boxes_greedy(preds_high, gt_boxes, iou_match)
    p_h = tp_h / (tp_h + fp_h) if (tp_h + fp_h) > 0 else 0.0
    r_h = tp_h / len(gt_boxes) if gt_boxes else 0.0

    # Render 4 panels
    p_gt = render_panel(
        img_bgr,
        gt_boxes,
        C_GT,
        f"1. Ground Truth ({image_path.name})",
        f"GT Count: {len(gt_boxes)} verified items | Reference standard",
    )
    p_low = render_panel(
        img_bgr,
        [b[:4] for b in preds_low],
        C_LOW,
        f"2. Low Threshold (conf={low_conf:.2f})",
        f"Detections: {len(preds_low)} | TP: {tp_l}, FP: {fp_l} | Prec: {p_l:.1%}, Rec: {r_l:.1%}",
    )
    p_rec = render_panel(
        img_bgr,
        [b[:4] for b in preds_rec],
        C_REC,
        f"3. Recommended Threshold (conf={rec_conf:.2f})",
        f"Detections: {len(preds_rec)} | TP: {tp_m}, FP: {fp_m} | Prec: {p_m:.1%}, Rec: {r_m:.1%}",
    )
    p_high = render_panel(
        img_bgr,
        [b[:4] for b in preds_high],
        C_HIGH,
        f"4. High Threshold (conf={high_conf:.2f})",
        f"Detections: {len(preds_high)} | TP: {tp_h}, FP: {fp_h} | Prec: {p_h:.1%}, Rec: {r_h:.1%}",
    )

    # Combine into 2x2 grid
    # Resize panels to common thumbnail width if large
    target_w = 700
    scale = target_w / p_gt.shape[1]
    dim = (target_w, int(p_gt.shape[0] * scale))

    p1 = cv2.resize(p_gt, dim, interpolation=cv2.INTER_AREA)
    p2 = cv2.resize(p_low, dim, interpolation=cv2.INTER_AREA)
    p3 = cv2.resize(p_rec, dim, interpolation=cv2.INTER_AREA)
    p4 = cv2.resize(p_high, dim, interpolation=cv2.INTER_AREA)

    top_row = np.hstack([p1, p2])
    bot_row = np.hstack([p3, p4])
    grid = np.vstack([top_row, bot_row])

    cv2.imwrite(str(output_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])


def run_single_image_pred(
    model: YOLO,
    img_bgr: np.ndarray,
    conf: float,
    nms_iou: float,
    device: str,
    imgsz: int,
) -> List[Tuple[int, int, int, int, float]]:
    """Run model prediction on single image, return list of (x1, y1, x2, y2, score)."""
    res = model.predict(
        img_bgr,
        conf=conf,
        iou=nms_iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]

    preds = []
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            preds.append((
                int(xyxy[i][0]),
                int(xyxy[i][1]),
                int(xyxy[i][2]),
                int(xyxy[i][3]),
                float(confs[i]),
            ))
    return preds


def plot_conf_curves(metrics_list: List[Dict], recommended_conf: float, output_png: pathlib.Path, output_jpg: pathlib.Path):
    """Plot Confidence vs Precision, Recall, F1 curves."""
    confs = [m["conf_threshold"] for m in metrics_list]
    precisions = [m["precision"] for m in metrics_list]
    recalls = [m["recall"] for m in metrics_list]
    f1s = [m["f1_score"] for m in metrics_list]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor("#0F141C")
    ax.set_facecolor("#161D27")

    # Plot lines with markers
    ax.plot(confs, precisions, marker="o", color="#38BDF8", linewidth=2.5, markersize=7, label="Precision")
    ax.plot(confs, recalls, marker="s", color="#34D399", linewidth=2.5, markersize=7, label="Recall (IoU >= 0.50)")
    ax.plot(confs, f1s, marker="^", color="#FB923C", linewidth=2.8, markersize=8, label="F1 Score")

    # Highlight recommended threshold
    rec_metric = next((m for m in metrics_list if abs(m["conf_threshold"] - recommended_conf) < 1e-4), metrics_list[0])
    ax.axvline(x=recommended_conf, color="#F43F5E", linestyle="--", linewidth=1.8, alpha=0.85)
    ax.scatter([recommended_conf], [rec_metric["f1_score"]], color="#F43F5E", s=180, zorder=5, edgecolors="#FFFFFF", linewidth=2)
    ax.annotate(
        f"Recommended: {recommended_conf:.2f}\n(F1={rec_metric['f1_score']:.3f}, P={rec_metric['precision']:.3f}, R={rec_metric['recall']:.3f})",
        xy=(recommended_conf, rec_metric["f1_score"]),
        xytext=(recommended_conf + 0.04, rec_metric["f1_score"] - 0.08),
        arrowprops=dict(facecolor="#F43F5E", shrink=0.08, width=1.5, headwidth=7),
        color="#FFFFFF",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1F2937", edgecolor="#F43F5E", alpha=0.95),
    )

    ax.set_title("Retail Detector: Confidence Threshold Trade-off (SKU-110K Test Split)", fontsize=14, fontweight="bold", color="#F3F4F6", pad=15)
    ax.set_xlabel("Confidence Threshold", fontsize=12, fontweight="medium", color="#E5E7EB", labelpad=10)
    ax.set_ylabel("Metric Value (0.0 – 1.0)", fontsize=12, fontweight="medium", color="#E5E7EB", labelpad=10)
    ax.set_xlim(0.0, 0.52)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(confs)
    ax.set_yticks(np.arange(0.0, 1.05, 0.1))

    ax.grid(True, color="#2D3748", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.tick_params(colors="#D1D5DB", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#374151")

    ax.legend(facecolor="#1F2937", edgecolor="#374151", labelcolor="#F9FAFB", fontsize=11, loc="center right")

    plt.tight_layout()
    plt.savefig(str(output_png), facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(str(output_jpg), facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=150)
    plt.close(fig)


def select_recommended_threshold(metrics_list: List[Dict]) -> float:
    """Recommend best practical confidence threshold.

    Prioritizes a balanced trade-off:
    - Does not blindly pick highest precision (e.g. conf=0.50 where recall drops).
    - Requires reasonable recall (>= 50%) and minimizes false positives while preserving shelf counting.
    - Maximizes F1 score among thresholds with recall >= 0.50.
    """
    candidates = [m for m in metrics_list if m["recall"] >= 0.50]
    if not candidates:
        candidates = metrics_list
    best = max(candidates, key=lambda m: m["f1_score"])
    return best["conf_threshold"]


def main():
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("STEP 3: CONFIDENCE THRESHOLD EVALUATION FOR RETAIL INVENTORY DETECTOR")
    print("================================================================================")
    print(f"Model: {args.model}")
    print(f"Test List: {args.test_list}")
    print(f"Annotations CSV: {args.ann_csv}")
    print(f"Thresholds: {args.thresholds}")
    print(f"Output Directory: {output_dir}")
    print("--------------------------------------------------------------------------------")

    # 1. Load test images
    test_images = []
    with open(args.test_list, "r", encoding="utf-8") as f:
        for line in f:
            p = pathlib.Path(line.strip())
            if p.is_file():
                test_images.append(p)
    if not test_images:
        raise FileNotFoundError(f"No test images found in {args.test_list}")
    print(f"Loaded {len(test_images)} unseen test images.")

    # 2. Load Ground Truth annotations
    gt_dict = load_gt_annotations(pathlib.Path(args.ann_csv))
    total_gt = sum(len(gt_dict.get(img_p.name, [])) for img_p in test_images)
    print(f"Total Ground-Truth Products across test set: {total_gt:,}")

    # 3. Load YOLO model
    print(f"Loading YOLO model from {args.model}...")
    model = YOLO(args.model)

    # 4. Sweep confidence thresholds
    metrics_summary = []

    print("\nRunning sweep across confidence thresholds...")
    print(f"{'Conf':>6} | {'Total':>7} | {'TP':>6} | {'FP':>6} | {'FN':>6} | {'Prec':>7} | {'Recall':>7} | {'F1':>7} | {'Coverage':>8} | {'Lat(ms)':>7}")
    print("-" * 88)

    for conf in sorted(args.thresholds):
        total_dets = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        latencies = []

        for img_p in test_images:
            gt_boxes = gt_dict.get(img_p.name, [])
            img_bgr = cv2.imread(str(img_p))
            if img_bgr is None:
                continue

            t0 = time.perf_counter()
            preds = run_single_image_pred(
                model=model,
                img_bgr=img_bgr,
                conf=conf,
                nms_iou=args.nms_iou,
                device=args.device,
                imgsz=args.imgsz,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            tp, fp, fn, _, _ = match_boxes_greedy(preds, gt_boxes, args.iou_match)
            total_dets += len(preds)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / total_gt if total_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        coverage = total_dets / total_gt if total_gt > 0 else 0.0
        avg_lat = float(np.mean(latencies)) if latencies else 0.0

        row = {
            "conf_threshold": conf,
            "total_detections": total_dets,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "detection_coverage": round(coverage, 4),
            "avg_latency_ms": round(avg_lat, 2),
        }
        metrics_summary.append(row)

        print(
            f"{conf:6.2f} | {total_dets:7d} | {total_tp:6d} | {total_fp:6d} | {total_fn:6d} | "
            f"{precision:7.4f} | {recall:7.4f} | {f1:7.4f} | {coverage:8.4f} | {avg_lat:7.2f}"
        )

    # 5. Select recommended threshold
    recommended_conf = select_recommended_threshold(metrics_summary)
    rec_metric = next(m for m in metrics_summary if abs(m["conf_threshold"] - recommended_conf) < 1e-4)

    print("-" * 88)
    print(f">>> RECOMMENDED CONFIDENCE THRESHOLD: conf = {recommended_conf:.2f}")
    print(
        f"    Precision: {rec_metric['precision']:.4f} ({rec_metric['precision']:.1%}) | "
        f"Recall: {rec_metric['recall']:.4f} ({rec_metric['recall']:.1%}) | "
        f"F1 Score: {rec_metric['f1_score']:.4f} | "
        f"Avg Latency: {rec_metric['avg_latency_ms']:.1f} ms"
    )

    # 6. Save JSON & CSV reports
    json_path = output_dir / "conf_thresholds_evaluation.json"
    csv_path = output_dir / "conf_thresholds_evaluation.csv"

    report_payload = {
        "model": args.model,
        "test_dataset_size": len(test_images),
        "total_ground_truth_instances": total_gt,
        "iou_matching_threshold": args.iou_match,
        "recommended_threshold": recommended_conf,
        "recommended_metrics": rec_metric,
        "threshold_metrics": metrics_summary,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"\n[Saved JSON Report] -> {json_path}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "conf_threshold", "total_detections", "true_positives", "false_positives",
            "false_negatives", "precision", "recall", "f1_score", "detection_coverage", "avg_latency_ms"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_summary)
    print(f"[Saved CSV Report]  -> {csv_path}")

    # 7. Generate Precision-Recall-F1 vs Confidence Plot
    plot_png = output_dir / "conf_vs_metrics.png"
    plot_jpg = output_dir / "conf_vs_metrics.jpg"
    plot_conf_curves(metrics_summary, recommended_conf, plot_png, plot_jpg)
    print(f"[Saved Plot]        -> {plot_png}")

    # 8. Generate Visual Examples on 3 Representative Shelf Images
    # Low threshold: 0.03 or 0.05
    # Recommended threshold: e.g. 0.20 or 0.25
    # High threshold: 0.50
    low_thresh = 0.03 if 0.03 in args.thresholds else 0.05
    high_thresh = 0.50

    representative_images = ["test_109.jpg", "test_1095.jpg", "test_104.jpg"]
    print(f"\nGenerating visual comparisons for 3 representative shelves: {representative_images}...")

    for rep_name in representative_images:
        matching_paths = [p for p in test_images if p.name == rep_name]
        if not matching_paths:
            continue
        rep_img_path = matching_paths[0]
        gt_boxes = gt_dict.get(rep_name, [])
        vis_out = output_dir / f"visual_comparison_{rep_img_path.stem}.jpg"

        create_visual_comparison_sheet(
            image_path=rep_img_path,
            gt_boxes=gt_boxes,
            model=model,
            low_conf=low_thresh,
            rec_conf=recommended_conf,
            high_conf=high_thresh,
            iou_match=args.iou_match,
            nms_iou=args.nms_iou,
            device=args.device,
            imgsz=args.imgsz,
            output_path=vis_out,
        )
        print(f"[Saved Visual Panel] -> {vis_out}")

    print("\nConfidence Threshold Evaluation Completed Successfully!")


if __name__ == "__main__":
    main()
