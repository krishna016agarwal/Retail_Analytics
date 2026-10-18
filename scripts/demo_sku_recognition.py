"""Demonstration of the SKU Recognition Foundation on Retail Shelf Images.

Workflow:
1. Detects visible product instances using retail_detector_exp2.pt (at recommended conf=0.30).
2. Extracts clean product crops for every detected bounding box.
3. Computes multi-zone spatial color-texture embeddings for crops.
4. Matches crops against a configurable store product catalog using cosine similarity.
5. Applies match threshold: outputs product ID, name, confidence, or assigns 'UNKNOWN'.
6. Generates visual overlay and structured JSON report.

Completely isolated from the crowd/queue pipeline (src/).
"""

import argparse
import json
import pathlib
import sys
import time
from collections import Counter
from typing import Dict, List

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

from inventory.catalog import SKUCatalog
from inventory.config import InventoryModelConfig
from inventory.shelf_detector import ShelfDetection, ShelfProductDetector
from inventory.shelf_visualizer import ShelfVisualizer
from inventory.sku_recognizer import SpatialColorTextureRecognizer


def parse_args():
    parser = argparse.ArgumentParser(description="SKU Recognition Foundation Demo")
    parser.add_argument(
        "--image",
        type=str,
        default="inventory_data/demo_images/SKU110K_fixed/images/test_1095.jpg",
        help="Path to input shelf image",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="inventory_data/custom_model/retail_detector_exp2.pt",
        help="Path to retail detector weights",
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default="inventory_data/catalogs/demo_store_catalog.json",
        help="Path to store product catalog JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/sku_recognition_demo",
        help="Output directory for annotated image and report",
    )
    parser.add_argument("--conf", type=float, default=0.30, help="Detector confidence cutoff (default: 0.30)")
    parser.add_argument("--match-thresh", type=float, default=0.65, help="SKU similarity cutoff for recognition")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device ('cpu')")
    return parser.parse_args()


def main():
    args = parse_args()
    img_path = pathlib.Path(args.image)
    if not img_path.is_file():
        raise FileNotFoundError(f"Input image not found: {img_path}")

    catalog_path = pathlib.Path(args.catalog)
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("RETAIL INVENTORY: SKU RECOGNITION FOUNDATION DEMO")
    print("================================================================================")
    print(f"Image:            {img_path.name}")
    print(f"Detector Model:   {args.model} (conf={args.conf})")
    print(f"Product Catalog:  {catalog_path.name} (match_threshold={args.match_thresh})")
    print("--------------------------------------------------------------------------------")

    # 1. Load Catalog and initialize Recognizer
    print("\n[1/4] Loading Store Product Catalog...")
    catalog = SKUCatalog.load_json(catalog_path)
    print(f"      Loaded catalog '{catalog.name}' with {len(catalog)} registered SKUs:")
    for item in catalog.list_items():
        print(f"      • [{item.sku_id}] {item.name} ({item.category})")

    recognizer = SpatialColorTextureRecognizer(catalog=catalog, match_threshold=args.match_thresh)
    recognizer.build_index()
    print(f"      Indexed reference embeddings for {len(catalog)} SKUs.")

    # 2. Run Retail Product Detector
    print("\n[2/4] Running Retail Product Detector on Shelf Frame...")
    frame = cv2.imread(str(img_path))
    if frame is None:
        raise ValueError(f"Could not load image: {img_path}")

    det_cfg = InventoryModelConfig(
        model_path=args.model,
        model_tier="retail_specific",
        device=args.device,
        confidence_threshold=args.conf,
        target_classes=None,  # custom model has single class 0 (product)
    )
    detector = ShelfProductDetector(det_cfg)

    t0 = time.perf_counter()
    batch = detector.detect(frame, frame_index=0)
    det_latency = (time.perf_counter() - t0) * 1000.0
    print(f"      Detector found {len(batch.product_detections)} visible product facings in {det_latency:.1f} ms.")

    # 3. Crop Products and Match Against Catalog
    print("\n[3/4] Cropping Products and Matching Against SKU Catalog...")
    t_rec0 = time.perf_counter()

    recognized_count = 0
    unknown_count = 0
    sku_counter = Counter()
    per_product_records = []

    for idx, det in enumerate(batch.product_detections):
        crop = det.crop_from_frame(frame)
        rec_res = recognizer.recognize_crop(crop)

        det.sku_id = rec_res.sku_id
        det.sku_name = rec_res.name
        det.sku_confidence = rec_res.confidence
        det.is_known_sku = rec_res.is_known

        if rec_res.is_known:
            recognized_count += 1
            sku_counter[f"{rec_res.name} [{rec_res.sku_id}]"] += 1
        else:
            unknown_count += 1
            sku_counter["UNKNOWN (Uncataloged Product)"] += 1

        record = {
            "product_index": idx + 1,
            "bbox": list(det.bbox),
            "detection_confidence": round(float(det.confidence), 4),
            "sku_id": det.sku_id,
            "sku_name": det.sku_name,
            "sku_confidence": round(float(det.sku_confidence), 4),
            "is_known_sku": det.is_known_sku,
            "top_candidates": [
                {"sku_id": c[0], "name": c[1], "similarity": c[2]} for c in rec_res.top_matches
            ],
        }
        per_product_records.append(record)

    rec_latency = (time.perf_counter() - t_rec0) * 1000.0
    avg_crop_ms = rec_latency / max(1, len(batch.product_detections))
    print(f"      Completed recognition for {len(batch.product_detections)} crops in {rec_latency:.1f} ms ({avg_crop_ms:.2f} ms/crop).")

    # 4. Generate Visual Overlay
    print("\n[4/4] Rendering Visual Annotations and Generating Reports...")
    visualizer = ShelfVisualizer(show_confidence=True, show_class_names=True)
    annotated = visualizer.annotate_frame(
        frame=frame,
        batch=batch,
        state=None,
        fps=0.0,
        shelf_roi=None,
        model_tier="retail_specific_sku",
    )

    # Add Top Banner for SKU Recognition Summary
    hud_h = 75
    h_ann, w_ann = annotated.shape[:2]
    sku_hud = np.zeros((hud_h, w_ann, 3), dtype=np.uint8)
    sku_hud[:] = (20, 24, 34)

    cv2.putText(
        sku_hud,
        f"SKU Recognition | Visible Facings: {len(batch.product_detections)} | Catalog: '{catalog.name}' (Match Thresh: {args.match_thresh})",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        sku_hud,
        f"Matched Catalog SKUs: {recognized_count} (Cyan) | UNKNOWN / Uncataloged: {unknown_count} (Orange) | Avg Crop Latency: {avg_crop_ms:.2f} ms",
        (15, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    combined = np.vstack([sku_hud, annotated])

    # Save annotated visual
    out_img_path = out_dir / f"{img_path.stem}_sku_annotated.jpg"
    cv2.imwrite(str(out_img_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"      [Saved Annotated Image] -> {out_img_path}")

    # Save JSON Report
    report_path = out_dir / f"{img_path.stem}_sku_report.json"
    report_data = {
        "image": str(img_path),
        "detector_model": args.model,
        "detector_confidence_threshold": args.conf,
        "sku_match_threshold": args.match_thresh,
        "catalog_name": catalog.name,
        "total_visible_facings": len(batch.product_detections),
        "recognized_skus_count": recognized_count,
        "unknown_skus_count": unknown_count,
        "recognition_rate": round(recognized_count / max(1, len(batch.product_detections)), 4),
        "sku_breakdown": dict(sku_counter),
        "timing": {
            "detection_ms": round(det_latency, 2),
            "recognition_ms": round(rec_latency, 2),
            "total_ms": round(det_latency + rec_latency, 2),
        },
        "detections": per_product_records,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"      [Saved JSON Report]     -> {report_path}")

    # Print Summary Table
    print("\n" + "=" * 80)
    print(f"SKU RECOGNITION SUMMARY: {img_path.name}")
    print("=" * 80)
    print(f"{'SKU Item / Product Category':<45} | {'Count':>6} | {'Share (%)':>10}")
    print("-" * 80)
    for sku_desc, count in sku_counter.most_common():
        share = count / max(1, len(batch.product_detections)) * 100.0
        print(f"{sku_desc:<45} | {count:6d} | {share:9.1f}%")
    print("-" * 80)
    print(
        f"{'TOTAL VISIBLE FACINGS':<45} | {len(batch.product_detections):6d} | "
        f"{'100.0%':>10}"
    )
    print(
        f"{'  -> Matched Catalog SKUs':<45} | {recognized_count:6d} | "
        f"{recognized_count / max(1, len(batch.product_detections)) * 100.0:9.1f}%"
    )
    print(
        f"{'  -> UNKNOWN (Uncataloged)':<45} | {unknown_count:6d} | "
        f"{unknown_count / max(1, len(batch.product_detections)) * 100.0:9.1f}%"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
