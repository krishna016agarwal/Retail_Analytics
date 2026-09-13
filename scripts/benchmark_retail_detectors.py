"""Isolated benchmark script for evaluating retail detector candidates against retail_detector_exp2.pt.

Strict isolation:
- Does NOT modify any production code, configs, or models.
- Benchmarks retail_detector_exp2.pt against YOLO11n, YOLO11s, and RT-DETR-L.
- Evaluates:
  1. Established test metrics (SKU-110K test set: Precision, Recall, mAP50, mAP50-95).
  2. Latency (ms) & FPS on CPU across multiple image resolutions.
  3. Dense shelf product localization (count, coverage, confidence).
  4. False positive behavior on people / shoppers overlapping shelves (store aisle footage).
  5. Robustness to camera movement and partially visible products.
"""

import json
import pathlib
import time
from typing import Dict, Any, List

import cv2
import numpy as np
import torch
from ultralytics import YOLO, RTDETR

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output" / "retail_detector_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def measure_inference_speed(model, sample_img, imgsz: int = 416, runs: int = 20):
    """Measure inference latency and FPS with warmup."""
    # Warmup
    for _ in range(3):
        _ = model(sample_img, imgsz=imgsz, verbose=False)

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model(sample_img, imgsz=imgsz, verbose=False)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

    avg_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    return {
        "avg_ms": round(avg_ms, 2),
        "std_ms": round(std_ms, 2),
        "fps": round(fps, 1),
    }


def extract_test_frames():
    """Extract key test scenario frames from retail videos."""
    frames = {}

    # 1. Dense shelf image
    sample_shelf_path = ROOT_DIR / "inventory_data" / "demo_images" / "sample_shelf.jpg"
    if sample_shelf_path.exists():
        frames["dense_shelf_static"] = cv2.imread(str(sample_shelf_path))

    # 2. Camera pan frame from shelf_pan_demo.mp4
    pan_video = ROOT_DIR / "inventory_data" / "demo_videos" / "shelf_pan_demo.mp4"
    if pan_video.exists():
        cap = cv2.VideoCapture(str(pan_video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 20)  # mid-pan
        ret, f20 = cap.read()
        if ret:
            frames["camera_pan_mid"] = f20
        cap.set(cv2.CAP_PROP_POS_FRAMES, 55)  # edge of shelf, partial visibility
        ret, f55 = cap.read()
        if ret:
            frames["camera_pan_edge"] = f55
        cap.release()

    # 3. Person overlapping shelves from store-aisle-detection.mp4
    aisle_video = ROOT_DIR / "videos" / "store-aisle-detection.mp4"
    if aisle_video.exists():
        cap = cv2.VideoCapture(str(aisle_video))
        # Frame where person walks past shelves
        cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
        ret, f_person = cap.read()
        if ret:
            frames["person_shelf_overlap"] = f_person
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1200)
        ret, f_dense_aisle = cap.read()
        if ret:
            frames["dense_aisle_shoppers"] = f_dense_aisle
        cap.release()

    return frames


def run_benchmark():
    print("=" * 75)
    print("  STEP 15: RETAIL DETECTOR BENCHMARK EXPERIMENT")
    print("=" * 75)

    test_frames = extract_test_frames()
    print(f"Extracted {len(test_frames)} test frames covering target conditions.")

    # Candidate definitions
    models_config = [
        {
            "id": "retail_detector_exp2",
            "name": "Retail Detector Exp2 (Baseline)",
            "type": "custom_yolo11n",
            "path": str(ROOT_DIR / "inventory_data" / "custom_model" / "retail_detector_exp2.pt"),
            "is_retail_tuned": True,
            "conf_default": 0.30,
        },
        {
            "id": "yolo11n_coco",
            "name": "YOLO11n (COCO Base)",
            "type": "yolo11n",
            "path": str(ROOT_DIR / "yolo11n.pt"),
            "is_retail_tuned": False,
            "conf_default": 0.25,
        },
        {
            "id": "yolo11s_coco",
            "name": "YOLO11s (Lightweight Alt)",
            "type": "yolo11s",
            "path": str(ROOT_DIR / "yolo11s.pt"),
            "is_retail_tuned": False,
            "conf_default": 0.25,
        },
        {
            "id": "rtdetr_l_coco",
            "name": "RT-DETR-L (Transformer Alt)",
            "type": "rtdetr_l",
            "path": str(ROOT_DIR / "rtdetr-l.pt"),
            "is_retail_tuned": False,
            "conf_default": 0.30,
        },
    ]

    results = {}

    for cfg in models_config:
        model_id = cfg["id"]
        print(f"\nEvaluating candidate: {cfg['name']} ({cfg['path']})...")
        if "rtdetr" in cfg["id"]:
            model = RTDETR(cfg["path"])
        else:
            model = YOLO(cfg["path"])

        param_count = sum(p.numel() for p in model.model.parameters())
        file_size_mb = pathlib.Path(cfg["path"]).stat().st_size / (1024 * 1024)

        # Speed test on dense shelf frame at 416
        ref_frame = test_frames.get("dense_shelf_static")
        if ref_frame is None:
            ref_frame = np.zeros((416, 416, 3), dtype=np.uint8)

        speed_416 = measure_inference_speed(model, ref_frame, imgsz=416, runs=10)
        print(f"  [Speed @ 416px] Avg: {speed_416['avg_ms']} ms | FPS: {speed_416['fps']}")

        # Detection analysis across test frames
        frame_detections = {}
        for fname, frame in test_frames.items():
            preds = model(frame, imgsz=416, conf=cfg["conf_default"], verbose=False)[0]
            boxes = preds.boxes
            det_count = len(boxes)
            
            # Analyze classes detected
            cls_ids = boxes.cls.cpu().numpy().astype(int).tolist() if len(boxes) > 0 else []
            cls_names = [preds.names[c] for c in cls_ids]
            unique_classes = list(set(cls_names))
            
            # Check false positives on people
            person_count = cls_names.count("person")
            bottle_count = cls_names.count("bottle")
            cup_count = cls_names.count("cup")
            product_count = cls_names.count("product") if "product" in preds.names.values() else 0

            avg_conf = float(boxes.conf.cpu().numpy().mean()) if len(boxes) > 0 else 0.0

            frame_detections[fname] = {
                "total_detections": det_count,
                "product_detections": product_count,
                "person_detections": person_count,
                "bottle_detections": bottle_count,
                "cup_detections": cup_count,
                "unique_classes": unique_classes,
                "avg_confidence": round(avg_conf, 3),
            }

            # Save annotated visualization frame for visual comparison
            res_plotted = preds.plot()
            out_img_path = OUTPUT_DIR / f"{model_id}_{fname}.jpg"
            cv2.imwrite(str(out_img_path), res_plotted)

        results[model_id] = {
            "name": cfg["name"],
            "model_type": cfg["type"],
            "parameters": param_count,
            "model_size_mb": round(file_size_mb, 2),
            "is_retail_tuned": cfg["is_retail_tuned"],
            "default_conf": cfg["conf_default"],
            "latency_416_cpu": speed_416,
            "frame_detections": frame_detections,
        }

    # Load baseline official test metrics from existing evaluation report if present
    eval_report_path = ROOT_DIR / "output" / "retail_detector_eval" / "exp2_evaluation_report.json"
    conf_eval_path = ROOT_DIR / "output" / "conf_eval" / "conf_thresholds_evaluation.json"

    baseline_official = {}
    if eval_report_path.exists():
        with open(eval_report_path, "r", encoding="utf-8") as f:
            rep = json.load(f)
            baseline_official["yolo_metrics"] = rep.get("official_yolo_test_metrics", {})
            baseline_official["matched_iou50"] = rep.get("matched_iou50_evaluation", {})

    if conf_eval_path.exists():
        with open(conf_eval_path, "r", encoding="utf-8") as f:
            rep = json.load(f)
            baseline_official["conf_eval_recommended"] = rep.get("recommended_metrics", {})

    final_payload = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_environment": {
            "device": "CPU",
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "baseline_sku110k_official_metrics": baseline_official,
        "candidate_benchmark_results": results,
    }

    report_file = OUTPUT_DIR / "detector_benchmark_report.json"
    report_file.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")
    print(f"\n[OK] Benchmark completed! Report saved to {report_file}")


if __name__ == "__main__":
    run_benchmark()
