"""Train retail-specific product detector using YOLO on SKU-110K staged subsets.

Features:
- Single detection class: product (class 0)
- Configurable epochs, batch size, resolution, device (default: CPU)
- Measures training duration, validation metrics (mAP50, mAP50-95, P, R)
- Automatically saves exported weights to inventory_data/custom_model/
"""

import argparse
import json
import os
import pathlib
import shutil
import sys
import time

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train retail product detector on SKU-110K subset")
    parser.add_argument(
        "--data",
        type=str,
        default="inventory_data/demo_images/SKU110K_fixed/sku110k_exp1.yaml",
        help="Path to dataset YAML",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=416, help="Image resolution for training")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or '0'")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader worker threads")
    parser.add_argument("--project", type=str, default="output/retail_model_train", help="Training log directory")
    parser.add_argument("--name", type=str, default="exp1_verification", help="Run name")
    parser.add_argument(
        "--weights-dest",
        type=str,
        default="inventory_data/custom_model/retail_detector_exp1.pt",
        help="Destination path for best weights",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = pathlib.Path(args.data).resolve()
    if not data_path.exists():
        print(f"[ERROR] Dataset YAML not found: {data_path}")
        return 1

    dest_weights = pathlib.Path(args.weights_dest).resolve()
    dest_weights.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  RETAIL PRODUCT DETECTOR TRAINING — STAGE 2 FOUNDATION")
    print("=" * 72)
    print(f"  Dataset Config   : {data_path}")
    print(f"  Base Model       : yolo11n.pt")
    print(f"  Target Class     : product (single class, ID 0)")
    print(f"  Epochs           : {args.epochs}")
    print(f"  Batch Size       : {args.batch}")
    print(f"  Image Size       : {args.imgsz}")
    print(f"  Device           : {args.device}")
    print(f"  Export Path      : {dest_weights}")
    print("=" * 72 + "\n")

    # Load base YOLO model
    print("[1/3] Loading base model (yolo11n.pt)...")
    model = YOLO("yolo11n.pt")

    # Train
    print("\n[2/3] Starting training run...")
    t0 = time.perf_counter()

    train_results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=True,
        plots=True,
        save=True,
    )

    train_time_sec = time.perf_counter() - t0
    print(f"\n[OK] Training completed in {train_time_sec:.1f}s ({train_time_sec/60:.2f} min).")

    # Locate best weights
    save_dir = getattr(train_results, "save_dir", None)
    if save_dir and pathlib.Path(save_dir).exists():
        run_dir = pathlib.Path(save_dir)
    elif (pathlib.Path("runs/detect") / args.project / args.name).exists():
        run_dir = pathlib.Path("runs/detect") / args.project / args.name
    elif (pathlib.Path(args.project) / args.name).exists():
        run_dir = pathlib.Path(args.project) / args.name
    else:
        run_dir = pathlib.Path("runs/detect") / args.name

    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    source_pt = best_pt if best_pt.exists() else last_pt

    if not source_pt.exists():
        print(f"[ERROR] Trained weights not found in {run_dir / 'weights'}")
        return 1

    shutil.copy2(source_pt, dest_weights)
    file_size_mb = dest_weights.stat().st_size / 1e6
    print(f"[+] Exported weights -> {dest_weights} ({file_size_mb:.2f} MB)")

    # Validate best model
    print("\n[3/3] Evaluating validation metrics...")
    eval_model = YOLO(str(dest_weights))
    val_metrics = eval_model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        verbose=False,
    )

    metrics_summary = {
        "dataset_yaml": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "training_time_sec": round(train_time_sec, 2),
        "model_size_mb": round(file_size_mb, 2),
        "precision": round(float(val_metrics.results_dict.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(val_metrics.results_dict.get("metrics/recall(B)", 0.0)), 4),
        "mAP50": round(float(val_metrics.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50-95": round(float(val_metrics.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
        "weights_path": str(dest_weights),
    }

    metrics_file = run_dir / "train_metrics.json"
    metrics_file.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("  EXPERIMENT 1 TRAINING & VALIDATION SUMMARY")
    print("=" * 72)
    print(f"  Training Time : {train_time_sec:.1f} s ({train_time_sec/60:.2f} min)")
    print(f"  Model Size    : {file_size_mb:.2f} MB")
    print(f"  Precision (B) : {metrics_summary['precision']:.4f}")
    print(f"  Recall (B)    : {metrics_summary['recall']:.4f}")
    print(f"  mAP@50 (B)    : {metrics_summary['mAP50']:.4f}")
    print(f"  mAP@50-95 (B) : {metrics_summary['mAP50-95']:.4f}")
    print(f"  Metrics JSON  : {metrics_file}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
