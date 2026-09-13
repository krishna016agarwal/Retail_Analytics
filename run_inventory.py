#!/usr/bin/env python3
"""Retail Shelf Analysis -- Inventory Foundation CLI (Stage 1).

Standalone entry point for the inventory/shelf detection module.
Does NOT import or modify any existing crowd/queue pipeline (src/ or main.py).

Usage examples
--------------
# Demo mode (generates and analyses a synthetic shelf image):
    python run_inventory.py --demo --no-show

# Single shelf image:
    python run_inventory.py --source inventory_data/demo_images/sample_shelf.jpg

# Shelf video:
    python run_inventory.py --source inventory_data/demo_videos/shelf.mp4

# Folder of shelf images:
    python run_inventory.py --source inventory_data/demo_images/

# Custom / retail-specific model (plug-in, no code changes):
    python run_inventory.py \\
        --source inventory_data/demo_images/ \\
        --model inventory_data/custom_model/retail_yolo.pt \\
        --model-tier retail_specific

# Limit temporal window and output dir:
    python run_inventory.py --source shelf.mp4 --temporal-window 20 \\
        --min-stable-frames 10 --output-dir output/my_run

# Headless / CI testing:
    python run_inventory.py --source inventory_data/demo_images/sample_shelf.jpg \\
        --no-show --max-frames 1

Model Tiers
-----------
    coco_baseline     Generic COCO objects (bottle, cup, etc.).
                      NOT retail brand or SKU recognition. Demo baseline only.
    retail_specific   Custom YOLO trained on retail shelf data.
                      Set --model to the .pt weights path.
    sku_recognition   Future: predicts specific brand/SKU labels.
"""

import argparse
import pathlib
import sys


# ---------------------------------------------------------------------------
# Sample shelf image generator (no Pillow required -- pure cv2/numpy)
# ---------------------------------------------------------------------------


def _create_sample_shelf_image(output_path: "pathlib.Path") -> None:
    """Generate a synthetic retail shelf image for demo / pipeline testing.

    Uses only OpenCV and NumPy -- no additional dependencies.
    Simulates a 3-row gondola shelf with labelled product facings.
    Replace with real shelf images for actual product detection experiments.
    """
    import cv2
    import numpy as np

    W, H = 960, 640
    img = np.full((H, W, 3), (210, 210, 210), dtype=np.uint8)

    # Shelf boards
    shelf_ys = [180, 360, 540]
    for sy in shelf_ys:
        cv2.rectangle(img, (40, sy), (W - 40, sy + 18), (150, 95, 55), -1)
        cv2.line(img, (40, sy), (W - 40, sy), (100, 60, 30), 1)

    # Product rows: (shelf_board_y, [(x, w, h, BGR, label), ...])
    product_rows = [
        (shelf_ys[0] - 120, [
            (70,  70, 100, (220, 60,  60),  "PROD-A"),
            (160, 70, 100, (60, 140, 220),  "PROD-B"),
            (250, 70, 100, (60, 200,  80),  "PROD-C"),
            (340, 70, 100, (220, 60,  60),  "PROD-A"),
            (430, 70, 100, (60, 140, 220),  "PROD-B"),
            (520, 70, 100, (80, 220, 200),  "PROD-D"),
            (610, 70, 100, (220, 60,  60),  "PROD-A"),
            (700, 70, 100, (220, 200,  60), "PROD-E"),
            (790, 70, 100, (60, 140, 220),  "PROD-B"),
        ]),
        (shelf_ys[1] - 100, [
            (70,  80, 80, (80,  80, 220),  "PROD-F"),
            (170, 80, 80, (220,160,  60),  "PROD-G"),
            (270, 80, 80, (80,  80, 220),  "PROD-F"),
            (370, 80, 80, (80, 220, 200),  "PROD-D"),
            (470, 80, 80, (220,160,  60),  "PROD-G"),
            (570, 80, 80, (80,  80, 220),  "PROD-F"),
            (670, 80, 80, (220,200,  60),  "PROD-E"),
            (770, 80, 80, (80, 220, 200),  "PROD-D"),
        ]),
        (shelf_ys[2] - 90, [
            ( 70, 90, 70, (200, 80, 180),  "PROD-H"),
            (180, 90, 70, (200, 80, 180),  "PROD-H"),
            (290, 90, 70, ( 80,180,  80),  "PROD-I"),
            (400, 90, 70, ( 80,180,  80),  "PROD-I"),
            (510, 90, 70, (200, 80, 180),  "PROD-H"),
            (620, 90, 70, (180,120,  60),  "PROD-J"),
            (730, 90, 70, (180,120,  60),  "PROD-J"),
            (840, 90, 70, ( 80,180,  80),  "PROD-I"),
        ]),
    ]

    for row_y, products in product_rows:
        for px, pw, ph, color, label in products:
            cv2.rectangle(img, (px, row_y), (px + pw, row_y + ph), color, -1)
            cv2.rectangle(img, (px, row_y), (px + pw, row_y + ph), (40, 40, 40), 1)
            cv2.putText(
                img, label, (px + 5, row_y + ph // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA,
            )
            # Simulated barcode strip
            cv2.rectangle(
                img,
                (px + 5, row_y + ph - 14),
                (px + pw - 5, row_y + ph - 4),
                (50, 50, 50), -1,
            )

    # Header text
    cv2.putText(
        img, "DEMO SHELF  --  SYNTHETIC  (pipeline testing only)",
        (60, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA,
    )
    cv2.putText(
        img, "Replace with real shelf images for actual product detection experiments.",
        (60, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 100, 100), 1, cv2.LINE_AA,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"[Demo] Synthetic shelf image created  ->  {output_path}")


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retail Shelf Analysis -- Inventory Foundation (Stage 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "NOTE: visible_facings is the camera-observable front-row count only.\n"
            "      It is NOT total physical inventory quantity."
        ),
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to shelf image, video file, or folder of images.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help=(
            "YOLO model weights path. "
            "Default: yolo11n.pt (COCO baseline -- generic objects ONLY, not retail SKUs). "
            "Retail model: set to path of custom-trained .pt file."
        ),
    )
    parser.add_argument(
        "--model-tier",
        type=str,
        default="coco_baseline",
        choices=["coco_baseline", "retail_specific", "sku_recognition"],
        help=(
            "Declared model capability tier. "
            "'coco_baseline' = generic COCO objects (demo, NOT SKU recognition). "
            "'retail_specific' = custom retail-trained model. "
            "'sku_recognition' = future."
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (lower for dense shelves).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.40,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device: 'cpu' or '0' for GPU.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Explicit output file path (auto-named in --output-dir if not set).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/shelf_analysis",
        help="Output directory for annotated results and JSON reports.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Run headless (no OpenCV display window).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames/images to process (None = all).",
    )
    parser.add_argument(
        "--temporal-window",
        type=int,
        default=15,
        help="Frames in the temporal stability rolling window.",
    )
    parser.add_argument(
        "--min-stable-frames",
        type=int,
        default=8,
        help="Minimum consecutive stable frames to confirm a shelf-state change.",
    )
    parser.add_argument(
        "--change-threshold",
        type=int,
        default=3,
        help="Minimum visible-count delta to trigger POSSIBLY_CHANGING state.",
    )
    parser.add_argument(
        "--shelf-roi",
        type=str,
        default=None,
        help=(
            "Optional shelf Region of Interest as 'x1,y1,x2,y2' pixel coordinates. "
            "If not set, the entire frame is the shelf area."
        ),
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help=(
            "Detect ALL model classes (bypasses the retail-plausible COCO class filter). "
            "Required when using a custom model with its own class set."
        ),
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate and run on a synthetic demo shelf image (no real images needed).",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=None,
        help="Display scale factor for OpenCV window (e.g., 0.3 or 0.5). Auto-scales by default to fit screen.",
    )
    parser.add_argument(
        "--display-max-h",
        type=int,
        default=700,
        help="Maximum window display height in pixels (default: 700 to fit standard screens).",
    )
    parser.add_argument(
        "--display-max-w",
        type=int,
        default=960,
        help="Maximum window display width in pixels (default: 960).",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Start the Inventory FastAPI REST API backend (port 8001) in background daemon thread for live dashboard telemetry.",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8001,
        help="Port number for Inventory FastAPI server (default: 8001).",
    )
    parser.add_argument(
        "--open-dashboard",
        action="store_true",
        help="Automatically open the web dashboard in your default browser.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    # ---- Resolve source ----
    source = args.source
    if source is None and not args.demo:
        if pathlib.Path("videos/inventory2.mp4").exists():
            source = "videos/inventory2.mp4"
            if args.model == "yolo11n.pt":
                args.model = "inventory_data/custom_model/retail_detector_exp2.pt"
                args.model_tier = "retail_specific"
        elif pathlib.Path("videos/inventory.mp4").exists():
            source = "videos/inventory.mp4"
            if args.model == "yolo11n.pt":
                args.model = "inventory_data/custom_model/retail_detector_exp2.pt"
                args.model_tier = "retail_specific"
        else:
            demo_path = pathlib.Path("inventory_data/demo_images/sample_shelf.jpg")
            if not demo_path.exists():
                _create_sample_shelf_image(demo_path)
            source = str(demo_path)
            print(f"[Demo mode] Source  ->  {source}")
    elif args.demo:
        demo_path = pathlib.Path("inventory_data/demo_images/sample_shelf.jpg")
        if not demo_path.exists():
            _create_sample_shelf_image(demo_path)
        source = str(demo_path)
        print(f"[Demo mode] Source  ->  {source}")

    # Auto-switch to retail_specific if custom retail weights are passed
    if "retail_detector" in args.model or "detect_product_empty_space" in args.model:
        args.model_tier = "retail_specific"

    # ---- Build config ----
    from inventory.config import (
        COCO_RETAIL_CLASS_IDS,
        InventoryConfig,
        InventoryModelConfig,
        InventoryTemporalConfig,
    )

    # For COCO baseline, use the retail-plausible class filter.
    # For custom/retail models, let the model use all its own classes.
    if args.model_tier == "coco_baseline" and not args.all_classes:
        target_classes = list(COCO_RETAIL_CLASS_IDS)
    else:
        target_classes = None

    shelf_roi = None
    if args.shelf_roi:
        try:
            parts = [int(v.strip()) for v in args.shelf_roi.split(",")]
            if len(parts) != 4:
                raise ValueError
            shelf_roi = tuple(parts)
        except ValueError:
            print("[ERROR] --shelf-roi must be 'x1,y1,x2,y2' (four integers).")
            return 1

    # Auto-calibrate resolution and confidence for retail_specific dense shelves
    effective_conf = args.conf
    effective_imgsz = args.imgsz

    if args.model_tier == "retail_specific":
        if args.conf == 0.25:
            # Calibrate threshold for dense retail shelves to detect all visible items
            effective_conf = 0.12
        if args.imgsz == 640:
            # Upgrade resolution to 1280 for wide camera views (e.g. 1650x754)
            effective_imgsz = 1280

    model_cfg = InventoryModelConfig(
        model_path=args.model,
        model_tier=args.model_tier,
        device=args.device,
        confidence_threshold=effective_conf,
        iou_threshold=args.iou,
        imgsz=effective_imgsz,
        target_classes=target_classes,
    )
    temporal_cfg = InventoryTemporalConfig(
        window_size=args.temporal_window,
        min_stable_frames=args.min_stable_frames,
        change_threshold=args.change_threshold,
    )
    config = InventoryConfig(
        model=model_cfg,
        temporal=temporal_cfg,
        shelf_roi=shelf_roi,
        output_dir=args.output_dir,
    )

    # ---- Startup banner ----
    print("=" * 68)
    print("  RETAIL SHELF ANALYSIS  --  INVENTORY FOUNDATION  (Stage 1)")
    print("=" * 68)
    print(f"  Source           : {source}")
    print(f"  Model            : {args.model}")
    print(f"  Model Tier       : {args.model_tier}")
    if args.model_tier == "coco_baseline":
        print("  [!] COCO baseline: detects generic objects (bottle, cup, etc.).")
        print("      This is NOT retail brand or SKU recognition.")
        print("      For retail detection, train a custom model and use --model-tier retail_specific.")
    print(f"  Confidence       : {effective_conf}  (auto-calibrated for dense shelves)")
    print(f"  Inference Size   : {effective_imgsz}px")
    print(f"  Display Modes    : Active Products = GREEN  |  Finished/Empty = RED COLUMN")
    print(f"  Dashboard Sync   : ACTIVE -> dashboard/public/evidence/")
    print(f"  Temporal Window  : {args.temporal_window} frames  |  "
          f"Min Stable: {args.min_stable_frames} frames")
    print(f"  Output Dir       : {args.output_dir}")
    if shelf_roi:
        print(f"  Shelf ROI        : {shelf_roi}")
    print("=" * 68 + "\n")

    # ---- Launch Inventory API in background thread if --api is set ----
    if args.api:
        import threading
        import uvicorn

        def _start_inv_api():
            uvicorn.run(
                "inventory.inventory_api:app",
                host="127.0.0.1",
                port=args.api_port,
                log_level="warning",
            )

        api_thread = threading.Thread(target=_start_inv_api, daemon=True, name="InventoryApiThread")
        api_thread.start()
        print(f"[+] Inventory REST API live at http://127.0.0.1:{args.api_port}/docs")
        print(f"[+] Web Dashboard accessible at http://localhost:3000 (Select 'Retail Inventory' tab)\n")

    if args.open_dashboard:
        import webbrowser
        webbrowser.open("http://localhost:3000")

    # ---- Run ----
    from inventory.shelf_pipeline import ShelfAnalysisPipeline

    pipeline = ShelfAnalysisPipeline(config)
    try:
        pipeline.run(
            source=source,
            output_path=args.output,
            show_display=not args.no_show,
            max_frames=args.max_frames,
            display_scale=args.display_scale,
            display_max_h=args.display_max_h,
            display_max_w=args.display_max_w,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
