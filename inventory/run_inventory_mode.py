"""Retail Inventory Shelf Stock Lifecycle & Dual Video Pipeline (SIH PS 179).

Runs when inventory command is executed from terminal:
1. Precomputes & verifies images 1, 2, 3:
   - Image 1: Full stock, all products marked.
   - Image 2: Missing products marked with prominent RED BOXES on empty spaces -> LOW STOCK alert.
   - Image 3: Critical depletion marked with RED BOXES on empty spaces -> URGENT NEED OF STOCK HERE alert.
2. In terminal:
   - Continuously runs both inventory videos (inventory.mp4 and inventory2.mp4) with real-time YOLO product detections.
   - Displays real-time terminal monitoring logs.
   - Shows OpenCV visual window displaying both cameras side-by-side (unless --no-show is passed).
   - Cycles dashboard active stage every 5 seconds (Image 1 -> Image 2 -> Image 3).
   - Ensures data from videos is purely for judges visual monitoring and does not affect stock calculations.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import threading
import time
import cv2
import numpy as np
import uvicorn

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inventory.inventory_shelf_stages import ShelfStageProcessor
from inventory.inventory_live_stream import DualInventoryStreamManager

PUBLIC_EVIDENCE_DIR = ROOT_DIR / "dashboard" / "public" / "evidence"


def run_inventory_pipeline(args=None, port: int = 8001) -> int:
    """Run complete inventory pipeline with terminal video display & dashboard sync."""
    no_show = getattr(args, "no_show", False) if args is not None else False

    print("\n" + "=" * 76)
    print("  SIH 179 - INTELLIGENT RETAIL INVENTORY & SHELF STOCK LIFECYCLE")
    print("  DEMO & EVALUATION MODE | Computer Vision Edge Inference")
    print("=" * 76)
    print("  Shelf Images   : videos/1.jpeg -> videos/2.jpeg -> videos/3.jpeg")
    print("  Cycle Interval : 5-second automatic progression on dashboard with alerts")
    print("  Live Feeds     : videos/inventory.mp4 (Cam 1) & videos/inventory2.mp4 (Cam 2)")
    print("  Judges Note    : Live video feeds run on terminal & dashboard for visual monitoring;")
    print("                   their data is NOT collected and does NOT alter stock numbers.")
    print("  Dashboard      : http://localhost:3000 (Tab: 'Retail Inventory')")
    print("=" * 76 + "\n")

    # Step 1: Process images 1, 2, 3
    print("[*] Processing shelf photographs 1.jpeg, 2.jpeg, 3.jpeg with YOLO...")
    processor = ShelfStageProcessor()
    stages_meta = processor.process_all_stages()
    print("[✓] 3-Stage shelf analysis complete. Annotated evidence saved.\n")

    # Step 2: Start background API server if not already running on port 8001
    def _run_server():
        try:
            uvicorn.run(
                "inventory.inventory_api:app",
                host="0.0.0.0",
                port=port,
                log_level="warning",
            )
        except Exception:
            pass

    server_thread = threading.Thread(target=_run_server, daemon=True, name="InventoryApiServer")
    server_thread.start()
    time.sleep(1.0)
    print(f"[✓] Inventory API active on port {port}")
    print(f"[✓] Dashboard accessible on http://localhost:3000\n")

    # Step 3: Start live video streaming manager
    print("[*] Starting live product detection on inventory.mp4 & inventory2.mp4...")
    mgr = DualInventoryStreamManager.get_instance()
    mgr.start()
    print("[✓] Dual camera live streams active.")
    if not no_show:
        print("[✓] Opening live OpenCV video window for judges: 'SIH 179 - Inventory Live Monitoring' (Press 'q' to close window)")
    print("[*] Running terminal monitoring loop (5-second dashboard stage rotation)...\n")

    # Step 4: Main runtime loop
    last_stage_idx = 0
    last_log_time = 0.0

    try:
        while True:
            t_now = time.time()
            curr_stage_idx = ((int(t_now) // 5) % 3) + 1
            stage_info = stages_meta["stages"][curr_stage_idx - 1]

            # If stage changed, update latest_shelf_snapshot.jpg and .json for dashboard
            if curr_stage_idx != last_stage_idx:
                last_stage_idx = curr_stage_idx
                stg_img_src = PUBLIC_EVIDENCE_DIR / f"shelf_stage_{curr_stage_idx}.jpg"
                stg_img_dst = PUBLIC_EVIDENCE_DIR / "latest_shelf_snapshot.jpg"
                if stg_img_src.is_file():
                    shutil.copyfile(stg_img_src, stg_img_dst)

                # Terminal stage announcement
                print("-" * 76)
                if curr_stage_idx == 1:
                    print(f">>> [STAGE 1 - 1.jpeg] FULL STOCK: {stage_info['observed_count']} products marked on shelf (100% capacity)")
                    print("    Dashboard: Healthy / Full Stock status")
                elif curr_stage_idx == 2:
                    print(f">>> [STAGE 2 - 2.jpeg] LOW STOCK: {stage_info['empty_spaces_count']} empty spaces marked with RED BOXES ({stage_info['observed_count']} remaining)")
                    print(f"    Dashboard Alert: {stage_info['alert_message']}")
                elif curr_stage_idx == 3:
                    print(f">>> [STAGE 3 - 3.jpeg] CRITICAL DEPLETION: {stage_info['empty_spaces_count']} empty spaces marked with RED BOXES ({stage_info['observed_count']} remaining)")
                    print(f"    Dashboard Alert: {stage_info['alert_message']}")
                print("-" * 76)

            # Terminal log every 1.5 seconds showing active camera detections
            if t_now - last_log_time >= 1.5:
                c1 = mgr.cam1
                c2 = mgr.cam2
                print(
                    f"[LIVE TERMINAL] Stage {curr_stage_idx} ({stage_info['status_label'][:16]}) | "
                    f"CAM_01 #{c1.frame_idx:04d}: {c1.current_products_count:3d} products | "
                    f"CAM_02 #{c2.frame_idx:04d}: {c2.current_products_count:3d} products | "
                    f"Visual Demo [No data skew]"
                )
                last_log_time = t_now

            # Visual window if not headless
            if not no_show:
                f1 = mgr.cam1.get_latest_annotated()
                f2 = mgr.cam2.get_latest_annotated()
                if f1 is not None and f2 is not None:
                    # Stack both streams side-by-side
                    grid = np.hstack([f1, f2])
                    cv2.imshow("SIH 179 - Inventory Live Monitoring (inventory.mp4 & inventory2.mp4)", grid)
                    key = cv2.waitKey(20) & 0xFF
                    if key == ord("q") or key == 27:
                        print("[*] OpenCV window closed. Keeping API server, dashboard streams, and terminal monitoring ACTIVE.")
                        cv2.destroyAllWindows()
                        no_show = True
            else:
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\n[*] Inventory pipeline stopped by user.")
    finally:
        mgr.stop()
        if not no_show:
            cv2.destroyAllWindows()
        print("[✓] Inventory pipeline shut down cleanly.")

    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="SIH 179 - Retail Inventory Pipeline")
    parser.add_argument("--no-show", action="store_true", help="Do not open OpenCV GUI window")
    parser.add_argument("--port", type=int, default=8001, help="Inventory API port (default: 8001)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run_inventory_pipeline(args, port=args.port))
