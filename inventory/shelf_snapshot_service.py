"""Shelf Snapshot & Product Stock Availability Service (SIH PS 179).

Performs single-frame edge inference on store shelf images or video streams (videos/inventory.mp4),
monitors individual product slot availability, detects whether each product is IN STOCK,
LOW STOCK, or SOLD OUT / FINISHED from the shelf, and generates marked visual evidence.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

from inventory.config import InventoryModelConfig
from inventory.shelf_detector import ShelfDetectionBatch, ShelfProductDetector

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PLANOGRAM = ROOT_DIR / "configs" / "shelf_planogram_config.json"
DEFAULT_MODEL = ROOT_DIR / "inventory_data" / "custom_model" / "retail_detector_exp2.pt"
DEFAULT_VIDEO = ROOT_DIR / "videos" / "inventory2.mp4" if (ROOT_DIR / "videos" / "inventory2.mp4").is_file() else (ROOT_DIR / "videos" / "inventory.mp4")
SNAPSHOT_PUBLIC_DIR = ROOT_DIR / "dashboard" / "public" / "evidence"


class ShelfSnapshotService:
    """Edge inference engine for product-level stock availability."""

    def __init__(
        self,
        planogram_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        conf_threshold: float = 0.18,
    ) -> None:
        self.planogram_path = planogram_path or DEFAULT_PLANOGRAM
        self.model_path = model_path or DEFAULT_MODEL
        self.conf_threshold = conf_threshold

        self.planogram = self._load_planogram()

        model_cfg = InventoryModelConfig(
            model_path=str(self.model_path),
            model_tier="retail_specific",
            device="cpu",
            confidence_threshold=self.conf_threshold,
            iou_threshold=0.40,
            imgsz=1280,
        )
        self.detector = ShelfProductDetector(model_cfg)
        SNAPSHOT_PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

    def _load_planogram(self) -> Dict[str, Any]:
        """Load planogram product slots from disk."""
        if self.planogram_path.is_file():
            try:
                with open(self.planogram_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[ShelfSnapshot] Error reading planogram: {e}")

        return {
            "camera_id": "CAM_CONVENIENCE_AISLE_1",
            "aisle_name": "Main Aisle — Snacks & Packaged Goods",
            "source_video": "videos/inventory.mp4",
            "default_scan_interval_sec": 10,
            "product_slots": [],
        }

    def analyze_frame(
        self,
        frame: np.ndarray,
        scan_id: Optional[str] = None,
        source_name: str = "inventory2.mp4",
        frame_index: int = 0,
    ) -> Dict[str, Any]:
        """Run product-level stock availability analysis on a single frame."""
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        scan_id = scan_id or f"SCAN-{int(time.time())}"

        # 1. Edge YOLO inference on frame
        batch: ShelfDetectionBatch = self.detector.detect(frame, frame_index=frame_index)
        inference_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        product_slots = self.planogram.get("product_slots", [])
        products_data = []
        total_observed = 0
        total_capacity = 0
        total_deficit = 0
        finished_count = 0
        low_count = 0
        in_stock_count = 0
        alerts = []

        annotated_frame = frame.copy()
        is_4k = w >= 2500
        box_thick = 3 if is_4k else 2
        f_scale = 0.50 if is_4k else 0.38
        f_thick = 2 if is_4k else 1

        # ─── Draw bounding boxes on EVERY detected product across the shelf ───
        for idx, det in enumerate(batch.product_detections, 1):
            bx1, by1, bx2, by2 = [int(v) for v in det.bbox]
            cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 220, 110), box_thick, cv2.LINE_AA)
            c_len = min(16 if is_4k else 8, (bx2 - bx1) // 3, (by2 - by1) // 3)
            for ox, oy, dx, dy in [(bx1, by1, 1, 1), (bx2, by1, -1, 1), (bx1, by2, 1, -1), (bx2, by2, -1, -1)]:
                cv2.line(annotated_frame, (ox, oy), (ox + dx * c_len, oy), (0, 255, 180), box_thick, cv2.LINE_AA)
                cv2.line(annotated_frame, (ox, oy), (ox, oy + dy * c_len), (0, 255, 180), box_thick, cv2.LINE_AA)
            label = f"P#{idx:03d} {int(det.confidence * 100)}%"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, f_scale, f_thick)
            pad = 3 if is_4k else 2
            pill_y1 = max(0, by1 - th - pad * 2 - bl)
            cv2.rectangle(annotated_frame, (bx1, pill_y1), (bx1 + tw + pad * 2, by1), (15, 20, 28), -1)
            cv2.putText(annotated_frame, label, (bx1 + pad, by1 - pad - bl), cv2.FONT_HERSHEY_SIMPLEX, f_scale, (0, 255, 180), f_thick, cv2.LINE_AA)

        for det in batch.person_detections:
            bx1, by1, bx2, by2 = [int(v) for v in det.bbox]
            cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 60, 220), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, "PERSON (OCCLUSION)", (bx1 + 4, max(24, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 60, 220), 1, cv2.LINE_AA)

        # Status color mapping (BGR)
        color_map = {
            "IN_STOCK": (0, 220, 110),       # Green
            "LOW_STOCK": (0, 180, 255),      # Amber/Yellow
            "SOLD_OUT": (35, 35, 235),       # Vibrant Red
        }

        # Second pass: map detected bounding boxes to product slots / shelf tiers
        for slot in product_slots:
            sid = slot["slot_id"]
            pname = slot["product_name"]
            cat = slot.get("category", "General")
            loc = slot.get("location", "Shelf Rack")
            x1_pct, y1_pct, x2_pct, y2_pct = slot["zone_bbox_pct"]
            cap = slot["capacity"]
            low_thresh = slot["low_stock_threshold"]

            zx1 = int(x1_pct * w)
            zy1 = int(y1_pct * h)
            zx2 = int(x2_pct * w)
            zy2 = int(y2_pct * h)

            # Find detections belonging to this product slot by center coordinate
            slot_dets = []
            for det in batch.product_detections:
                bx1, by1, bx2, by2 = det.bbox
                cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    slot_dets.append(det)

            count = len(slot_dets)
            occupancy_pct = round((count / max(cap, 1)) * 100.0, 1)
            deficit = max(0, cap - count)

            total_observed += count
            total_capacity += cap
            total_deficit += deficit

            # Determine product status
            if count == 0:
                stock_status = "SOLD_OUT"
                status_label = "SOLD OUT / FINISHED"
                is_finished = True
                finished_count += 1
                severity = "HIGH"
                recommendation = f"URGENT: {pname} is completely SOLD OUT from shelf at {loc}. Restock {cap} units immediately."
            elif count <= low_thresh:
                stock_status = "LOW_STOCK"
                status_label = "LOW STOCK"
                is_finished = False
                low_count += 1
                severity = "MEDIUM"
                recommendation = f"LOW STOCK: Only {count} units of {pname} left at {loc}. Restock {deficit} units."
            else:
                stock_status = "IN_STOCK"
                status_label = "IN STOCK"
                is_finished = False
                in_stock_count += 1
                severity = "NONE"
                recommendation = f"Normal stock. {count} units available at {loc}."

            if stock_status in ("SOLD_OUT", "LOW_STOCK"):
                alerts.append({
                    "alert_id": f"ALT-{sid}-{int(time.time())}",
                    "slot_id": sid,
                    "product_name": pname,
                    "category": cat,
                    "location": loc,
                    "severity": severity,
                    "status": stock_status,
                    "is_finished": is_finished,
                    "message": recommendation,
                    "observed": count,
                    "capacity": cap,
                    "deficit": deficit,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

            theme_color = color_map[stock_status]

            # ─── Visual Marking for Shelf Tier / Slot on Frame ───
            if stock_status == "SOLD_OUT":
                # Mark the EMPTY / SOLD OUT slot with a prominent red hatched box
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (20, 20, 200), -1)
                cv2.addWeighted(overlay, 0.25, annotated_frame, 0.75, 0, annotated_frame)
                cv2.rectangle(annotated_frame, (zx1, zy1), (zx2, zy2), (30, 30, 240), 3)

                tag_y = max(zy1 + (38 if is_4k else 25), 30)
                tag_text = f"[!] SOLD OUT / FINISHED: {pname}"
                (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7 if is_4k else 0.55, 2)
                cv2.rectangle(annotated_frame, (zx1, tag_y - th - 6), (zx1 + tw + 10, tag_y + 6), (15, 15, 25), -1)
                cv2.rectangle(annotated_frame, (zx1, tag_y - th - 6), (zx1 + tw + 10, tag_y + 6), (30, 30, 240), 2)
                cv2.putText(
                    annotated_frame,
                    tag_text,
                    (zx1 + 5, tag_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7 if is_4k else 0.55,
                    (0, 100, 255),
                    2,
                    cv2.LINE_AA,
                )

            elif stock_status == "LOW_STOCK":
                # Subtle amber border
                cv2.rectangle(annotated_frame, (zx1, zy1), (zx2, zy2), (0, 180, 255), 1)
                tag_y = max(zy1 + (32 if is_4k else 22), 24)
                tag_text = f"LOW: {sid} — {count}/{cap} units ({pname})"
                (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6 if is_4k else 0.48, 1)
                cv2.rectangle(annotated_frame, (zx1 + 8, tag_y - th - 4), (zx1 + tw + 16, tag_y + 4), (20, 24, 33), -1)
                cv2.putText(
                    annotated_frame,
                    tag_text,
                    (zx1 + 12, tag_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 if is_4k else 0.48,
                    (0, 180, 255),
                    1,
                    cv2.LINE_AA,
                )

            else:
                # Subtle tier guideline and inventory counter
                cv2.line(annotated_frame, (zx1, zy1), (zx2, zy1), (60, 90, 80), 1, cv2.LINE_AA)
                tag_y = max(zy1 + (26 if is_4k else 18), 20)
                tier_badge = f"{sid}: {count}/{cap} ({occupancy_pct}%) — {pname}"
                (tw, th), _ = cv2.getTextSize(tier_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.55 if is_4k else 0.42, 1)
                cv2.rectangle(annotated_frame, (zx1 + 6, tag_y - th - 3), (zx1 + tw + 12, tag_y + 3), (12, 18, 24), -1)
                cv2.putText(
                    annotated_frame,
                    tier_badge,
                    (zx1 + 9, tag_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 if is_4k else 0.42,
                    (0, 220, 140),
                    1,
                    cv2.LINE_AA,
                )

            products_data.append({
                "slot_id": sid,
                "product_name": pname,
                "category": cat,
                "location": loc,
                "zone_bbox": [zx1, zy1, zx2, zy2],
                "capacity": cap,
                "observed_count": count,
                "deficit": deficit,
                "occupancy_pct": occupancy_pct,
                "status": stock_status,
                "status_label": status_label,
                "is_finished": is_finished,
                "severity": severity,
                "recommendation": recommendation,
            })

        # ─── Top Store Audit HUD ───
        hud_h = 100 if is_4k else 68
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (12, 15, 22), -1)
        cv2.addWeighted(overlay, 0.85, annotated_frame, 0.15, 0, annotated_frame)
        cv2.line(annotated_frame, (0, hud_h), (w, hud_h), (0, 215, 255), 2)

        title_scale = 0.85 if is_4k else 0.62
        stats_scale = 0.68 if is_4k else 0.50
        cv2.putText(
            annotated_frame,
            f"RETAIL INVENTORY AUDIT — {self.planogram.get('aisle_name', 'Main Aisle')} ({source_name})",
            (15, 38 if is_4k else 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            title_scale,
            (0, 215, 255),
            2,
            cv2.LINE_AA,
        )

        hud_stats = (
            f"All Visible Products: {len(batch.product_detections)} facings  |  "
            f"Monitored Tiers: {len(products_data)}  |  "
            f"IN STOCK: {in_stock_count}  |  "
            f"LOW: {low_count}  |  "
            f"SOLD OUT: {finished_count}  |  "
            f"Edge AI: {inference_ms}ms"
        )
        status_hud_color = (0, 0, 255) if finished_count > 0 else ((0, 180, 255) if low_count > 0 else (0, 220, 110))
        cv2.putText(
            annotated_frame,
            hud_stats,
            (15, 78 if is_4k else 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            stats_scale,
            status_hud_color,
            2 if is_4k else 1,
            cv2.LINE_AA,
        )

        # ─── Save Evidence ───
        snapshot_filename = "latest_shelf_snapshot.jpg"
        save_path = SNAPSHOT_PUBLIC_DIR / snapshot_filename
        cv2.imwrite(str(save_path), annotated_frame)

        overall_status = "SOLD_OUT" if finished_count > 0 else ("LOW_STOCK" if low_count > 0 else "HEALTHY")
        overall_occupancy = round((total_observed / max(total_capacity, 1)) * 100.0, 1)

        result_payload = {
            "scan_id": scan_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "aisle_name": self.planogram.get("aisle_name", "Main Aisle"),
            "camera_id": self.planogram.get("camera_id", "CAM_CONVENIENCE_AISLE_1"),
            "source": source_name,
            "frame_index": frame_index,
            "inference_time_ms": inference_ms,
            "total_products_monitored": len(products_data),
            "in_stock_products_count": in_stock_count,
            "low_stock_products_count": low_count,
            "finished_products_count": finished_count,
            "total_observed_facings": total_observed,
            "total_capacity": total_capacity,
            "total_deficit": total_deficit,
            "overall_occupancy_pct": overall_occupancy,
            "overall_status": overall_status,
            "products": products_data,
            "alerts": alerts,
            "snapshot_image_url": f"/evidence/{snapshot_filename}?t={int(time.time())}",
        }

        # Sync JSON fallback
        json_path = SNAPSHOT_PUBLIC_DIR / "latest_shelf_snapshot.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result_payload, f, indent=2)
        except Exception:
            pass

        return result_payload

    def capture_and_analyze_from_video(
        self,
        video_path: Optional[Path] = None,
        frame_offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Sample a frame from videos/inventory.mp4 and run product availability analysis."""
        vpath = video_path or DEFAULT_VIDEO
        if not vpath.is_file():
            # Fallback check
            vpath = ROOT_DIR / "inventory_data" / "demo_videos" / "shelf_pan_demo.mp4"

        cap = cv2.VideoCapture(str(vpath))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100

        target_idx = frame_offset if frame_offset is not None else 0
        target_idx = target_idx % max(total_frames, 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            raise RuntimeError(f"Could not read frame {target_idx} from {vpath}")

        return self.analyze_frame(
            frame,
            source_name=vpath.name,
            frame_index=target_idx,
        )


if __name__ == "__main__":
    service = ShelfSnapshotService()
    print("[+] Running product stock availability test on videos/inventory.mp4...")
    res = service.capture_and_analyze_from_video()
    print("\n--- PRODUCT AVAILABILITY AUDIT ---")
    print(f"Overall Status   : {res['overall_status']}")
    print(f"Total Monitored  : {res['total_products_monitored']} products")
    print(f"IN STOCK         : {res['in_stock_products_count']}")
    print(f"LOW STOCK        : {res['low_stock_products_count']}")
    print(f"SOLD OUT/FINISHED: {res['finished_products_count']}\n")
    for p in res["products"]:
        mark = "[FINISHED]" if p["is_finished"] else f"[{p['status_label']}]"
        print(f"  {mark:<12s} {p['product_name']:<45s} | {p['observed_count']:2d}/{p['capacity']:2d} | Loc: {p['location']}")
        if p["is_finished"]:
            print(f"    -> {p['recommendation']}")
    print(f"\nAnnotated snapshot saved -> {SNAPSHOT_PUBLIC_DIR / 'latest_shelf_snapshot.jpg'}")
