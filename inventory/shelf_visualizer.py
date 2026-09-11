"""Shelf annotation and visualisation module.

Completely isolated from src.* — depends only on OpenCV, numpy, and
inventory.* modules. No imports from the crowd/queue pipeline.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from inventory.shelf_detector import ShelfDetection, ShelfDetectionBatch
from inventory.shelf_state import ObservationStatus, ShelfStateSnapshot


# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------
_C_PRODUCT      = (0,  220, 140)     # emerald green  — product box
_C_SKU_KNOWN    = (255, 200,   0)     # cyan           — matched catalog SKU
_C_SKU_UNKNOWN  = (  0, 140, 255)     # orange         — uncataloged / UNKNOWN product
_C_PERSON       = (0,   60, 220)     # red            — person / occlusion warning
_C_UNCERTAIN    = (0,  165, 255)     # amber
_C_CHANGING     = (0,  200, 255)     # orange-amber
_C_CHANGED      = (0,   50, 230)     # red
_C_STABLE       = (0,  220, 140)     # green
_C_ROI          = (200, 200,   0)    # cyan-ish ROI border
_C_HUD_BG       = ( 20,  24,  33)   # dark slate
_C_ACCENT       = (  0, 215, 255)    # cyan accent
_C_TEXT         = (240, 240, 240)    # near-white text
_C_DISCLAIMER   = (160, 160, 160)    # grey small-print

_STATUS_COLORS = {
    ObservationStatus.INITIALIZING:     (160, 160, 160),
    ObservationStatus.STABLE:           _C_STABLE,
    ObservationStatus.UNCERTAIN:        _C_UNCERTAIN,
    ObservationStatus.POSSIBLY_CHANGING: _C_CHANGING,
    ObservationStatus.STATE_CHANGED:    _C_CHANGED,
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


class ShelfVisualizer:
    """Annotates shelf frames with detection boxes, HUD, and state information.

    Does not import anything from src/.
    """

    def __init__(
        self,
        show_confidence: bool = True,
        show_class_names: bool = True,
    ) -> None:
        self.show_confidence = show_confidence
        self.show_class_names = show_class_names

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate_frame(
        self,
        frame: np.ndarray,
        batch: ShelfDetectionBatch,
        state: Optional[ShelfStateSnapshot] = None,
        fps: float = 0.0,
        shelf_roi: Optional[Tuple[int, int, int, int]] = None,
        model_tier: str = "coco_baseline",
        product_slots: Optional[List[dict]] = None,
        slot_counts: Optional[dict] = None,
        finished_slots: Optional[List[dict]] = None,
        active_alert_message: Optional[str] = None,
    ) -> np.ndarray:
        """Produce a fully annotated shelf frame with green product boxes and red finished columns."""
        output = frame.copy()

        # 1. Draw shelf column slots (highlight finished / empty space in red column)
        if product_slots:
            self._draw_shelf_columns(output, product_slots, slot_counts or {}, finished_slots or [])

        # 2. Draw shelf ROI if defined
        if shelf_roi:
            self._draw_shelf_roi(output, shelf_roi)

        # 3. Draw detected products in green boxes
        for det in batch.product_detections:
            self._draw_box(output, det, is_person=False)

        # 4. Draw persons (occlusion signal)
        for det in batch.person_detections:
            self._draw_box(output, det, is_person=True)

        # 5. Draw telemetry HUD and active alert banner
        self._draw_hud(output, batch, state, fps, model_tier, finished_slots, active_alert_message)
        return output

    # ------------------------------------------------------------------
    # Internal drawing helpers
    # ------------------------------------------------------------------

    def _draw_shelf_columns(
        self,
        frame: np.ndarray,
        product_slots: List[dict],
        slot_counts: dict,
        finished_slots: List[dict],
    ) -> None:
        """Draw shelf columns/zones. If a slot is finished/empty, render a prominent RED COLUMN."""
        h, w = frame.shape[:2]
        finished_ids = {s.get("slot_id") for s in finished_slots}

        for slot in product_slots:
            sid = slot.get("slot_id", "")
            pname = slot.get("product_name", sid)
            loc = slot.get("location", "")
            x1_pct, y1_pct, x2_pct, y2_pct = slot.get("zone_bbox_pct", [0, 0, 1, 1])
            cap = slot.get("capacity", 10)
            low_th = slot.get("low_stock_threshold", 3)
            count = slot_counts.get(sid, 0)

            zx1 = max(0, min(w - 1, int(x1_pct * w)))
            zy1 = max(0, min(h - 1, int(y1_pct * h)))
            zx2 = max(zx1 + 2, min(w, int(x2_pct * w)))
            zy2 = max(zy1 + 2, min(h, int(y2_pct * h)))

            is_sold_out = (sid in finished_ids) or (count == 0)

            if is_sold_out:
                # ─── RED COLUMN: Product Finished / Empty Space with blank shelf behind it ───
                overlay = frame.copy()
                # Translucent red wash over the shelf column
                cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (20, 20, 230), -1)
                cv2.addWeighted(overlay, 0.32, frame, 0.68, 0, frame)

                # Outer bold red border
                cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (30, 30, 245), 3, cv2.LINE_AA)

                # Corner accent lines
                c_len = min(28, (zx2 - zx1) // 3, (zy2 - zy1) // 3)
                for ox, oy, dx, dy in [
                    (zx1, zy1, 1, 1),
                    (zx2, zy1, -1, 1),
                    (zx1, zy2, 1, -1),
                    (zx2, zy2, -1, -1),
                ]:
                    cv2.line(frame, (ox, oy), (ox + dx * c_len, oy), (0, 0, 255), 4, cv2.LINE_AA)
                    cv2.line(frame, (ox, oy), (ox, oy + dy * c_len), (0, 0, 255), 4, cv2.LINE_AA)

                # Diagonal hatch marks indicating vacant / empty void
                step = max(35, (zy2 - zy1) // 5)
                for dy in range(zy1 + step, zy2, step):
                    x_start = zx1
                    y_start = dy
                    x_end = min(zx2, zx1 + (zy2 - dy))
                    y_end = min(zy2, dy + (zx2 - zx1))
                    cv2.line(frame, (x_start, y_start), (x_end, y_end), (40, 40, 210), 2, cv2.LINE_AA)

                # Prominent red header label on empty column
                tag_y = max(zy1 + 26, 32)
                tag_text = f"[!] FINISHED / EMPTY SPACE: {pname}"
                (tw, th), bl = cv2.getTextSize(tag_text, _FONT, 0.48, 2)
                cv2.rectangle(frame, (zx1, tag_y - th - 6), (min(w - 1, zx1 + tw + 14), tag_y + 6), (15, 15, 25), -1)
                cv2.rectangle(frame, (zx1, tag_y - th - 6), (min(w - 1, zx1 + tw + 14), tag_y + 6), (30, 30, 245), 2)
                cv2.putText(
                    frame,
                    tag_text,
                    (zx1 + 6, tag_y),
                    _FONT,
                    0.48,
                    (50, 180, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif count <= low_th:
                # Amber dashed zone for low stock
                cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 180, 255), 1, cv2.LINE_AA)
                low_label = f"LOW: {pname[:22]} ({count}/{cap})"
                (tw, th), bl = cv2.getTextSize(low_label, _FONT, 0.40, 1)
                cv2.rectangle(frame, (zx1, zy2 - th - 6), (zx1 + tw + 8, zy2), (18, 22, 32), -1)
                cv2.putText(frame, low_label, (zx1 + 4, zy2 - 4), _FONT, 0.40, (0, 180, 255), 1, cv2.LINE_AA)
            else:
                # Normal in-stock zone: subtle border
                cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 160, 90), 1, cv2.LINE_AA)

    def _draw_box(
        self, frame: np.ndarray, det: ShelfDetection, is_person: bool
    ) -> None:
        x1, y1, x2, y2 = det.bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if is_person:
            color = _C_PERSON
        else:
            # Active products present on shelf are drawn in vibrant GREEN
            color = _C_PRODUCT

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        self._corner_accents(frame, (x1, y1, x2, y2), color)

        parts: List[str] = []
        if det.track_id is not None:
            parts.append(f"#{det.track_id}")

        if is_person:
            parts.append("PERSON (customer)")
        elif det.sku_name is not None:
            if det.is_known_sku:
                score_str = f"{det.sku_confidence * 100:.0f}%" if det.sku_confidence is not None else ""
                parts.append(f"{det.sku_name} ({score_str})" if score_str else det.sku_name)
            else:
                score_str = f" ({det.sku_confidence * 100:.0f}%)" if det.sku_confidence is not None else ""
                parts.append(f"UNKNOWN{score_str}")
        else:
            cname = det.class_name if (det.class_name and det.class_name.lower() != "product") else ""
            if cname:
                parts.append(cname)
            if self.show_confidence:
                parts.append(f"{det.confidence * 100:.0f}%")
            elif not parts:
                parts.append("product")

        label_text = " ".join(parts) if parts else "product"
        self._label_pill(frame, label_text, (x1, y1), color)

    def _corner_accents(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        length: int = 12,
        thickness: int = 2,
    ) -> None:
        x1, y1, x2, y2 = bbox
        for ox, oy, dx, dy in [
            (x1, y1,  1,  1),
            (x2, y1, -1,  1),
            (x1, y2,  1, -1),
            (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (ox, oy), (ox + dx * length, oy), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (ox, oy), (ox, oy + dy * length), color, thickness, cv2.LINE_AA)

    def _label_pill(
        self,
        frame: np.ndarray,
        text: str,
        anchor: Tuple[int, int],
        color: Tuple[int, int, int],
    ) -> None:
        scale, thick = 0.40, 1
        (tw, th), bl = cv2.getTextSize(text, _FONT, scale, thick)
        x, y = anchor
        pad = 2
        bg_y1 = max(0, y - th - pad * 2 - bl)
        cv2.rectangle(frame, (x, bg_y1), (x + tw + pad * 2, y), (18, 22, 32), -1)
        cv2.putText(frame, text, (x + pad, y - pad - bl), _FONT, scale, color, thick, cv2.LINE_AA)

    def _draw_shelf_roi(
        self, frame: np.ndarray, roi: Tuple[int, int, int, int]
    ) -> None:
        x1, y1, x2, y2 = roi
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), _C_ROI, -1)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), _C_ROI, 2, cv2.LINE_AA)
        cv2.putText(frame, "SHELF ROI", (x1 + 4, y1 + 16), _FONT, 0.45, _C_ROI, 1, cv2.LINE_AA)

    def _draw_hud(
        self,
        frame: np.ndarray,
        batch: ShelfDetectionBatch,
        state: Optional[ShelfStateSnapshot],
        fps: float,
        model_tier: str,
        finished_slots: Optional[List[dict]] = None,
        active_alert_message: Optional[str] = None,
    ) -> None:
        h, w = frame.shape[:2]
        hud_h = 138

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), _C_HUD_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.line(frame, (0, hud_h), (w, hud_h), _C_ACCENT, 1, cv2.LINE_AA)

        # Title row
        cv2.putText(
            frame, "RETAIL SHELF MONITOR  |  LIVE INVENTORY & EMPTY SPACE DETECTOR",
            (10, 22), _FONT, 0.50, _C_ACCENT, 1, cv2.LINE_AA,
        )

        # Model tier badge (right-aligned)
        tier_label = f"[{model_tier.upper()}]"
        tier_color = (110, 110, 255) if model_tier == "coco_baseline" else _C_ACCENT
        cv2.putText(frame, tier_label, (w - 220, 22), _FONT, 0.44, tier_color, 1, cv2.LINE_AA)

        # Marked products count (GREEN products)
        cv2.putText(
            frame,
            f"Marked Products: {batch.visible_count} facings  (GREEN: In Stock)",
            (10, 48), _FONT, 0.52, (0, 235, 120), 1, cv2.LINE_AA,
        )

        # Shelf state & slots summary
        n_finished = len(finished_slots) if finished_slots else 0
        if n_finished > 0:
            status_text = f"Shelf Status: {n_finished} PRODUCT(S) FINISHED / EMPTY SPACE"
            status_color = (40, 40, 255)
        else:
            status_text = "Shelf Status: ALL MONITORED SLOTS STABLE"
            status_color = (0, 220, 140)

        cv2.putText(
            frame, status_text,
            (10, 72), _FONT, 0.52, status_color, 1, cv2.LINE_AA,
        )

        # Alert banner row
        if active_alert_message:
            cv2.rectangle(frame, (10, 84), (w - 240, 114), (20, 20, 180), -1)
            cv2.rectangle(frame, (10, 84), (w - 240, 114), (40, 40, 255), 1)
            cv2.putText(
                frame, f"[!] {active_alert_message}",
                (16, 105), _FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )
        elif n_finished > 0:
            p_names = ", ".join(s.get("product_name", "")[:25] for s in finished_slots[:2])
            alert_msg = f"[!] ALERT: Product finished & empty space detected: {p_names} -> Dashboard Alert Sent!"
            cv2.rectangle(frame, (10, 84), (w - 240, 114), (20, 20, 180), -1)
            cv2.rectangle(frame, (10, 84), (w - 240, 114), (40, 40, 255), 1)
            cv2.putText(
                frame, alert_msg,
                (16, 105), _FONT, 0.43, (255, 255, 255), 1, cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                f"Camera observable front-row facings: {batch.visible_count}  |  Dashboard Live Sync: ACTIVE",
                (10, 102), _FONT, 0.44, _C_TEXT, 1, cv2.LINE_AA,
            )

        # Right column: performance + customer flag
        if fps > 0:
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 140, 48), _FONT, 0.50, _C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(
            frame, f"AI: {batch.inference_time_ms:.0f}ms",
            (w - 140, 68), _FONT, 0.44, _C_TEXT, 1, cv2.LINE_AA,
        )
        if batch.person_present:
            cv2.putText(
                frame, "CUSTOMER PRESENT",
                (w - 220, 92), _FONT, 0.44, (0, 140, 255), 1, cv2.LINE_AA,
            )

        # Bottom disclaimer
        cv2.putText(
            frame,
            "Green = Active Product  |  Red Column = Finished / Empty Shelf Space  |  Alerts synced to Dashboard",
            (10, hud_h - 6), _FONT, 0.38, (0, 215, 255), 1, cv2.LINE_AA,
        )

