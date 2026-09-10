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
    ) -> np.ndarray:
        """Produce a fully annotated shelf frame.

        Args:
            frame: Input BGR image.
            batch: Detection results for this frame.
            state: Temporal state snapshot (None for single-image mode).
            fps: Current processing FPS (0.0 for image mode).
            shelf_roi: Optional shelf ROI rectangle (x1, y1, x2, y2).
            model_tier: Model capability tier label shown on HUD.

        Returns:
            Annotated BGR image (copy of input — input is not modified).
        """
        output = frame.copy()

        if shelf_roi:
            self._draw_shelf_roi(output, shelf_roi)

        for det in batch.product_detections:
            self._draw_box(output, det, is_person=False)

        for det in batch.person_detections:
            self._draw_box(output, det, is_person=True)

        self._draw_hud(output, batch, state, fps, model_tier)
        return output

    # ------------------------------------------------------------------
    # Internal drawing helpers
    # ------------------------------------------------------------------

    def _draw_box(
        self, frame: np.ndarray, det: ShelfDetection, is_person: bool
    ) -> None:
        x1, y1, x2, y2 = det.bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if is_person:
            color = _C_PERSON
        elif det.temporal_state == "POSSIBLY_CHANGING":
            color = (0, 80, 240)  # Orange-red for displacement / activity
        elif det.temporal_state == "UNCERTAIN":
            color = (0, 185, 255)  # Amber for newly appearing / unconfirmed
        elif det.sku_name is not None:
            color = _C_SKU_KNOWN if det.is_known_sku else _C_SKU_UNKNOWN
        else:
            color = _C_PRODUCT

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        self._corner_accents(frame, (x1, y1, x2, y2), color)

        parts: List[str] = []
        if det.track_id is not None:
            parts.append(f"#{det.track_id}")

        if is_person:
            parts.append("PERSON (occlusion)")
        elif det.sku_name is not None:
            if det.is_known_sku:
                score_str = f"{det.sku_confidence * 100:.0f}%" if det.sku_confidence is not None else ""
                parts.append(f"{det.sku_name} ({score_str})" if score_str else det.sku_name)
            else:
                score_str = f" ({det.sku_confidence * 100:.0f}%)" if det.sku_confidence is not None else ""
                parts.append(f"UNKNOWN{score_str}")
        else:
            if self.show_class_names:
                parts.append(det.class_name)
            if self.show_confidence:
                parts.append(f"{det.confidence * 100:.1f}%")

        if not is_person and det.temporal_state:
            parts.append(det.temporal_state)

        self._label_pill(frame, " | ".join(parts), (x1, y1), color)

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
        scale, thick = 0.45, 1
        (tw, th), bl = cv2.getTextSize(text, _FONT, scale, thick)
        x, y = anchor
        pad = 3
        bg_y1 = max(0, y - th - pad * 2 - bl)
        cv2.rectangle(frame, (x, bg_y1), (x + tw + pad * 2, y), (20, 20, 20), -1)
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
    ) -> None:
        h, w = frame.shape[:2]
        hud_h = 138

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), _C_HUD_BG, -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.line(frame, (0, hud_h), (w, hud_h), _C_ACCENT, 1, cv2.LINE_AA)

        # Title row
        cv2.putText(
            frame, "RETAIL SHELF MONITOR  |  INVENTORY FOUNDATION  [Stage 1]",
            (10, 22), _FONT, 0.50, _C_ACCENT, 1, cv2.LINE_AA,
        )

        # Model tier badge (right-aligned)
        tier_label = f"[{model_tier.upper()}]"
        tier_color = (110, 110, 255) if model_tier == "coco_baseline" else _C_ACCENT
        cv2.putText(frame, tier_label, (w - 220, 22), _FONT, 0.44, tier_color, 1, cv2.LINE_AA)

        # Visible facings (main metric)
        cv2.putText(
            frame,
            f"Visible Facings: {batch.visible_count}   (camera-observable front row only)",
            (10, 48), _FONT, 0.52, _C_TEXT, 1, cv2.LINE_AA,
        )

        # Shelf state
        status_str = state.observation_status.value if state else "N/A"
        status_color = _STATUS_COLORS.get(
            state.observation_status, _C_TEXT
        ) if state else _C_TEXT
        cv2.putText(
            frame, f"Shelf State: {status_str}",
            (10, 72), _FONT, 0.52, status_color, 1, cv2.LINE_AA,
        )

        # Stable facings + occupancy
        if state:
            cv2.putText(
                frame,
                f"Stable Facings: {state.stable_visible_count}     "
                f"Occupancy: {state.shelf_occupancy_pct:.0f}%     "
                f"Window: {state.clear_frame_count}/{state.window_size} clear frames",
                (10, 96), _FONT, 0.44, _C_TEXT, 1, cv2.LINE_AA,
            )

            # Alert row
            if state.state_change_detected:
                cv2.putText(
                    frame, f"! STATE CHANGE: {state.change_description}",
                    (10, 118), _FONT, 0.47, _C_CHANGED, 1, cv2.LINE_AA,
                )
            elif state.uncertainty_reason:
                short_reason = state.uncertainty_reason[:72]
                cv2.putText(
                    frame, f"UNCERTAIN: {short_reason}",
                    (10, 118), _FONT, 0.43, _C_UNCERTAIN, 1, cv2.LINE_AA,
                )

        # Right column: performance + person flag
        if fps > 0:
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 48), _FONT, 0.50, _C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(
            frame, f"Inf: {batch.inference_time_ms:.0f}ms",
            (w - 130, 68), _FONT, 0.44, _C_TEXT, 1, cv2.LINE_AA,
        )
        if batch.person_present:
            cv2.putText(
                frame, "PERSON IN FRAME",
                (w - 210, 92), _FONT, 0.44, _C_PERSON, 1, cv2.LINE_AA,
            )

        # Small-print disclaimer
        cv2.putText(
            frame,
            "* Visible facings = camera-observable product instances only -- NOT physical inventory quantity",
            (10, hud_h - 5), _FONT, 0.37, _C_DISCLAIMER, 1, cv2.LINE_AA,
        )
