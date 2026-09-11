"""Visualization and HUD overlay module for Retail Analytics.

Provides clean rendering of person bounding boxes, persistent tracking IDs,
motion trajectory trails, and a real-time performance HUD.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from configs.config import VisualizerConfig
from src.detector import DetectionBatch
from src.tracker import TrackingBatch, TrajectoryManager


class Visualizer:
    """Visualizes detections, tracks, trajectories, and system metrics on video frames."""

    def __init__(self, config: Optional[VisualizerConfig] = None):
        """Initialize visualizer with display configuration."""
        self.config = config or VisualizerConfig()

    def draw_detections(
        self,
        frame: np.ndarray,
        batch: DetectionBatch,
        fps: float,
        model_name: str = "yolo11n",
    ) -> np.ndarray:
        """Annotate frame in detection-only mode (Phase 1).

        Args:
            frame: Input BGR image.
            batch: DetectionBatch containing detected people.
            fps: Smoothed frames per second.
            model_name: Model name for display on HUD.

        Returns:
            Annotated BGR frame.
        """
        output = frame.copy()
        height, width = output.shape[:2]

        # Draw bounding boxes and labels for each detection
        for det in batch.detections:
            x1, y1, x2, y2 = det.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                self.config.box_color,
                self.config.box_thickness,
                lineType=cv2.LINE_AA,
            )
            self._draw_corner_accents(output, (x1, y1, x2, y2), self.config.hud_accent_color)
            label = f"Person {det.confidence * 100:.1f}%"
            self._draw_label_pill(output, label, (x1, y1))

        # Top HUD in Detection mode
        self._draw_hud(
            output,
            primary_text=f"People Detected: {batch.count}",
            mode_label=f"{model_name.upper()} (CPU)",
            fps=fps,
        )

        return output

    def draw_tracks(
        self,
        frame: np.ndarray,
        batch: TrackingBatch,
        trajectory_manager: Optional[TrajectoryManager] = None,
        draw_trajectories: bool = True,
        fps: float = 0.0,
        model_name: str = "yolo11n",
        tracker_label: str = "BYTETRACK",
        footfall_metrics: Optional[object] = None,
        entrance_line_y: Optional[int] = None,
        entry_direction: str = "down",
        queue_metrics: Optional[object] = None,
    ) -> np.ndarray:
        """Annotate frame in multi-object tracking and footfall analytics mode.

        Draws:
        - Virtual entrance line with direction indicator
        - Queue zone with dynamic congestion color and metrics pill
        - Bounding boxes with corner accents
        - Persistent tracking labels: 'Person {track_id} | {conf}%'
        - Motion trajectory trails
        - Footfall, occupancy, and dwell HUD (IN, OUT, OCC, PEAK, AVG DWELL, MAX DWELL, FPS)

        Args:
            frame: Input BGR image.
            batch: TrackingBatch containing tracked persons.
            trajectory_manager: Optional TrajectoryManager to query spatial trails.
            draw_trajectories: Whether to render trajectory trails.
            fps: Current smoothed FPS.
            model_name: Model name for HUD display.
            tracker_label: Label for the active tracker ('BYTETRACK' / 'BOTSORT').
            footfall_metrics: Optional FootfallMetrics from EntryExitCounter.
            entrance_line_y: Optional Y-coordinate for the virtual entrance line.
            entry_direction: Direction representing entry ('down' or 'up').
            queue_metrics: Optional QueueMetrics from QueueAnalytics.

        Returns:
            Annotated BGR frame.
        """
        output = frame.copy()
        height, width = output.shape[:2]

        # 1. Draw virtual entrance boundary line first (behind bounding boxes)
        if entrance_line_y is not None:
            self.draw_entrance_line(
                output,
                line_y=entrance_line_y,
                entry_direction=entry_direction,
            )

        # 1b. Draw queue zone (behind bounding boxes and trails)
        if queue_metrics is not None:
            self.draw_queue_zone(output, queue_metrics)

        # 2. Draw trajectory trails
        if draw_trajectories and trajectory_manager is not None:
            for track in batch.tracks:
                trail = trajectory_manager.get_recent_trail(track.track_id)
                self._draw_trajectory_trail(output, trail)

        # 3. Draw bounding boxes and persistent ID pills
        for track in batch.tracks:
            x1, y1, x2, y2 = track.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)

            # Primary bounding box
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                self.config.box_color,
                self.config.box_thickness,
                lineType=cv2.LINE_AA,
            )

            # Corner accents
            self._draw_corner_accents(output, (x1, y1, x2, y2), self.config.hud_accent_color)

            # Foot ground contact point marker
            fcx, fcy = track.bottom_center
            cv2.circle(output, (fcx, fcy), 3, (0, 255, 255), -1)

            # Persistent Tracking ID Label Pill (e.g., 'Person 1 | 85.2%')
            label = f"{track.display_id} | {track.confidence * 100:.1f}%"
            self._draw_label_pill(output, label, (x1, y1))

        # 4. Top HUD in Tracking / Analytics mode
        if footfall_metrics is not None:
            hud_info = (
                f"IN: {footfall_metrics.total_entries}  |  "
                f"OUT: {footfall_metrics.total_exits}  |  "
                f"OCC: {footfall_metrics.current_occupancy}  |  "
                f"PEAK: {footfall_metrics.peak_occupancy}  |  "
                f"AVG DWELL: {footfall_metrics.avg_dwell_time:.1f}s  |  "
                f"MAX DWELL: {footfall_metrics.max_dwell_time:.1f}s"
            )
        else:
            hud_info = f"Tracked: {batch.count} | Unique: {batch.total_unique_ids}"

        self._draw_hud(
            output,
            primary_text=hud_info,
            mode_label=f"{tracker_label.upper()} (CPU)",
            fps=fps,
        )

        return output

    def draw_entrance_line(
        self,
        img: np.ndarray,
        line_y: int,
        entry_direction: str = "down",
        line_color: Tuple[int, int, int] = (0, 165, 255),
    ) -> None:
        """Draw virtual entrance boundary line with direction indicators."""
        height, width = img.shape[:2]
        if not (0 <= line_y < height):
            return

        # Draw line with shadow and primary stroke
        cv2.line(img, (0, line_y), (width, line_y), (15, 15, 15), 4, cv2.LINE_AA)
        cv2.line(img, (0, line_y), (width, line_y), line_color, 2, cv2.LINE_AA)

        # Label pill on left side
        dir_hint = "IN: v | OUT: ^" if entry_direction == "down" else "IN: ^ | OUT: v"
        label = f"ENTRANCE LINE (y={line_y}) [{dir_hint}]"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.42
        thick = 1
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)

        pill_y1 = max(0, line_y - th - 8)
        pill_y2 = line_y
        pill_x1 = 16
        pill_x2 = pill_x1 + tw + 12

        # Draw background pill
        cv2.rectangle(img, (pill_x1, pill_y1), (pill_x2, pill_y2), (20, 24, 33), -1)
        cv2.rectangle(img, (pill_x1, pill_y1), (pill_x2, pill_y2), line_color, 1)

        # Text
        cv2.putText(
            img,
            label,
            (pill_x1 + 6, pill_y2 - 5),
            font,
            scale,
            (255, 255, 255),
            thick,
            cv2.LINE_AA,
        )

    def draw_queue_zone(
        self,
        img: np.ndarray,
        queue_metrics: object,
    ) -> None:
        """Draw rectangular queue zone with congestion color coding and metrics pill."""
        x1, y1, x2, y2 = queue_metrics.zone_bbox
        height, width = img.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(width - 1, int(x2)), min(height - 1, int(y2))

        if x2 <= x1 or y2 <= y1:
            return

        # Visual distinction based on congestion level
        level = queue_metrics.congestion_level
        if level == "HIGH":
            zone_color = (50, 50, 240)    # Vibrant Red
        elif level == "MEDIUM":
            zone_color = (0, 190, 255)   # Amber / Orange
        else:
            zone_color = (80, 220, 100)   # Emerald Green

        # Subtle translucent fill (alpha = 0.15)
        sub_overlay = img[y1:y2, x1:x2].copy()
        cv2.rectangle(sub_overlay, (0, 0), (x2 - x1, y2 - y1), zone_color, -1)
        cv2.addWeighted(sub_overlay, 0.15, img[y1:y2, x1:x2], 0.85, 0, img[y1:y2, x1:x2])

        # Zone border rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), zone_color, 2, lineType=cv2.LINE_AA)

        # Corner accent brackets
        accent_len = 16
        cv2.line(img, (x1, y1), (x1 + accent_len, y1), zone_color, 3, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + accent_len), zone_color, 3, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2 - accent_len, y1), zone_color, 3, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + accent_len), zone_color, 3, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1 + accent_len, y2), zone_color, 3, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - accent_len), zone_color, 3, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2 - accent_len, y2), zone_color, 3, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - accent_len), zone_color, 3, cv2.LINE_AA)

        # Label Pill: QUEUE ZONE | QUEUE: X | STATUS: LOW/MEDIUM/HIGH
        label = f"QUEUE ZONE | QUEUE: {queue_metrics.current_queue_length} | STATUS: {level}"
        if level == "HIGH":
            label += f" | {queue_metrics.recommendation}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.42
        thick = 1
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)

        pill_y1 = max(0, y1 - th - 8)
        pill_y2 = y1
        pill_x1 = x1
        pill_x2 = min(width - 1, x1 + tw + 12)

        # Draw dark background pill with colored border
        cv2.rectangle(img, (pill_x1, pill_y1), (pill_x2, pill_y2), (20, 24, 33), -1)
        cv2.rectangle(img, (pill_x1, pill_y1), (pill_x2, pill_y2), zone_color, 1)

        # Text inside pill
        cv2.putText(
            img,
            label,
            (pill_x1 + 6, pill_y2 - 5),
            font,
            scale,
            (255, 255, 255),
            thick,
            cv2.LINE_AA,
        )

    def draw_calibration_overlay(
        self,
        img: np.ndarray,
        is_calibrating: bool = False,
        drag_start: Optional[Tuple[int, int]] = None,
        drag_current: Optional[Tuple[int, int]] = None,
        notification_text: Optional[str] = None,
    ) -> None:
        """Render interactive queue calibration guides, drag rectangle, and status banner."""
        h, w = img.shape[:2]

        # 1. Calibration mode banner
        if is_calibrating:
            banner_text = "CALIBRATION MODE: Click and drag mouse to define Queue Zone | Press 'C' to resume"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.50
            thick = 1
            (bw, bh), _ = cv2.getTextSize(banner_text, font, scale, thick)
            bx1 = max(10, (w - bw) // 2 - 16)
            bx2 = min(w - 10, bx1 + bw + 32)
            by1 = 12
            by2 = by1 + bh + 16

            # Amber translucent pill
            sub = img[by1:by2, bx1:bx2].copy()
            cv2.rectangle(sub, (0, 0), (bx2 - bx1, by2 - by1), (0, 140, 255), -1)
            cv2.addWeighted(sub, 0.85, img[by1:by2, bx1:bx2], 0.15, 0, img[by1:by2, bx1:bx2])
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 220, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(img, banner_text, (bx1 + 16, by2 - 9), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

        # 2. Render active drag preview
        if drag_start is not None and drag_current is not None:
            x1 = min(drag_start[0], drag_current[0])
            y1 = min(drag_start[1], drag_current[1])
            x2 = max(drag_start[0], drag_current[0])
            y2 = max(drag_start[1], drag_current[1])

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            if x2 > x1 and y2 > y1:
                # Translucent fill
                sub = img[y1:y2, x1:x2].copy()
                cv2.rectangle(sub, (0, 0), (x2 - x1, y2 - y1), (0, 255, 255), -1)
                cv2.addWeighted(sub, 0.25, img[y1:y2, x1:x2], 0.75, 0, img[y1:y2, x1:x2])

                # Bright yellow boundary with corner accents
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2, lineType=cv2.LINE_AA)
                self._draw_corner_accents(img, (x1, y1, x2, y2), (0, 255, 255), length=16, thickness=3)

                # Size label
                size_label = f"Queue Zone: {x2 - x1}x{y2 - y1} px ({x1},{y1})-({x2},{y2})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(size_label, font, 0.40, 1)
                py1 = max(0, y1 - th - 6)
                py2 = y1
                px1 = x1
                px2 = min(w - 1, x1 + tw + 10)
                cv2.rectangle(img, (px1, py1), (px2, py2), (20, 24, 33), -1)
                cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 255), 1)
                cv2.putText(img, size_label, (px1 + 5, py2 - 4), font, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Notification pill (e.g., "[✓] Queue zone saved")
        if notification_text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            (nw, nh), _ = cv2.getTextSize(notification_text, font, 0.45, 1)
            nx1 = 16
            ny1 = h - nh - 20
            nx2 = nx1 + nw + 16
            ny2 = ny1 + nh + 12
            cv2.rectangle(img, (nx1, ny1), (nx2, ny2), (20, 40, 20), -1)
            cv2.rectangle(img, (nx1, ny1), (nx2, ny2), (50, 220, 100), 1, lineType=cv2.LINE_AA)
            cv2.putText(img, notification_text, (nx1 + 8, ny2 - 6), font, 0.45, (80, 255, 140), 1, cv2.LINE_AA)

    def _draw_trajectory_trail(
        self,
        img: np.ndarray,
        trail: List[Tuple[int, int]],
    ) -> None:
        """Render a smooth motion trail connecting historical trajectory points."""
        if len(trail) < 2:
            return

        # Draw anti-aliased polyline connecting trail points
        points = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            img,
            [points],
            isClosed=False,
            color=self.config.trail_color,
            thickness=self.config.trail_thickness,
            lineType=cv2.LINE_AA,
        )

        # Small dot at start of trail
        cv2.circle(img, trail[0], 2, (150, 150, 150), -1)

    def _draw_corner_accents(
        self,
        img: np.ndarray,
        bbox: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        length: int = 12,
        thickness: int = 3,
    ) -> None:
        """Draw accent corners on bounding boxes for high visual polish."""
        x1, y1, x2, y2 = bbox
        # Top-Left
        cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness, cv2.LINE_AA)
        # Top-Right
        cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness, cv2.LINE_AA)

    def _draw_label_pill(
        self, img: np.ndarray, text: str, origin: Tuple[int, int]
    ) -> None:
        """Draw filled label background pill with text."""
        x, y = origin
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = self.config.font_scale
        thick = self.config.font_thickness

        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thick)
        pad_x, pad_y = 6, 4

        # Place label above box if space permits, else inside top edge
        box_y1 = max(0, y - text_h - pad_y * 2)
        box_y2 = y
        box_x2 = min(img.shape[1], x + text_w + pad_x * 2)

        if y - text_h - pad_y * 2 < 0:
            box_y1 = y
            box_y2 = y + text_h + pad_y * 2

        # Draw dark background pill
        cv2.rectangle(
            img,
            (x, box_y1),
            (box_x2, box_y2),
            self.config.label_bg_color,
            -1,
        )

        # Draw label text
        text_origin_y = box_y2 - pad_y - baseline // 2
        cv2.putText(
            img,
            text,
            (x + pad_x, text_origin_y),
            font,
            scale,
            self.config.label_text_color,
            thick,
            lineType=cv2.LINE_AA,
        )

    def _draw_hud(
        self,
        img: np.ndarray,
        primary_text: str,
        mode_label: str,
        fps: float,
    ) -> None:
        """Draw modern, translucent 2-row HUD bar displaying metrics and status."""
        height, width = img.shape[:2]
        hud_h = 56

        # Draw translucent background banner
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (width, hud_h), self.config.hud_bg_color, -1)
        cv2.addWeighted(
            overlay,
            self.config.hud_alpha,
            img,
            1.0 - self.config.hud_alpha,
            0,
            img,
        )

        # Bottom accent border for HUD
        cv2.line(
            img,
            (0, hud_h),
            (width, hud_h),
            self.config.hud_accent_color,
            1,
            cv2.LINE_AA,
        )

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Row 1: Brand & Model (Left) + FPS & Quit Prompt (Right)
        brand_text = f"RETAIL ANALYTICS | {mode_label}"
        cv2.putText(
            img,
            brand_text,
            (16, 22),
            font,
            0.42,
            self.config.hud_accent_color,
            1,
            cv2.LINE_AA,
        )

        fps_text = f"FPS: {fps:4.1f}"
        hint_text = "[Q] Quit"

        cv2.putText(
            img,
            fps_text,
            (max(width - 190, 220), 22),
            font,
            0.44,
            (80, 240, 140),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            img,
            hint_text,
            (width - 80, 22),
            font,
            0.40,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        # Row 2: Store Metrics (IN, OUT, OCC, PEAK, AVG DWELL, MAX DWELL)
        base_scale = 0.44
        thick = 1
        (tw, _), _ = cv2.getTextSize(primary_text, font, base_scale, thick)

        # Adaptively scale down if text exceeds frame width
        scale = base_scale
        if tw > (width - 32) and tw > 0:
            scale = base_scale * ((width - 32) / tw)

        cv2.putText(
            img,
            primary_text,
            (16, 46),
            font,
            scale,
            (255, 255, 255),
            thick,
            cv2.LINE_AA,
        )
