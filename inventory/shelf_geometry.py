"""Automatic Shelf Geometry Discovery Module.

Discovers physical shelf boundaries and perspective-aware physical tier polygons
from visual structural cues (horizontal shelf lips, rails, and structural edges)
using classical computer vision with temporal consensus across sampled video frames.

Separates physical shelf geometry from product detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ShelfBoundaryLine:
    """A physical horizontal or perspective-tilted shelf boundary line."""

    line_id: str
    slope: float  # m in y = m * x + c
    intercept: float  # c in y = m * x + c (full frame coordinates)
    y_mean: float  # y at mid-width
    y_norm: float  # y_mean normalized to [0, 1]
    confidence: float  # [0.0, 1.0]
    temporal_support: float  # ratio of sampled frames confirming this boundary
    points: Tuple[Tuple[int, int], Tuple[int, int]]  # ((0, y_left), (w, y_right))

    def evaluate_y(self, x: float) -> float:
        """Evaluate Y coordinate on this boundary line at horizontal coordinate x."""
        return self.slope * x + self.intercept

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_id": self.line_id,
            "slope": round(self.slope, 4),
            "intercept": round(self.intercept, 2),
            "y_mean": round(self.y_mean, 1),
            "y_norm": round(self.y_norm, 3),
            "confidence": round(self.confidence, 3),
            "temporal_support": round(self.temporal_support, 3),
            "points": list(self.points),
        }


@dataclass
class DetectedTierRegion:
    """A physical shelf tier region between two consecutive physical shelf boundaries."""

    tier_id: str
    name: str
    tier_index: int
    polygon: List[Tuple[int, int]]  # 4 vertices: [top-left, top-right, bottom-right, bottom-left]
    roi_norm: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized bounding box
    boundary_top: ShelfBoundaryLine
    boundary_bottom: ShelfBoundaryLine
    height_px_mean: float

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point (x, y) lies vertically between top and bottom boundary lines."""
        y_top = self.boundary_top.evaluate_y(x)
        y_bot = self.boundary_bottom.evaluate_y(x)
        return y_top <= y <= y_bot

    def vertical_overlap_ratio(self, bx1: float, by1: float, bx2: float, by2: float) -> float:
        """Compute vertical overlap fraction of a bounding box with this tier."""
        cx = (bx1 + bx2) / 2.0
        bh = max(1.0, by2 - by1)
        y_top = self.boundary_top.evaluate_y(cx)
        y_bot = self.boundary_bottom.evaluate_y(cx)
        overlap = max(0.0, min(by2, y_bot) - max(by1, y_top))
        return overlap / bh

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier_id": self.tier_id,
            "name": self.name,
            "tier_index": self.tier_index,
            "polygon": [list(pt) for pt in self.polygon],
            "roi_norm": [round(v, 4) for v in self.roi_norm],
            "height_px_mean": round(self.height_px_mean, 1),
            "boundary_top_id": self.boundary_top.line_id,
            "boundary_bottom_id": self.boundary_bottom.line_id,
        }


@dataclass
class ShelfGeometryResult:
    """Complete result of shelf geometry discovery for a video."""

    shelf_id: str
    geometry_source: str  # "AUTO_DISCOVERY" | "CONFIGURED_FALLBACK"
    boundaries: List[ShelfBoundaryLine] = field(default_factory=list)
    tiers: List[DetectedTierRegion] = field(default_factory=list)
    shelf_roi_norm: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    geometry_confidence: float = 0.0
    fallback_reason: Optional[str] = None
    message: str = ""
    sampled_frames_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shelf_id": self.shelf_id,
            "geometry_source": self.geometry_source,
            "geometry_confidence": round(self.geometry_confidence, 3),
            "tier_count": len(self.tiers),
            "boundary_count": len(self.boundaries),
            "fallback_reason": self.fallback_reason,
            "message": self.message,
            "shelf_roi_norm": [round(v, 4) for v in self.shelf_roi_norm],
            "sampled_frames_count": self.sampled_frames_count,
            "boundaries": [b.to_dict() for b in self.boundaries],
            "tiers": [t.to_dict() for t in self.tiers],
        }


class AutomaticShelfGeometryDetector:
    """Discovers physical shelf boundaries and perspective tier polygons from video structure.

    Applies bilateral smoothing, horizontal Sobel gradients, horizontal morphological closing,
    temporal consensus across sampled frames, and perspective-aware line fitting.
    """

    def __init__(
        self,
        min_tier_height_ratio: float = 0.11,
        max_perspective_angle_deg: float = 14.0,
        sample_frames_count: int = 8,
        min_confidence_threshold: float = 0.40,
    ) -> None:
        self.min_tier_height_ratio = min_tier_height_ratio
        self.max_perspective_angle_deg = max_perspective_angle_deg
        self.sample_frames_count = sample_frames_count
        self.min_confidence_threshold = min_confidence_threshold

    def detect_from_video(
        self,
        video_path: str | Path,
        shelf_id: str = "SHELF-01",
        configured_fallback: Optional[Dict[str, Any]] = None,
    ) -> ShelfGeometryResult:
        """Analyze video and discover physical shelf boundaries with temporal consensus."""
        v_str = str(video_path)
        cap = cv2.VideoCapture(v_str)
        if not cap.isOpened():
            return self._fallback(
                shelf_id,
                configured_fallback,
                f"Cannot open video source: {v_str}",
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if total_frames <= 0 or w <= 0 or h <= 0:
            cap.release()
            return self._fallback(
                shelf_id,
                configured_fallback,
                f"Invalid video metadata (frames={total_frames}, w={w}, h={h})",
            )

        # Scale down for efficient real-time analysis while retaining full structural accuracy
        proc_w = min(w, 1280)
        scale = proc_w / float(w)
        proc_h = int(h * scale)

        max_sample_idx = min(total_frames - 1, max(10, int(total_frames * 0.85)))
        sample_indices = np.linspace(0, max_sample_idx, self.sample_frames_count, dtype=int)

        kernel_w = max(18, int(proc_w * 0.035))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))

        frame_edges: List[np.ndarray] = []
        frame_line_sets: List[List[Tuple[float, float, float, float]]] = []
        acc_edges = np.zeros((proc_h, proc_w), dtype=np.float32)

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            small_frame = cv2.resize(frame, (proc_w, proc_h)) if scale < 1.0 else frame
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            # Bilateral filter reduces product texture while maintaining strong shelf borders
            filtered = cv2.bilateralFilter(gray, 7, 50, 50)
            grad_y = cv2.Sobel(filtered, cv2.CV_16S, 0, 1, ksize=3)
            abs_grad = cv2.convertScaleAbs(grad_y)
            morph = cv2.morphologyEx(abs_grad, cv2.MORPH_CLOSE, kernel_h)

            acc_edges += morph.astype(np.float32)
            frame_edges.append(morph)

            # Detect lines in individual frame for temporal consensus
            _, frame_thresh = cv2.threshold(morph, 35, 255, cv2.THRESH_BINARY)
            lines = cv2.HoughLinesP(
                frame_thresh,
                rho=1,
                theta=np.pi / 180,
                threshold=35,
                minLineLength=int(proc_w * 0.16),
                maxLineGap=int(proc_w * 0.05),
            )
            f_lines: List[Tuple[float, float, float, float]] = []
            if lines is not None:
                for l in lines:
                    x1, y1, x2, y2 = l.reshape(-1)
                    dx = x2 - x1
                    dy = y2 - y1
                    if abs(dx) < 1e-4:
                        continue
                    m = dy / float(dx)
                    if abs(np.degrees(np.arctan(m))) <= self.max_perspective_angle_deg:
                        c = y1 - m * x1
                        length = float(np.sqrt(dx * dx + dy * dy))
                        ym = m * (proc_w / 2.0) + c
                        f_lines.append((m, c, ym, length))
            frame_line_sets.append(f_lines)

        cap.release()

        valid_sample_count = len(frame_edges)
        if valid_sample_count < 2:
            return self._fallback(
                shelf_id,
                configured_fallback,
                "Insufficient valid sampled frames for temporal consensus",
            )

        acc_edges = (acc_edges / valid_sample_count).astype(np.uint8)

        # 1. 1D Horizontal Edge Profile across central 80% bay width
        x_start, x_end = int(proc_w * 0.10), int(proc_w * 0.90)
        row_profile = np.mean(acc_edges[:, x_start:x_end], axis=1)
        k_smooth = int(proc_h * 0.016) | 1
        prof_smooth = np.convolve(row_profile, np.ones(k_smooth) / k_smooth, mode="same")

        # 2. Local maxima detection for physical shelf boundary peaks
        min_dist_px = int(proc_h * self.min_tier_height_ratio)
        baseline_thresh = max(14.0, float(np.mean(prof_smooth) * 0.60))

        candidate_peaks: List[Tuple[int, float]] = []
        for y in range(int(proc_h * 0.05), int(proc_h * 0.96)):
            if (
                prof_smooth[y] >= baseline_thresh
                and prof_smooth[y] >= prof_smooth[y - 1]
                and prof_smooth[y] >= prof_smooth[y + 1]
            ):
                if not candidate_peaks or (y - candidate_peaks[-1][0]) >= min_dist_px:
                    candidate_peaks.append((y, float(prof_smooth[y])))
                elif prof_smooth[y] > candidate_peaks[-1][1]:
                    candidate_peaks[-1] = (y, float(prof_smooth[y]))

        if len(candidate_peaks) < 2:
            return self._fallback(
                shelf_id,
                configured_fallback,
                f"Insufficient physical shelf boundary peaks detected ({len(candidate_peaks)} found)",
            )

        # 3. Perspective line fitting & temporal consensus for each peak
        band_px = int(proc_h * 0.05)
        max_slope = np.tan(np.radians(self.max_perspective_angle_deg))
        boundary_lines: List[ShelfBoundaryLine] = []

        for b_idx, (y_pk, pk_val) in enumerate(candidate_peaks):
            # Evaluate temporal support across individual sampled frames
            confirming_frames = 0
            slopes_all: List[float] = []
            intercepts_all: List[float] = []
            weights_all: List[float] = []

            for f_lines in frame_line_sets:
                matching_in_frame = [
                    l for l in f_lines if abs(l[2] - y_pk) <= band_px and abs(l[0]) <= max_slope
                ]
                if matching_in_frame:
                    confirming_frames += 1
                    for m, c, ym, length in matching_in_frame:
                        slopes_all.append(m)
                        intercepts_all.append(c)
                        weights_all.append(length)

            temporal_ratio = confirming_frames / float(valid_sample_count)

            # Fit line parameters using weighted average of confirming segments
            if slopes_all and sum(weights_all) > 0:
                avg_m = float(np.average(slopes_all, weights=weights_all))
                avg_c_proc = float(np.average(intercepts_all, weights=weights_all))
                weight_score = min(1.0, sum(weights_all) / (proc_w * 1.5))
            else:
                avg_m = 0.0
                avg_c_proc = float(y_pk)
                weight_score = 0.35

            # Map line parameters back to original full frame resolution
            avg_c_full = avg_c_proc / scale
            y_mid_full = avg_m * (w / 2.0) + avg_c_full
            y_left_full = int(avg_c_full)
            y_right_full = int(avg_m * w + avg_c_full)

            # Boundary confidence combines temporal support, edge strength, and line weight
            prominence = min(1.0, pk_val / (np.mean(prof_smooth) * 1.8 + 1e-6))
            bound_conf = float(np.clip(
                0.40 * temporal_ratio + 0.35 * prominence + 0.25 * weight_score,
                0.0,
                1.0,
            ))

            boundary_lines.append(
                ShelfBoundaryLine(
                    line_id=f"BOUND-{b_idx+1:02d}",
                    slope=avg_m,
                    intercept=avg_c_full,
                    y_mean=y_mid_full,
                    y_norm=y_mid_full / float(h),
                    confidence=bound_conf,
                    temporal_support=temporal_ratio,
                    points=((0, y_left_full), (w, y_right_full)),
                )
            )

        # Sort boundaries strictly top to bottom
        boundary_lines.sort(key=lambda b: b.y_mean)

        # Filter out boundary lines with poor temporal consensus (< 0.25) if enough strong lines exist
        strong_boundaries = [b for b in boundary_lines if b.temporal_support >= 0.25]
        if len(strong_boundaries) >= 2:
            boundary_lines = strong_boundaries

        if len(boundary_lines) < 2:
            return self._fallback(
                shelf_id,
                configured_fallback,
                f"Fewer than 2 stable shelf boundaries survived temporal consensus ({len(boundary_lines)} remaining)",
            )

        # 4. Construct Physical Shelf Tiers (Polygons between adjacent boundaries)
        tiers: List[DetectedTierRegion] = []
        for i in range(len(boundary_lines) - 1):
            top_b = boundary_lines[i]
            bot_b = boundary_lines[i + 1]

            # 4-point perspective polygon in full resolution: [TL, TR, BR, BL]
            poly = [
                (0, max(0, int(top_b.intercept))),
                (w, max(0, int(top_b.slope * w + top_b.intercept))),
                (w, min(h, int(bot_b.slope * w + bot_b.intercept))),
                (0, min(h, int(bot_b.intercept))),
            ]

            roi_y1 = max(0.0, min(poly[0][1], poly[1][1]) / float(h))
            roi_y2 = min(1.0, max(poly[2][1], poly[3][1]) / float(h))
            roi_norm = (0.02, roi_y1, 0.98, roi_y2)
            height_mean = abs(bot_b.y_mean - top_b.y_mean)

            tiers.append(
                DetectedTierRegion(
                    tier_id=f"{shelf_id}-TIER-{i+1:02d}",
                    name=f"TIER-{i+1:02d}",
                    tier_index=i + 1,
                    polygon=poly,
                    roi_norm=roi_norm,
                    boundary_top=top_b,
                    boundary_bottom=bot_b,
                    height_px_mean=height_mean,
                )
            )

        overall_roi = (
            0.02,
            max(0.0, boundary_lines[0].y_norm),
            0.98,
            min(1.0, boundary_lines[-1].y_norm),
        )

        overall_confidence = float(np.mean([b.confidence for b in boundary_lines]))

        if overall_confidence < self.min_confidence_threshold:
            return self._fallback(
                shelf_id,
                configured_fallback,
                f"Discovered geometry confidence ({overall_confidence:.2f}) below threshold ({self.min_confidence_threshold:.2f})",
            )

        return ShelfGeometryResult(
            shelf_id=shelf_id,
            geometry_source="AUTO_DISCOVERY",
            boundaries=boundary_lines,
            tiers=tiers,
            shelf_roi_norm=overall_roi,
            geometry_confidence=overall_confidence,
            fallback_reason=None,
            message=f"Discovered {len(boundary_lines)} physical shelf boundaries and {len(tiers)} physical tiers via temporal consensus.",
            sampled_frames_count=valid_sample_count,
        )

    def _fallback(
        self,
        shelf_id: str,
        configured_fallback: Optional[Dict[str, Any]],
        reason: str,
    ) -> ShelfGeometryResult:
        """Transparent fallback to preconfigured shelf geometry when automatic discovery is unviable."""
        logger.warning("[ShelfGeometry] Fallback triggered for %s: %s", shelf_id, reason)
        if not configured_fallback:
            return ShelfGeometryResult(
                shelf_id=shelf_id,
                geometry_source="CONFIGURED_FALLBACK",
                geometry_confidence=0.0,
                fallback_reason=reason,
                message=f"SHELF GEOMETRY FALLBACK: {reason}. No configured fallback available.",
            )

        raw_tiers = configured_fallback.get("tiers", [])
        shelf_roi = tuple(configured_fallback.get("shelf_roi", (0.02, 0.12, 0.98, 0.88)))

        detected_tiers: List[DetectedTierRegion] = []
        boundaries: List[ShelfBoundaryLine] = []

        for idx, t in enumerate(raw_tiers):
            t_roi = tuple(t.get("roi", shelf_roi))
            t_id = t.get("tier_id", f"{shelf_id}-TIER-{idx+1:02d}")
            t_name = t.get("name", f"TIER-{idx+1:02d}")

            b_top = ShelfBoundaryLine(
                line_id=f"BOUND-{idx+1:02d}-TOP",
                slope=0.0,
                intercept=t_roi[1] * 1000.0,
                y_mean=t_roi[1] * 1000.0,
                y_norm=t_roi[1],
                confidence=0.50,
                temporal_support=1.0,
                points=((0, int(t_roi[1] * 1000.0)), (1000, int(t_roi[1] * 1000.0))),
            )
            b_bot = ShelfBoundaryLine(
                line_id=f"BOUND-{idx+1:02d}-BOT",
                slope=0.0,
                intercept=t_roi[3] * 1000.0,
                y_mean=t_roi[3] * 1000.0,
                y_norm=t_roi[3],
                confidence=0.50,
                temporal_support=1.0,
                points=((0, int(t_roi[3] * 1000.0)), (1000, int(t_roi[3] * 1000.0))),
            )

            poly = [
                (int(t_roi[0] * 1000), int(t_roi[1] * 1000)),
                (int(t_roi[2] * 1000), int(t_roi[1] * 1000)),
                (int(t_roi[2] * 1000), int(t_roi[3] * 1000)),
                (int(t_roi[0] * 1000), int(t_roi[3] * 1000)),
            ]

            detected_tiers.append(
                DetectedTierRegion(
                    tier_id=t_id,
                    name=t_name,
                    tier_index=idx + 1,
                    polygon=poly,
                    roi_norm=t_roi,
                    boundary_top=b_top,
                    boundary_bottom=b_bot,
                    height_px_mean=abs(t_roi[3] - t_roi[1]) * 1000.0,
                )
            )

        return ShelfGeometryResult(
            shelf_id=shelf_id,
            geometry_source="CONFIGURED_FALLBACK",
            boundaries=boundaries,
            tiers=detected_tiers,
            shelf_roi_norm=shelf_roi,
            geometry_confidence=0.0,
            fallback_reason=reason,
            message=f"SHELF GEOMETRY FALLBACK: {reason}. Using preconfigured shelf tiers.",
            sampled_frames_count=0,
        )
