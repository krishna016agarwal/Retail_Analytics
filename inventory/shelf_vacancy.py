"""Generic Shelf Vacancy Detection Engine (Step 29).

Enhancements (Step 29):
1. Visual Vacancy Confidence: Computes a deterministic, bounded [0.0, 1.0] visual vacancy
   confidence derived from temporal persistence, gap significance (width ratio), and
   geometric stability. Explicitly represents confidence in visual observation (NOT stockout probability).
2. Spatial-Temporal Gap Tracking: Tracks individual candidate gaps per tier across frames
   using horizontal coordinates. Prevents jumping/unstable candidates from accumulating persistence.
3. Replenishment Signal Generation: Distinguishes raw vacancy candidates from actionable
   replenishment verification recommendations (active when confirmed, unoccluded, geometrically stable, and confidence >= threshold).
4. Multiple Vacancy Regions: Tracks and reports distinct, independent empty spaces within each physical tier.
5. Preserves all Step 28 foundations: Physical shelf tier partitioning, 2D horizontal occupancy projection,
   shopper occlusion freezing, optical flow camera-motion filtering, and zero SKU/brand assumptions.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class VacancyTemporalState(str, Enum):
    """Temporal lifecycle states for shelf vacancy confirmation."""

    NORMAL = "NORMAL"
    TEMPORARY_VACANCY = "TEMPORARY_VACANCY"
    VACANCY_CONFIRMED = "VACANCY_CONFIRMED"
    RESTORED = "RESTORED"


@dataclass
class VacantRegion:
    """An internal vacant / empty space detected within a physical shelf tier."""

    region_id: str
    row_id: str  # Kept for backward-compatibility (maps to tier_id)
    x1: int
    y1: int
    x2: int
    y2: int
    x1_pct: float
    y1_pct: float
    x2_pct: float
    y2_pct: float
    width_px: float
    width_pct: float
    width_multiple: float
    tier_id: str = ""
    is_occluded_by_person: bool = False
    # Step 29 Signals:
    persistence_frames: int = 1
    geometric_stability: float = 1.0  # [0.0, 1.0]
    vacancy_confidence: float = 0.0  # [0.0, 1.0] Visual Vacancy Confidence
    replenishment_recommended: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "row_id": self.row_id,
            "tier_id": self.tier_id or self.row_id,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "x1_pct": round(self.x1_pct, 4),
            "y1_pct": round(self.y1_pct, 4),
            "x2_pct": round(self.x2_pct, 4),
            "y2_pct": round(self.y2_pct, 4),
            "width_px": round(self.width_px, 1),
            "width_pct": round(self.width_pct, 4),
            "width_multiple": round(self.width_multiple, 2),
            "is_occluded_by_person": self.is_occluded_by_person,
            "persistence_frames": self.persistence_frames,
            "geometric_stability": round(self.geometric_stability, 3),
            "vacancy_confidence": round(self.vacancy_confidence, 3),
            "replenishment_recommended": self.replenishment_recommended,
        }


@dataclass
class ShelfTier:
    """A physical horizontal shelf tier (e.g. Tier 1 Top, Tier 2 Mid, Tier 3 Lower)."""

    tier_id: str
    name: str
    roi: Tuple[float, float, float, float]  # (x1_pct, y1_pct, x2_pct, y2_pct)
    y_min_px: int
    y_max_px: int
    product_count: int
    median_product_width: float
    occupancy_pct: float
    status: str  # "OCCUPIED" | "VACANT" | "UNCERTAIN"
    temporal_state: str = VacancyTemporalState.NORMAL.value
    products: List[Dict[str, Any]] = field(default_factory=list)
    gaps: List[VacantRegion] = field(default_factory=list)
    rejected_pseudo_gaps: List[Dict[str, Any]] = field(default_factory=list)
    polygon: Optional[List[Tuple[int, int]]] = None
    boundary_top: Optional[Any] = None
    boundary_bottom: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier_id": self.tier_id,
            "name": self.name,
            "roi": list(self.roi),
            "polygon": [list(p) for p in self.polygon] if self.polygon else None,
            "y_min_px": self.y_min_px,
            "y_max_px": self.y_max_px,
            "product_count": self.product_count,
            "median_product_width": round(self.median_product_width, 1),
            "occupancy_pct": round(self.occupancy_pct, 1),
            "status": self.status,
            "temporal_state": self.temporal_state,
            "gaps": [g.to_dict() for g in self.gaps],
            "rejected_pseudo_gaps": self.rejected_pseudo_gaps,
        }


# Backward compatibility alias
@dataclass
class ShelfRow:
    """Backward compatibility container mapping to ShelfTier."""

    row_id: str
    y_mean_px: float
    y_min_px: int
    y_max_px: int
    product_count: int
    median_product_width: float
    products: List[Dict[str, Any]] = field(default_factory=list)
    gaps: List[VacantRegion] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_id": self.row_id,
            "y_mean_px": round(self.y_mean_px, 1),
            "y_min_px": self.y_min_px,
            "y_max_px": self.y_max_px,
            "product_count": self.product_count,
            "median_product_width": round(self.median_product_width, 1),
            "gaps": [g.to_dict() for g in self.gaps],
        }


@dataclass
class ShelfVacancySnapshot:
    """Frame-level spatial vacancy and occupancy evaluation."""

    shelf_id: str
    status: str  # "VACANT" | "OCCUPIED" | "UNCERTAIN"
    vacancy_detected: bool
    vacancy_score: float
    occupancy_pct: float
    detected_facings_count: int
    temporal_state: str = VacancyTemporalState.NORMAL.value
    vacant_regions: List[VacantRegion] = field(default_factory=list)
    unoccluded_vacant_regions: List[VacantRegion] = field(default_factory=list)
    rejected_pseudo_gaps: List[Dict[str, Any]] = field(default_factory=list)
    tiers: List[ShelfTier] = field(default_factory=list)
    rows: List[ShelfRow] = field(default_factory=list)  # Backward compatibility
    is_occluded: bool = False
    is_camera_moving: bool = False
    uncertainty_reason: str = ""
    geometry_source: str = "AUTO_DISCOVERY"
    geometry_confidence: float = 1.0
    geometry_message: str = ""
    boundary_lines: List[Dict[str, Any]] = field(default_factory=list)
    model_empty_boxes: List[Tuple[Tuple[int, int, int, int], float]] = field(default_factory=list)
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shelf_id": self.shelf_id,
            "status": self.status,
            "vacancy_detected": self.vacancy_detected,
            "vacancy_score": round(self.vacancy_score, 3),
            "occupancy_pct": round(self.occupancy_pct, 1),
            "detected_facings_count": self.detected_facings_count,
            "temporal_state": self.temporal_state,
            "vacant_regions": [r.to_dict() for r in self.vacant_regions],
            "unoccluded_vacant_regions": [r.to_dict() for r in self.unoccluded_vacant_regions],
            "rejected_pseudo_gaps": self.rejected_pseudo_gaps,
            "tiers": [t.to_dict() for t in self.tiers],
            "rows": [r.to_dict() for r in self.rows],
            "is_occluded": self.is_occluded,
            "is_camera_moving": self.is_camera_moving,
            "uncertainty_reason": self.uncertainty_reason,
            "geometry_source": self.geometry_source,
            "geometry_confidence": round(self.geometry_confidence, 3),
            "geometry_message": self.geometry_message,
            "boundary_lines": self.boundary_lines,
            "model_empty_boxes_count": len(self.model_empty_boxes),
            "model_name": self.model_name,
        }


class ShelfVacancyEngine:
    """Physical Tier & 2D Occupancy Shelf Vacancy Detector.

    Applies physical tier segmentation and 2D horizontal occupancy projection
    to robustly reject pseudo-gaps caused by partial detections or height variance.
    """

    def __init__(
        self,
        shelf_id: str = "SHELF-01",
        roi: Optional[Tuple[float, float, float, float]] = (0.02, 0.12, 0.98, 0.88),
        tiers: Optional[List[Dict[str, Any]]] = None,
        geometry_result: Optional[Any] = None,
        min_gap_multiplier: float = 1.75,
        min_absolute_gap_px: int = 45,
        max_normal_spacing_ratio: float = 0.40,
        row_vertical_tolerance_ratio: float = 0.45,
        min_products_per_row: int = 3,
        min_products_per_tier: int = 3,
        ignore_boundary_margin_pct: float = 0.03,
    ) -> None:
        self.shelf_id = shelf_id
        self.geometry_result = geometry_result
        if geometry_result and hasattr(geometry_result, "shelf_roi_norm"):
            self.roi = tuple(geometry_result.shelf_roi_norm)
        else:
            self.roi = roi

        self.min_gap_multiplier = min_gap_multiplier
        self.min_absolute_gap_px = min_absolute_gap_px
        self.max_normal_spacing_ratio = max_normal_spacing_ratio
        self.row_vertical_tolerance_ratio = row_vertical_tolerance_ratio
        self.min_products_per_row = min_products_per_row
        self.min_products_per_tier = min_products_per_tier
        self.ignore_boundary_margin_pct = ignore_boundary_margin_pct

        if geometry_result and hasattr(geometry_result, "tiers") and geometry_result.tiers:
            self.tier_configs = []
            for t in geometry_result.tiers:
                self.tier_configs.append({
                    "tier_id": t.tier_id,
                    "name": t.name,
                    "roi": tuple(t.roi_norm),
                    "polygon": t.polygon,
                    "boundary_top": t.boundary_top,
                    "boundary_bottom": t.boundary_bottom,
                })
        elif tiers:
            self.tier_configs = tiers
        elif roi:
            self.tier_configs = [
                {
                    "tier_id": f"{shelf_id}-TIER-01",
                    "name": "TIER-01",
                    "roi": roi,
                }
            ]
        else:
            self.tier_configs = [
                {
                    "tier_id": f"{shelf_id}-TIER-01",
                    "name": "TIER-01",
                    "roi": (0.0, 0.0, 1.0, 1.0),
                }
            ]

    def analyze_frame(
        self,
        product_boxes: List[Tuple[int, int, int, int]],
        frame_w: int,
        frame_h: int,
        person_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        camera_motion_mag: float = 0.0,
        camera_motion_threshold: float = 6.0,
        model_empty_boxes: Optional[List[Tuple[Tuple[int, int, int, int], float]]] = None,
        model_name: str = "",
    ) -> ShelfVacancySnapshot:
        """Analyze a frame's product detections using 2D physical tier occupancy."""
        if frame_w <= 0 or frame_h <= 0 or not product_boxes:
            return ShelfVacancySnapshot(
                shelf_id=self.shelf_id,
                status="OCCUPIED",
                vacancy_detected=False,
                vacancy_score=0.0,
                occupancy_pct=0.0,
                detected_facings_count=0,
                model_empty_boxes=model_empty_boxes or [],
                model_name=model_name,
            )

        # 1. Resolve Overall Shelf ROI
        rx1 = int((self.roi[0] if self.roi else 0.0) * frame_w)
        ry1 = int((self.roi[1] if self.roi else 0.0) * frame_h)
        rx2 = int((self.roi[2] if self.roi else 1.0) * frame_w)
        ry2 = int((self.roi[3] if self.roi else 1.0) * frame_h)
        shelf_w = max(1, rx2 - rx1)

        # 2. Filter products inside overall shelf ROI
        shelf_products: List[Tuple[int, int, int, int]] = []
        for box in product_boxes:
            bx1, by1, bx2, by2 = box
            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                shelf_products.append(box)

        total_facings = len(shelf_products)
        if total_facings == 0:
            return ShelfVacancySnapshot(
                shelf_id=self.shelf_id,
                status="UNCERTAIN",
                vacancy_detected=False,
                vacancy_score=0.0,
                occupancy_pct=0.0,
                detected_facings_count=0,
                uncertainty_reason="No products detected in shelf ROI",
            )

        # 3. Check Shopper Occlusion & Camera Motion
        persons = person_boxes or []
        is_shelf_occluded = False
        for pbx in persons:
            px1, py1, px2, py2 = pbx
            ix1 = max(rx1, px1)
            iy1 = max(ry1, py1)
            ix2 = min(rx2, px2)
            iy2 = min(ry2, py2)
            if ix2 > ix1 and iy2 > iy1:
                is_shelf_occluded = True
                break

        is_camera_moving = camera_motion_mag > camera_motion_threshold

        # 4. Physical Tier Assignment & 2D Occupancy Analysis
        shelf_tiers: List[ShelfTier] = []
        shelf_rows: List[ShelfRow] = []
        all_vacant_regions: List[VacantRegion] = []
        all_rejected_pseudo_gaps: List[Dict[str, Any]] = []

        total_tier_width_sum = 0.0
        total_tier_occupied_sum = 0.0

        for t_cfg in self.tier_configs:
            t_id = t_cfg["tier_id"]
            t_name = t_cfg.get("name", t_id)
            t_roi = t_cfg.get("roi", self.roi or (0.0, 0.0, 1.0, 1.0))
            t_poly = t_cfg.get("polygon")
            b_top = t_cfg.get("boundary_top")
            b_bot = t_cfg.get("boundary_bottom")

            tx1 = int(t_roi[0] * frame_w)
            ty1 = int(t_roi[1] * frame_h)
            tx2 = int(t_roi[2] * frame_w)
            ty2 = int(t_roi[3] * frame_h)
            tier_w = max(1, tx2 - tx1)

            # Assign product boxes to this physical tier (perspective-aware if boundaries exist):
            tier_products: List[Tuple[int, int, int, int]] = []
            for b in shelf_products:
                bx1, by1, bx2, by2 = b
                b_cx = (bx1 + bx2) / 2.0
                b_cy = (by1 + by2) / 2.0
                b_h = max(1, by2 - by1)

                if b_top is not None and b_bot is not None and hasattr(b_top, "evaluate_y"):
                    y_top = b_top.evaluate_y(b_cx)
                    y_bot = b_bot.evaluate_y(b_cx)
                    v_overlap = max(0.0, min(by2, y_bot) - max(by1, y_top))
                    if (y_top <= b_cy <= y_bot) or (v_overlap / b_h >= 0.30):
                        h_overlap = max(0, min(bx2, tx2) - max(bx1, tx1))
                        if h_overlap > 0:
                            tier_products.append(b)
                else:
                    v_overlap = max(0, min(by2, ty2) - max(by1, ty1))
                    if (ty1 <= b_cy <= ty2) or (v_overlap / b_h >= 0.30):
                        h_overlap = max(0, min(bx2, tx2) - max(bx1, tx1))
                        if h_overlap > 0:
                            tier_products.append(b)

            if len(tier_products) < self.min_products_per_tier:
                continue

            # Sort products horizontally along the shelf tier
            sorted_x = sorted(tier_products, key=lambda b: b[0])
            widths = [b[2] - b[0] for b in sorted_x]
            median_w = float(np.median(widths)) if widths else 50.0
            y_min = min(b[1] for b in sorted_x)
            y_max = max(b[3] for b in sorted_x)

            # Thresholds
            gap_threshold = max(self.min_gap_multiplier * median_w, float(self.min_absolute_gap_px))
            spacing_tolerance = self.max_normal_spacing_ratio * median_w
            margin_px = int(self.ignore_boundary_margin_pct * frame_w)

            # --- 2D HORIZONTAL OCCUPANCY PROJECTION & INTERVAL MERGING ---
            merged_intervals: List[List[Any]] = []  # [x1, x2, [boxes]]
            for b in sorted_x:
                bx1, bx2 = b[0], b[2]
                if not merged_intervals:
                    merged_intervals.append([bx1, bx2, [b]])
                else:
                    last = merged_intervals[-1]
                    if bx1 <= last[1] + 3:
                        last[1] = max(last[1], bx2)
                        last[2].append(b)
                    else:
                        merged_intervals.append([bx1, bx2, [b]])

            tier_occupied_px = sum(iv[1] - iv[0] for iv in merged_intervals)
            tier_occupancy_pct = min(100.0, (tier_occupied_px / tier_w) * 100.0)

            total_tier_width_sum += tier_w
            total_tier_occupied_sum += tier_occupied_px

            tier_gaps: List[VacantRegion] = []
            tier_rejected_pseudo: List[Dict[str, Any]] = []

            # Scan internal gaps between merged horizontal occupied intervals
            for j in range(len(merged_intervals) - 1):
                gx1 = merged_intervals[j][1]
                gx2 = merged_intervals[j + 1][0]
                gap_px = gx2 - gx1

                if gap_px <= spacing_tolerance:
                    # Normal product spacing — ignore
                    continue

                # Ignore boundary edges
                if gx1 < tx1 + margin_px or gx2 > tx2 - margin_px:
                    continue

                # Check if candidate gap meets the vacancy threshold
                if gap_px >= gap_threshold:
                    # --- 2D CROSS-ROW OCCUPANCY VALIDATION ---
                    is_occupied_in_overlapping_region = False
                    occupying_box = None

                    for prod in shelf_products:
                        pbx1, pby1, pbx2, pby2 = prod
                        h_inter = max(0, min(gx2, pbx2) - max(gx1, pbx1))
                        v_inter = max(0, min(pby2, ty2) - max(pby1, ty1))
                        if h_inter >= 0.35 * (pbx2 - pbx1) and v_inter > 0:
                            is_occupied_in_overlapping_region = True
                            occupying_box = prod
                            break

                    if is_occupied_in_overlapping_region:
                        rej_record = {
                            "tier_id": t_id,
                            "x1": gx1,
                            "x2": gx2,
                            "y1": y_min,
                            "y2": y_max,
                            "gap_width_px": round(gap_px, 1),
                            "gap_multiple": round(gap_px / max(median_w, 1.0), 2),
                            "reason": "REJECTED — PRODUCT OCCUPANCY EXISTS IN OVERLAPPING VERTICAL REGION",
                            "occupying_box": list(occupying_box) if occupying_box else [],
                        }
                        tier_rejected_pseudo.append(rej_record)
                        all_rejected_pseudo_gaps.append(rej_record)
                        continue

                    # Check shopper occlusion for this specific genuine gap
                    gap_occluded = False
                    for pbx in persons:
                        px1, py1, px2, py2 = pbx
                        x_int = max(0, min(gx2, px2) - max(gx1, px1))
                        y_int = max(0, min(y_max, py2) - max(y_min, py1))
                        if x_int > 0 and y_int > 0:
                            gap_occluded = True
                            break

                    gap_multiple = float(gap_px / max(median_w, 1.0))
                    reg = VacantRegion(
                        region_id=f"GAP-{t_id}-{len(tier_gaps) + 1:02d}",
                        row_id=t_id,
                        tier_id=t_id,
                        x1=gx1,
                        y1=y_min,
                        x2=gx2,
                        y2=y_max,
                        x1_pct=gx1 / frame_w,
                        y1_pct=y_min / frame_h,
                        x2_pct=gx2 / frame_w,
                        y2_pct=y_max / frame_h,
                        width_px=float(gap_px),
                        width_pct=float(gap_px / frame_w),
                        width_multiple=gap_multiple,
                        is_occluded_by_person=gap_occluded,
                    )
                    tier_gaps.append(reg)
                    all_vacant_regions.append(reg)

            tier_status = "VACANT" if any(not g.is_occluded_by_person for g in tier_gaps) else "OCCUPIED"
            if is_shelf_occluded or is_camera_moving:
                tier_status = "UNCERTAIN"

            st_obj = ShelfTier(
                tier_id=t_id,
                name=t_name,
                roi=t_roi,
                polygon=t_poly,
                boundary_top=b_top,
                boundary_bottom=b_bot,
                y_min_px=y_min,
                y_max_px=y_max,
                product_count=len(sorted_x),
                median_product_width=median_w,
                occupancy_pct=tier_occupancy_pct,
                status=tier_status,
                products=[{"bbox": list(b)} for b in sorted_x],
                gaps=tier_gaps,
                rejected_pseudo_gaps=tier_rejected_pseudo,
            )
            shelf_tiers.append(st_obj)

            shelf_rows.append(
                ShelfRow(
                    row_id=t_id,
                    y_mean_px=float((y_min + y_max) / 2.0),
                    y_min_px=y_min,
                    y_max_px=y_max,
                    product_count=len(sorted_x),
                    median_product_width=median_w,
                    products=[{"bbox": list(b)} for b in sorted_x],
                    gaps=tier_gaps,
                )
            )

        # 5. Overall Shelf Metrics
        unoccluded_gaps = [g for g in all_vacant_regions if not g.is_occluded_by_person]
        has_vacancy = len(unoccluded_gaps) > 0

        overall_occupancy_pct = (
            min(100.0, (total_tier_occupied_sum / max(1.0, total_tier_width_sum)) * 100.0)
            if total_tier_width_sum > 0
            else 0.0
        )
        total_gap_width = sum(g.width_px for g in unoccluded_gaps)
        vacancy_score = min(1.0, total_gap_width / max(1.0, total_tier_width_sum))

        if is_shelf_occluded or is_camera_moving:
            status = "UNCERTAIN"
            reason = "Shopper occluding shelf" if is_shelf_occluded else "Significant camera motion"
        elif has_vacancy:
            status = "VACANT"
            reason = f"{len(unoccluded_gaps)} genuine internal vacant gap(s) confirmed by 2D occupancy"
        else:
            status = "OCCUPIED"
            reason = "Products packed across physical tiers without significant vacancies"

        geo_src = self.geometry_result.geometry_source if self.geometry_result else "CONFIGURED_FALLBACK"
        geo_conf = self.geometry_result.geometry_confidence if self.geometry_result else 1.0
        geo_msg = self.geometry_result.message if self.geometry_result else ""
        b_lines = [b.to_dict() for b in self.geometry_result.boundaries] if (self.geometry_result and hasattr(self.geometry_result, "boundaries")) else []

        return ShelfVacancySnapshot(
            shelf_id=self.shelf_id,
            status=status,
            vacancy_detected=has_vacancy,
            vacancy_score=vacancy_score,
            occupancy_pct=overall_occupancy_pct,
            detected_facings_count=total_facings,
            vacant_regions=all_vacant_regions,
            unoccluded_vacant_regions=unoccluded_gaps,
            rejected_pseudo_gaps=all_rejected_pseudo_gaps,
            tiers=shelf_tiers,
            rows=shelf_rows,
            is_occluded=is_shelf_occluded,
            is_camera_moving=is_camera_moving,
            uncertainty_reason=reason,
            geometry_source=geo_src,
            geometry_confidence=geo_conf,
            geometry_message=geo_msg,
            boundary_lines=b_lines,
            model_empty_boxes=model_empty_boxes or [],
            model_name=model_name,
        )


class ShelfVacancyTracker:
    """Temporal & Spatial Gap Tracker with Visual Vacancy Confidence Scoring (Step 29).

    Tracks independent empty regions per tier across consecutive frames.
    Calculates:
    - Persistence frames
    - Geometric stability score (penalizing spatial jitter / jumps in center X and width)
    - Visual Vacancy Confidence (deterministic bounded score [0.0, 1.0])
    - Replenishment recommendation signal (triggered only when confirmed, stable, and confidence >= threshold)

    Guarantees:
    - If a candidate gap jumps to an unrelated coordinate span, persistence resets.
    - If a shopper occludes the region, confidence drops/freezes and status is UNCERTAIN.
    - If camera is moving significantly, confirmation is frozen.
    """

    def __init__(
        self,
        min_consecutive_frames: int = 10,
        recovery_frames: int = 5,
        min_replenishment_confidence: float = 0.70,
        max_center_drift_ratio: float = 0.25,
        stability_history_window: int = 5,
        confidence_persistence_weight: float = 0.40,
        confidence_size_weight: float = 0.30,
        confidence_stability_weight: float = 0.30,
    ) -> None:
        self.min_consecutive_frames = min_consecutive_frames
        self.recovery_frames = recovery_frames
        self.min_replenishment_confidence = min_replenishment_confidence
        self.max_center_drift_ratio = max_center_drift_ratio
        self.stability_history_window = stability_history_window
        self.w_persist = confidence_persistence_weight
        self.w_size = confidence_size_weight
        self.w_stab = confidence_stability_weight

        self._state = VacancyTemporalState.NORMAL
        self._consecutive_vacant_frames = 0
        self._consecutive_occupied_frames = 0
        self._confirmed_alert_active = False

        # Spatial tracks for candidate gaps: dict mapping tier_id -> list of TrackedGap dicts
        self._tier_gap_tracks: Dict[str, List[Dict[str, Any]]] = {}

    @property
    def current_state(self) -> VacancyTemporalState:
        return self._state

    @property
    def is_confirmed(self) -> bool:
        return self._state == VacancyTemporalState.VACANCY_CONFIRMED

    def update(self, snapshot: ShelfVacancySnapshot) -> ShelfVacancySnapshot:
        """Update spatial-temporal tracking, geometric stability, and vacancy confidence."""
        # 1. Occlusion / Camera-Motion Freeze
        if snapshot.is_occluded or snapshot.is_camera_moving:
            snapshot.temporal_state = self._state.value
            for t in snapshot.tiers:
                t.temporal_state = self._state.value
                for g in t.gaps:
                    g.vacancy_confidence = 0.0
                    g.replenishment_recommended = False
            return snapshot

        # 2. Spatial-Temporal Gap Tracking per Physical Tier
        any_confirmed_replenishment = False
        any_active_vacancy_candidate = False

        for tier in snapshot.tiers:
            t_id = tier.tier_id
            active_tracks = self._tier_gap_tracks.setdefault(t_id, [])
            updated_tracks = []
            med_w = max(1.0, tier.median_product_width)

            matched_track_indices = set()

            for g in tier.gaps:
                gx1, gx2 = g.x1, g.x2
                gcx = (gx1 + gx2) / 2.0
                gw = float(gx2 - gx1)

                # Attempt spatial match with an existing track in this tier
                best_idx = None
                best_dist = float("inf")

                for idx, trk in enumerate(active_tracks):
                    if idx in matched_track_indices:
                        continue
                    trk_cx, trk_w = trk["history"][-1]
                    drift_ratio = abs(gcx - trk_cx) / med_w

                    # Calculate horizontal IoU
                    inter = max(0.0, min(gx2, trk["x2"]) - max(gx1, trk["x1"]))
                    union = max(gx2, trk["x2"]) - min(gx1, trk["x1"])
                    iou = inter / max(1.0, union)

                    # Condition: center drift <= max_center_drift_ratio OR horizontal overlap
                    if (drift_ratio <= self.max_center_drift_ratio or iou >= 0.20) and drift_ratio <= 0.45:
                        if drift_ratio < best_dist:
                            best_dist = drift_ratio
                            best_idx = idx

                if best_idx is not None:
                    # MATCH FOUND: Continuation of an existing candidate gap
                    matched_track_indices.add(best_idx)
                    trk = active_tracks[best_idx]
                    trk["history"].append((gcx, gw))
                    if len(trk["history"]) > self.stability_history_window:
                        trk["history"].pop(0)

                    trk["persistence"] += 1
                    trk["x1"] = gx1
                    trk["x2"] = gx2

                    # Calculate Geometric Stability across history
                    history = trk["history"]
                    if len(history) >= 2:
                        cx_shifts = [abs(history[i][0] - history[i - 1][0]) / med_w for i in range(1, len(history))]
                        w_shifts = [abs(history[i][1] - history[i - 1][1]) / max(1.0, history[i - 1][1]) for i in range(1, len(history))]
                        mean_cx_shift = float(np.mean(cx_shifts))
                        mean_w_shift = float(np.mean(w_shifts))
                        # Stability: 1.0 when shifts <= 0.05, decays to 0 as shifts approach 0.25
                        stab_score = max(0.0, min(1.0, 1.0 - 2.5 * mean_cx_shift - 1.0 * mean_w_shift))
                    else:
                        stab_score = 0.85

                    # Calculate Visual Vacancy Confidence Components
                    s_persist = min(1.0, trk["persistence"] / float(self.min_consecutive_frames))
                    s_size = min(1.0, max(0.0, (g.width_multiple - 1.0) / 2.0))
                    s_stab = stab_score

                    confidence = (
                        self.w_persist * s_persist
                        + self.w_size * s_size
                        + self.w_stab * s_stab
                    )
                    confidence = round(max(0.0, min(1.0, confidence)), 3)

                    # Replenishment signal logic
                    is_replenish = (
                        (not g.is_occluded_by_person)
                        and (trk["persistence"] >= self.min_consecutive_frames)
                        and (confidence >= self.min_replenishment_confidence)
                    )

                    g.region_id = trk["track_id"]
                    g.persistence_frames = trk["persistence"]
                    g.geometric_stability = stab_score
                    g.vacancy_confidence = 0.0 if g.is_occluded_by_person else confidence
                    g.replenishment_recommended = is_replenish

                    updated_tracks.append(trk)

                    if not g.is_occluded_by_person:
                        any_active_vacancy_candidate = True
                        if is_replenish:
                            any_confirmed_replenishment = True

                else:
                    # NEW OR JUMPED GAP: Initialize new spatial track
                    track_id = f"GAP-{t_id}-{len(updated_tracks) + 1:02d}"
                    new_trk = {
                        "track_id": track_id,
                        "history": [(gcx, gw)],
                        "persistence": 1,
                        "x1": gx1,
                        "x2": gx2,
                    }
                    updated_tracks.append(new_trk)

                    # Initial frame scores
                    s_persist = 1.0 / float(self.min_consecutive_frames)
                    s_size = min(1.0, max(0.0, (g.width_multiple - 1.0) / 2.0))
                    s_stab = 0.70  # Neutral prior on first observation

                    confidence = (
                        self.w_persist * s_persist
                        + self.w_size * s_size
                        + self.w_stab * s_stab
                    )
                    confidence = round(max(0.0, min(1.0, confidence)), 3)

                    g.region_id = track_id
                    g.persistence_frames = 1
                    g.geometric_stability = s_stab
                    g.vacancy_confidence = 0.0 if g.is_occluded_by_person else confidence
                    g.replenishment_recommended = False

                    if not g.is_occluded_by_person:
                        any_active_vacancy_candidate = True

            self._tier_gap_tracks[t_id] = updated_tracks

        # 3. State Progression across Shelf
        if any_confirmed_replenishment:
            self._consecutive_vacant_frames += 1
            self._consecutive_occupied_frames = 0
            self._state = VacancyTemporalState.VACANCY_CONFIRMED
            self._confirmed_alert_active = True

        elif any_active_vacancy_candidate:
            self._consecutive_vacant_frames += 1
            self._consecutive_occupied_frames = 0
            if self._state == VacancyTemporalState.NORMAL:
                self._state = VacancyTemporalState.TEMPORARY_VACANCY
            elif self._consecutive_vacant_frames >= self.min_consecutive_frames:
                # If persistent across frames, confirm state even if confidence threshold is pending
                self._state = VacancyTemporalState.VACANCY_CONFIRMED
                self._confirmed_alert_active = True

        else:
            # Clear frame without unoccluded vacancies
            self._consecutive_occupied_frames += 1
            self._consecutive_vacant_frames = 0

            if self._state == VacancyTemporalState.TEMPORARY_VACANCY:
                self._state = VacancyTemporalState.NORMAL

            elif (
                self._state == VacancyTemporalState.VACANCY_CONFIRMED
                and self._consecutive_occupied_frames >= self.recovery_frames
            ):
                self._state = VacancyTemporalState.RESTORED
                self._confirmed_alert_active = False

            elif self._state == VacancyTemporalState.RESTORED:
                if self._consecutive_occupied_frames >= self.recovery_frames * 2:
                    self._state = VacancyTemporalState.NORMAL

        snapshot.temporal_state = self._state.value
        for t in snapshot.tiers:
            t.temporal_state = self._state.value

        if self._state == VacancyTemporalState.VACANCY_CONFIRMED:
            snapshot.status = "VACANT"
        elif self._state == VacancyTemporalState.NORMAL and not snapshot.is_occluded:
            snapshot.status = "OCCUPIED"

        return snapshot

    def reset(self) -> None:
        """Reset temporal tracker and active spatial tracks."""
        self._state = VacancyTemporalState.NORMAL
        self._consecutive_vacant_frames = 0
        self._consecutive_occupied_frames = 0
        self._confirmed_alert_active = False
        self._tier_gap_tracks.clear()


def annotate_vacancy_frame(
    frame: np.ndarray,
    snapshot: ShelfVacancySnapshot,
    roi_pct: Optional[Tuple[float, float, float, float]] = None,
    active_alert: bool = False,
    show_rejected_pseudo_gaps: bool = True,
    model_name: Optional[str] = None,
) -> np.ndarray:
    """Render visual overlay of physical shelf tiers, 2D occupancy, and replenishment alerts."""
    import cv2

    vis = frame.copy()
    h, w = vis.shape[:2]

    # 1. Draw Physical Tier ROI boundaries (Perspective Polygon or Cyan Rectangle)
    for t in snapshot.tiers:
        if t.polygon and len(t.polygon) >= 4:
            pts = np.array(t.polygon, dtype=np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=(255, 200, 0), thickness=2)
            tx1, ty1 = t.polygon[0]
        else:
            tx1 = int(t.roi[0] * w)
            ty1 = int(t.roi[1] * h)
            tx2 = int(t.roi[2] * w)
            ty2 = int(t.roi[3] * h)
            cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), (255, 200, 0), 2)

        tier_lbl = f"{t.name} [{t.occupancy_pct:.0f}% Occupied]"
        # Background pill for tier label
        (tw, th), _ = cv2.getTextSize(tier_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        lbl_y1 = max(0, ty1)
        cv2.rectangle(vis, (tx1, lbl_y1), (tx1 + tw + 12, lbl_y1 + 24), (20, 25, 35), -1)
        cv2.rectangle(vis, (tx1, lbl_y1), (tx1 + tw + 12, lbl_y1 + 24), (255, 200, 0), 1)
        cv2.putText(
            vis,
            tier_lbl,
            (tx1 + 6, lbl_y1 + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 220, 50),
            1,
            cv2.LINE_AA,
        )

    if not snapshot.tiers and roi_pct:
        rx1, ry1 = int(roi_pct[0] * w), int(roi_pct[1] * h)
        rx2, ry2 = int(roi_pct[2] * w), int(roi_pct[3] * h)
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 200, 0), 2)

    # 2. Draw detected products in subtle green
    for t in snapshot.tiers:
        for p in t.products:
            bx1, by1, bx2, by2 = p["bbox"]
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 220, 120), 1)

    # 2b. Draw direct model-detected empty spaces
    model_empty = getattr(snapshot, "model_empty_boxes", []) or []
    for item in model_empty:
        if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], (tuple, list)):
            box, conf = item
        elif isinstance(item, (tuple, list)) and len(item) == 4:
            box, conf = item, 0.5
        else:
            continue
        ex1, ey1, ex2, ey2 = box
        # Outline and badge for empty space detection
        cv2.rectangle(vis, (ex1, ey1), (ex2, ey2), (0, 215, 255), 2)
        conf_str = f" ({int(conf * 100)}%)" if conf > 0 else ""
        lbl = f"MODEL EMPTY SPACE{conf_str}"
        (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        by1 = max(0, ey1 - lh - 6)
        cv2.rectangle(vis, (ex1, by1), (ex1 + lw + 8, by1 + lh + 6), (15, 25, 45), -1)
        cv2.rectangle(vis, (ex1, by1), (ex1 + lw + 8, by1 + lh + 6), (0, 215, 255), 1)
        cv2.putText(vis, lbl, (ex1 + 4, by1 + lh + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 235, 255), 1, cv2.LINE_AA)

    # 3. Draw Translucent Overlays for pseudo-gaps, genuine vacancies, and model empty spaces
    overlay = vis.copy()
    if model_empty:
        for item in model_empty:
            box = item[0] if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], (tuple, list)) else item
            if isinstance(box, (tuple, list)) and len(box) == 4:
                ex1, ey1, ex2, ey2 = box
                cv2.rectangle(overlay, (ex1, ey1), (ex2, ey2), (0, 140, 255), -1)
    if show_rejected_pseudo_gaps:
        for r_gap in snapshot.rejected_pseudo_gaps:
            rx1, ry1, rx2, ry2 = r_gap["x1"], r_gap["y1"], r_gap["x2"], r_gap["y2"]
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (180, 50, 180), -1)

    for gap in snapshot.vacant_regions:
        gx1, gy1, gx2, gy2 = gap.x1, gap.y1, gap.x2, gap.y2
        if gap.is_occluded_by_person:
            cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), (180, 130, 70), -1)
        elif gap.replenishment_recommended:
            cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), (0, 40, 240), -1)
        else:
            cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), (0, 140, 255), -1)

    # Blend overlay with 35% opacity
    cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

    # Outline rejected pseudo-gaps
    if show_rejected_pseudo_gaps:
        for r_gap in snapshot.rejected_pseudo_gaps:
            rx1, ry1, rx2, ry2 = r_gap["x1"], r_gap["y1"], r_gap["x2"], r_gap["y2"]
            cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (220, 80, 220), 2)
            lbl = "REJECTED PSEUDO-GAP: PRODUCT OCCUPIED IN OVERLAPPING REGION"
            cv2.putText(vis, lbl, (rx1 + 4, (ry1 + ry2) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 200, 255), 1, cv2.LINE_AA)

    # Outline genuine gaps and render multi-line badge tags
    for gap in snapshot.vacant_regions:
        gx1, gy1, gx2, gy2 = gap.x1, gap.y1, gap.x2, gap.y2
        conf_pct = int(gap.vacancy_confidence * 100)

        if gap.is_occluded_by_person:
            cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (220, 180, 100), 2)
            lines = [
                "SHOPPER OCCLUSION",
                "ANALYSIS UNCERTAIN",
            ]
            box_bg = (30, 40, 55)
            border_c = (220, 180, 100)
            txt_c = (220, 200, 150)
        elif gap.replenishment_recommended:
            cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 50, 255), 3)
            lines = [
                "PERSISTENT VACANCY",
                f"Visual Vacancy Confidence: {conf_pct}%",
                "REPLENISHMENT RECOMMENDED",
            ]
            box_bg = (10, 15, 80)
            border_c = (0, 50, 255)
            txt_c = (255, 255, 255)
        else:
            cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 140, 255), 2)
            lines = [
                "VACANCY CANDIDATE",
                f"Confidence: {conf_pct}%",
                f"Persistence: {gap.persistence_frames} frames",
            ]
            box_bg = (15, 35, 55)
            border_c = (0, 140, 255)
            txt_c = (255, 255, 255)

        # Draw structured multi-line badge inside or above gap
        badge_h = len(lines) * 16 + 8
        badge_w = 220
        by_start = max(10, gy1 - badge_h - 4) if (gy1 - badge_h - 4) > 80 else gy1 + 6
        bx_start = max(10, min(gx1 + 4, w - badge_w - 10))

        cv2.rectangle(vis, (bx_start, by_start), (bx_start + badge_w, by_start + badge_h), box_bg, -1)
        cv2.rectangle(vis, (bx_start, by_start), (bx_start + badge_w, by_start + badge_h), border_c, 1)

        for l_idx, line in enumerate(lines):
            y_pos = by_start + 15 + (l_idx * 16)
            font_scale = 0.38 if l_idx > 0 else 0.42
            font_weight = 1 if l_idx > 0 else 2
            cv2.putText(
                vis,
                line,
                (bx_start + 6, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                txt_c,
                font_weight,
                cv2.LINE_AA,
            )

    # 4. Top Telemetry HUD Header
    hud_h = 75
    hud_bg = vis.copy()
    cv2.rectangle(hud_bg, (0, 0), (w, hud_h), (15, 20, 28), -1)
    cv2.addWeighted(hud_bg, 0.88, vis, 0.12, 0, vis)
    cv2.line(vis, (0, hud_h), (w, hud_h), (60, 70, 85), 1)

    st_color = (0, 220, 120) if snapshot.status == "OCCUPIED" else ((0, 60, 230) if snapshot.status == "VACANT" else (0, 180, 255))
    cv2.circle(vis, (25, 26), 8, st_color, -1)
    cv2.putText(vis, f"{snapshot.shelf_id} STATUS: {snapshot.status}", (42, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

    # Telemetry badge for geometry source
    geo_src = getattr(snapshot, "geometry_source", "AUTO_DISCOVERY")
    geo_conf = getattr(snapshot, "geometry_confidence", 1.0)
    if geo_src == "CONFIGURED_FALLBACK":
        geo_lbl = "SHELF GEOMETRY: CONFIGURED FALLBACK"
        geo_c = (0, 140, 255)
    else:
        geo_lbl = f"SHELF GEOMETRY: AUTO-DETECTED ({int(geo_conf * 100)}% Conf)"
        geo_c = (0, 220, 120)
    (gw, gh), _ = cv2.getTextSize(geo_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    gx = 360
    cv2.rectangle(vis, (gx, 15), (gx + gw + 16, 39), (25, 35, 45), -1)
    cv2.rectangle(vis, (gx, 15), (gx + gw + 16, 39), geo_c, 1)
    cv2.putText(vis, geo_lbl, (gx + 8, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.42, geo_c, 1, cv2.LINE_AA)

    # Telemetry badge for model name
    active_m_name = model_name or getattr(snapshot, "model_name", "")
    if active_m_name:
        short_m = Path(active_m_name).name
        m_lbl = f"MODEL: {short_m}"
        (mw, mh), _ = cv2.getTextSize(m_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        mx = gx + gw + 28
        cv2.rectangle(vis, (mx, 15), (mx + mw + 16, 39), (25, 35, 45), -1)
        cv2.rectangle(vis, (mx, 15), (mx + mw + 16, 39), (0, 215, 255), 1)
        cv2.putText(vis, m_lbl, (mx + 8, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 235, 255), 1, cv2.LINE_AA)

    rej_count = len(snapshot.rejected_pseudo_gaps)
    empty_cnt = len(model_empty)
    empty_str = f"Model Empty Spaces: {empty_cnt}  |  " if empty_cnt > 0 else ""
    sub_txt = (
        f"Temporal: {snapshot.temporal_state}  |  "
        f"Tiers: {len(snapshot.tiers)} monitored  |  "
        f"{empty_str}"
        f"Gaps: {len(snapshot.unoccluded_vacant_regions)} detected, {rej_count} rejected pseudo  |  "
        f"Occupancy: {snapshot.occupancy_pct:.1f}%  |  "
        f"Products: {snapshot.detected_facings_count}"
    )
    cv2.putText(vis, sub_txt, (42, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 200, 215), 1, cv2.LINE_AA)

    # Right side indicator for Uncertainty or Confirmed Alert
    if snapshot.is_camera_moving:
        badge_w = 310
        bx = w - badge_w - 20
        cv2.rectangle(vis, (bx, 14), (bx + badge_w, 60), (20, 45, 80), -1)
        cv2.rectangle(vis, (bx, 14), (bx + badge_w, 60), (0, 180, 255), 2)
        cv2.putText(vis, "CAMERA MOTION", (bx + 12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, "ANALYSIS PAUSED / UNCERTAIN", (bx + 12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 230, 255), 1, cv2.LINE_AA)
    elif snapshot.is_occluded:
        badge_w = 290
        bx = w - badge_w - 20
        cv2.rectangle(vis, (bx, 14), (bx + badge_w, 60), (25, 40, 50), -1)
        cv2.rectangle(vis, (bx, 14), (bx + badge_w, 60), (220, 180, 100), 2)
        cv2.putText(vis, "SHOPPER OCCLUSION", (bx + 12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, "ANALYSIS UNCERTAIN", (bx + 12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 200, 150), 1, cv2.LINE_AA)
    elif active_alert or snapshot.temporal_state == VacancyTemporalState.VACANCY_CONFIRMED.value:
        banner_w = min(480, w - 40)
        bx1 = w - banner_w - 20
        by1 = 12
        by2 = hud_h - 12
        cv2.rectangle(vis, (bx1, by1), (bx1 + banner_w, by2), (10, 20, 150), -1)
        cv2.rectangle(vis, (bx1, by1), (bx1 + banner_w, by2), (0, 60, 255), 2)
        cv2.putText(vis, f"{snapshot.shelf_id} - REPLENISHMENT RECOMMENDED", (bx1 + 14, by1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, "Persistent vacancy confirmed via Visual Vacancy Confidence", (bx1 + 14, by1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 240), 1, cv2.LINE_AA)

    return vis
