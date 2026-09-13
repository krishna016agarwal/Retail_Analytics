"""Inventory Alerts and Actionable Insights Module for Retail Shelves.

Consumes SKU inventory aggregation states and inventory lifecycle events to detect
actionable operational conditions:
1. LOW_STOCK: Stable visible facings <= low_stock_threshold.
2. POSSIBLE_STOCKOUT: Previously visible SKU has 0 stable facings for consecutive frames.
   Labeled clearly as POSSIBLE_STOCKOUT / OUT_OF_VIEW_PENDING_VERIFICATION (requires_verification=True).
3. RAPID_REMOVAL: Multiple removals for the same SKU in a temporal window (requires verification for pan artifacts).
4. PRODUCT_MOVEMENT: Sustained product movements/displacements for a SKU.
5. SKU_RECOGNITION_UNCERTAIN: High proportion of visible facings unclassified (UNKNOWN).

Strictly preserves camera-observable front-row visible facing semantics:
- Never claims total physical inventory or warehouse back-stock.
- Never claims definite stockout or definite theft/pick during moving camera sequences.
"""

from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from inventory.catalog import SKUCatalog
from inventory.inventory_aggregator import SKUInventoryStats, SKUShelfStatus
from inventory.inventory_events import InventoryChangeEvent, InventoryEventType


class InventoryAlertType(str, Enum):
    """Categorized inventory alerts."""

    LOW_STOCK = "LOW_STOCK"
    POSSIBLE_STOCKOUT = "POSSIBLE_STOCKOUT"
    RAPID_REMOVAL = "RAPID_REMOVAL"
    PRODUCT_MOVEMENT = "PRODUCT_MOVEMENT"
    SKU_RECOGNITION_UNCERTAIN = "SKU_RECOGNITION_UNCERTAIN"


class AlertSeverity(str, Enum):
    """Alert severity level."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class InventoryAlert:
    """Structured record of an inventory alert."""

    alert_id: int
    frame_index: int
    timestamp_sec: float
    alert_type: InventoryAlertType
    sku_id: Optional[str]
    sku_name: Optional[str]
    severity: AlertSeverity
    current_stable_facings: int
    active_facings: int
    reason: str
    supporting_event_count: int = 1
    requires_verification: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize alert to JSON-compatible dictionary."""
        d = asdict(self)
        d["alert_type"] = self.alert_type.value
        d["severity"] = self.severity.value
        return d


@dataclass
class InventoryAlertConfig:
    """Configurable thresholds for Inventory Alert Detection."""

    low_stock_threshold: int = 2
    stockout_consecutive_frames: int = 10
    rapid_removal_count: int = 3
    rapid_removal_window_frames: int = 20
    movement_alert_count: int = 3
    movement_window_frames: int = 20
    uncertain_sku_ratio_threshold: float = 0.30
    uncertain_sku_min_count: int = 15
    alert_cooldown_frames: int = 15


class InventoryAlertDetector:
    """Detects actionable inventory alerts from SKU aggregation states and events."""

    def __init__(
        self,
        config: Optional[InventoryAlertConfig] = None,
        catalog: Optional[SKUCatalog] = None,
    ) -> None:
        self.config = config or InventoryAlertConfig()
        self.catalog = catalog

        self._alert_counter: int = 0
        self._all_alerts: List[InventoryAlert] = []
        self._active_alerts_current_frame: List[InventoryAlert] = []

        # Tracking state for condition detection
        self._sku_ever_observed: Set[str] = set()
        self._consecutive_zero_stable_frames: Dict[str, int] = {}
        self._sku_panned_out_of_view: Set[str] = set()
        self._removals_history: Deque[Tuple[int, str]] = deque()  # (frame_idx, sku_id)
        self._movements_history: Deque[Tuple[int, str]] = deque()  # (frame_idx, sku_id)

        # Cooldown management: (alert_type, sku_id) -> last_fired_frame
        self._alert_cooldowns: Dict[Tuple[str, Optional[str]], int] = {}

    @property
    def all_alerts(self) -> List[InventoryAlert]:
        return list(self._all_alerts)

    @property
    def active_alerts_current_frame(self) -> List[InventoryAlert]:
        return list(self._active_alerts_current_frame)

    def process_frame(
        self,
        sku_stats: Dict[str, SKUInventoryStats],
        frame_events: List[InventoryChangeEvent],
        frame_index: int,
        timestamp_sec: float = 0.0,
        person_present: bool = False,
        person_occluding_shelf: bool = False,
    ) -> List[InventoryAlert]:
        """Evaluate inventory alerts for the current frame observation."""
        self._active_alerts_current_frame = []
        new_alerts: List[InventoryAlert] = []
        active_occlusion = person_present or person_occluding_shelf

        # Record incoming events into sliding history buffers
        for ev in frame_events:
            if ev.event_type == InventoryEventType.PRODUCT_REMOVED and ev.sku_id:
                self._removals_history.append((frame_index, ev.sku_id))
            elif ev.event_type == InventoryEventType.PRODUCT_MOVED and ev.sku_id:
                self._movements_history.append((frame_index, ev.sku_id))
            elif ev.event_type == InventoryEventType.OUT_OF_VIEW_PAN_EXIT and ev.sku_id:
                self._sku_panned_out_of_view.add(ev.sku_id)

        # Evict events outside rolling windows
        while (
            self._removals_history
            and self._removals_history[0][0] < frame_index - self.config.rapid_removal_window_frames
        ):
            self._removals_history.popleft()

        while (
            self._movements_history
            and self._movements_history[0][0] < frame_index - self.config.movement_window_frames
        ):
            self._movements_history.popleft()

        # ----------------------------------------------------------------------
        # 1. Evaluate Per-SKU Alerts (LOW_STOCK, POSSIBLE_STOCKOUT, RAPID_REMOVAL, MOVEMENT)
        # ----------------------------------------------------------------------
        for sku_id, stat in sku_stats.items():
            if sku_id == "UNKNOWN":
                continue

            # Track if SKU has ever been seen in camera view
            if stat.current_visible_facings > 0:
                self._sku_ever_observed.add(sku_id)

            # --- Condition A: LOW_STOCK ---
            if (
                stat.status == SKUShelfStatus.LOW_STOCK
                and 1 <= stat.stable_facings <= self.config.low_stock_threshold
                and not active_occlusion
            ):
                alert = self._create_alert_if_eligible(
                    alert_type=InventoryAlertType.LOW_STOCK,
                    sku_id=sku_id,
                    sku_name=stat.sku_name,
                    severity=AlertSeverity.HIGH,
                    stable_facings=stat.stable_facings,
                    active_facings=stat.current_visible_facings,
                    reason=(
                        f"Visible facings ({stat.stable_facings}) at or below low-stock threshold "
                        f"({self.config.low_stock_threshold})"
                    ),
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    supporting_count=stat.stable_facings,
                    requires_verification=False,
                )
                if alert:
                    new_alerts.append(alert)

            # --- Condition B: POSSIBLE_STOCKOUT (Camera Out-of-View vs Stockout) ---
            # Suppressed while shoppers are actively blocking the shelf or when SKU panned out of view
            if sku_id in self._sku_ever_observed:
                if stat.stable_facings == 0:
                    is_panned_out = sku_id in self._sku_panned_out_of_view
                    if not active_occlusion and not is_panned_out:
                        self._consecutive_zero_stable_frames[sku_id] = (
                            self._consecutive_zero_stable_frames.get(sku_id, 0) + 1
                        )
                    zero_count = self._consecutive_zero_stable_frames.get(sku_id, 0)
                    if zero_count >= self.config.stockout_consecutive_frames and not active_occlusion and not is_panned_out:
                        alert = self._create_alert_if_eligible(
                            alert_type=InventoryAlertType.POSSIBLE_STOCKOUT,
                            sku_id=sku_id,
                            sku_name=stat.sku_name,
                            severity=AlertSeverity.HIGH,
                            stable_facings=0,
                            active_facings=stat.current_visible_facings,
                            reason=(
                                f"Zero stable visible facings for {zero_count} consecutive frames "
                                f"[POSSIBLE_STOCKOUT / OUT_OF_VIEW_PENDING_VERIFICATION: May have panned out of view]"
                            ),
                            frame_index=frame_index,
                            timestamp_sec=timestamp_sec,
                            supporting_count=zero_count,
                            requires_verification=True,
                        )
                        if alert:
                            new_alerts.append(alert)
                else:
                    self._consecutive_zero_stable_frames[sku_id] = 0
                    self._sku_panned_out_of_view.discard(sku_id)

            # --- Condition C: RAPID_REMOVAL ---
            # Suppressed during active occlusion
            removals_in_window = sum(
                1 for f_idx, s_id in self._removals_history if s_id == sku_id
            )
            if removals_in_window >= self.config.rapid_removal_count and not active_occlusion:
                alert = self._create_alert_if_eligible(
                    alert_type=InventoryAlertType.RAPID_REMOVAL,
                    sku_id=sku_id,
                    sku_name=stat.sku_name,
                    severity=AlertSeverity.HIGH,
                    stable_facings=stat.stable_facings,
                    active_facings=stat.current_visible_facings,
                    reason=(
                        f"{removals_in_window} product removals in last "
                        f"{self.config.rapid_removal_window_frames} frames "
                        f"(Requires verification: verify if customer pick or camera pan exit)"
                    ),
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    supporting_count=removals_in_window,
                    requires_verification=True,
                )
                if alert:
                    new_alerts.append(alert)

            # --- Condition D: PRODUCT_MOVEMENT ---
            moves_in_window = sum(
                1 for f_idx, s_id in self._movements_history if s_id == sku_id
            )
            if moves_in_window >= self.config.movement_alert_count:
                alert = self._create_alert_if_eligible(
                    alert_type=InventoryAlertType.PRODUCT_MOVEMENT,
                    sku_id=sku_id,
                    sku_name=stat.sku_name,
                    severity=AlertSeverity.MEDIUM,
                    stable_facings=stat.stable_facings,
                    active_facings=stat.current_visible_facings,
                    reason=(
                        f"{moves_in_window} sustained displacement events in last "
                        f"{self.config.movement_window_frames} frames (shelf reorganization or customer handling)"
                    ),
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    supporting_count=moves_in_window,
                    requires_verification=False,
                )
                if alert:
                    new_alerts.append(alert)

        # ----------------------------------------------------------------------
        # 2. Evaluate Global Alert: SKU_RECOGNITION_UNCERTAIN
        # ----------------------------------------------------------------------
        total_active = sum(s.current_visible_facings for s in sku_stats.values())
        unknown_stat = sku_stats.get("UNKNOWN")
        unknown_active = unknown_stat.current_visible_facings if unknown_stat else 0

        if total_active > 0:
            unknown_ratio = unknown_active / float(total_active)
            if (
                unknown_active >= self.config.uncertain_sku_min_count
                and unknown_ratio >= self.config.uncertain_sku_ratio_threshold
            ):
                alert = self._create_alert_if_eligible(
                    alert_type=InventoryAlertType.SKU_RECOGNITION_UNCERTAIN,
                    sku_id="UNKNOWN",
                    sku_name="Unclassified Products",
                    severity=AlertSeverity.MEDIUM,
                    stable_facings=unknown_stat.stable_facings if unknown_stat else 0,
                    active_facings=unknown_active,
                    reason=(
                        f"{unknown_active} of {total_active} active facings ({unknown_ratio * 100:.1f}%) "
                        f"remain UNKNOWN or have low match confidence"
                    ),
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    supporting_count=unknown_active,
                    requires_verification=True,
                )
                if alert:
                    new_alerts.append(alert)

        self._active_alerts_current_frame = new_alerts
        return new_alerts

    def _create_alert_if_eligible(
        self,
        alert_type: InventoryAlertType,
        sku_id: Optional[str],
        sku_name: Optional[str],
        severity: AlertSeverity,
        stable_facings: int,
        active_facings: int,
        reason: str,
        frame_index: int,
        timestamp_sec: float,
        supporting_count: int,
        requires_verification: bool,
    ) -> Optional[InventoryAlert]:
        """Generate alert if cooldown interval has elapsed."""
        key = (alert_type.value, sku_id)
        last_fired = self._alert_cooldowns.get(key, -999)

        if frame_index - last_fired < self.config.alert_cooldown_frames:
            return None  # Suppressed by cooldown

        self._alert_counter += 1
        self._alert_cooldowns[key] = frame_index

        alert = InventoryAlert(
            alert_id=self._alert_counter,
            frame_index=frame_index,
            timestamp_sec=round(timestamp_sec, 2),
            alert_type=alert_type,
            sku_id=sku_id,
            sku_name=sku_name,
            severity=severity,
            current_stable_facings=stable_facings,
            active_facings=active_facings,
            reason=reason,
            supporting_event_count=supporting_count,
            requires_verification=requires_verification,
        )
        self._all_alerts.append(alert)
        return alert

    # ----------------------------------------------------------------------
    # Query APIs for Operational Questions
    # ----------------------------------------------------------------------

    def query_skus_needing_attention(self) -> List[Dict[str, Any]]:
        """Which SKUs need operational attention?"""
        attention_dict: Dict[str, Dict[str, Any]] = {}
        for a in self._all_alerts:
            if not a.sku_id or a.sku_id == "UNKNOWN":
                continue
            if a.sku_id not in attention_dict:
                attention_dict[a.sku_id] = {
                    "sku_id": a.sku_id,
                    "sku_name": a.sku_name,
                    "alert_types": set(),
                    "total_alerts": 0,
                    "requires_verification": False,
                }
            item = attention_dict[a.sku_id]
            item["alert_types"].add(a.alert_type.value)
            item["total_alerts"] += 1
            if a.requires_verification:
                item["requires_verification"] = True

        return [
            {
                "sku_id": v["sku_id"],
                "sku_name": v["sku_name"],
                "alert_types": list(v["alert_types"]),
                "total_alerts": v["total_alerts"],
                "requires_verification": v["requires_verification"],
            }
            for v in attention_dict.values()
        ]

    def query_low_stock_skus(self) -> List[Dict[str, Any]]:
        """Which SKUs are low on visible facings?"""
        return [
            a.to_dict()
            for a in self._all_alerts
            if a.alert_type == InventoryAlertType.LOW_STOCK
        ]

    def query_possible_stockouts(self) -> List[Dict[str, Any]]:
        """Which SKUs may be out of view or possible stockout?"""
        return [
            a.to_dict()
            for a in self._all_alerts
            if a.alert_type == InventoryAlertType.POSSIBLE_STOCKOUT
        ]

    def query_repeated_removals(self) -> List[Dict[str, Any]]:
        """Which SKUs had repeated removals?"""
        return [
            a.to_dict()
            for a in self._all_alerts
            if a.alert_type == InventoryAlertType.RAPID_REMOVAL
        ]

    def query_significant_movement(self) -> List[Dict[str, Any]]:
        """Which SKUs had significant displacement/movement?"""
        return [
            a.to_dict()
            for a in self._all_alerts
            if a.alert_type == InventoryAlertType.PRODUCT_MOVEMENT
        ]

    def query_uncertain_recognition(self) -> List[Dict[str, Any]]:
        """Which alerts flag poor recognition confidence / UNKNOWN facings?"""
        return [
            a.to_dict()
            for a in self._all_alerts
            if a.alert_type == InventoryAlertType.SKU_RECOGNITION_UNCERTAIN
        ]

    # ----------------------------------------------------------------------
    # Reporting
    # ----------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Produce alert counts by type and severity."""
        by_type: Dict[str, int] = {t.value: 0 for t in InventoryAlertType}
        by_severity: Dict[str, int] = {s.value: 0 for s in AlertSeverity}
        by_sku: Dict[str, int] = {}
        verification_count = 0

        for a in self._all_alerts:
            by_type[a.alert_type.value] += 1
            by_severity[a.severity.value] += 1
            target = a.sku_name or a.sku_id or "UNKNOWN"
            by_sku[target] = by_sku.get(target, 0) + 1
            if a.requires_verification:
                verification_count += 1

        return {
            "total_alerts_fired": len(self._all_alerts),
            "alerts_requiring_verification": verification_count,
            "alerts_by_type": by_type,
            "alerts_by_severity": by_severity,
            "alerts_by_sku": by_sku,
            "semantics_notice": (
                "Camera-observable front-row visible facings only. Alerts such as POSSIBLE_STOCKOUT "
                "and RAPID_REMOVAL during camera motion must be verified against field-of-view changes. "
                "Do NOT claim warehouse back-stock or definite physical out-of-stock."
            ),
        }

    def get_full_report(
        self, final_sku_stats: Optional[Dict[str, SKUInventoryStats]] = None
    ) -> Dict[str, Any]:
        """Produce full alert audit report."""
        return {
            "alert_summary": self.get_summary(),
            "thresholds": asdict(self.config),
            "actionable_queries": {
                "skus_needing_attention": self.query_skus_needing_attention(),
                "low_stock_alerts": self.query_low_stock_skus(),
                "possible_stockouts": self.query_possible_stockouts(),
                "rapid_removals": self.query_repeated_removals(),
                "significant_movements": self.query_significant_movement(),
                "uncertain_recognition_alerts": self.query_uncertain_recognition(),
            },
            "final_sku_inventory_state": (
                [s.to_dict() for s in final_sku_stats.values()]
                if final_sku_stats
                else []
            ),
            "chronological_alert_history": [a.to_dict() for a in self._all_alerts],
        }
