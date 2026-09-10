"""SKU-Level Inventory Aggregation and Shelf Status Module.

Converts track-level detections, temporal facing states, and inventory events
into a structured SKU-level inventory summary suitable for retail systems.

Key Principles:
1. Maintains per-SKU metrics:
   - current visible facings, stable facings, uncertain facings, possibly changing facings
   - cumulative event counts: appeared, removed, moved, sku_changes
   - last seen frame and video timestamp
2. Strictly counts ACTIVE persistent tracks per frame (does NOT sum across frames).
3. Maintains camera-observable front-row visible facing semantics (no back-stock claims).
4. Classifies shelf status per catalog SKU:
   - IN_STOCK: stable visible facings >= in_stock_threshold
   - LOW_STOCK: 1 <= stable visible facings <= low_stock_threshold
   - OUT_OF_VIEW: 0 active stable visible facings (not currently observed on shelf)
   - UNKNOWN: unmapped / uncataloged detections
5. Configurable thresholds.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from inventory.catalog import SKUCatalog, SKUCatalogItem
from inventory.inventory_events import InventoryChangeEvent, InventoryEventType
from inventory.shelf_state import ProductTemporalSnapshot, ProductTemporalState


class SKUShelfStatus(str, Enum):
    """Shelf presence status for a catalog SKU."""

    IN_STOCK = "IN_STOCK"                # Ample visible facings (>= in_stock_threshold)
    LOW_STOCK = "LOW_STOCK"              # Very few visible facings (1 <= facings <= low_stock_threshold)
    OUT_OF_VIEW = "OUT_OF_VIEW"          # 0 visible facings currently observed on shelf
    UNKNOWN = "UNKNOWN"                  # Unmapped / uncataloged product detections


@dataclass
class InventoryAggregatorConfig:
    """Configurable thresholds for SKU inventory aggregation."""

    in_stock_threshold: int = 3          # Stable facings required for IN_STOCK
    low_stock_threshold: int = 2         # Maximum stable facings considered LOW_STOCK
    snapshot_interval: int = 15          # Store detailed frame snapshot every N frames


@dataclass
class SKUInventoryStats:
    """Current inventory metrics and event history for a single SKU."""

    sku_id: str
    sku_name: str
    category: str = "General"
    status: SKUShelfStatus = SKUShelfStatus.OUT_OF_VIEW

    # Current frame active facing counts (strictly non-cumulative)
    current_visible_facings: int = 0     # Total active tracks in current frame
    stable_facings: int = 0              # Tracks in STABLE state
    uncertain_facings: int = 0           # Tracks in UNCERTAIN state
    possibly_changing_facings: int = 0   # Tracks in POSSIBLY_CHANGING state

    # Cumulative lifecycle events for this SKU
    products_appeared: int = 0
    products_removed: int = 0
    products_moved: int = 0
    sku_changes: int = 0

    # Temporal visibility tracking
    last_seen_frame: Optional[int] = None
    last_seen_timestamp: Optional[float] = None
    active_track_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to a JSON-serializable dictionary."""
        d = asdict(self)
        d["status"] = self.status.value
        return d


class SKUInventoryAggregator:
    """Aggregates track detections, temporal states, and events at the SKU level."""

    def __init__(
        self,
        catalog: Optional[SKUCatalog] = None,
        config: Optional[InventoryAggregatorConfig] = None,
    ) -> None:
        self.catalog = catalog
        self.config = config or InventoryAggregatorConfig()

        self._stats: Dict[str, SKUInventoryStats] = {}
        self._cumulative_unique_tracks_by_sku: Dict[str, Set[int]] = {}
        self._frame_snapshots: List[Dict[str, Any]] = []
        self._all_events_log: List[InventoryChangeEvent] = []

        # Initialize registered catalog items
        if self.catalog is not None:
            for item in self.catalog.list_items():
                self._stats[item.sku_id] = SKUInventoryStats(
                    sku_id=item.sku_id,
                    sku_name=item.name,
                    category=item.category,
                    status=SKUShelfStatus.OUT_OF_VIEW,
                )
                self._cumulative_unique_tracks_by_sku[item.sku_id] = set()

        # Always initialize UNKNOWN bucket
        self._stats["UNKNOWN"] = SKUInventoryStats(
            sku_id="UNKNOWN",
            sku_name="Unclassified Products",
            category="Uncataloged",
            status=SKUShelfStatus.UNKNOWN,
        )
        self._cumulative_unique_tracks_by_sku["UNKNOWN"] = set()

    @property
    def current_stats(self) -> Dict[str, SKUInventoryStats]:
        """Return reference to current per-SKU inventory stats."""
        return self._stats

    def update(
        self,
        snapshot: ProductTemporalSnapshot,
        events: List[InventoryChangeEvent],
        frame_index: int,
        timestamp_sec: float = 0.0,
    ) -> Dict[str, SKUInventoryStats]:
        """Update SKU statistics for the current frame.

        Args:
            snapshot: ProductTemporalSnapshot containing active tracks and states.
            events: InventoryChangeEvents detected in this frame.
            frame_index: Current frame sequence index.
            timestamp_sec: Current video timestamp in seconds.

        Returns:
            Dictionary mapping sku_id -> SKUInventoryStats for the current frame.
        """
        # 1. Reset per-frame active facing counters
        for sku_id, stat in self._stats.items():
            stat.current_visible_facings = 0
            stat.stable_facings = 0
            stat.uncertain_facings = 0
            stat.possibly_changing_facings = 0
            stat.active_track_ids.clear()

        # 2. Count active tracks per SKU (strictly each active track counts exactly once)
        for record in snapshot.active_tracks:
            sku_id = record.sku_id
            if not record.is_known_sku or not sku_id or sku_id not in self._stats:
                sku_id = "UNKNOWN"

            stat = self._stats[sku_id]
            stat.current_visible_facings += 1
            stat.active_track_ids.append(record.track_id)
            stat.last_seen_frame = frame_index
            stat.last_seen_timestamp = timestamp_sec

            self._cumulative_unique_tracks_by_sku[sku_id].add(record.track_id)

            if record.state == ProductTemporalState.STABLE:
                stat.stable_facings += 1
            elif record.state == ProductTemporalState.UNCERTAIN:
                stat.uncertain_facings += 1
            elif record.state == ProductTemporalState.POSSIBLY_CHANGING:
                stat.possibly_changing_facings += 1

        # 3. Update cumulative lifecycle events per SKU
        for ev in events:
            self._all_events_log.append(ev)
            target_sku = ev.sku_id if (ev.sku_id and ev.sku_id in self._stats) else "UNKNOWN"
            stat = self._stats[target_sku]

            if ev.event_type == InventoryEventType.PRODUCT_APPEARED:
                stat.products_appeared += 1
            elif ev.event_type == InventoryEventType.PRODUCT_REMOVED:
                stat.products_removed += 1
            elif ev.event_type == InventoryEventType.PRODUCT_MOVED:
                stat.products_moved += 1
            elif ev.event_type == InventoryEventType.SKU_CHANGED:
                stat.sku_changes += 1
                if ev.previous_sku_id and ev.previous_sku_id in self._stats:
                    self._stats[ev.previous_sku_id].sku_changes += 1

        # 4. Classify current shelf status per SKU
        for sku_id, stat in self._stats.items():
            if sku_id == "UNKNOWN":
                stat.status = (
                    SKUShelfStatus.UNKNOWN
                    if stat.current_visible_facings > 0
                    else SKUShelfStatus.OUT_OF_VIEW
                )
            else:
                if stat.stable_facings >= self.config.in_stock_threshold:
                    stat.status = SKUShelfStatus.IN_STOCK
                elif stat.stable_facings >= 1:
                    stat.status = SKUShelfStatus.LOW_STOCK
                else:
                    stat.status = SKUShelfStatus.OUT_OF_VIEW

        # 5. Periodically record frame snapshot for audit reports
        if (
            frame_index % self.config.snapshot_interval == 0
            or frame_index == snapshot.frame_index
        ):
            self._record_frame_snapshot(frame_index, timestamp_sec)

        return self._stats

    def _record_frame_snapshot(self, frame_index: int, timestamp_sec: float) -> None:
        """Store a structured frame-level snapshot of SKU counts."""
        snap = {
            "frame_index": frame_index,
            "timestamp_sec": round(timestamp_sec, 2),
            "total_active_facings": sum(s.current_visible_facings for s in self._stats.values()),
            "total_stable_facings": sum(s.stable_facings for s in self._stats.values()),
            "skus_visible": {
                s.sku_id: {
                    "sku_name": s.sku_name,
                    "stable": s.stable_facings,
                    "uncertain": s.uncertain_facings,
                    "changing": s.possibly_changing_facings,
                    "status": s.status.value,
                }
                for s in self._stats.values()
                if s.current_visible_facings > 0
            },
        }
        self._frame_snapshots.append(snap)

    # ----------------------------------------------------------------------
    # Query APIs for Inventory Questions
    # ----------------------------------------------------------------------

    def query_visible_skus(self) -> List[Dict[str, Any]]:
        """Which SKUs are currently visible on the shelf?"""
        return [
            {
                "sku_id": s.sku_id,
                "sku_name": s.sku_name,
                "status": s.status.value,
                "current_visible_facings": s.current_visible_facings,
                "stable_facings": s.stable_facings,
            }
            for s in self._stats.values()
            if s.current_visible_facings > 0 and s.sku_id != "UNKNOWN"
        ]

    def query_low_stock_skus(self) -> List[Dict[str, Any]]:
        """Which SKUs are potentially low on visible facings?"""
        return [
            {
                "sku_id": s.sku_id,
                "sku_name": s.sku_name,
                "stable_facings": s.stable_facings,
                "threshold": self.config.in_stock_threshold,
                "status": s.status.value,
            }
            for s in self._stats.values()
            if s.status == SKUShelfStatus.LOW_STOCK
        ]

    def query_appeared_products(self) -> List[Dict[str, Any]]:
        """Which products appeared (promoted to STABLE)?"""
        return [
            {
                "sku_id": s.sku_id,
                "sku_name": s.sku_name,
                "products_appeared": s.products_appeared,
            }
            for s in self._stats.values()
            if s.products_appeared > 0
        ]

    def query_removed_products(self) -> List[Dict[str, Any]]:
        """Which products disappeared (removed after grace period)?"""
        return [
            {
                "sku_id": s.sku_id,
                "sku_name": s.sku_name,
                "products_removed": s.products_removed,
            }
            for s in self._stats.values()
            if s.products_removed > 0
        ]

    def query_moved_products(self) -> List[Dict[str, Any]]:
        """Which products moved (sustained displacement)?"""
        return [
            {
                "sku_id": s.sku_id,
                "sku_name": s.sku_name,
                "products_moved": s.products_moved,
            }
            for s in self._stats.values()
            if s.products_moved > 0
        ]

    # ----------------------------------------------------------------------
    # Reporting & Export
    # ----------------------------------------------------------------------

    def get_shelf_summary(self) -> Dict[str, Any]:
        """Produce top-level shelf inventory status summary."""
        total_active = sum(s.current_visible_facings for s in self._stats.values())
        total_stable = sum(s.stable_facings for s in self._stats.values())
        total_uncertain = sum(s.uncertain_facings for s in self._stats.values())
        total_changing = sum(s.possibly_changing_facings for s in self._stats.values())

        catalog_skus = [s for s in self._stats.values() if s.sku_id != "UNKNOWN"]
        in_stock_count = sum(1 for s in catalog_skus if s.status == SKUShelfStatus.IN_STOCK)
        low_stock_count = sum(1 for s in catalog_skus if s.status == SKUShelfStatus.LOW_STOCK)
        out_of_view_count = sum(1 for s in catalog_skus if s.status == SKUShelfStatus.OUT_OF_VIEW)

        unknown_stat = self._stats.get("UNKNOWN")
        unknown_active = unknown_stat.current_visible_facings if unknown_stat else 0

        event_totals = {
            "appeared": sum(s.products_appeared for s in self._stats.values()),
            "removed": sum(s.products_removed for s in self._stats.values()),
            "moved": sum(s.products_moved for s in self._stats.values()),
            "sku_changes": sum(s.sku_changes for s in self._stats.values()) // 2,  # avoid double count
        }

        return {
            "total_active_visible_facings": total_active,
            "total_stable_facings": total_stable,
            "total_uncertain_facings": total_uncertain,
            "total_changing_facings": total_changing,
            "unknown_facings_count": unknown_active,
            "catalog_skus_registered": len(catalog_skus),
            "skus_in_stock_count": in_stock_count,
            "skus_low_stock_count": low_stock_count,
            "skus_out_of_view_count": out_of_view_count,
            "skus_currently_observed_count": in_stock_count + low_stock_count,
            "event_totals": event_totals,
            "interpretation_notice": (
                "Camera-observable front-row visible facings only. "
                "Does NOT represent total physical inventory or back-stock."
            ),
        }

    def get_full_report(self) -> Dict[str, Any]:
        """Produce the comprehensive JSON report."""
        return {
            "shelf_summary": self.get_shelf_summary(),
            "thresholds": {
                "in_stock_threshold": self.config.in_stock_threshold,
                "low_stock_threshold": self.config.low_stock_threshold,
            },
            "per_sku_inventory": [s.to_dict() for s in self._stats.values()],
            "queries": {
                "currently_visible_skus": self.query_visible_skus(),
                "low_stock_alerts": self.query_low_stock_skus(),
                "products_appeared": self.query_appeared_products(),
                "products_removed": self.query_removed_products(),
                "products_moved": self.query_moved_products(),
            },
            "frame_snapshots": self._frame_snapshots,
        }
