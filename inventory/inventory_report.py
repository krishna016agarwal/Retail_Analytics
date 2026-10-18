"""Store-Level Inventory Reporting & Operational Snapshot Module.

Consolidates outputs from:
- SKUInventoryAggregator (per-SKU facings and shelf status)
- InventoryAlertDetector (actionable alerts, severity, verification flags)
- InventoryEventDetector (recent lifecycle events)
- ProductTemporalState (facing stability breakdowns)

Produces a clean, structured operational snapshot ready for dashboard/API consumption
and executive store management.

Key Guarantees:
1. Strictly preserves camera-observable front-row visible-facing semantics.
   Never describes counts as total physical inventory or back-stock.
2. Implements deterministic shelf health classification:
   - HEALTHY: No HIGH or MEDIUM operational issues.
   - ATTENTION_REQUIRED: Unresolved HIGH/MEDIUM operational issues exist.
   - VERIFICATION_REQUIRED: One or more alerts require physical/camera verification
     (e.g., distinguishing moving-camera out-of-view from physical stockouts).
3. Provides JSON serialization and human-readable managerial summary text.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from inventory.inventory_aggregator import SKUInventoryStats, SKUShelfStatus
from inventory.inventory_alerts import AlertSeverity, InventoryAlert, InventoryAlertType
from inventory.inventory_events import InventoryChangeEvent


class ShelfHealthStatus(str, Enum):
    """Deterministic store-level shelf operational health rating."""

    HEALTHY = "HEALTHY"
    ATTENTION_REQUIRED = "ATTENTION_REQUIRED"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"


@dataclass
class SKUSnapshotItem:
    """Detailed operational snapshot for an individual catalog SKU."""

    sku_id: str
    product_name: str
    category: str
    shelf_status: str
    current_active_facings: int
    stable_facings: int
    uncertain_facings: int
    changing_facings: int
    appeared_count: int
    removed_count: int
    moved_count: int
    sku_change_count: int
    active_alerts: List[Dict[str, Any]] = field(default_factory=list)
    requires_verification: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StoreInventoryReport:
    """Consolidated store-level inventory snapshot."""

    timestamp_iso: str
    video_source: str
    frame_index: int
    timestamp_sec: float
    total_frames: int

    # Overall Facings Breakdown (Camera-Observable Front Row Only)
    total_active_visible_facings: int
    stable_facings: int
    uncertain_facings: int
    possibly_changing_facings: int
    total_unknown_facings: int

    # Catalog Presence Metrics
    catalog_skus_registered: int
    catalog_skus_visible: int
    in_stock_skus_count: int
    low_stock_skus_count: int
    out_of_view_skus_count: int

    # Alert Metrics
    total_alerts_fired: int
    active_alerts_count: int
    verification_required_count: int
    alerts_by_severity: Dict[str, int]
    alerts_by_type: Dict[str, int]

    # Health & Operational Status
    shelf_health: ShelfHealthStatus
    health_reason: str

    # Detailed Breakdowns
    sku_inventory_summary: List[SKUSnapshotItem] = field(default_factory=list)
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    active_alerts: List[Dict[str, Any]] = field(default_factory=list)

    # Operational notice
    semantics_notice: str = (
        "Visible facings represent camera-observable front-row products only. "
        "Do NOT interpret as total physical store inventory or back-stock quantity. "
        "In moving-camera scenarios, items exiting camera field-of-view are labeled "
        "OUT_OF_VIEW / POSSIBLE_STOCKOUT pending verification."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        d = asdict(self)
        d["shelf_health"] = self.shelf_health.value
        return d

    def save_json(self, path: Union[str, Path]) -> None:
        """Save report to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_text(self) -> str:
        """Generate clean, manager-oriented human-readable summary text."""
        lines = [
            "=" * 78,
            "STORE SHELF INVENTORY & OPERATIONAL HEALTH REPORT",
            "=" * 78,
            f"Generated:       {self.timestamp_iso} (Frame {self.frame_index + 1}/{self.total_frames} @ {self.timestamp_sec:.2f}s)",
            f"Video Source:    {self.video_source}",
            f"Shelf Health:    [{self.shelf_health.value}] - {self.health_reason}",
            "-" * 78,
            "1. CAMERA-OBSERVABLE FRONT-ROW FACING SUMMARY",
            f"   • Total Active Visible Facings:     {self.total_active_visible_facings}",
            f"     - Confirmed STABLE Facings:       {self.stable_facings}",
            f"     - UNCERTAIN / New Facings:        {self.uncertain_facings}",
            f"     - POSSIBLY_CHANGING (Motion):     {self.possibly_changing_facings}",
            f"   • Unclassified (UNKNOWN) Facings:   {self.total_unknown_facings}",
            "-" * 78,
            "2. SKU CATALOG PRESENCE & COMPLIANCE",
            f"   • Registered Catalog SKUs:          {self.catalog_skus_registered}",
            f"   • Currently Visible SKUs:           {self.catalog_skus_visible}",
            f"   • IN_STOCK SKUs (>=3 facings):      {self.in_stock_skus_count}",
            f"   • LOW_STOCK SKUs (1-2 facings):     {self.low_stock_skus_count}",
            f"   • OUT_OF_VIEW SKUs (0 facings):     {self.out_of_view_skus_count}",
            "-" * 78,
            "3. ACTIONABLE ALERTS & VERIFICATION STATUS",
            f"   • Total Alerts Fired (Cumulative):  {self.total_alerts_fired}",
            f"   • Active Alerts Current Snapshot:   {self.active_alerts_count}",
            f"   • Alerts Requiring Verification:    {self.verification_required_count}",
            f"   • Severity Breakdown:               HIGH: {self.alerts_by_severity.get('HIGH', 0)}, "
            f"MEDIUM: {self.alerts_by_severity.get('MEDIUM', 0)}, LOW: {self.alerts_by_severity.get('LOW', 0)}",
            "-" * 78,
            "4. PER-SKU INVENTORY BREAKDOWN",
            f"   {'SKU Name':<28s} {'Status':<12s} {'Stable':<8s} {'Active':<8s} {'Appeared':<9s} {'Removed':<8s} {'Alerts'}",
            f"   {'-'*28} {'-'*12} {'-'*8} {'-'*8} {'-'*9} {'-'*8} {'-'*10}",
        ]

        for s in self.sku_inventory_summary:
            v_flag = " [VERIF]" if s.requires_verification else ""
            alert_types = [a["alert_type"] for a in s.active_alerts]
            alert_str = (", ".join(alert_types) + v_flag) if alert_types else "None"
            lines.append(
                f"   {s.product_name:<28s} {s.shelf_status:<12s} {s.stable_facings:<8d} "
                f"{s.current_active_facings:<8d} {s.appeared_count:<9d} {s.removed_count:<8d} {alert_str}"
            )

        lines.extend([
            "-" * 78,
            "5. OPERATIONAL NOTICE & SEMANTICS",
            f"   {self.semantics_notice}",
            "=" * 78,
        ])
        return "\n".join(lines)

    def save_text(self, path: Union[str, Path]) -> None:
        """Save text report to a file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.to_text())


class StoreInventoryReportBuilder:
    """Factory for building structured StoreInventoryReports from pipeline state."""

    @staticmethod
    def build_report(
        sku_stats: Dict[str, SKUInventoryStats],
        all_alerts: List[InventoryAlert],
        recent_events: List[InventoryChangeEvent],
        frame_index: int,
        timestamp_sec: float,
        total_frames: int,
        video_source: str,
        catalog_skus_registered: int,
    ) -> StoreInventoryReport:
        """Compile and evaluate a comprehensive StoreInventoryReport."""
        # 1. Facing calculations
        tot_active = sum(s.current_visible_facings for s in sku_stats.values())
        tot_stable = sum(s.stable_facings for s in sku_stats.values())
        tot_unc = sum(s.uncertain_facings for s in sku_stats.values())
        tot_chg = sum(s.possibly_changing_facings for s in sku_stats.values())

        unknown_stat = sku_stats.get("UNKNOWN")
        unknown_count = unknown_stat.current_visible_facings if unknown_stat else 0

        # 2. Catalog SKU status counts
        catalog_items = [s for s in sku_stats.values() if s.sku_id != "UNKNOWN"]
        in_stock_count = sum(1 for s in catalog_items if s.status == SKUShelfStatus.IN_STOCK)
        low_stock_count = sum(1 for s in catalog_items if s.status == SKUShelfStatus.LOW_STOCK)
        out_of_view_count = sum(1 for s in catalog_items if s.status == SKUShelfStatus.OUT_OF_VIEW)
        visible_count = in_stock_count + low_stock_count

        # 3. Alert categorization
        alerts_by_sev = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0}
        alerts_by_type: Dict[str, int] = {}
        verif_count = 0

        # Map alerts by SKU
        alerts_by_sku: Dict[str, List[Dict[str, Any]]] = {}
        for a in all_alerts:
            alerts_by_sev[a.severity.value] = alerts_by_sev.get(a.severity.value, 0) + 1
            alerts_by_type[a.alert_type.value] = alerts_by_type.get(a.alert_type.value, 0) + 1
            if a.requires_verification:
                verif_count += 1

            if a.sku_id:
                if a.sku_id not in alerts_by_sku:
                    alerts_by_sku[a.sku_id] = []
                alerts_by_sku[a.sku_id].append(a.to_dict())

        # 4. Deterministic Shelf Health Rating
        if verif_count > 0:
            shelf_health = ShelfHealthStatus.VERIFICATION_REQUIRED
            health_reason = (
                f"{verif_count} alerts require camera/field-of-view verification "
                f"(distinguishing camera panning out-of-view from physical stockout)."
            )
        elif alerts_by_sev.get("HIGH", 0) > 0:
            shelf_health = ShelfHealthStatus.ATTENTION_REQUIRED
            health_reason = (
                f"{alerts_by_sev['HIGH']} HIGH-severity inventory conditions detected "
                f"(low stock or rapid removals)."
            )
        elif alerts_by_sev.get("MEDIUM", 0) > 0 or low_stock_count > 0:
            shelf_health = ShelfHealthStatus.ATTENTION_REQUIRED
            health_reason = (
                f"{low_stock_count} SKUs low on visible facings or experiencing shelf movement."
            )
        else:
            shelf_health = ShelfHealthStatus.HEALTHY
            health_reason = "All registered visible SKUs have ample stable facings with no active alerts."

        # 5. Build per-SKU snapshot items
        sku_snapshots: List[SKUSnapshotItem] = []
        for s in sku_stats.values():
            sku_alerts = alerts_by_sku.get(s.sku_id, [])
            req_verif = any(a.get("requires_verification", False) for a in sku_alerts)
            item = SKUSnapshotItem(
                sku_id=s.sku_id,
                product_name=s.sku_name,
                category=s.category,
                shelf_status=s.status.value,
                current_active_facings=s.current_visible_facings,
                stable_facings=s.stable_facings,
                uncertain_facings=s.uncertain_facings,
                changing_facings=s.possibly_changing_facings,
                appeared_count=s.products_appeared,
                removed_count=s.products_removed,
                moved_count=s.products_moved,
                sku_change_count=s.sku_changes,
                active_alerts=sku_alerts,
                requires_verification=req_verif,
            )
            sku_snapshots.append(item)

        # 6. Recent events formatting
        recent_events_dicts = [ev.to_dict() for ev in recent_events[-15:]]

        return StoreInventoryReport(
            timestamp_iso=datetime.now().isoformat(),
            video_source=video_source,
            frame_index=frame_index,
            timestamp_sec=round(timestamp_sec, 2),
            total_frames=total_frames,
            total_active_visible_facings=tot_active,
            stable_facings=tot_stable,
            uncertain_facings=tot_unc,
            possibly_changing_facings=tot_chg,
            total_unknown_facings=unknown_count,
            catalog_skus_registered=catalog_skus_registered,
            catalog_skus_visible=visible_count,
            in_stock_skus_count=in_stock_count,
            low_stock_skus_count=low_stock_count,
            out_of_view_skus_count=out_of_view_count,
            total_alerts_fired=len(all_alerts),
            active_alerts_count=len(all_alerts),
            verification_required_count=verif_count,
            alerts_by_severity=alerts_by_sev,
            alerts_by_type=alerts_by_type,
            shelf_health=shelf_health,
            health_reason=health_reason,
            sku_inventory_summary=sku_snapshots,
            recent_events=recent_events_dicts,
            active_alerts=[a.to_dict() for a in all_alerts],
        )
