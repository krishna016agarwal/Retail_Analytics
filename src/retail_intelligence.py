"""Retail Intelligence Engine for Edge and Cloud Analytics (SIH Problem Statement 179).

Produces explainable, deterministic retail operational intelligence:
- Queue congestion prediction and billing counter opening recommendations
- Configured staff capacity load tracking and staff reassignment recommendations
- Edge crowd spike statistical anomaly detection
- Alert state tracking with cooldown deduplication
- Historical store traffic pattern analysis
Guarantees privacy: operates strictly on anonymous aggregates. Zero LLM for core decisions.
"""

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
import time
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
import uuid

from configs.config import RetailIntelligenceConfig, ZoneConfig
from src.queue_analytics import QueueMetrics
from src.shopper_analytics import FootfallMetrics
from src.zone_analytics import ZoneMetrics


@dataclass
class RetailAlert:
    """Structured operational alert for retail store managers and dashboards.

    Attributes:
        alert_id: Unique alert identifier.
        timestamp: ISO-8601 UTC timestamp string.
        store_id: Retail branch identifier.
        device_id: Edge device identifier.
        camera_id: Camera identifier.
        zone_id: Specific store zone or queue zone ID.
        type: Alert classification ('QUEUE_CONGESTION', 'STAFFING', 'CROWD_SPIKE').
        severity: Urgency level ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL').
        title: Short human-readable alert title.
        message: Detailed explanation of condition and metrics.
        current_value: Current observed metric value.
        predicted_value: Forecasted metric value (e.g., predicted queue in 3 min).
        threshold: Configured threshold that triggered the alert.
        recommendation: Clear operational action recommended to staff.
        status: Alert state ('ACTIVE', 'RESOLVED').
    """

    alert_id: str
    timestamp: str
    store_id: str
    device_id: str
    camera_id: str
    zone_id: str
    type: str
    severity: str
    title: str
    message: str
    current_value: Union[int, float]
    predicted_value: Optional[Union[int, float]] = None
    threshold: Union[int, float] = 0.0
    recommendation: str = ""
    status: str = "ACTIVE"

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary representation."""
        return asdict(self)


class AlertManager:
    """Manages alert deduplication, cooldown windows, and lifecycle states."""

    def __init__(self, cooldown_seconds: float = 45.0):
        """Initialize alert manager.

        Args:
            cooldown_seconds: Minimum seconds before a duplicate alert can fire.
        """
        self.cooldown_seconds = float(cooldown_seconds)
        # Map alert_key -> last trigger timestamp (monotonic seconds)
        self._last_trigger_time: Dict[str, float] = {}
        # Map alert_key -> active RetailAlert
        self._active_alerts: Dict[str, RetailAlert] = {}
        # History of all generated alerts
        self._alert_history: List[RetailAlert] = []
        # Queue of alerts resolved in latest cycle
        self._recently_resolved: List[RetailAlert] = []

    def _make_key(self, alert_type: str, zone_id: str) -> str:
        """Generate deduplication key based on alert type and zone."""
        return f"{alert_type}:{zone_id}"

    def should_fire(
        self,
        alert_type: str,
        zone_id: str,
        severity: str,
        current_time: float,
    ) -> bool:
        """Determine if an alert should fire or be suppressed by cooldown.

        Fires if:
        1. Alert has not fired before, or
        2. Previous alert was resolved/cleared, or
        3. Severity has changed (e.g., MEDIUM -> HIGH), or
        4. Cooldown duration has elapsed since previous firing.
        """
        key = self._make_key(alert_type, zone_id)
        if key not in self._last_trigger_time:
            return True

        # Check if severity upgraded
        active = self._active_alerts.get(key)
        if active is not None and active.status == "ACTIVE" and active.severity != severity:
            return True

        elapsed = current_time - self._last_trigger_time[key]
        return elapsed >= self.cooldown_seconds

    def record_alert(self, alert: RetailAlert, current_time: float) -> RetailAlert:
        """Register newly fired alert and update cooldown tracking."""
        key = self._make_key(alert.type, alert.zone_id)
        self._last_trigger_time[key] = current_time
        self._active_alerts[key] = alert
        self._alert_history.append(alert)
        return alert

    def resolve_alert(self, alert_type: str, zone_id: str) -> Optional[RetailAlert]:
        """Mark an active alert as resolved when condition clears."""
        key = self._make_key(alert_type, zone_id)
        active = self._active_alerts.get(key)
        if active and active.status == "ACTIVE":
            active.status = "RESOLVED"
            self._recently_resolved.append(active)
            return active
        return None

    def pop_recently_resolved(self) -> List[RetailAlert]:
        """Return and clear the list of alerts resolved during the current evaluation cycle."""
        resolved = list(self._recently_resolved)
        self._recently_resolved.clear()
        return resolved

    def get_active_alerts(self) -> List[RetailAlert]:
        """Return list of currently active alerts."""
        return [a for a in self._active_alerts.values() if a.status == "ACTIVE"]

    def get_resolved_alerts(self) -> List[RetailAlert]:
        """Return list of historical alerts that have transitioned to RESOLVED."""
        return [a for a in self._alert_history if a.status == "RESOLVED"]

    def get_alert_history(self, limit: int = 50) -> List[RetailAlert]:
        """Return recent alert history ordered latest first."""
        return list(reversed(self._alert_history[-limit:]))

    def reset(self) -> None:
        """Clear all active alerts and cooldown states."""
        self._last_trigger_time.clear()
        self._active_alerts.clear()
        self._alert_history.clear()
        self._recently_resolved.clear()


class RetailIntelligenceEngine:
    """Core retail intelligence engine combining queue, staffing, and crowd logic."""

    def __init__(
        self,
        config: Optional[RetailIntelligenceConfig] = None,
        store_id: str = "store_001",
        device_id: str = "edge_device_01",
        camera_id: str = "CAM_01",
    ):
        """Initialize intelligence engine with configurations.

        Args:
            config: RetailIntelligenceConfig instance. If None, defaults are used.
            store_id: Retail branch identifier.
            device_id: Edge processing unit identifier.
            camera_id: Primary or default camera identifier.
        """
        self.config = config or RetailIntelligenceConfig()
        self.store_id = str(store_id)
        self.device_id = str(device_id)
        self.camera_id = str(camera_id)

        # Alert manager with configurable cooldown
        self.alert_manager = AlertManager(cooldown_seconds=self.config.cooldown_seconds)

        # Queue tracking history: deque of (monotonic_time, queue_length)
        self._queue_history: Deque[Tuple[float, int]] = deque(maxlen=60)

        # Zone occupancy rolling history: zone_id -> deque of (monotonic_time, occupancy)
        self._zone_history: Dict[str, Deque[Tuple[float, int]]] = {}

    def calculate_queue_growth(
        self,
        current_queue: int,
        current_time: float,
        window_seconds: float = 30.0,
        min_elapsed_seconds: float = 5.0,
    ) -> Tuple[float, str]:
        """Calculate queue growth rate (people per minute) and qualitative trend.

        Uses rolling linear least-squares regression over historical samples within
        window_seconds. Requires a minimum observation baseline of min_elapsed_seconds
        (default 5.0s) to prevent tiny dt noise or single-frame flicker from generating
        extreme extrapolated rates.

        Args:
            current_queue: Current count of persons in queue zone.
            current_time: Monotonic or video timestamp in seconds.
            window_seconds: Time horizon for calculating rate of change (default 30.0s).
            min_elapsed_seconds: Minimum required elapsed duration to estimate per-minute slope (default 5.0s).

        Returns:
            Tuple of (growth_rate_per_minute, trend_str).
        """
        self._queue_history.append((float(current_time), int(current_queue)))

        # Prune old samples beyond double the window
        prune_cutoff = current_time - (window_seconds * 2.0)
        while self._queue_history and self._queue_history[0][0] < prune_cutoff:
            self._queue_history.popleft()

        cutoff = current_time - window_seconds
        window_samples = [(t, q) for t, q in self._queue_history if cutoff <= t <= current_time]

        if len(window_samples) < 2:
            return 0.0, "STABLE"

        elapsed = window_samples[-1][0] - window_samples[0][0]
        if elapsed < min_elapsed_seconds or elapsed <= 0.0:
            # Insufficient elapsed time to reliably compute per-minute rate; avoid noise extrapolation
            return 0.0, "STABLE"

        # Linear least-squares regression across all window samples
        n = len(window_samples)
        mean_t = sum(t for t, _ in window_samples) / n
        mean_q = sum(q for _, q in window_samples) / n

        ss_tt = sum((t - mean_t) ** 2 for t, _ in window_samples)
        ss_tq = sum((t - mean_t) * (q - mean_q) for t, q in window_samples)

        if ss_tt <= 1e-5:
            # All timestamps identical/duplicate
            return 0.0, "STABLE"

        slope_per_sec = ss_tq / ss_tt
        raw_growth_rate = slope_per_sec * 60.0

        # Physical retail queue bound: rate cannot realistically exceed 30 people/min per counter
        clamped_growth_rate = max(-30.0, min(30.0, raw_growth_rate))
        growth_rate = round(clamped_growth_rate, 2)

        if growth_rate >= 0.5:
            trend = "GROWING"
        elif growth_rate <= -0.5:
            trend = "SHRINKING"
        else:
            trend = "STABLE"

        return growth_rate, trend

    def predict_queue(
        self,
        current_queue: int,
        growth_rate: float,
        prediction_minutes: Optional[float] = None,
    ) -> int:
        """Forecast queue length forward in time based on current growth rate.

        Formula: max(0, round(current_queue + growth_rate * prediction_minutes))

        Args:
            current_queue: Active queue count.
            growth_rate: People per minute rate of change.
            prediction_minutes: Forward projection horizon (defaults to config: 3.0 min).

        Returns:
            Predicted integer queue length.
        """
        horizon = prediction_minutes if prediction_minutes is not None else self.config.prediction_minutes
        predicted = current_queue + (growth_rate * horizon)
        return max(0, int(round(predicted)))

    def evaluate_queue_intelligence(
        self,
        queue_metrics: QueueMetrics,
        current_time: float,
        timestamp_iso: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Optional[RetailAlert]]:
        """Compute queue analytics, predict congestion, and generate alerts.

        Alert Rule:
        Triggers QUEUE_CONGESTION when:
        (predicted_queue >= threshold AND growth_rate >= 0.0)
        OR (current_queue >= threshold AND growth_rate >= 0.0)
        """
        current_q = queue_metrics.current_queue_length
        growth_rate, trend = self.calculate_queue_growth(current_q, current_time)
        pred_q = self.predict_queue(current_q, growth_rate)

        threshold = self.config.queue_high_threshold
        ts = timestamp_iso or datetime.now(timezone.utc).isoformat()

        # Assess congestion risk
        if pred_q >= threshold or current_q >= threshold:
            congestion_risk = "HIGH"
        elif pred_q >= (threshold * 0.7) or current_q >= (threshold * 0.7):
            congestion_risk = "MEDIUM"
        else:
            congestion_risk = "LOW"

        intel_summary = {
            "current_queue": current_q,
            "peak_queue": queue_metrics.peak_queue_length,
            "growth_rate_per_min": growth_rate,
            "trend": trend,
            "predicted_queue_3min": pred_q,
            "congestion_risk": congestion_risk,
            "threshold": threshold,
            "average_wait_time": queue_metrics.average_wait_time,
        }

        # Check alert condition: predicted crosses threshold with active or upward trend
        should_alert = (pred_q >= threshold and growth_rate > 0.0) or (current_q >= threshold and trend != "SHRINKING")

        alert = None
        zone_key = "checkout_queue"
        if should_alert:
            severity = "CRITICAL" if pred_q >= (threshold * 1.5) else "HIGH"
            if self.alert_manager.should_fire("QUEUE_CONGESTION", zone_key, severity, current_time):
                alert = RetailAlert(
                    alert_id=f"alert_q_{uuid.uuid4().hex[:8]}",
                    timestamp=ts,
                    store_id=self.store_id,
                    device_id=self.device_id,
                    camera_id=self.camera_id,
                    zone_id=zone_key,
                    type="QUEUE_CONGESTION",
                    severity=severity,
                    title="Checkout congestion predicted",
                    message=(
                        f"Queue is growing at {growth_rate:+.1f} people/min and is predicted "
                        f"to reach {pred_q} within 3 minutes (threshold: {threshold})."
                    ),
                    current_value=current_q,
                    predicted_value=pred_q,
                    threshold=threshold,
                    recommendation="Open additional billing counter",
                    status="ACTIVE",
                )
                self.alert_manager.record_alert(alert, current_time)
        else:
            self.alert_manager.resolve_alert("QUEUE_CONGESTION", zone_key)

        return intel_summary, alert

    def evaluate_staffing_intelligence(
        self,
        zone_metrics_dict: Dict[str, ZoneMetrics],
        current_time: float,
        timestamp_iso: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], List[RetailAlert]]:
        """Evaluate staffing coverage and generate rebalance alerts.

        IMPORTANT:
        YOLO detects persons without distinguishing employees from shoppers.
        Staffing intelligence strictly calculates:
        shopper_load = shopper_count / expected_staff (based on configured staff capacity).
        """
        ts = timestamp_iso or datetime.now(timezone.utc).isoformat()
        max_load = self.config.max_shoppers_per_staff
        zone_summaries = {}
        alerts: List[RetailAlert] = []

        for z_id, z_metric in zone_metrics_dict.items():
            staff = max(1, z_metric.expected_staff)
            shoppers = z_metric.current_shoppers
            load = round(shoppers / staff, 2)

            is_overloaded = load >= max_load and shoppers >= 2
            zone_summaries[z_id] = {
                "name": z_metric.name,
                "current_shoppers": shoppers,
                "expected_staff": staff,
                "shopper_load_per_staff": load,
                "load_threshold": max_load,
                "is_overloaded": is_overloaded,
            }

            if is_overloaded:
                severity = "HIGH" if load >= (max_load * 1.5) else "MEDIUM"
                if self.alert_manager.should_fire("STAFFING", z_id, severity, current_time):
                    alert = RetailAlert(
                        alert_id=f"alert_st_{uuid.uuid4().hex[:8]}",
                        timestamp=ts,
                        store_id=self.store_id,
                        device_id=self.device_id,
                        camera_id=self.camera_id,
                        zone_id=z_id,
                        type="STAFFING",
                        severity=severity,
                        title="Low staff coverage",
                        message=(
                            f"Shopper load in {z_metric.name} is {load:.1f} per staff "
                            f"({shoppers} shoppers / {staff} configured staff, limit: {max_load})."
                        ),
                        current_value=load,
                        predicted_value=None,
                        threshold=max_load,
                        recommendation=f"Move 1 staff member to {z_metric.name}",
                        status="ACTIVE",
                    )
                    self.alert_manager.record_alert(alert, current_time)
                    alerts.append(alert)
            else:
                self.alert_manager.resolve_alert("STAFFING", z_id)

        return zone_summaries, alerts

    def evaluate_crowd_intelligence(
        self,
        zone_metrics_dict: Dict[str, ZoneMetrics],
        current_time: float,
        timestamp_iso: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], List[RetailAlert]]:
        """Evaluate zone occupancy against rolling statistical baseline to detect crowd spikes.

        Classified as: Edge intelligence / statistical anomaly detection.
        """
        ts = timestamp_iso or datetime.now(timezone.utc).isoformat()
        spike_thresh = self.config.spike_percentage_threshold
        min_occ = self.config.min_occupancy_for_spike
        summaries = {}
        alerts: List[RetailAlert] = []

        for z_id, z_metric in zone_metrics_dict.items():
            if z_id not in self._zone_history:
                self._zone_history[z_id] = deque(maxlen=self.config.baseline_window_size)

            current_occ = z_metric.current_shoppers
            history = self._zone_history[z_id]

            # Compute rolling baseline from previous history (excluding current frame)
            if history:
                baseline = sum(h[1] for h in history) / len(history)
            else:
                baseline = float(current_occ)

            # Record current observation for future frames
            history.append((current_time, current_occ))

            # Calculate relative increase over baseline
            if baseline > 0.5:
                increase_ratio = (current_occ - baseline) / baseline
            else:
                increase_ratio = float(current_occ) if current_occ >= min_occ else 0.0

            is_spike = current_occ >= min_occ and increase_ratio >= spike_thresh
            summaries[z_id] = {
                "name": z_metric.name,
                "current_occupancy": current_occ,
                "rolling_baseline": round(baseline, 1),
                "increase_percentage": round(increase_ratio * 100, 1),
                "is_spike": is_spike,
            }

            if is_spike:
                severity = "HIGH" if increase_ratio >= 1.0 else "MEDIUM"
                if self.alert_manager.should_fire("CROWD_SPIKE", z_id, severity, current_time):
                    alert = RetailAlert(
                        alert_id=f"alert_cs_{uuid.uuid4().hex[:8]}",
                        timestamp=ts,
                        store_id=self.store_id,
                        device_id=self.device_id,
                        camera_id=self.camera_id,
                        zone_id=z_id,
                        type="CROWD_SPIKE",
                        severity=severity,
                        title="High shopper concentration",
                        message=(
                            f"Shopper concentration in {z_metric.name} spiked to {current_occ} "
                            f"(rolling baseline: {baseline:.1f}, increase: +{increase_ratio * 100:.0f}%)."
                        ),
                        current_value=current_occ,
                        predicted_value=None,
                        threshold=round(baseline * (1 + spike_thresh), 1),
                        recommendation="Monitor zone / deploy floor staff",
                        status="ACTIVE",
                    )
                    self.alert_manager.record_alert(alert, current_time)
                    alerts.append(alert)
            else:
                self.alert_manager.resolve_alert("CROWD_SPIKE", z_id)

        return summaries, alerts

    def process_analytics(
        self,
        queue_metrics: Optional[QueueMetrics] = None,
        zone_metrics_dict: Optional[Dict[str, ZoneMetrics]] = None,
        footfall_metrics: Optional[FootfallMetrics] = None,
        current_time: Optional[float] = None,
        timestamp_iso: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Aggregate all analytical inputs and execute complete intelligence evaluation.

        Args:
            queue_metrics: Optional QueueMetrics from queue analytics.
            zone_metrics_dict: Optional dict of ZoneMetrics from zone analytics.
            footfall_metrics: Optional FootfallMetrics from entry/exit counter.
            current_time: Monotonic or video seconds.
            timestamp_iso: Real UTC ISO-8601 timestamp string.

        Returns:
            Dictionary containing queue, staffing, crowd intelligence, and generated alerts.
        """
        now_time = current_time if current_time is not None else time.time()
        now_iso = timestamp_iso or datetime.now(timezone.utc).isoformat()

        queue_summary: Dict[str, Any] = {}
        queue_alert: Optional[RetailAlert] = None
        if queue_metrics is not None:
            queue_summary, queue_alert = self.evaluate_queue_intelligence(
                queue_metrics=queue_metrics,
                current_time=now_time,
                timestamp_iso=now_iso,
            )

        staffing_summary: Dict[str, Any] = {}
        staffing_alerts: List[RetailAlert] = []
        crowd_summary: Dict[str, Any] = {}
        crowd_alerts: List[RetailAlert] = []

        if zone_metrics_dict:
            staffing_summary, staffing_alerts = self.evaluate_staffing_intelligence(
                zone_metrics_dict=zone_metrics_dict,
                current_time=now_time,
                timestamp_iso=now_iso,
            )
            crowd_summary, crowd_alerts = self.evaluate_crowd_intelligence(
                zone_metrics_dict=zone_metrics_dict,
                current_time=now_time,
                timestamp_iso=now_iso,
            )

        new_alerts: List[RetailAlert] = []
        if queue_alert:
            new_alerts.append(queue_alert)
        new_alerts.extend(staffing_alerts)
        new_alerts.extend(crowd_alerts)

        resolved_alerts = self.alert_manager.pop_recently_resolved()
        active_alerts = self.alert_manager.get_active_alerts()

        footfall_summary = {}
        if footfall_metrics is not None:
            footfall_summary = {
                "total_entries": footfall_metrics.total_entries,
                "total_exits": footfall_metrics.total_exits,
                "current_occupancy": footfall_metrics.current_occupancy,
                "peak_occupancy": footfall_metrics.peak_occupancy,
                "avg_dwell_time": footfall_metrics.avg_dwell_time,
            }

        return {
            "timestamp": now_iso,
            "store_id": self.store_id,
            "device_id": self.device_id,
            "queue_intelligence": queue_summary,
            "staffing_intelligence": staffing_summary,
            "crowd_intelligence": crowd_summary,
            "footfall_intelligence": footfall_summary,
            "active_alerts": [a.to_dict() for a in active_alerts],
            "new_alerts": [a.to_dict() for a in new_alerts],
            "resolved_alerts": [a.to_dict() for a in resolved_alerts],
        }

    @staticmethod
    def derive_live_summaries(
        latest_snapshot: Optional[Dict[str, Any]],
        recent_snapshots: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Derive real-time queue and crowd intelligence metrics from live snapshots.

        Returns:
            Tuple of (queue_intelligence_dict, crowd_intelligence_dict).
        """
        if not latest_snapshot:
            return (
                {
                    "current_queue": 0,
                    "peak_queue": 0,
                    "growth_rate_per_min": 0.0,
                    "trend": "STABLE",
                    "predicted_queue_3min": 0,
                    "congestion_risk": "LOW",
                    "average_wait_time": 0.0,
                },
                {
                    "current_occupancy": 0,
                    "baseline_occupancy": 0.0,
                    "increase_percentage": 0.0,
                    "is_spike": False,
                    "spike_status": "NOMINAL",
                },
            )

        # Queue Intelligence
        current_q = int(latest_snapshot.get("queue_length", 0) or 0)
        peak_q = int(latest_snapshot.get("peak_queue", current_q) or current_q)
        avg_wait = float(latest_snapshot.get("avg_wait", 0.0) or 0.0)

        growth_rate = 0.0
        trend = "STABLE"

        # Collect chronological samples from recent snapshots
        samples: List[Tuple[float, int]] = []
        if recent_snapshots:
            for s in reversed(recent_snapshots):
                q = s.get("queue_length")
                if q is None:
                    continue
                ts_str = s.get("created_at") or s.get("timestamp")
                if not ts_str:
                    continue
                t_val: Optional[float] = None
                if isinstance(ts_str, str) and not str(ts_str).startswith("T+"):
                    try:
                        t_val = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00")).timestamp()
                    except Exception:
                        pass
                if t_val is None and isinstance(ts_str, str) and str(ts_str).startswith("T+"):
                    try:
                        clean = str(ts_str).replace("T+", "").replace("s", "")
                        t_val = float(clean)
                    except Exception:
                        pass
                if t_val is not None:
                    samples.append((t_val, int(q)))

        if len(samples) >= 2:
            span_sec = samples[-1][0] - samples[0][0]
            # Require at least 5.0 seconds of elapsed duration to establish a per-minute trend
            if span_sec >= 5.0:
                n = len(samples)
                mean_t = sum(t for t, _ in samples) / n
                mean_q = sum(q for _, q in samples) / n
                ss_tt = sum((t - mean_t) ** 2 for t, _ in samples)
                ss_tq = sum((t - mean_t) * (q - mean_q) for t, q in samples)
                if ss_tt > 1e-5:
                    slope = ss_tq / ss_tt
                    raw_rate = slope * 60.0
                    growth_rate = round(max(-30.0, min(30.0, raw_rate)), 2)

        if growth_rate >= 0.5:
            trend = "GROWING"
        elif growth_rate <= -0.5:
            trend = "SHRINKING"
        else:
            trend = "STABLE"

        predicted_q = max(0, int(round(current_q + growth_rate * 3.0)))
        if predicted_q >= 8 or current_q >= 8:
            risk = "HIGH"
        elif predicted_q >= 5 or current_q >= 5:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        queue_intel = {
            "current_queue": current_q,
            "peak_queue": peak_q,
            "growth_rate_per_min": growth_rate,
            "trend": trend,
            "predicted_queue_3min": predicted_q,
            "congestion_risk": risk,
            "average_wait_time": avg_wait,
        }

        # Crowd Intelligence
        current_occ = int(latest_snapshot.get("occupancy", 0) or 0)
        recent_occ = [int(s.get("occupancy", 0) or 0) for s in recent_snapshots[:10]]
        if recent_occ:
            baseline = round(sum(recent_occ) / len(recent_occ), 1)
        else:
            baseline = float(current_occ)

        if baseline > 0:
            increase_pct = round(((current_occ - baseline) / baseline) * 100.0, 1)
        else:
            increase_pct = 0.0

        is_spike = current_occ >= 5 and increase_pct >= 50.0

        crowd_intel = {
            "current_occupancy": current_occ,
            "baseline_occupancy": baseline,
            "increase_percentage": increase_pct,
            "is_spike": is_spike,
            "spike_status": "SPIKE DETECTED" if is_spike else "NOMINAL",
        }

        return queue_intel, crowd_intel

    @staticmethod
    def calculate_historical_patterns(
        snapshots: List[Dict[str, Any]],
        zone_snapshots: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Compute store traffic patterns across hourly intervals from historical data.

        Strict Rules:
        - Uses actual timestamps, not relative T+0 / T+5 video timestamps.
        - If historical records are insufficient, returns "Insufficient historical data".
        - Never fabricates busy hours or statements.

        Args:
            snapshots: List of telemetry snapshots with valid 'created_at' or ISO timestamps.
            zone_snapshots: Optional list of zone snapshots.

        Returns:
            Dictionary with patterns analysis or insufficient data notification.
        """
        if not snapshots or len(snapshots) < 3:
            return {
                "status": "insufficient_data",
                "message": "Insufficient historical data",
            }

        # Parse timestamps and group by hour (0 to 23)
        hourly_occupancy: Dict[int, List[int]] = {h: [] for h in range(24)}
        hourly_queue: Dict[int, List[int]] = {h: [] for h in range(24)}
        valid_records = 0

        for s in snapshots:
            # Use real timestamp (created_at or valid ISO timestamp)
            ts_str = s.get("created_at") or s.get("timestamp")
            if not ts_str or str(ts_str).startswith("T+"):
                continue

            try:
                # Handle ISO-8601 strings with optional 'Z' or offset
                clean_ts = str(ts_str).replace("Z", "+00:00")
                dt = datetime.fromisoformat(clean_ts)
                hour = dt.hour
                occ = int(s.get("occupancy", 0))
                q_len = int(s.get("queue_length", 0))
                hourly_occupancy[hour].append(occ)
                hourly_queue[hour].append(q_len)
                valid_records += 1
            except Exception:
                continue

        # Need at least 3 valid records across the dataset
        if valid_records < 3:
            return {
                "status": "insufficient_data",
                "message": "Insufficient historical data",
            }

        # Calculate hourly averages for active hours
        avg_occ_by_hour: Dict[str, float] = {}
        avg_q_by_hour: Dict[str, float] = {}
        for h in range(24):
            if hourly_occupancy[h]:
                avg_occ_by_hour[f"{h:02d}:00"] = round(
                    sum(hourly_occupancy[h]) / len(hourly_occupancy[h]), 1
                )
            if hourly_queue[h]:
                avg_q_by_hour[f"{h:02d}:00"] = round(
                    sum(hourly_queue[h]) / len(hourly_queue[h]), 1
                )

        if not avg_occ_by_hour:
            return {
                "status": "insufficient_data",
                "message": "Insufficient historical data",
            }

        # Identify busiest hour
        busiest_hour_key = max(avg_occ_by_hour, key=avg_occ_by_hour.get)
        b_hour_int = int(busiest_hour_key.split(":")[0])
        busiest_hour = f"{b_hour_int:02d}:00 - {(b_hour_int + 1) % 24:02d}:00"

        # Analyze zone historical data if available
        peak_occ_by_zone: Dict[str, int] = {}
        zone_totals: Dict[str, int] = {}
        if zone_snapshots:
            for zs in zone_snapshots:
                z_name = zs.get("zone_name") or zs.get("zone_id", "Unknown")
                shoppers = int(zs.get("current_shoppers", 0))
                peak_occ_by_zone[z_name] = max(
                    peak_occ_by_zone.get(z_name, 0), shoppers
                )
                zone_totals[z_name] = zone_totals.get(z_name, 0) + shoppers

        busiest_zone = (
            max(zone_totals, key=zone_totals.get) if zone_totals else "N/A"
        )

        # Identify peak queue periods
        peak_queue_periods = []
        sorted_q_hours = sorted(
            avg_q_by_hour.items(), key=lambda item: item[1], reverse=True
        )
        for hour_str, q_val in sorted_q_hours[:3]:
            if q_val > 0:
                h_int = int(hour_str.split(":")[0])
                peak_queue_periods.append(
                    {
                        "period": f"{h_int:02d}:00 - {(h_int + 1) % 24:02d}:00",
                        "average_queue": q_val,
                    }
                )

        return {
            "status": "success",
            "busiest_hour": busiest_hour,
            "busiest_zone": busiest_zone,
            "average_occupancy_by_hour": avg_occ_by_hour,
            "peak_occupancy_by_zone": peak_occ_by_zone,
            "average_queue_by_hour": avg_q_by_hour,
            "peak_queue_periods": peak_queue_periods,
            "total_analyzed_records": valid_records,
        }

    def reset(self) -> None:
        """Reset state across alert manager and historical sliding windows."""
        self.alert_manager.reset()
        self._queue_history.clear()
        self._zone_history.clear()
