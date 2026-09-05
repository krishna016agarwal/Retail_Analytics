import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
  Users, 
  UserPlus, 
  UserMinus, 
  TrendingUp, 
  AlertCircle, 
  Clock, 
  Hourglass,
  Building2,
  Database
} from 'lucide-react';

import Header from './components/Header';
import ActionCenter from './components/ActionCenter';
import KPICard from './components/KPICard';
import AnalyticsChart from './components/AnalyticsChart';
import QueueStatus from './components/QueueStatus';
import StoreZones from './components/StoreZones';
import MultiCameraSection from './components/MultiCameraSection';
import SyncStatus from './components/SyncStatus';
import SystemPipeline from './components/SystemPipeline';
import HeatmapCard from './components/HeatmapCard';
import HistoricalTable from './components/HistoricalTable';
import AlertBanner from './components/AlertBanner';

import { 
  getHealth, 
  getLatestAnalytics, 
  getAnalytics, 
  getSyncStatus, 
  getLatestIntelligence,
  getAlerts,
  getZones,
  getPatterns,
  classifyApiError 
} from './api/client';
import { formatNumber, formatSeconds } from './utils/formatters';

export default function App() {
  // Primary telemetry state
  const [latestSnapshot, setLatestSnapshot] = useState(null);
  const [historicalData, setHistoricalData] = useState([]);
  const [syncStatus, setSyncStatus] = useState({});
  const [healthStatus, setHealthStatus] = useState({ status: 'unknown', database: 'unknown' });

  // Phase 8 Retail Intelligence state
  const [intelligenceData, setIntelligenceData] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [zones, setZones] = useState([]);
  const [patterns, setPatterns] = useState(null);

  // UI & lifecycle state
  const [loading, setLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  // Auto-refresh configuration (default: 5 seconds)
  const [autoRefreshInterval, setAutoRefreshInterval] = useState(5);
  const [countdown, setCountdown] = useState(5);

  const countdownTimerRef = useRef(null);

  /**
   * Main telemetry fetching function calling all central & intelligence endpoints in parallel
   */
  const fetchAllTelemetry = useCallback(async (isManualTrigger = false) => {
    if (isManualTrigger) {
      setIsRefreshing(true);
    }

    try {
      const [
        healthRes, 
        latestRes, 
        historyRes, 
        syncRes, 
        intelRes, 
        alertsRes, 
        zonesRes, 
        patternsRes
      ] = await Promise.allSettled([
        getHealth(),
        getLatestAnalytics(),
        getAnalytics({ limit: 100 }),
        getSyncStatus(),
        getLatestIntelligence(),
        getAlerts({ limit: 50 }),
        getZones(),
        getPatterns({ limit: 200 }),
      ]);

      // Process health check strictly per requirements: do not infer from sync
      if (healthRes.status === 'fulfilled' && healthRes.value) {
        const val = healthRes.value;
        const statusStr = String(val.status || '').toLowerCase();
        const dbStr = String(val.database || '').toLowerCase();
        setHealthStatus({
          status: statusStr === 'healthy' ? 'healthy' : 'unhealthy',
          database: dbStr === 'connected' ? 'connected' : 'disconnected',
        });
      } else {
        const rejectionErr = healthRes.status === 'rejected' ? healthRes.reason : null;
        const isTimeout = rejectionErr?.code === 'ECONNABORTED' ||
                          (rejectionErr?.message && rejectionErr.message.includes('timeout'));
        setHealthStatus({
          status: isTimeout ? 'cold_start' : 'unhealthy',
          database: 'disconnected',
        });
      }

      // Process latest snapshot
      if (latestRes.status === 'fulfilled') {
        setLatestSnapshot(latestRes.value);
      }

      // Process historical data
      if (historyRes.status === 'fulfilled') {
        const histVal = historyRes.value;
        const histList = Array.isArray(histVal) 
          ? histVal 
          : (Array.isArray(histVal?.items) ? histVal.items : (Array.isArray(histVal?.snapshots) ? histVal.snapshots : []));
        setHistoricalData(histList);
      }

      // Process sync status
      if (syncRes.status === 'fulfilled') {
        setSyncStatus(syncRes.value || {});
      }

      // Process Retail Intelligence
      if (intelRes.status === 'fulfilled' && intelRes.value) {
        setIntelligenceData(intelRes.value);
      }

      // Process Alerts
      let rawAlerts = [];
      if (alertsRes.status === 'fulfilled' && alertsRes.value) {
        if (Array.isArray(alertsRes.value)) {
          rawAlerts = alertsRes.value;
        } else if (Array.isArray(alertsRes.value.alerts)) {
          rawAlerts = alertsRes.value.alerts;
        }
      } else if (intelRes.status === 'fulfilled' && Array.isArray(intelRes.value?.active_alerts)) {
        rawAlerts = intelRes.value.active_alerts;
      }
      setAlerts(rawAlerts);

      // Process Zones
      if (zonesRes.status === 'fulfilled' && Array.isArray(zonesRes.value)) {
        setZones(zonesRes.value);
      } else if (intelRes.status === 'fulfilled' && Array.isArray(intelRes.value?.zones)) {
        setZones(intelRes.value.zones);
      }

      // Process Patterns
      if (patternsRes.status === 'fulfilled') {
        setPatterns(patternsRes.value);
      }

      // If at least one call fulfilled successfully, clear global network error
      const anySuccess = [healthRes, latestRes, historyRes, syncRes, intelRes].some(
        (r) => r.status === 'fulfilled'
      );

      if (anySuccess) {
        setError(null);
        setLastUpdated(new Date().toISOString());
      } else {
        // All primary calls failed: check rejection reason
        const firstError = (latestRes.status === 'rejected' ? latestRes.reason : null) ||
                           (healthRes.status === 'rejected' ? healthRes.reason : null) ||
                           new Error('Failed to connect to Retail API');
        setError(classifyApiError(firstError));
      }
    } catch (err) {
      setError(classifyApiError(err));
    } finally {
      setLoading(false);
      setIsRefreshing(false);
      setCountdown(autoRefreshInterval);
    }
  }, [autoRefreshInterval]);

  // Initial load
  useEffect(() => {
    fetchAllTelemetry();
  }, [fetchAllTelemetry]);

  // Countdown and periodic auto-refresh effect
  useEffect(() => {
    if (autoRefreshInterval <= 0) {
      setCountdown(0);
      return;
    }

    setCountdown(autoRefreshInterval);

    countdownTimerRef.current = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          fetchAllTelemetry();
          return autoRefreshInterval;
        }
        return prev - 1;
      });
    }, 1000);

    return () => {
      if (countdownTimerRef.current) {
        clearInterval(countdownTimerRef.current);
      }
    };
  }, [autoRefreshInterval, fetchAllTelemetry]);

  // Derive active KPI metric values safely from latestSnapshot or history
  const occupancy = latestSnapshot?.occupancy ?? 0;
  const entries = latestSnapshot?.entries ?? 0;
  const exits = latestSnapshot?.exits ?? 0;
  const peakOccupancy = latestSnapshot?.peak_occupancy ?? 0;
  const queueLength = latestSnapshot?.queue_length ?? 0;
  const peakQueue = latestSnapshot?.peak_queue ?? 0;
  const avgDwell = latestSnapshot?.avg_dwell ?? 0;
  const avgWait = latestSnapshot?.avg_wait ?? 0;
  const storeId = latestSnapshot?.store_id || intelligenceData?.store_id || 'store_001';
  const deviceId = latestSnapshot?.device_id || intelligenceData?.device_id || 'edge_device_01';

  // Overall connection status strictly from /health endpoint & cold-start state
  const isHealthy = healthStatus.status === 'healthy';
  const isColdStart = healthStatus.status === 'cold_start' || error?.type === 'COLD_START';

  let connectionStatus = 'offline';
  if (isColdStart) {
    connectionStatus = 'cold_start';
  } else if (isHealthy) {
    connectionStatus = 'healthy';
  } else {
    connectionStatus = 'offline';
  }

  // Queue Intelligence derived metrics with fallback
  const queueData = intelligenceData?.queue || {
    current_queue: queueLength,
    peak_queue: peakQueue,
    growth_rate_per_min: 0.0,
    trend: 'STABLE',
    predicted_queue_3min: queueLength,
    congestion_risk: queueLength >= 8 ? 'HIGH' : queueLength >= 4 ? 'MEDIUM' : 'LOW',
    average_wait_time: avgWait,
  };

  // Crowd Intelligence derived metrics with fallback
  const crowdData = intelligenceData?.crowd || {
    current_occupancy: occupancy,
    baseline_occupancy: occupancy,
    increase_percentage: 0.0,
    is_spike: false,
    spike_status: 'NOMINAL',
  };

  return (
    <div className="min-h-screen bg-[#070b14] text-slate-100 px-4 py-6 md:px-8 max-w-7xl mx-auto">
      {/* Top Header */}
      <Header
        storeId={storeId}
        deviceId={deviceId}
        healthStatus={healthStatus}
        connectionStatus={connectionStatus}
        lastUpdated={lastUpdated}
        isRefreshing={isRefreshing}
        onRefresh={() => fetchAllTelemetry(true)}
        autoRefreshInterval={autoRefreshInterval}
        setAutoRefreshInterval={setAutoRefreshInterval}
        countdown={countdown}
      />

      {/* Alert Banner for Render cold start or network errors */}
      <AlertBanner error={error} onRetry={() => fetchAllTelemetry(true)} />

      {/* Phase 8: Retail Action Center */}
      <ActionCenter alerts={alerts} loading={loading} />

      {/* 7 KPI Metric Cards Grid */}
      <section className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3.5 mb-6">
        {/* 1. Current Occupancy */}
        <KPICard
          title="Occupancy"
          subtitle="Live inside store"
          value={formatNumber(occupancy)}
          icon={Users}
          accentColor="cyan"
          badgeText="LIVE"
          badgeType={occupancy > 15 ? 'warning' : 'positive'}
          loading={loading}
        />

        {/* 2. Cumulative Entries / Latest Entries */}
        <KPICard
          title="Latest Entries"
          subtitle="Cumulative footfall"
          value={formatNumber(entries)}
          icon={UserPlus}
          accentColor="emerald"
          badgeText="INFLOW"
          badgeType="positive"
          loading={loading}
        />

        {/* 3. Cumulative Exits / Latest Exits */}
        <KPICard
          title="Latest Exits"
          subtitle="Cumulative departed"
          value={formatNumber(exits)}
          icon={UserMinus}
          accentColor="rose"
          badgeText="OUTFLOW"
          badgeType="neutral"
          loading={loading}
        />

        {/* 4. Peak Occupancy */}
        <KPICard
          title="Peak Occupancy"
          subtitle="Maximum reached"
          value={formatNumber(peakOccupancy)}
          icon={TrendingUp}
          accentColor="amber"
          badgeText="RECORD"
          badgeType="neutral"
          loading={loading}
        />

        {/* 5. Queue Length */}
        <KPICard
          title="Queue Length"
          subtitle="Shoppers waiting"
          value={formatNumber(queueLength)}
          icon={AlertCircle}
          accentColor={queueLength >= 8 ? 'rose' : queueLength >= 4 ? 'amber' : 'emerald'}
          badgeText={queueLength >= 8 ? 'HIGH' : queueLength >= 4 ? 'MED' : 'LOW'}
          badgeType={queueLength >= 8 ? 'alert' : queueLength >= 4 ? 'warning' : 'positive'}
          loading={loading}
        />

        {/* 6. Average Dwell Time */}
        <KPICard
          title="Avg Dwell Time"
          subtitle="Engagement duration"
          value={formatSeconds(avgDwell)}
          icon={Clock}
          accentColor="indigo"
          badgeText="ENGAGE"
          badgeType="neutral"
          loading={loading}
        />

        {/* 7. Average Wait Time */}
        <KPICard
          title="Avg Queue Wait"
          subtitle="Checkout delay"
          value={formatSeconds(avgWait)}
          icon={Hourglass}
          accentColor="sky"
          badgeText="CHECKOUT"
          badgeType={avgWait > 60 ? 'warning' : 'neutral'}
          loading={loading}
        />
      </section>

      {/* Main Analytics: Interactive Recharts + Side Status Column (Queue Intelligence + Sync) */}
      <section className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {/* Left 2 Cols: Interactive Recharts */}
        <div className="lg:col-span-2">
          <AnalyticsChart snapshots={historicalData} loading={loading} />
        </div>

        {/* Right 1 Col: Queue Intelligence Card + Sync Status Stack */}
        <div className="space-y-6">
          <QueueStatus
            queueData={queueData}
            queueLength={queueLength}
            peakQueue={peakQueue}
            avgWait={avgWait}
          />
          <SyncStatus
            syncData={syncStatus}
            healthData={healthStatus}
            latestSnapshot={latestSnapshot}
            isRefreshing={isRefreshing}
          />
        </div>
      </section>

      {/* Phase 8: Store Zones & Crowd Intelligence */}
      <StoreZones
        zones={zones}
        crowdData={crowdData}
        liveOccupancy={occupancy}
      />

      {/* Phase 8: Multi-Camera Architecture Section (Simulation Mode Tagged) */}
      <MultiCameraSection />

      {/* System Architecture Pipeline & Spatial Heatmap Cards */}
      <section className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {/* Left 2 Cols: Offline-First Edge AI Pipeline */}
        <div className="lg:col-span-2">
          <SystemPipeline
            isCloudConnected={healthStatus.status === 'healthy'}
            isSyncing={isRefreshing}
            totalSnapshots={syncStatus.total_snapshots || historicalData.length}
          />
        </div>

        {/* Right 1 Col: Edge-Generated Shopper Heatmap Card */}
        <div>
          <HeatmapCard />
        </div>
      </section>

      {/* Historical Telemetry Snapshots Table */}
      <section className="mb-8">
        <HistoricalTable
          snapshots={historicalData}
          loading={loading}
          sourceLabel={
            !import.meta.env.VITE_CENTRAL_API_URL?.includes('onrender.com')
              ? 'Edge Telemetry Records'
              : 'Central PostgreSQL Records'
          }
        />
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 pt-6 pb-8 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-500">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-emerald-400" />
          <span>SIH Problem Statement 179 • Intelligent Retail Analytics System</span>
        </div>
        <div>
          <span>Central Cloud: </span>
          <a
            href="https://retail-analytics-api-6ni3.onrender.com/docs"
            target="_blank"
            rel="noreferrer"
            className="text-cyan-400 hover:underline font-mono"
          >
            retail-analytics-api-6ni3.onrender.com/docs
          </a>
        </div>
      </footer>
    </div>
  );
}
