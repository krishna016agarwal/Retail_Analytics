import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
  Users, 
  UserPlus, 
  UserMinus, 
  TrendingUp, 
  AlertCircle, 
  Clock, 
  Hourglass,
  Layers,
  Sparkles,
  BarChart2,
  ShieldAlert,
  ShoppingCart,
  Activity
} from 'lucide-react';
import InventoryDashboard from './components/InventoryDashboard';

import Header from './components/Header';
import ActionCenter from './components/ActionCenter';
import KPICard from './components/KPICard';
import LiveDepartmentTable from './components/LiveDepartmentTable';
import HourlyTrafficSection from './components/HourlyTrafficSection';
import AnalyticsChart from './components/AnalyticsChart';
import QueueStatus from './components/QueueStatus';
import MultiCameraSection from './components/MultiCameraSection';
import SyncStatus from './components/SyncStatus';
import SystemPipeline from './components/SystemPipeline';
import HistoricalTable from './components/HistoricalTable';
import AlertBanner from './components/AlertBanner';

import { 
  getHealth, 
  getLatestAnalytics, 
  getAnalytics, 
  getSyncStatus, 
  getLatestIntelligence,
  getAlerts,
  getDepartments,
  getCameras,
  getHourlyPatterns,
  getSimulationClock,
  getActiveBackendInfo,
  classifyApiError 
} from './api/client';
import { formatNumber, formatSeconds } from './utils/formatters';

export default function App() {
  // Telemetry state
  const [latestSnapshot, setLatestSnapshot] = useState(null);
  const [historicalData, setHistoricalData] = useState([]);
  const [syncStatus, setSyncStatus] = useState({});
  const [healthStatus, setHealthStatus] = useState({ status: 'unknown', database: 'unknown' });
  const [backendInfo, setBackendInfo] = useState(getActiveBackendInfo());

  // 4-Camera Multi-Stream & Department State
  const [departments, setDepartments] = useState([]);
  const [cameras, setCameras] = useState([]);
  const [hourlyTraffic, setHourlyTraffic] = useState(null);
  const [clockInfo, setClockInfo] = useState({ simulated_store_time: '17:00:00', demo_start_time: '17:00:00' });

  // Intelligence & Alerts state
  const [intelligenceData, setIntelligenceData] = useState(null);
  const [alerts, setAlerts] = useState([]);

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
   * Main telemetry fetching function calling all endpoints in parallel
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
        deptRes,
        camsRes,
        hourlyRes,
        clockRes
      ] = await Promise.allSettled([
        getHealth(),
        getLatestAnalytics(),
        getAnalytics({ limit: 100 }),
        getSyncStatus(),
        getLatestIntelligence(),
        getAlerts({ limit: 50 }),
        getDepartments(),
        getCameras(),
        getHourlyPatterns({ limit: 2000 }),
        getSimulationClock(),
      ]);

      // Update backend status indicator (Central vs Edge)
      setBackendInfo(getActiveBackendInfo());

      // Health status
      if (healthRes.status === 'fulfilled' && healthRes.value) {
        const val = healthRes.value;
        setHealthStatus({
          status: String(val.status || '').toLowerCase() === 'healthy' ? 'healthy' : 'unhealthy',
          database: String(val.database || '').toLowerCase() === 'connected' ? 'connected' : 'disconnected',
        });
      }

      // Latest snapshot
      if (latestRes.status === 'fulfilled' && latestRes.value) {
        setLatestSnapshot(latestRes.value);
      }

      // Historical data
      if (historyRes.status === 'fulfilled' && historyRes.value) {
        const hVal = historyRes.value;
        const list = Array.isArray(hVal) ? hVal : (hVal?.items || hVal?.snapshots || []);
        setHistoricalData(list);
      }

      // Sync status
      if (syncRes.status === 'fulfilled') {
        setSyncStatus(syncRes.value || {});
      }

      // Intelligence data
      if (intelRes.status === 'fulfilled' && intelRes.value) {
        setIntelligenceData(intelRes.value);
      }

      // Alerts
      if (alertsRes.status === 'fulfilled' && alertsRes.value) {
        const aVal = alertsRes.value;
        setAlerts(Array.isArray(aVal) ? aVal : (aVal?.alerts || []));
      } else if (intelRes.status === 'fulfilled' && Array.isArray(intelRes.value?.active_alerts)) {
        setAlerts(intelRes.value.active_alerts);
      }

      // Departments
      if (deptRes.status === 'fulfilled' && Array.isArray(deptRes.value)) {
        setDepartments(deptRes.value);
      }

      // Cameras
      if (camsRes.status === 'fulfilled' && Array.isArray(camsRes.value)) {
        setCameras(camsRes.value);
      }

      // Hourly patterns
      if (hourlyRes.status === 'fulfilled' && hourlyRes.value) {
        setHourlyTraffic(hourlyRes.value);
      }

      // Simulation clock
      if (clockRes.status === 'fulfilled' && clockRes.value) {
        setClockInfo(clockRes.value);
      } else if (intelRes.status === 'fulfilled' && intelRes.value?.simulation_clock) {
        setClockInfo(intelRes.value.simulation_clock);
      }

      // Error clearance
      const anySuccess = [healthRes, latestRes, historyRes, deptRes].some((r) => r.status === 'fulfilled');
      if (anySuccess) {
        setError(null);
        setLastUpdated(new Date().toISOString());
      } else {
        const firstErr = (latestRes.status === 'rejected' ? latestRes.reason : null) || new Error('Connection failed');
        setError(classifyApiError(firstErr));
      }
    } catch (err) {
      setError(classifyApiError(err));
    } finally {
      setLoading(false);
      setIsRefreshing(false);
      setCountdown(autoRefreshInterval);
    }
  }, [autoRefreshInterval]);

  useEffect(() => {
    fetchAllTelemetry();
  }, [fetchAllTelemetry]);

  // Auto-refresh timer
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
      if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
    };
  }, [autoRefreshInterval, fetchAllTelemetry]);

  // Derived metrics
  const occupancy = latestSnapshot?.occupancy ?? (intelligenceData?.store_totals?.total_live_shoppers ?? 0);
  const entries = latestSnapshot?.entries ?? (intelligenceData?.store_totals?.total_footfall ?? 0);
  const exits = latestSnapshot?.exits ?? 0;
  const peakOccupancy = latestSnapshot?.peak_occupancy ?? occupancy;
  const queueLength = latestSnapshot?.queue_length ?? (intelligenceData?.queue?.current_queue ?? 0);
  const peakQueue = latestSnapshot?.peak_queue ?? queueLength;
  const avgDwell = latestSnapshot?.avg_dwell ?? 0;
  const avgWait = latestSnapshot?.avg_wait ?? (intelligenceData?.queue?.average_wait_time ?? 0);
  const storeId = latestSnapshot?.store_id || 'store_001';
  const deviceId = latestSnapshot?.device_id || 'edge_device_01';

  // Queue Intelligence
  const queueData = intelligenceData?.queue || {
    current_queue: queueLength,
    peak_queue: peakQueue,
    growth_rate_per_min: 0.0,
    trend: 'STABLE',
    predicted_queue_3min: queueLength,
    congestion_risk: queueLength >= 6 ? 'HIGH' : queueLength >= 3 ? 'MEDIUM' : 'LOW',
    average_wait_time: avgWait,
  };

  // Crowd Intelligence
  const crowdData = intelligenceData?.crowd || {
    current_occupancy: occupancy,
    baseline_occupancy: occupancy,
    increase_percentage: 0.0,
    is_spike: false,
    spike_status: 'NOMINAL',
  };

  // ── Top-level view state ──────────────────────────────────────────────────
  const [mainTab, setMainTab] = useState(() => {
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      if (params.get('tab') === 'inventory') return 'inventory';
    }
    return 'crowd';
  });

  return (
    <div className="min-h-screen bg-[#070b14] text-slate-100 px-4 py-6 md:px-8 max-w-7xl mx-auto">
      {/* Top Header with Offline-First Badge */}
      <Header
        storeId={storeId}
        deviceId={deviceId}
        healthStatus={healthStatus}
        backendSourceInfo={backendInfo}
        connectionStatus={backendInfo.isOnlineCentral ? 'healthy' : 'offline'}
        lastUpdated={lastUpdated}
        isRefreshing={isRefreshing}
        onRefresh={() => fetchAllTelemetry(true)}
        autoRefreshInterval={autoRefreshInterval}
        setAutoRefreshInterval={setAutoRefreshInterval}
        countdown={countdown}
      />

      {/* Alert Banner for cold starts or connection errors */}
      <AlertBanner error={error} onRetry={() => fetchAllTelemetry(true)} />

      {/* ================================================================= */}
      {/* PRIMARY MODULE TAB BAR                                             */}
      {/* ================================================================= */}
      <div className="flex gap-1 mb-6 border-b border-slate-800/80 overflow-x-auto">
        <button
          id="tab-crowd-queue"
          onClick={() => setMainTab('crowd')}
          className={`flex items-center gap-2 px-5 py-3 text-sm font-semibold transition-colors whitespace-nowrap border-b-2 -mb-px
            ${mainTab === 'crowd'
              ? 'border-emerald-500 text-emerald-300'
              : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-600'}`}
        >
          <Activity className="h-4 w-4" />
          Crowd &amp; Queue Analytics
        </button>
        <button
          id="tab-inventory"
          onClick={() => setMainTab('inventory')}
          className={`flex items-center gap-2 px-5 py-3 text-sm font-semibold transition-colors whitespace-nowrap border-b-2 -mb-px
            ${mainTab === 'inventory'
              ? 'border-violet-500 text-violet-300'
              : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-600'}`}
        >
          <ShoppingCart className="h-4 w-4" />
          Shelf Inventory
        </button>
      </div>

      {/* ================================================================= */}
      {/* INVENTORY MODULE (conditionally rendered)                          */}
      {/* ================================================================= */}
      {mainTab === 'inventory' && (
        <div className="mb-12">
          <InventoryDashboard />
        </div>
      )}

      {/* ================================================================= */}
      {/* CROWD / QUEUE MODULE (conditionally rendered)                      */}
      {/* ================================================================= */}
      {mainTab === 'crowd' && (<>    {/* begin crowd-queue content */}

      {/* ========================================================================= */}
      {/* SECTION A: LIVE — WHAT IS HAPPENING NOW?                                 */}
      {/* ========================================================================= */}
      <section className="mb-8">
        <div className="flex items-center gap-2 mb-3">
          <span className="flex h-2.5 w-2.5 rounded-full bg-emerald-400 animate-pulse" />
          <h2 className="text-sm font-bold uppercase tracking-wider text-emerald-400">
            A. Live — What is happening now?
          </h2>
          <span className="text-xs text-slate-400 font-mono">
            (Live Shoppers, Store Footfall, Current Queue, Live Departments)
          </span>
        </div>

        {/* Top KPI Cards Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3 mb-4">
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
          <KPICard
            title="Store Footfall"
            subtitle="Cumulative entries"
            value={formatNumber(entries)}
            icon={UserPlus}
            accentColor="emerald"
            badgeText="INFLOW"
            badgeType="positive"
            loading={loading}
          />
          <KPICard
            title="Exits"
            subtitle="Cumulative departures"
            value={formatNumber(exits)}
            icon={UserMinus}
            accentColor="rose"
            badgeText="OUTFLOW"
            badgeType="neutral"
            loading={loading}
          />
          <KPICard
            title="Peak Occupancy"
            subtitle="Record simultaneous"
            value={formatNumber(peakOccupancy)}
            icon={TrendingUp}
            accentColor="amber"
            badgeText="PEAK"
            badgeType="neutral"
            loading={loading}
          />
          <KPICard
            title="Checkout Queue"
            subtitle="Shoppers waiting"
            value={formatNumber(queueLength)}
            icon={AlertCircle}
            accentColor={queueLength >= 6 ? 'rose' : queueLength >= 3 ? 'amber' : 'emerald'}
            badgeText={queueLength >= 6 ? 'HIGH' : queueLength >= 3 ? 'MED' : 'LOW'}
            badgeType={queueLength >= 6 ? 'alert' : queueLength >= 3 ? 'warning' : 'positive'}
            loading={loading}
          />
          <KPICard
            title="Avg Dwell Time"
            subtitle="Shopper visit time"
            value={formatSeconds(avgDwell)}
            icon={Clock}
            accentColor="indigo"
            badgeText="ENGAGE"
            badgeType="neutral"
            loading={loading}
          />
          <KPICard
            title="Avg Queue Wait"
            subtitle="Checkout delay"
            value={formatSeconds(avgWait)}
            icon={Hourglass}
            accentColor="sky"
            badgeText="WAIT"
            badgeType={avgWait > 60 ? 'warning' : 'neutral'}
            loading={loading}
          />
        </div>

        {/* Live Department View (Food, Electronics, Grocery) */}
        <LiveDepartmentTable departments={departments} loading={loading} />
      </section>

      {/* ========================================================================= */}
      {/* SECTION B: ANALYTICS — WHAT HAS HAPPENED?                                 */}
      {/* ========================================================================= */}
      <section className="mb-8">
        <div className="flex items-center gap-2 mb-3">
          <BarChart2 className="h-4 w-4 text-cyan-400" />
          <h2 className="text-sm font-bold uppercase tracking-wider text-cyan-400">
            B. Analytics — What has happened?
          </h2>
          <span className="text-xs text-slate-400 font-mono">
            (Hourly Traffic, Footfall Trends, Peak Hours, Historical Observations)
          </span>
        </div>

        {/* Department Traffic by Hour */}
        <HourlyTrafficSection hourlyData={hourlyTraffic} loading={loading} />

        {/* Time-Series Charts */}
        <div className="mb-6">
          <AnalyticsChart snapshots={historicalData} loading={loading} />
        </div>
      </section>

      {/* ========================================================================= */}
      {/* SECTION C: PREDICTIONS — WHAT WILL HAPPEN?                                */}
      {/* ========================================================================= */}
      <section className="mb-8">
        <div className="flex items-center gap-2 mb-3">
          <Sparkles className="h-4 w-4 text-purple-400" />
          <h2 className="text-sm font-bold uppercase tracking-wider text-purple-400">
            C. Predictions — What will happen?
          </h2>
          <span className="text-xs text-slate-400 font-mono">
            (3-Min Queue Forecast, Congestion Risk, Traffic Anomaly Detection)
          </span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Checkout Queue Intelligence & 3-Min Prediction */}
          <QueueStatus
            queueData={queueData}
            queueLength={queueLength}
            peakQueue={peakQueue}
            avgWait={avgWait}
          />

          {/* Crowd Intelligence / Anomaly Detection Card */}
          <div className="glass-panel p-5 border border-slate-800 rounded-xl flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-3 border-b border-slate-800/80">
                <div className="flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-purple-400" />
                  <h3 className="text-sm font-semibold text-white">Crowd Traffic Anomaly Detection</h3>
                </div>
                <span
                  className={`px-2.5 py-0.5 text-xs font-bold uppercase rounded-full border ${
                    crowdData.is_spike
                      ? 'bg-rose-500/15 border-rose-500/30 text-rose-300 animate-pulse'
                      : 'bg-emerald-500/15 border-emerald-500/30 text-emerald-400'
                  }`}
                >
                  {crowdData.is_spike ? 'SPIKE DETECTED' : 'NOMINAL'}
                </span>
              </div>

              <div className="mt-4 flex items-baseline justify-between">
                <div>
                  <span className="text-3xl font-extrabold text-white font-mono">
                    {crowdData.current_occupancy}
                  </span>
                  <span className="text-xs text-slate-400 ml-1.5">current shoppers</span>
                </div>
                <div className="text-right">
                  <span className="text-[10px] text-slate-400 block uppercase tracking-wider">
                    Rolling Baseline
                  </span>
                  <span className="text-sm font-bold text-slate-200 font-mono">
                    {typeof crowdData.baseline_occupancy === 'number'
                      ? crowdData.baseline_occupancy.toFixed(1)
                      : crowdData.baseline_occupancy}
                  </span>
                </div>
              </div>

              <div className="mt-4 p-3 rounded-lg bg-slate-950/50 border border-slate-800/60 flex items-center justify-between text-xs">
                <span className="text-slate-300">Increase vs Rolling Baseline:</span>
                <span
                  className={`font-mono font-bold text-sm ${
                    crowdData.increase_percentage >= 50
                      ? 'text-rose-400'
                      : crowdData.increase_percentage > 0
                      ? 'text-amber-300'
                      : 'text-slate-400'
                  }`}
                >
                  {crowdData.increase_percentage > 0 ? `+${crowdData.increase_percentage}%` : `${crowdData.increase_percentage}%`}
                </span>
              </div>
            </div>

            <div className="mt-4 pt-3 border-t border-slate-800/80 text-xs text-slate-300">
              {crowdData.is_spike ? (
                <span className="text-rose-300 font-medium">
                  ⚠️ High crowd concentration (+50% over baseline). Recommendation: Deploy floor staff.
                </span>
              ) : (
                <span className="text-emerald-400 font-medium">
                  ✓ Shopper density within normal statistical variance. No crowd spikes detected.
                </span>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* ========================================================================= */}
      {/* SECTION D: ACTIONS — WHAT SHOULD THE MANAGER DO?                         */}
      {/* ========================================================================= */}
      <section className="mb-8">
        <div className="flex items-center gap-2 mb-3">
          <ShieldAlert className="h-4 w-4 text-amber-400" />
          <h2 className="text-sm font-bold uppercase tracking-wider text-amber-400">
            D. Actions — What should the manager do?
          </h2>
          <span className="text-xs text-slate-400 font-mono">
            (Operational Staff Recommendations, Additional Counter Alerts)
          </span>
        </div>

        <ActionCenter alerts={alerts} loading={loading} />
      </section>

      {/* ========================================================================= */}
      {/* 4-CAMERA STREAMS INGESTION & PIPELINE STATUS                              */}
      {/* ========================================================================= */}
      <MultiCameraSection cameras={cameras} clockInfo={clockInfo} />

      {/* Offline-First Sync & Edge Pipeline Architecture */}
      <section className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <div className="lg:col-span-2">
          <SystemPipeline
            isCloudConnected={backendInfo.isOnlineCentral}
            isSyncing={isRefreshing}
            totalSnapshots={syncStatus.total_snapshots || historicalData.length}
          />
        </div>
        <div>
          <SyncStatus
            syncData={syncStatus}
            healthData={healthStatus}
            latestSnapshot={latestSnapshot}
            isRefreshing={isRefreshing}
          />
        </div>
      </section>

      {/* Raw Historical Observations Table */}
      <section className="mb-8">
        <HistoricalTable
          snapshots={historicalData}
          loading={loading}
          sourceLabel={
            backendInfo.isOnlineCentral ? 'Central PostgreSQL Records' : 'Local SQLite Edge Records'
          }
        />
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 pt-6 pb-8 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-500 font-mono">
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
            className="text-cyan-400 hover:underline"
          >
            retail-analytics-api-6ni3.onrender.com/docs
          </a>
        </div>
      </footer>
    </>)}  {/* end crowd-queue content */}
    </div>
  );
}
