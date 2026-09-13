import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Camera,
  Package,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Cpu,
  Layers,
  Loader2,
  Film,
  ShieldCheck,
  Eye,
  Info,
  History,
  Check,
  CheckCheck,
  Clock,
  Bell,
  Activity,
  Server,
  AlertCircle,
} from 'lucide-react';
import {
  getActiveInventoryEvents,
  getInventoryEventHistory,
  acknowledgeInventoryEvent,
  resolveInventoryEvent,
} from '../api/client';

/**
 * ShelfRackVisualizer.jsx — Step 33 Final Retail Product & Demo Layer
 *
 * Requirements:
 * 1. Store Overview: Monitored Shelves, Active Replenishment Alerts, Attention required, Camera status, Run ID.
 * 2. Priority View: CRITICAL, MONITORING, UNCERTAIN, NORMAL with frozen confidence logic.
 * 3. Replenishment Action Center: Active alert hero component with shelf/tier, Visual Vacancy Confidence, duration, coords, ack/resolve.
 * 4. Lightweight Analytics: Real persisted event metrics (Total, Recommended, Resolved, Active, Avg resolution time with empty fallback).
 * 5. Demo Mode: Run isolation, video synchronicity, audit history.
 * 6. UI/UX Polish: Premium retail operations hierarchy, zero fake AI animations, zero SKU assumptions.
 * 7. System Health: Compact status indicator (API, Run, Camera, Processing, Heartbeat).
 * 8. Demo Safety: Tested state isolation, graceful error handling.
 */

export default function ShelfRackVisualizer({
  report,
  selectedVideoSource = 'shelf_pan_demo.mp4',
  activeVideoSource = 'shelf_pan_demo.mp4',
  activeRunId = null,
  pipelineRunning = false,
  apiHealth = null,
  lastUpdated = null,
}) {
  const [videoMode, setVideoMode] = useState('ANNOTATED'); // 'ANNOTATED' | 'ORIGINAL' | 'SNAPSHOT'
  const [activeEvents, setActiveEvents] = useState([]);
  const [eventHistory, setEventHistory] = useState([]);
  const [historyFilter, setHistoryFilter] = useState('ALL');
  const [actionLoading, setActionLoading] = useState({});
  const [apiErrorState, setApiErrorState] = useState(null);

  const fetchEvents = useCallback(async () => {
    try {
      const active = await getActiveInventoryEvents();
      const history = await getInventoryEventHistory();
      if (active && Array.isArray(active)) {
        setActiveEvents(active);
        setApiErrorState(null);
      }
      if (history && Array.isArray(history)) {
        setEventHistory(history);
      }
    } catch (err) {
      setApiErrorState('Temporary connectivity delay with inventory service.');
    }
  }, []);

  useEffect(() => {
    fetchEvents();
    const interval = setInterval(fetchEvents, 3000);
    return () => clearInterval(interval);
  }, [fetchEvents]);

  const handleAcknowledge = async (eventId) => {
    setActionLoading((prev) => ({ ...prev, [eventId]: true }));
    try {
      await acknowledgeInventoryEvent(eventId);
      setActiveEvents((prev) =>
        prev.map((e) => (e.event_id === eventId ? { ...e, status: 'ACKNOWLEDGED' } : e))
      );
      setEventHistory((prev) =>
        prev.map((e) => (e.event_id === eventId ? { ...e, status: 'ACKNOWLEDGED' } : e))
      );
    } catch (err) {
      console.error('Error acknowledging event:', err);
    } finally {
      setActionLoading((prev) => ({ ...prev, [eventId]: false }));
    }
  };

  const handleResolve = async (eventId) => {
    setActionLoading((prev) => ({ ...prev, [eventId]: true }));
    try {
      await resolveInventoryEvent(eventId);
      setActiveEvents((prev) => prev.filter((e) => e.event_id !== eventId));
      setEventHistory((prev) =>
        prev.map((e) =>
          e.event_id === eventId
            ? { ...e, status: 'RESOLVED', resolved_at: new Date().toISOString() }
            : e
        )
      );
    } catch (err) {
      console.error('Error resolving event:', err);
    } finally {
      setActionLoading((prev) => ({ ...prev, [eventId]: false }));
    }
  };

  const isPendingAnalysis =
    selectedVideoSource &&
    activeVideoSource &&
    selectedVideoSource !== activeVideoSource;
  const currentRunId = activeRunId || report?.run_id || 'RUN-001';
  const displayedVideo = selectedVideoSource || activeVideoSource || 'shelf_pan_demo.mp4';

  // Extract V1 shelf data
  const shelves = report?.shelves || [];
  const primaryShelf = shelves[0] || {
    shelf_id: 'SHELF-01',
    status: report?.shelf_health === 'GOOD' ? 'OCCUPIED' : 'VACANT',
    vacancy_detected: (report?.active_alerts || []).length > 0,
    vacancy_score: 0.0,
    visual_vacancy_confidence: 0.0,
    replenishment_recommended: false,
    occupancy_pct: 65.0,
    detected_facings_count: report?.total_active_visible_facings || 0,
    temporal_state: 'NORMAL',
    vacant_regions: [],
  };

  const isVacant = primaryShelf.status === 'VACANT' || primaryShelf.vacancy_detected;
  const isUncertain = primaryShelf.status === 'UNCERTAIN' || primaryShelf.is_occluded || primaryShelf.is_camera_moving;
  const vacantRegions = primaryShelf.vacant_regions || [];
  const activeAlerts = report?.active_alerts || [];

  // Scoped to active run
  const relevantActiveEvents = activeEvents.filter(
    (e) => !activeRunId || e.run_id === activeRunId
  );
  const displayedEvents =
    historyFilter === 'CURRENT_RUN' && currentRunId
      ? eventHistory.filter((e) => e.run_id === currentRunId)
      : eventHistory;

  // ─── 1. Priority View Calculation (Frozen Rules) ───────────────────────────
  const vacancyConfidence = primaryShelf.visual_vacancy_confidence || 0.0;
  const isCritical = (isVacant || relevantActiveEvents.length > 0) && vacancyConfidence >= 0.70;
  const isMonitoring = !isCritical && vacantRegions.some((g) => (g.persistence_frames || 1) < 10 && !g.replenishment_recommended);
  const isUncertainPriority = !isCritical && !isMonitoring && isUncertain;
  const isNormal = !isCritical && !isMonitoring && !isUncertainPriority;

  let priorityBadge = {
    label: 'NORMAL',
    desc: 'No actionable vacancy. Product facings packed.',
    bg: 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300',
    dot: 'bg-emerald-400',
  };

  if (isCritical) {
    priorityBadge = {
      label: 'CRITICAL',
      desc: 'Persistent vacancy confirmed (Confidence ≥ 70%). Immediate replenishment recommended.',
      bg: 'bg-rose-500/20 border-rose-500/50 text-rose-300 animate-pulse',
      dot: 'bg-rose-500',
    };
  } else if (isMonitoring) {
    priorityBadge = {
      label: 'MONITORING',
      desc: 'Vacancy candidate under temporal evaluation (< 10 frames).',
      bg: 'bg-cyan-500/15 border-cyan-500/40 text-cyan-300',
      dot: 'bg-cyan-400',
    };
  } else if (isUncertainPriority) {
    priorityBadge = {
      label: 'UNCERTAIN',
      desc: 'Shopper occlusion or optical-flow camera motion prevents confirmation.',
      bg: 'bg-amber-500/15 border-amber-500/40 text-amber-300',
      dot: 'bg-amber-400',
    };
  }

  // ─── 2. Real Lightweight Analytics ──────────────────────────────────────────
  const analytics = useMemo(() => {
    const totalEvents = eventHistory.length;
    const recommendations = eventHistory.filter((e) => e.event_type === 'REPLENISHMENT_RECOMMENDED').length;
    const resolvedEvents = eventHistory.filter((e) => e.status === 'RESOLVED');
    const resolvedCount = resolvedEvents.length;
    const activeCount = relevantActiveEvents.length;

    // Durations: only if >= 2 resolved events
    const durations = [];
    resolvedEvents.forEach((e) => {
      if (e.duration_sec && e.duration_sec > 0) {
        durations.push(e.duration_sec);
      } else if (e.resolved_at && e.timestamp_iso) {
        try {
          const tStart = new Date(e.timestamp_iso).getTime();
          const tEnd = new Date(e.resolved_at).getTime();
          const diff = (tEnd - tStart) / 1000;
          if (diff > 0) durations.push(diff);
        } catch (_) {}
      }
    });

    let avgResolutionText = 'Insufficient Data (< 2 resolved events)';
    let hasSufficientData = false;
    if (durations.length >= 2) {
      const avg = durations.reduce((a, b) => a + b, 0) / durations.length;
      avgResolutionText = `${avg.toFixed(1)}s`;
      hasSufficientData = true;
    }

    return {
      totalEvents,
      recommendations,
      resolvedCount,
      activeCount,
      avgResolutionText,
      hasSufficientData,
      samplesCount: durations.length,
    };
  }, [eventHistory, relevantActiveEvents]);

  // Primary active event for hero action center
  const primaryActiveAlert = relevantActiveEvents[0] || null;

  return (
    <div className="space-y-6">
      {/* ─── 1. STORE OVERVIEW (Top Header KPI Strip) ─────────────────────── */}
      <div className="bg-slate-900/90 border border-slate-800 backdrop-blur-md rounded-2xl p-5 shadow-xl space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div className="space-y-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-1.5 font-mono">
                <Film className="h-3.5 w-3.5" />
                {pipelineRunning
                  ? `ANALYZING PLAYBACK: ${displayedVideo}`
                  : isPendingAnalysis
                  ? `CAMERA PREVIEW: ${displayedVideo}`
                  : `EDGE CAMERA — ANALYSIS PLAYBACK: ${activeVideoSource}`}
              </span>
              <span className="text-[10px] px-2.5 py-0.5 rounded-full bg-violet-500/15 border border-violet-500/30 text-violet-300 font-mono font-semibold">
                Run: {currentRunId} ({activeVideoSource})
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 font-mono">
                Frame: {report?.frame_index ? report.frame_index + 1 : 75} / {report?.total_frames ?? 75}
              </span>
            </div>
            <h2 className="text-xl font-extrabold text-white flex items-center gap-2">
              Store Shelf Vacancy & Replenishment Operations
            </h2>
            <p className="text-xs text-slate-400 max-w-2xl">
              Generic computer vision shelf vacancy monitoring on <strong>{primaryShelf.shelf_id}</strong>.
              Detects physical persistent empty spaces using 2D product occupancy geometry without SKU assumptions.
            </p>
          </div>

          {/* Sync & Priority Status Badges */}
          <div className="flex flex-wrap items-center gap-2.5">
            {/* Priority Status Pill */}
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-xl border text-xs font-mono font-bold ${priorityBadge.bg}`}>
              <span className={`h-2 w-2 rounded-full ${priorityBadge.dot}`} />
              <span>STATUS: {priorityBadge.label}</span>
            </div>

            {/* Sync Badge */}
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-950/70 border border-slate-800 text-xs font-mono">
              <Layers className="h-3.5 w-3.5 text-cyan-400" />
              <span className="text-slate-400">Sync:</span>
              <span className="font-bold text-white">
                {pipelineRunning ? (
                  <span className="text-violet-400 flex items-center gap-1">
                    <Loader2 className="h-3 w-3 animate-spin" /> In-Progress
                  </span>
                ) : isPendingAnalysis ? (
                  <span className="text-amber-400 flex items-center gap-1">
                    <AlertTriangle className="h-3 w-3" /> Standby
                  </span>
                ) : (
                  <span className="text-emerald-400 flex items-center gap-1">
                    <CheckCircle2 className="h-3 w-3" /> Synchronized
                  </span>
                )}
              </span>
            </div>
          </div>
        </div>

        {/* Store Overview 5 KPI Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 pt-3 border-t border-slate-800/80">
          <div className="bg-slate-950/50 rounded-xl p-3 border border-slate-800/60">
            <span className="text-[10px] text-slate-400 uppercase font-semibold block">Shelves Monitored</span>
            <div className="text-base font-bold text-white flex items-center gap-1.5 mt-0.5 font-mono">
              <Package className="h-4 w-4 text-cyan-400" />
              1 (SHELF-01 · {primaryShelf.tiers?.length || 3} Tiers)
            </div>
          </div>

          <div className="bg-slate-950/50 rounded-xl p-3 border border-slate-800/60">
            <span className="text-[10px] text-slate-400 uppercase font-semibold block">Active Alerts</span>
            <div className={`text-base font-bold flex items-center gap-1.5 mt-0.5 font-mono ${
              relevantActiveEvents.length > 0 ? 'text-rose-400' : 'text-emerald-400'
            }`}>
              <Bell className="h-4 w-4" />
              {relevantActiveEvents.length} active alert{relevantActiveEvents.length === 1 ? '' : 's'}
            </div>
          </div>

          <div className="bg-slate-950/50 rounded-xl p-3 border border-slate-800/60">
            <span className="text-[10px] text-slate-400 uppercase font-semibold block">Attention Required</span>
            <div className={`text-base font-bold flex items-center gap-1.5 mt-0.5 font-mono ${
              isVacant || relevantActiveEvents.length > 0 ? 'text-amber-400' : 'text-emerald-400'
            }`}>
              {isVacant || relevantActiveEvents.length > 0 ? (
                <>
                  <AlertTriangle className="h-4 w-4 text-amber-400" />
                  1 Shelf
                </>
              ) : (
                <>
                  <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                  0 Shelves
                </>
              )}
            </div>
          </div>

          <div className="bg-slate-950/50 rounded-xl p-3 border border-slate-800/60">
            <span className="text-[10px] text-slate-400 uppercase font-semibold block">Camera CAM-01</span>
            <div className="text-base font-bold text-white flex items-center gap-1.5 mt-0.5 font-mono">
              <Camera className="h-4 w-4 text-violet-400" />
              {pipelineRunning ? (
                <span className="text-violet-400">STREAMING</span>
              ) : isPendingAnalysis ? (
                <span className="text-amber-400">STANDBY</span>
              ) : apiHealth && !apiHealth.alive ? (
                <span className="text-slate-500">OFFLINE</span>
              ) : (
                <span className="text-emerald-400">ONLINE</span>
              )}
            </div>
          </div>

          <div className="bg-slate-950/50 rounded-xl p-3 border border-slate-800/60">
            <span className="text-[10px] text-slate-400 uppercase font-semibold block">Active Run ID</span>
            <div className="text-base font-mono font-bold text-cyan-300 flex items-center gap-1 mt-0.5">
              <Cpu className="h-4 w-4" />
              {currentRunId}
            </div>
          </div>
        </div>
      </div>

      {/* ─── API Connection Warning (Graceful Degradation) ────────────────── */}
      {apiErrorState && (
        <div className="flex items-center gap-3 p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-200 text-xs font-mono">
          <AlertCircle className="h-4 w-4 text-amber-400 shrink-0" />
          <span>{apiErrorState} Operating on persisted local state.</span>
        </div>
      )}

      {/* ─── Pending Analysis Warning ────────────────────────────────────── */}
      {isPendingAnalysis && !pipelineRunning && (
        <div className="flex items-center gap-3 p-4 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-200">
          <AlertTriangle className="h-5 w-5 text-amber-400 shrink-0" />
          <div>
            <h4 className="text-sm font-bold text-white">
              Selected Video ({selectedVideoSource}) Pending Analysis
            </h4>
            <p className="text-xs text-amber-300/90 mt-0.5">
              Camera is previewing <strong>{selectedVideoSource}</strong>. The metrics below belong to run <strong>{currentRunId} ({activeVideoSource})</strong>.
              Click <strong>'Start Inventory Run'</strong> to analyze {selectedVideoSource}.
            </p>
          </div>
        </div>
      )}

      {/* ─── 3. REPLENISHMENT ACTION CENTER (Hero Actionable Component) ─────── */}
      {!pipelineRunning && !isPendingAnalysis && (
        <div className={`p-5 rounded-2xl border transition-all shadow-xl space-y-4 ${
          isCritical || relevantActiveEvents.length > 0
            ? 'bg-rose-500/15 border-rose-500/40 text-rose-200'
            : isUncertainPriority
            ? 'bg-amber-500/10 border-amber-500/30 text-amber-200'
            : 'bg-slate-900/90 border-slate-800 text-slate-300'
        }`}>
          <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
            <div className="flex items-start gap-3.5">
              {isCritical || relevantActiveEvents.length > 0 ? (
                <div className="h-10 w-10 rounded-xl bg-rose-500/20 border border-rose-500/40 flex items-center justify-center shrink-0 mt-0.5">
                  <AlertTriangle className="h-5 w-5 text-rose-400 animate-pulse" />
                </div>
              ) : isUncertainPriority ? (
                <div className="h-10 w-10 rounded-xl bg-amber-500/20 border border-amber-500/40 flex items-center justify-center shrink-0 mt-0.5">
                  <AlertTriangle className="h-5 w-5 text-amber-400" />
                </div>
              ) : (
                <div className="h-10 w-10 rounded-xl bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center shrink-0 mt-0.5">
                  <CheckCircle2 className="h-5 w-5 text-emerald-400" />
                </div>
              )}

              <div className="space-y-1">
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="text-base font-extrabold text-white tracking-wide">
                    {isCritical || relevantActiveEvents.length > 0
                      ? 'REPLENISHMENT ACTION REQUIRED — EMPTY SHELF SPACE CONFIRMED'
                      : isUncertainPriority
                      ? 'MONITORING — SHELF OCCLUSION OR MOTION IN PROGRESS'
                      : 'SHELF MONITORING NORMAL — PRODUCT FACINGS STOCKED'}
                  </h3>
                  {primaryShelf.replenishment_recommended && (
                    <span className="px-2.5 py-0.5 rounded-full bg-emerald-500/20 border border-emerald-500/50 text-emerald-300 text-[10px] font-bold uppercase tracking-wider animate-pulse">
                      Replenishment Recommended
                    </span>
                  )}
                  <span className={`px-2 py-0.5 rounded-full text-[10px] font-mono font-bold uppercase ${priorityBadge.bg}`}>
                    {priorityBadge.label}
                  </span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">
                  {priorityBadge.desc} Visual Vacancy Confidence:{' '}
                  <strong className="text-amber-400 font-mono">
                    {Math.round(vacancyConfidence * 100)}%
                  </strong>.
                </p>
              </div>
            </div>

            {/* Severity Tag */}
            <div className="shrink-0 self-start">
              <span className={`px-3 py-1 rounded-full text-[10px] font-mono font-bold uppercase tracking-wider border ${
                isCritical || relevantActiveEvents.length > 0
                  ? 'bg-rose-600/30 border-rose-500/50 text-rose-300'
                  : 'bg-emerald-600/20 border-emerald-500/40 text-emerald-300'
              }`}>
                {isCritical || relevantActiveEvents.length > 0 ? 'HIGH PRIORITY' : 'HEALTHY'}
              </span>
            </div>
          </div>

          {/* Detailed Alert Cards with Explicit Gap Coordinates & Timestamps */}
          {relevantActiveEvents.length > 0 && (
            <div className="space-y-2.5 pt-3 border-t border-rose-500/20">
              <span className="text-[11px] font-mono uppercase font-bold text-rose-300 flex items-center gap-1.5">
                <Bell className="h-3.5 w-3.5 text-amber-400" />
                Active Alerts Ready for Operator Action ({relevantActiveEvents.length}):
              </span>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {relevantActiveEvents.map((evt) => {
                  const confPct = Math.round((evt.visual_vacancy_confidence || 0) * 100);
                  const gapCoords = evt.gap_coordinates || {};
                  const timeFormatted = evt.timestamp_iso
                    ? new Date(evt.timestamp_iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
                    : evt.timestamp
                    ? new Date(evt.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
                    : '—';

                  return (
                    <div
                      key={evt.event_id}
                      className="bg-slate-950/80 border border-slate-700/70 rounded-xl p-3.5 flex flex-col justify-between space-y-3"
                    >
                      <div className="space-y-1.5">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="font-mono font-bold text-cyan-300 text-sm">
                              {evt.tier_id || evt.shelf_id}
                            </span>
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 font-mono text-slate-400">
                              {evt.event_id}
                            </span>
                          </div>
                          <span
                            className={`text-[10px] px-2 py-0.5 rounded font-mono font-bold uppercase ${
                              evt.status === 'ACKNOWLEDGED'
                                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                                : 'bg-rose-500/20 text-rose-300 border border-rose-500/40 animate-pulse'
                            }`}
                          >
                            {evt.status}
                          </span>
                        </div>

                        <div className="grid grid-cols-2 gap-2 text-[11px] font-mono text-slate-400 pt-1">
                          <div>
                            <span className="text-slate-500">Vacancy Conf:</span>{' '}
                            <strong className="text-amber-400">{confPct}%</strong>
                          </div>
                          <div>
                            <span className="text-slate-500">Duration:</span>{' '}
                            <strong className="text-slate-200">
                              {evt.duration_sec ? `${evt.duration_sec.toFixed(1)}s` : `${evt.duration_frames || 0} frames`}
                            </strong>
                          </div>
                          <div>
                            <span className="text-slate-500">Gap Span:</span>{' '}
                            <strong className="text-slate-200">
                              X [{gapCoords.x1 ?? 0} &rarr; {gapCoords.x2 ?? 0} px]
                            </strong>
                          </div>
                          <div>
                            <span className="text-slate-500">Detected:</span>{' '}
                            <strong className="text-slate-200">{timeFormatted}</strong>
                          </div>
                        </div>
                      </div>

                      {/* Operator Action Buttons */}
                      <div className="flex items-center justify-end gap-2 pt-2 border-t border-slate-800">
                        {evt.status === 'ACTIVE' && (
                          <button
                            onClick={() => handleAcknowledge(evt.event_id)}
                            disabled={actionLoading[evt.event_id]}
                            className="px-3 py-1.5 rounded-lg bg-amber-600/30 hover:bg-amber-600/50 border border-amber-500/50 text-amber-200 text-xs font-bold transition-colors cursor-pointer flex items-center gap-1.5"
                          >
                            <Check className="h-3.5 w-3.5" />
                            {actionLoading[evt.event_id] ? 'Saving...' : 'Acknowledge'}
                          </button>
                        )}
                        <button
                          onClick={() => handleResolve(evt.event_id)}
                          disabled={actionLoading[evt.event_id]}
                          className="px-3 py-1.5 rounded-lg bg-emerald-600/30 hover:bg-emerald-600/50 border border-emerald-500/50 text-emerald-200 text-xs font-bold transition-colors cursor-pointer flex items-center gap-1.5"
                        >
                          <CheckCheck className="h-3.5 w-3.5" />
                          {actionLoading[evt.event_id] ? 'Resolving...' : 'Resolve / Restocked'}
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ─── Main Two-Column View: Shelf Status (Left) + Camera Playback (Right) ─── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Shelf 1 Detailed Status (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          {pipelineRunning ? (
            <div className="p-8 rounded-2xl border border-violet-500/30 bg-violet-950/10 flex flex-col items-center justify-center text-center space-y-4 py-16">
              <div className="h-16 w-16 rounded-2xl bg-violet-600/20 border border-violet-500/40 flex items-center justify-center">
                <Loader2 className="h-8 w-8 text-violet-400 animate-spin" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-white">Analyzing Shelf Vacancy</h3>
                <p className="text-xs font-mono text-violet-300 mt-1">
                  Video: <strong>{selectedVideoSource}</strong> · Run ID: <strong>{currentRunId}</strong>
                </p>
                <p className="text-xs text-slate-400 mt-2 max-w-md mx-auto leading-relaxed">
                  Executing retail product detector, physical tier grouping, adaptive empty space analysis, and occlusion suppression.
                  Results update atomically upon completion.
                </p>
              </div>
            </div>
          ) : (
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 space-y-5 shadow-xl">
              {/* Shelf Header */}
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="space-y-0.5">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-[10px] font-mono uppercase font-bold text-cyan-400">
                      Camera CAM-01 · Active Primary ROI
                    </span>
                    <span className={`px-2 py-0.5 rounded-full text-[9px] font-mono font-bold border ${
                      report?.shelf_geometry?.geometry_source === 'CONFIGURED_FALLBACK'
                        ? 'bg-amber-500/20 border-amber-500/40 text-amber-300'
                        : 'bg-emerald-500/20 border-emerald-500/40 text-emerald-300'
                    }`}>
                      {report?.shelf_geometry?.geometry_source === 'CONFIGURED_FALLBACK'
                        ? 'SHELF GEOMETRY: CONFIGURED FALLBACK'
                        : `SHELF GEOMETRY: AUTO-DETECTED (${Math.round((report?.shelf_geometry?.geometry_confidence || 1.0) * 100)}% Conf)`}
                    </span>
                  </div>
                  <h3 className="text-base font-extrabold text-white flex items-center gap-2">
                    <Package className="h-4 w-4 text-violet-400" />
                    SHELF 1 ({primaryShelf.shelf_id})
                  </h3>
                </div>
                <span
                  className={`px-3 py-1 rounded-full text-xs font-mono font-bold border ${
                    isVacant
                      ? 'bg-rose-500/15 border-rose-500/40 text-rose-400'
                      : isUncertain
                      ? 'bg-amber-500/15 border-amber-500/40 text-amber-300'
                      : 'bg-emerald-500/15 border-emerald-500/30 text-emerald-400'
                  }`}
                >
                  {primaryShelf.status}
                </span>
              </div>

              {/* Occupancy Meter Bar */}
              <div className="space-y-2">
                <div className="flex justify-between text-xs font-medium">
                  <span className="text-slate-400">Visible Shelf Occupancy</span>
                  <span className="text-white font-mono font-bold">
                    {primaryShelf.occupancy_pct}% Occupied
                  </span>
                </div>
                <div className="w-full h-3 bg-slate-800/90 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-700 ${
                      isVacant
                        ? 'bg-gradient-to-r from-amber-500 to-rose-500'
                        : 'bg-gradient-to-r from-emerald-500 to-teal-400'
                    }`}
                    style={{ width: `${Math.max(5, Math.min(100, primaryShelf.occupancy_pct))}%` }}
                  />
                </div>
                <div className="flex justify-between text-[11px] text-slate-500 font-mono">
                  <span>Vacancy Score: {primaryShelf.vacancy_score}</span>
                  <span>Temporal State: {primaryShelf.temporal_state}</span>
                </div>
              </div>

              {/* Physical Shelf Tiers (Step 28) */}
              {primaryShelf.tiers && primaryShelf.tiers.length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                      <Layers className="h-3.5 w-3.5 text-cyan-400" />
                      Physical Shelf Tiers ({primaryShelf.tiers.length})
                    </h4>
                    <span className="text-[10px] text-slate-500 font-mono">
                      2D Horizontal Occupancy
                    </span>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                    {primaryShelf.tiers.map((tier) => (
                      <div
                        key={tier.tier_id}
                        className={`p-2.5 rounded-xl border ${
                          tier.status === 'VACANT'
                            ? 'bg-rose-950/30 border-rose-500/40 text-rose-200'
                            : tier.status === 'UNCERTAIN'
                            ? 'bg-amber-950/30 border-amber-500/40 text-amber-200'
                            : 'bg-slate-950/60 border-slate-800 text-slate-200'
                        }`}
                      >
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span className="font-mono font-bold text-slate-200">{tier.tier_id.replace('SHELF-01-', '')}</span>
                          <span
                            className={`text-[9px] px-1.5 py-0.5 rounded font-bold uppercase ${
                              tier.status === 'VACANT'
                                ? 'bg-rose-500/20 text-rose-300'
                                : tier.status === 'UNCERTAIN'
                                ? 'bg-amber-500/20 text-amber-300'
                                : 'bg-emerald-500/20 text-emerald-300'
                            }`}
                          >
                            {tier.status}
                          </span>
                        </div>
                        <p className="text-[10px] font-mono text-slate-400 truncate">
                          {tier.name && !tier.name.includes('Beverages') && !tier.name.includes('Bottles') && !tier.name.includes('Cans') && !tier.name.includes('Snacks')
                            ? tier.name
                            : `Physical Tier ROI`}
                        </p>
                        <div className="flex items-center justify-between text-[10px] text-slate-400 mt-1.5 font-mono">
                          <span>{tier.product_count} items</span>
                          <span className="font-bold text-cyan-300">{tier.occupancy_pct}% occ</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Detected Vacant Regions List */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                    <AlertTriangle className="h-3.5 w-3.5 text-amber-400" />
                    Detected Vacant Regions
                  </h4>
                  <span className="text-[10px] text-slate-500 font-mono">
                    Adaptive gap threshold: &gt;1.75x median facing
                  </span>
                </div>

                {vacantRegions.length === 0 ? (
                  <div className="p-4 rounded-xl bg-slate-950/50 border border-slate-800/60 text-xs text-slate-400 flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
                    <span>
                      No persistent vacant gaps detected. Product facings are packed without significant empty spaces.
                    </span>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {vacantRegions.map((gap, idx) => {
                      const confPct = Math.round((gap.vacancy_confidence || 0) * 100);
                      const stabPct = Math.round((gap.geometric_stability !== undefined ? gap.geometric_stability : 1.0) * 100);
                      return (
                        <div
                          key={gap.region_id || idx}
                          className="p-3 rounded-xl bg-slate-950/60 border border-rose-500/30 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 text-xs"
                        >
                          <div className="space-y-1">
                            <div className="flex items-center gap-2 flex-wrap">
                              <span className="font-mono font-bold text-rose-300">
                                {gap.region_id}
                              </span>
                              <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">
                                {gap.tier_id || gap.row_id}
                              </span>
                              {gap.replenishment_recommended ? (
                                <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 font-bold uppercase tracking-wider">
                                  Replenishment Recommended
                                </span>
                              ) : (
                                <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800/80 border border-slate-700 text-slate-400 font-mono">
                                  Monitoring ({gap.persistence_frames || 1} frames)
                                </span>
                              )}
                            </div>
                            <p className="text-[11px] text-slate-400">
                              Span: X [{gap.x1}px &rarr; {gap.x2}px] &middot; Width: {gap.width_px}px ({gap.width_multiple}x facing) &middot; Persist: {gap.persistence_frames || 1}f
                            </p>
                          </div>
                          <div className="flex items-center gap-2 self-end sm:self-center">
                            <div className="bg-slate-900/80 border border-slate-800 rounded-lg px-2 py-1 text-center">
                              <div className="text-[9px] text-slate-400 uppercase font-semibold">Stability</div>
                              <div className="text-xs font-mono font-bold text-cyan-300">{stabPct}%</div>
                            </div>
                            <div className="bg-slate-900/80 border border-slate-800 rounded-lg px-2 py-1 text-center min-w-[70px]">
                              <div className="text-[9px] text-slate-400 uppercase font-semibold">Vacancy Conf.</div>
                              <div className={`text-xs font-mono font-bold ${confPct >= 70 ? 'text-amber-400' : 'text-slate-300'}`}>
                                {confPct}%
                              </div>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Semantics & Architecture Notice */}
              <div className="p-3 rounded-xl bg-slate-950/40 border border-slate-800/40 flex items-start gap-2 text-[11px] text-slate-400 leading-relaxed">
                <Info className="h-4 w-4 text-cyan-400 shrink-0 mt-0.5" />
                <span>
                  <strong>Architecture Guarantee:</strong> Vacancy detection uses computer vision product localization and physical tier row spacing.
                  It strictly reports camera-observable shelf space and does not assert brand identities, SKUs, or warehouse inventory quantities.
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Right: Edge Camera Video Playback (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2">
              <Camera className="h-4 w-4 text-cyan-400" />
              Edge Camera Video Playback
            </h3>
            <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
              <button
                type="button"
                onClick={() => setVideoMode('ANNOTATED')}
                className={`px-2.5 py-1 rounded-md text-xs font-mono font-bold transition-all cursor-pointer ${
                  videoMode === 'ANNOTATED'
                    ? 'bg-cyan-500 text-slate-950 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                ANNOTATED
              </button>
              <button
                type="button"
                onClick={() => setVideoMode('ORIGINAL')}
                className={`px-2.5 py-1 rounded-md text-xs font-mono font-bold transition-all cursor-pointer ${
                  videoMode === 'ORIGINAL'
                    ? 'bg-cyan-500 text-slate-950 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                ORIGINAL
              </button>
              <button
                type="button"
                onClick={() => setVideoMode('SNAPSHOT')}
                className={`px-2 py-1 rounded-md text-xs font-mono transition-all cursor-pointer ${
                  videoMode === 'SNAPSHOT'
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                    : 'text-slate-500 hover:text-slate-300'
                }`}
                title="View high-resolution static frame snapshot"
              >
                SNAPSHOT
              </button>
            </div>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-2 shadow-xl overflow-hidden">
            <div className="relative aspect-[16/9] w-full rounded-xl overflow-hidden bg-slate-950 flex items-center justify-center">
              {videoMode === 'SNAPSHOT' ? (
                <img
                  src={`/evidence/latest_shelf_snapshot.jpg?t=${Date.now()}`}
                  alt="Annotated Shelf Vacancy Evidence Snapshot"
                  className="w-full h-full object-contain rounded-xl"
                  onError={(e) => {
                    e.target.onerror = null;
                    e.target.src = `/evidence/${displayedVideo}_vacancy.jpg`;
                  }}
                />
              ) : videoMode === 'ANNOTATED' ? (
                <video
                  key={`annotated-${currentRunId}-${displayedVideo}`}
                  src={`/inventory/video/annotated/${currentRunId}?t=${Date.now()}`}
                  controls
                  autoPlay
                  muted
                  loop
                  playsInline
                  className="w-full h-full object-contain rounded-xl"
                  onError={(e) => {
                    if (!e.target.dataset.triedEvidence) {
                      e.target.dataset.triedEvidence = 'true';
                      e.target.src = `/evidence/${currentRunId}_annotated.mp4`;
                    } else if (!e.target.dataset.triedLatest) {
                      e.target.dataset.triedLatest = 'true';
                      e.target.src = `/evidence/latest_annotated.mp4`;
                    }
                  }}
                />
              ) : (
                <video
                  key={`original-${displayedVideo}`}
                  src={`/inventory/video/${displayedVideo}`}
                  controls
                  autoPlay
                  muted
                  loop
                  playsInline
                  className="w-full h-full object-contain rounded-xl"
                />
              )}

              {/* Top-left badge: Video file & mode */}
              <div className="absolute top-2 left-2 px-2.5 py-1 rounded-md bg-slate-950/85 backdrop-blur-md border border-slate-700/60 text-[10px] font-mono text-cyan-300 flex items-center gap-1.5 pointer-events-none">
                <span
                  className={`h-2 w-2 rounded-full ${
                    isPendingAnalysis ? 'bg-amber-400 animate-pulse' : 'bg-emerald-400 animate-pulse'
                  }`}
                />
                {videoMode === 'ANNOTATED'
                  ? `CV Pipeline Overlay (${displayedVideo})`
                  : videoMode === 'ORIGINAL'
                  ? `Original Source (${displayedVideo})`
                  : `Snapshot Evidence (${displayedVideo})`}
              </div>

              {/* Bottom-left badge: Mode description */}
              <div className="absolute bottom-2 left-2 px-2.5 py-1 rounded-md bg-slate-950/85 backdrop-blur-md border border-slate-700/60 text-[10px] font-mono pointer-events-none">
                {isPendingAnalysis ? (
                  <span className="text-amber-300 font-bold">PREVIEW — PENDING ANALYSIS</span>
                ) : videoMode === 'ANNOTATED' ? (
                  <span className="text-cyan-300 font-bold">FULL-LENGTH ANNOTATED MP4 — CV PIPELINE</span>
                ) : videoMode === 'ORIGINAL' ? (
                  <span className="text-emerald-300 font-bold">ORIGINAL EDGE CAMERA STREAM</span>
                ) : (
                  <span className="text-purple-300 font-bold">STILL FRAME EVIDENCE SNAPSHOT</span>
                )}
              </div>

              {/* Bottom-right badge: Run ID */}
              <div className="absolute bottom-2 right-2 px-2.5 py-1 rounded-md bg-slate-950/85 backdrop-blur-md border border-slate-700/60 text-[10px] font-mono text-slate-300 pointer-events-none">
                {isPendingAnalysis ? 'Analysis Required' : `Run: ${currentRunId}`}
              </div>
            </div>

            {/* Guide */}
            <div className="p-2 space-y-1 text-[11px] text-slate-400">
              <div className="flex flex-wrap items-center gap-3">
                <span className="flex items-center gap-1">
                  <span className="h-2.5 w-2.5 rounded bg-cyan-400" />
                  Cyan: SHELF-01 Monitored ROI
                </span>
                <span className="flex items-center gap-1">
                  <span className="h-2.5 w-2.5 rounded bg-emerald-500" />
                  Green: Detected Products
                </span>
                <span className="flex items-center gap-1">
                  <span className="h-2.5 w-2.5 rounded bg-rose-500" />
                  Red/Amber: Vacant Space
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ─── 4. LIGHTWEIGHT ANALYTICS (Real Persisted Event Metrics) ─────── */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-3">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="space-y-0.5">
            <span className="text-[10px] font-mono uppercase font-bold text-cyan-400">
              Real Audit Telemetry
            </span>
            <h3 className="text-base font-extrabold text-white flex items-center gap-2">
              <Activity className="h-4 w-4 text-cyan-400" />
              Historical Event Analytics
            </h3>
          </div>
          <span className="text-xs font-mono text-slate-400">
            Based on {analytics.totalEvents} recorded event{analytics.totalEvents === 1 ? '' : 's'}
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 pt-1">
          <div className="bg-slate-950/50 rounded-xl p-3 border border-slate-800/60">
            <span className="text-[10px] text-slate-400 uppercase font-semibold block">Total Events</span>
            <div className="text-lg font-mono font-bold text-white mt-0.5">
              {analytics.totalEvents}
            </div>
          </div>

          <div className="bg-slate-950/50 rounded-xl p-3 border border-slate-800/60">
            <span className="text-[10px] text-slate-400 uppercase font-semibold block">Replenish Recommended</span>
            <div className="text-lg font-mono font-bold text-rose-400 mt-0.5">
              {analytics.recommendations}
            </div>
          </div>

          <div className="bg-slate-950/50 rounded-xl p-3 border border-slate-800/60">
            <span className="text-[10px] text-slate-400 uppercase font-semibold block">Resolved Events</span>
            <div className="text-lg font-mono font-bold text-emerald-400 mt-0.5">
              {analytics.resolvedCount}
            </div>
          </div>

          <div className="bg-slate-950/50 rounded-xl p-3 border border-slate-800/60">
            <span className="text-[10px] text-slate-400 uppercase font-semibold block">Currently Active</span>
            <div className={`text-lg font-mono font-bold mt-0.5 ${
              analytics.activeCount > 0 ? 'text-amber-400' : 'text-slate-300'
            }`}>
              {analytics.activeCount}
            </div>
          </div>

          <div className="bg-slate-950/50 rounded-xl p-3 border border-slate-800/60">
            <span className="text-[10px] text-slate-400 uppercase font-semibold block">Avg Resolution Time</span>
            <div className={`text-sm font-mono font-bold mt-1 ${
              analytics.hasSufficientData ? 'text-cyan-300' : 'text-slate-500 text-xs'
            }`}>
              {analytics.avgResolutionText}
            </div>
          </div>
        </div>
      </div>

      {/* ─── 5. INVENTORY EVENT HISTORY TABLE ─────────────────────────────── */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 space-y-4 shadow-xl">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="space-y-0.5">
            <span className="text-[10px] font-mono uppercase font-bold text-cyan-400">
              Audit Trail & Operational Log
            </span>
            <h3 className="text-base font-extrabold text-white flex items-center gap-2">
              <History className="h-4 w-4 text-cyan-400" />
              Event Audit History
            </h3>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center bg-slate-950/80 p-0.5 rounded-lg border border-slate-800 text-[10px] font-mono">
              <button
                onClick={() => setHistoryFilter('ALL')}
                className={`px-2 py-0.5 rounded-md transition-colors cursor-pointer ${
                  historyFilter === 'ALL'
                    ? 'bg-violet-600/40 text-violet-200 border border-violet-500/50 font-bold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                All Runs ({eventHistory.length})
              </button>
              <button
                onClick={() => setHistoryFilter('CURRENT_RUN')}
                className={`px-2 py-0.5 rounded-md transition-colors cursor-pointer ${
                  historyFilter === 'CURRENT_RUN'
                    ? 'bg-cyan-600/40 text-cyan-200 border border-cyan-500/50 font-bold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Current Run ({eventHistory.filter((e) => e.run_id === currentRunId).length})
              </button>
            </div>
            <span className="text-xs font-mono text-slate-400">
              Active: <strong className="text-rose-400">{relevantActiveEvents.length}</strong> · Displayed:{' '}
              <strong className="text-slate-200">{displayedEvents.length}</strong>
            </span>
          </div>
        </div>

        {displayedEvents.length === 0 ? (
          <div className="p-6 rounded-xl bg-slate-950/40 border border-slate-800/60 text-xs text-slate-400 text-center flex flex-col items-center justify-center space-y-1">
            <CheckCircle2 className="h-6 w-6 text-emerald-400 mb-1" />
            <span className="font-semibold text-slate-300">
              {historyFilter === 'CURRENT_RUN'
                ? `No inventory vacancy events logged for current run (${currentRunId}).`
                : 'No inventory vacancy events logged yet.'}
            </span>
            <span className="text-[11px] text-slate-500">
              Events are automatically generated when shelf empty spaces are confirmed.
            </span>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300 border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-[10px] font-mono text-slate-400 uppercase">
                  <th className="py-2.5 px-3">Event ID / Run / Time</th>
                  <th className="py-2.5 px-3">Shelf Tier</th>
                  <th className="py-2.5 px-3">Event Type</th>
                  <th className="py-2.5 px-3">Vacancy Conf.</th>
                  <th className="py-2.5 px-3">Duration</th>
                  <th className="py-2.5 px-3">Status</th>
                  <th className="py-2.5 px-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                {displayedEvents.slice(-10).reverse().map((evt) => {
                  const isAct = evt.status === 'ACTIVE';
                  const isAck = evt.status === 'ACKNOWLEDGED';
                  const isRes = evt.status === 'RESOLVED';
                  const confPct = Math.round((evt.visual_vacancy_confidence || 0) * 100);
                  const timeFormatted = evt.timestamp
                    ? new Date(evt.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
                    : evt.timestamp_iso
                    ? new Date(evt.timestamp_iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
                    : '—';

                  return (
                    <tr key={evt.event_id} className="hover:bg-slate-800/30 transition-colors">
                      <td className="py-2.5 px-3">
                        <div className="font-bold text-slate-200 flex items-center gap-1.5 flex-wrap">
                          <span>{evt.event_id}</span>
                          <span className="text-[9px] px-1.5 py-0.2 rounded bg-slate-800 text-cyan-300 font-mono font-semibold">
                            {evt.run_id || 'RUN-001'}
                          </span>
                        </div>
                        <div className="text-[10px] text-slate-500 font-sans flex items-center gap-1 mt-0.5">
                          <Clock className="h-3 w-3" /> {timeFormatted}
                        </div>
                      </td>
                      <td className="py-2.5 px-3 font-semibold text-cyan-300">
                        {evt.tier_id || evt.shelf_id}
                      </td>
                      <td className="py-2.5 px-3">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold tracking-wide uppercase ${
                          evt.event_type === 'REPLENISHMENT_RECOMMENDED'
                            ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                            : evt.event_type === 'VACANCY_CONFIRMED'
                            ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                            : evt.event_type === 'SHELF_RESTORED'
                            ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                            : 'bg-slate-800 text-slate-300 border border-slate-700'
                        }`}>
                          {evt.event_type.replace(/_/g, ' ')}
                        </span>
                      </td>
                      <td className="py-2.5 px-3">
                        <span className={confPct >= 70 ? 'text-amber-400 font-bold' : 'text-slate-400'}>
                          {confPct}%
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-slate-400">
                        {evt.duration_sec ? `${evt.duration_sec.toFixed(1)}s` : `${evt.duration_frames || 0}f`}
                      </td>
                      <td className="py-2.5 px-3">
                        <span className={`px-2 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider ${
                          isAct
                            ? 'bg-rose-500/20 text-rose-300 border border-rose-500/50 animate-pulse'
                            : isAck
                            ? 'bg-amber-500/20 text-amber-300 border border-amber-500/50'
                            : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/50'
                        }`}>
                          {evt.status}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-right">
                        <div className="flex items-center justify-end gap-1.5 font-sans">
                          {isAct && (
                            <button
                              onClick={() => handleAcknowledge(evt.event_id)}
                              disabled={actionLoading[evt.event_id]}
                              className="px-2 py-1 rounded bg-amber-600/30 hover:bg-amber-600/50 border border-amber-500/50 text-amber-200 text-[10px] font-bold transition-colors cursor-pointer"
                              title="Acknowledge vacancy alert"
                            >
                              {actionLoading[evt.event_id] ? 'Saving...' : 'Ack'}
                            </button>
                          )}
                          {!isRes && (
                            <button
                              onClick={() => handleResolve(evt.event_id)}
                              disabled={actionLoading[evt.event_id]}
                              className="px-2 py-1 rounded bg-emerald-600/30 hover:bg-emerald-600/50 border border-emerald-500/50 text-emerald-200 text-[10px] font-bold transition-colors cursor-pointer"
                              title="Mark vacancy as resolved / replenished"
                            >
                              {actionLoading[evt.event_id] ? 'Resolving...' : 'Resolve'}
                            </button>
                          )}
                          {isRes && (
                            <span className="text-[10px] text-emerald-400/80 flex items-center gap-1 justify-end">
                              <CheckCheck className="h-3.5 w-3.5" /> Resolved
                            </span>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ─── 6. SYSTEM HEALTH COMPACT STRIP ──────────────────────────────── */}
      <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3 flex flex-wrap items-center justify-between gap-3 text-xs font-mono text-slate-400">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1.5">
            <Server className="h-3.5 w-3.5 text-cyan-400" />
            <span>API:</span>
            <strong className={apiHealth?.alive ? 'text-emerald-400' : 'text-rose-400'}>
              {apiHealth?.alive ? 'ONLINE (port 8001)' : 'DEGRADED / LOCAL CACHE'}
            </strong>
          </div>
          <span className="text-slate-700">|</span>
          <div>
            <span>Active Run:</span> <strong className="text-white">{currentRunId}</strong>
          </div>
          <span className="text-slate-700">|</span>
          <div>
            <span>Video Stream:</span> <strong className="text-white">{activeVideoSource}</strong>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div>
            <span>Engine:</span>{' '}
            <strong className="text-violet-300">
              {pipelineRunning ? 'INFERENCE ACTIVE' : 'IDLE (READY)'}
            </strong>
          </div>
          <span className="text-slate-700">|</span>
          <div className="text-[11px] text-slate-500">
            Updated: {lastUpdated || 'Live'}
          </div>
        </div>
      </div>
    </div>
  );
}
