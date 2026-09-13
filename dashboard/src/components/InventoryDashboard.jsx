import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Package,
  AlertTriangle,
  Clock,
  RefreshCw,
  RotateCcw,
  ShoppingCart,
  Wifi,
  WifiOff,
  Play,
  CheckCircle,
  Loader2,
  ClipboardList,
  History,
  Layers,
} from 'lucide-react';
import InventoryOverview from './InventoryOverview';
import SKUInventoryTable from './SKUInventoryTable';
import InventoryAlertsPanel from './InventoryAlertsPanel';
import InventoryRecentEvents from './InventoryRecentEvents';
import InventoryActionCenter from './InventoryActionCenter';
import InventoryRunHistory from './InventoryRunHistory';
import InventoryEvidenceViewer from './InventoryEvidenceViewer';
import InventoryRunControl from './InventoryRunControl';
import ShelfRackVisualizer from './ShelfRackVisualizer';
import {
  getInventoryReport,
  triggerInventoryRun,
  getInventoryHealth,
} from '../api/client';

// ─── Sub-tab definitions ──────────────────────────────────────────────────────
const TABS = [
  { id: 'rack',        label: 'Shelf 1 Vacancy Monitor', Icon: Package       },
  { id: 'history',     label: 'Run History',             Icon: History        },
];


// ─── Source badge ─────────────────────────────────────────────────────────────
function SourceBadge({ source, apiAlive, pipelineRunning }) {
  if (pipelineRunning) {
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-violet-500/15 border border-violet-500/30 text-violet-300 text-[10px] font-bold uppercase animate-pulse">
        <Loader2 className="h-3 w-3 animate-spin" />
        PIPELINE RUNNING
      </span>
    );
  }
  if (source === 'LIVE' || source === 'PIPELINE') {
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 text-[10px] font-bold uppercase">
        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
        LIVE / API
      </span>
    );
  }
  if (source === 'DISK') {
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-cyan-500/15 border border-cyan-500/30 text-cyan-400 text-[10px] font-bold uppercase">
        <Wifi className="h-3 w-3" />
        API / DISK CACHE
      </span>
    );
  }
  // DEMO fallback
  return (
    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-amber-500/15 border border-amber-500/30 text-amber-300 text-[10px] font-bold uppercase">
      <WifiOff className="h-3 w-3" />
      DEMO / STATIC
    </span>
  );
}

// ─── API status strip ─────────────────────────────────────────────────────────
function ApiStatusStrip({ apiHealth }) {
  if (!apiHealth) return null;
  return (
    <div className={`flex flex-wrap items-center gap-2 px-3 py-1.5 rounded-lg border text-[10px] font-mono
      ${apiHealth.alive
        ? 'bg-emerald-500/5 border-emerald-500/20 text-emerald-400'
        : 'bg-slate-800/50 border-slate-700/50 text-slate-500'}`}>
      {apiHealth.alive
        ? <CheckCircle className="h-3 w-3" />
        : <WifiOff className="h-3 w-3" />}
      <span>
        Inventory API: {apiHealth.alive ? 'ONLINE (port 8001)' : 'OFFLINE — showing cached/demo data'}
      </span>
      {apiHealth.alive && (
        <>
          <span className="text-slate-500">·</span>
          <span>Cache: {apiHealth.cacheSource}</span>
          {apiHealth.cacheTimestamp && (
            <>
              <span className="text-slate-500">·</span>
              <span>Last run: {apiHealth.cacheTimestamp.slice(0, 19).replace('T', ' ')}</span>
            </>
          )}
        </>
      )}
    </div>
  );
}

// ─── Main Inventory Dashboard ─────────────────────────────────────────────────
export default function InventoryDashboard() {
  const [activeTab,          setActiveTab]          = useState('rack');
  const [report,             setReport]             = useState(null);
  const [reportSource,       setReportSource]       = useState('NONE');
  const [apiHealth,          setApiHealth]          = useState(null);
  const [loading,            setLoading]            = useState(true);
  const [error,              setError]              = useState(null);
  const [lastUpdated,        setLastUpdated]        = useState(null);
  const [refreshing,         setRefreshing]         = useState(false);
  const [runTriggered,       setRunTriggered]       = useState(false);
  const [pipelineRunning,    setPipelineRunning]    = useState(false);
  const [runEvidenceModal,   setRunEvidenceModal]   = useState(null);
  const [selectedVideoSource, setSelectedVideoSource] = useState('shelf_pan_demo.mp4');
  const [activeVideoSource,  setActiveVideoSource]  = useState('shelf_pan_demo.mp4');
  const [activeRunId,        setActiveRunId]        = useState(null);
  const [targetRunId,        setTargetRunId]        = useState(null);

  // ─── Fetch report + health ────────────────────────────────────────────────
  const load = useCallback(async (manual = false) => {
    if (manual) setRefreshing(true);
    else        setLoading(true);
    setError(null);

    try {
      // Parallel: health check + report fetch
      const [healthResult, reportResult] = await Promise.allSettled([
        getInventoryHealth(),
        getInventoryReport(),
      ]);

      // Health
      if (healthResult.status === 'fulfilled' && healthResult.value) {
        setApiHealth(healthResult.value);
        setPipelineRunning(healthResult.value.pipelineRunning ?? false);
      }

      // Report
      if (reportResult.status === 'fulfilled' && reportResult.value?.data) {
        const repData = reportResult.value.data;
        setReport(repData);
        setReportSource(reportResult.value.source);
        setLastUpdated(new Date().toLocaleTimeString());
        if (repData.video_source) {
          const vName = repData.video_source.split(/[/\\]/).pop();
          if (vName) {
            setActiveVideoSource(vName);
            setSelectedVideoSource((prev) => prev || vName);
          }
        }
        if (repData.run_id) {
          setActiveRunId(repData.run_id);
        }
      } else {
        setError('No inventory report available. Start the inventory API (run_inventory_api.py) or run demo_inventory_report.py first.');
      }
    } catch (err) {
      setError(err.message ?? 'Unexpected error loading inventory report.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  // ─── Guarded Run Completion Handler ──────────────────────────────────────
  // Guarantees that we ONLY accept and display the report when its run_id matches the completed run.
  const handleRunComplete = useCallback(async (completedRunId) => {
    setRefreshing(true);
    setError(null);
    const expectedId = completedRunId || targetRunId;

    let retries = 0;
    while (retries < 8) {
      try {
        const [healthResult, reportResult] = await Promise.allSettled([
          getInventoryHealth(),
          getInventoryReport(),
        ]);

        if (healthResult.status === 'fulfilled' && healthResult.value) {
          setApiHealth(healthResult.value);
        }

        if (reportResult.status === 'fulfilled' && reportResult.value?.data) {
          const repData = reportResult.value.data;
          // Guard: verify returned report matches the newly completed run ID
          if (!expectedId || repData.run_id === expectedId) {
            setReport(repData);
            setReportSource(reportResult.value.source);
            setActiveRunId(repData.run_id);
            if (repData.video_source) {
              const vName = repData.video_source.split(/[/\\]/).pop();
              if (vName) {
                setActiveVideoSource(vName);
                setSelectedVideoSource(vName);
              }
            }
            setLastUpdated(new Date().toLocaleTimeString());
            setPipelineRunning(false);
            setRefreshing(false);
            setTargetRunId(null);
            return;
          } else {
            console.log(`[InventoryDashboard] Guard waiting: got ${repData.run_id}, expecting ${expectedId} (retry ${retries + 1}/8)`);
          }
        }
      } catch (err) {
        console.warn('[InventoryDashboard] Error during guarded fetch:', err);
      }
      retries += 1;
      await new Promise((resolve) => setTimeout(resolve, 350));
    }

    // Retries completed: fallback to refresh latest
    await load(true);
    setPipelineRunning(false);
    setTargetRunId(null);
  }, [targetRunId, load]);

  // ─── Trigger fresh pipeline run ───────────────────────────────────────────
  const handleRunPipeline = useCallback(async () => {
    setRunTriggered(true);
    const resp = await triggerInventoryRun(null, selectedVideoSource);
    if (resp?.run_id) {
      setTargetRunId(resp.run_id);
      setPipelineRunning(true);
    } else if (resp) {
      setPipelineRunning(true);
    } else {
      setError('Could not trigger pipeline — is the inventory API running? (python run_inventory_api.py)');
    }
    setTimeout(() => setRunTriggered(false), 3000);
  }, [selectedVideoSource]);

  // ─── Derived ──────────────────────────────────────────────────────────────
  const allAlerts  = report?.active_alerts ?? [];
  const skuSummary = report?.sku_inventory_summary ?? [];
  const events     = report?.recent_events ?? [];
  const highAlerts = allAlerts.filter((a) => a.severity === 'HIGH').length;
  const verifCount = allAlerts.filter((a) => a.requires_verification).length;
  const isLive     = reportSource === 'LIVE' || reportSource === 'PIPELINE' || reportSource === 'DISK';

  return (
    <div className="space-y-5">
      {/* ── Panel header ── */}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <Package className="h-5 w-5 text-cyan-400" />
            Shelf Vacancy & Empty-Space Monitor (Step 25)
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            SHELF-01 · Generic Vacancy Detection · Adaptive Product Geometry · Temporal Confirmation
          </p>
        </div>


        <div className="flex flex-wrap items-center gap-2">
          {/* Source badge */}
          <SourceBadge source={reportSource} apiAlive={apiHealth?.alive} pipelineRunning={pipelineRunning} />

          {/* Last updated */}
          {lastUpdated && (
            <span className="text-[10px] font-mono text-slate-500">Updated {lastUpdated}</span>
          )}

          {/* Run Pipeline button (only when API is live) */}
          {apiHealth?.alive && !pipelineRunning && (
            <button
              id="btn-run-pipeline"
              onClick={handleRunPipeline}
              disabled={runTriggered || pipelineRunning}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-violet-600/20 border border-violet-500/40
                         text-xs font-semibold text-violet-300 hover:bg-violet-600/30 transition-colors disabled:opacity-50"
            >
              <Play className="h-3.5 w-3.5" />
              {runTriggered ? 'Starting…' : 'Run Pipeline'}
            </button>
          )}

          {/* Reload report button */}
          <button
            id="btn-reload-report"
            onClick={() => load(true)}
            disabled={refreshing || pipelineRunning}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700/60
                       text-xs font-semibold text-slate-200 hover:bg-slate-700 transition-colors disabled:opacity-50"
          >
            {refreshing
              ? <RotateCcw className="h-3.5 w-3.5 animate-spin" />
              : <RefreshCw className="h-3.5 w-3.5" />}
            Refresh
          </button>
        </div>
      </div>

      {/* ── API status strip ── */}
      <ApiStatusStrip apiHealth={apiHealth} />

      {/* ── Error banner ── */}
      {error && (
        <div className="flex items-start gap-2.5 p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-sm text-rose-300">
          <AlertTriangle className="h-5 w-5 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold">Could not load inventory report</p>
            <p className="text-xs text-rose-400/80 mt-0.5">{error}</p>
            <p className="text-[10px] text-rose-400/60 mt-1 font-mono">
              Start API: <code>python run_inventory_api.py</code> &nbsp;|&nbsp;
              Or run:    <code>python scripts/demo_inventory_report.py</code>
            </p>
          </div>
        </div>
      )}

      {/* ── Pipeline-running notice ── */}
      {pipelineRunning && (
        <div className="flex items-center gap-2.5 p-3 rounded-xl bg-violet-500/10 border border-violet-500/30 text-sm text-violet-300 animate-pulse">
          <Loader2 className="h-4 w-4 animate-spin shrink-0" />
          <span>
            Pipeline is running (retail_detector_exp2.pt → ByteTrack → SKU recognition → aggregation → alerts).
            Report will refresh automatically when complete.
          </span>
        </div>
      )}

      {/* ── Quick-stat strip ── */}
      {!loading && report && !pipelineRunning && (
        <div className="flex flex-wrap gap-3">
          {highAlerts > 0 && (
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-rose-500/15 border border-rose-500/30 text-rose-300 text-xs font-bold animate-pulse">
              <AlertTriangle className="h-3.5 w-3.5" />
              {highAlerts} HIGH alert{highAlerts > 1 ? 's' : ''}
            </span>
          )}
          {verifCount > 0 && (
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-500/15 border border-amber-500/30 text-amber-300 text-xs font-bold">
              {verifCount} requiring field verification
            </span>
          )}
          {highAlerts === 0 && verifCount === 0 && (
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 text-xs font-bold">
              <CheckCircle className="h-3.5 w-3.5" />
              No high-severity alerts
            </span>
          )}
          {/* Show where data came from */}
          <div className="ml-auto flex items-center gap-2 text-[10px] font-mono">
            <span className="text-slate-400">
              Active Run: <strong className="text-white">{activeRunId || 'RUN'}</strong> ({activeVideoSource})
            </span>
            {selectedVideoSource !== activeVideoSource && (
              <span className="px-2 py-0.5 rounded bg-amber-500/15 border border-amber-500/30 text-amber-300 font-bold">
                Pending: {selectedVideoSource}
              </span>
            )}
          </div>
        </div>
      )}

      {!loading && pipelineRunning && (
        <div className="flex items-center gap-2.5 px-3.5 py-2 rounded-xl bg-violet-500/10 border border-violet-500/30 text-xs text-violet-300">
          <Loader2 className="h-4 w-4 animate-spin text-violet-400 shrink-0" />
          <span>
            Executing Retail Analysis on <strong>{selectedVideoSource}</strong> ({targetRunId || 'Active Run'}). Previous metrics suppressed until completion.
          </span>
        </div>
      )}

      {/* ── Run Control Section ── */}
      <InventoryRunControl
        selectedVideoSource={selectedVideoSource}
        activeVideoSource={activeVideoSource}
        onRunComplete={handleRunComplete}
        onRunStart={({ video, runId }) => {
          setSelectedVideoSource(video);
          if (runId) setTargetRunId(runId);
          setPipelineRunning(true);
        }}
        onVideoChange={(vid) => setSelectedVideoSource(vid)}
        apiAlive={apiHealth?.alive}
      />

      {/* ── Sub-tab bar ── */}
      <div className="flex gap-1 border-b border-slate-800/80 pb-0 overflow-x-auto">
        {TABS.map(({ id, label, Icon }) => {
          // Show a red dot on alert tabs when there are high-severity unresolved items
          const alertPing = (id === 'alerts' || id === 'operations') && highAlerts > 0 && !pipelineRunning;
          return (
            <button
              key={id}
              id={`inventory-tab-${id}`}
              onClick={() => setActiveTab(id)}
              className={`relative flex items-center gap-1.5 px-4 py-2.5 text-xs font-semibold transition-colors whitespace-nowrap
                border-b-2 -mb-px
                ${activeTab === id
                  ? 'border-violet-500 text-violet-300'
                  : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-600'}`}
            >
              <Icon className="h-3.5 w-3.5" />
              {label}
              {alertPing && (
                <span className="absolute top-1.5 right-1.5 h-1.5 w-1.5 rounded-full bg-rose-500 animate-pulse" />
              )}
            </button>
          );
        })}
      </div>

      {/* ── Tab content ── */}
      <div>
        {activeTab === 'rack' && (
          <ShelfRackVisualizer
            report={report}
            selectedVideoSource={selectedVideoSource}
            activeVideoSource={activeVideoSource}
            activeRunId={activeRunId}
            pipelineRunning={pipelineRunning}
            apiHealth={apiHealth}
            lastUpdated={lastUpdated}
          />
        )}
        {activeTab === 'overview' && (
          <InventoryOverview report={report} loading={loading} pipelineRunning={pipelineRunning} />
        )}
        {activeTab === 'skus' && (
          <SKUInventoryTable skuSummary={skuSummary} loading={loading} pipelineRunning={pipelineRunning} />
        )}
        {activeTab === 'operations' && (
          <InventoryActionCenter report={report} loading={loading} pipelineRunning={pipelineRunning} />
        )}
        {activeTab === 'alerts' && (
          <InventoryAlertsPanel alerts={allAlerts} loading={loading} pipelineRunning={pipelineRunning} />
        )}
        {activeTab === 'events' && (
          <InventoryRecentEvents events={events} loading={loading} pipelineRunning={pipelineRunning} />
        )}
        {activeTab === 'history' && (
          <InventoryRunHistory report={report} onViewEvidence={setRunEvidenceModal} />
        )}
      </div>

      {/* ── Run Evidence Modal Replay ── */}
      {runEvidenceModal && (
        <InventoryEvidenceViewer
          evidence={runEvidenceModal}
          onClose={() => setRunEvidenceModal(null)}
        />
      )}
    </div>
  );
}
