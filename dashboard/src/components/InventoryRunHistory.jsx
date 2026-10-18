/**
 * InventoryRunHistory.jsx — Step 13: Run History & Snapshot Comparison
 *
 * Provides historical run tracking and delta comparisons for retail managers.
 * - Compares the Current Live Pipeline Run against earlier runs.
 * - Historical runs are visibly labeled "DEMO HISTORICAL SNAPSHOT" everywhere
 *   and are not presented as actual model measurements or physical ground truth.
 * - Generates metric deltas and SKU comparisons dynamically from data.
 * - Connects to the existing InventoryEvidenceViewer for video replay.
 * - Purely presentation layer: does not touch detector, tracker, or model weights.
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  History,
  GitCompare,
  Film,
  AlertTriangle,
  CheckCircle2,
  ShieldAlert,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  Clock,
  Video,
  Layers,
  Package,
  Calendar,
  Sparkles,
  Info,
  ChevronRight,
  Check,
} from 'lucide-react';

// Health badge styles
const HEALTH_STYLES = {
  GOOD: {
    badge: 'text-emerald-400 bg-emerald-500/15 border-emerald-500/30',
    dot: 'bg-emerald-400',
    label: 'Healthy Shelf',
  },
  ATTENTION_NEEDED: {
    badge: 'text-amber-300 bg-amber-500/15 border-amber-500/30',
    dot: 'bg-amber-400',
    label: 'Attention Needed',
  },
  VERIFICATION_REQUIRED: {
    badge: 'text-rose-300 bg-rose-500/15 border-rose-500/30',
    dot: 'bg-rose-400',
    label: 'Verification Required',
  },
};

// Status badge styles
const STATUS_STYLES = {
  IN_STOCK: 'text-emerald-300 bg-emerald-500/10 border-emerald-500/25',
  LOW_STOCK: 'text-amber-300 bg-amber-500/10 border-amber-500/25',
  OUT_OF_VIEW: 'text-rose-300 bg-rose-500/10 border-rose-500/25',
};

export default function InventoryRunHistory({ report, onViewEvidence }) {
  const [runsData, setRunsData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedPrevRunId, setSelectedPrevRunId] = useState('RUN-001');

  // Load run history dataset (re-fetches when a new report is generated)
  useEffect(() => {
    fetch(`/run_history.json?t=${Date.now()}`)
      .then((res) => {
        if (!res.ok) return fetch(`/inventory_run_history.json?t=${Date.now()}`);
        return res;
      })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (data?.runs) {
          setRunsData(data.runs);
        }
      })
      .catch((err) => {
        console.warn('Could not load run_history.json, using fallback:', err);
      })
      .finally(() => setLoading(false));
  }, [report?.timestamp_iso]);

  // Merge the latest run dynamically with the live report if available
  const runs = useMemo(() => {
    if (!runsData || runsData.length === 0) return [];
    return runsData.map((run) => {
      if (run.is_latest && report) {
        // Keep RUN-003 strictly in sync with live report
        const skus = (report.sku_inventory_summary || []).map((s) => ({
          sku_id: s.sku_id,
          product_name: s.product_name,
          category: s.category,
          stable_facings: s.stable_facings ?? 0,
          shelf_status: s.shelf_status ?? 'UNKNOWN',
        }));

        return {
          ...run,
          timestamp: report.timestamp_iso || run.timestamp,
          active_visible_facings: report.total_active_visible_facings ?? run.active_visible_facings,
          stable_facings: report.stable_facings ?? run.stable_facings,
          uncertain_facings: report.uncertain_facings ?? run.uncertain_facings,
          unknown_facings: report.total_unknown_facings ?? run.unknown_facings,
          catalog_skus_visible: report.catalog_skus_visible ?? run.catalog_skus_visible,
          in_stock_skus: report.in_stock_skus_count ?? run.in_stock_skus,
          low_stock_skus: report.low_stock_skus_count ?? run.low_stock_skus,
          out_of_view_skus: report.out_of_view_skus_count ?? run.out_of_view_skus,
          total_alerts: report.total_alerts_fired ?? run.total_alerts,
          high_alerts: report.alerts_by_severity?.HIGH ?? run.high_alerts,
          medium_alerts: report.alerts_by_severity?.MEDIUM ?? run.medium_alerts,
          verification_required: report.verification_required_count ?? run.verification_required,
          shelf_health: report.shelf_health || run.shelf_health,
          health_reason: report.health_reason || run.health_reason,
          skus: skus.length > 0 ? skus : run.skus,
        };
      }
      return run;
    });
  }, [runsData, report]);

  // Current (latest) run and previous runs
  const currentRun = useMemo(() => runs.find((r) => r.is_latest) || runs[runs.length - 1], [runs]);
  const previousRuns = useMemo(() => runs.filter((r) => !r.is_latest), [runs]);

  // Selected previous run for comparison
  const selectedPrevRun = useMemo(() => {
    return previousRuns.find((r) => r.run_id === selectedPrevRunId) || previousRuns[0] || null;
  }, [previousRuns, selectedPrevRunId]);

  // Helper to compute comparison metric diff
  const computeDiff = (currentVal = 0, prevVal = 0, invertGood = false) => {
    const diff = currentVal - prevVal;
    const isUp = diff > 0;
    const isDown = diff < 0;
    const isNeutral = diff === 0;

    let color = 'text-slate-400';
    if (!isNeutral) {
      if (!invertGood) {
        // Higher is better (e.g. stable facings, in-stock skus)
        color = isUp ? 'text-emerald-400' : 'text-rose-400';
      } else {
        // Lower is better (e.g. alerts, unknown facings, out-of-view)
        color = isDown ? 'text-emerald-400' : 'text-rose-400';
      }
    }

    return { diff, isUp, isDown, isNeutral, color };
  };

  // 10 Comparison metrics definition
  const comparisonMetrics = useMemo(() => {
    if (!currentRun || !selectedPrevRun) return [];
    return [
      {
        key: 'active_visible_facings',
        label: 'Active Visible Facings',
        cur: currentRun.active_visible_facings,
        prev: selectedPrevRun.active_visible_facings,
        invertGood: false,
      },
      {
        key: 'stable_facings',
        label: 'Stable Facings',
        cur: currentRun.stable_facings,
        prev: selectedPrevRun.stable_facings,
        invertGood: false,
      },
      {
        key: 'unknown_facings',
        label: 'Unknown Facings',
        cur: currentRun.unknown_facings,
        prev: selectedPrevRun.unknown_facings,
        invertGood: true,
      },
      {
        key: 'catalog_skus_visible',
        label: 'Visible Catalog SKUs',
        cur: currentRun.catalog_skus_visible,
        prev: selectedPrevRun.catalog_skus_visible,
        invertGood: false,
      },
      {
        key: 'in_stock_skus',
        label: 'IN_STOCK SKUs',
        cur: currentRun.in_stock_skus,
        prev: selectedPrevRun.in_stock_skus,
        invertGood: false,
      },
      {
        key: 'low_stock_skus',
        label: 'LOW_STOCK SKUs',
        cur: currentRun.low_stock_skus,
        prev: selectedPrevRun.low_stock_skus,
        invertGood: true,
      },
      {
        key: 'out_of_view_skus',
        label: 'OUT_OF_VIEW SKUs',
        cur: currentRun.out_of_view_skus,
        prev: selectedPrevRun.out_of_view_skus,
        invertGood: true,
      },
      {
        key: 'total_alerts',
        label: 'Total Alerts',
        cur: currentRun.total_alerts,
        prev: selectedPrevRun.total_alerts,
        invertGood: true,
      },
      {
        key: 'high_alerts',
        label: 'HIGH Alerts',
        cur: currentRun.high_alerts,
        prev: selectedPrevRun.high_alerts,
        invertGood: true,
      },
      {
        key: 'verification_required',
        label: 'Verification Required',
        cur: currentRun.verification_required,
        prev: selectedPrevRun.verification_required,
        invertGood: true,
      },
    ];
  }, [currentRun, selectedPrevRun]);

  // Dynamic SKU comparison list
  const skuComparisons = useMemo(() => {
    if (!currentRun || !selectedPrevRun) return [];
    const prevSkuMap = new Map((selectedPrevRun.skus || []).map((s) => [s.sku_id, s]));

    return (currentRun.skus || []).map((curSku) => {
      const prevSku = prevSkuMap.get(curSku.sku_id) || {
        stable_facings: 0,
        shelf_status: 'UNKNOWN',
      };
      const facingDiff = (curSku.stable_facings ?? 0) - (prevSku.stable_facings ?? 0);
      const statusChanged = curSku.shelf_status !== prevSku.shelf_status;

      let transitionBadge = null;
      if (statusChanged) {
        if (prevSku.shelf_status === 'IN_STOCK' && curSku.shelf_status === 'OUT_OF_VIEW') {
          transitionBadge = { text: 'IN_STOCK → OUT_OF_VIEW', style: 'text-rose-300 bg-rose-500/20 border-rose-500/40' };
        } else if (prevSku.shelf_status === 'IN_STOCK' && curSku.shelf_status === 'LOW_STOCK') {
          transitionBadge = { text: 'IN_STOCK → LOW_STOCK', style: 'text-amber-300 bg-amber-500/20 border-amber-500/40' };
        } else if (prevSku.shelf_status === 'LOW_STOCK' && curSku.shelf_status === 'IN_STOCK') {
          transitionBadge = { text: 'LOW_STOCK → IN_STOCK', style: 'text-emerald-300 bg-emerald-500/20 border-emerald-500/40' };
        } else {
          transitionBadge = { text: `${prevSku.shelf_status} → ${curSku.shelf_status}`, style: 'text-slate-300 bg-slate-700/40 border-slate-600/40' };
        }
      }

      return {
        sku_id: curSku.sku_id,
        product_name: curSku.product_name,
        category: curSku.category,
        prevStable: prevSku.stable_facings ?? 0,
        curStable: curSku.stable_facings ?? 0,
        facingDiff,
        prevStatus: prevSku.shelf_status,
        curStatus: curSku.shelf_status,
        statusChanged,
        transitionBadge,
      };
    });
  }, [currentRun, selectedPrevRun]);

  // Handle opening video evidence for a run
  const handleOpenRunEvidence = (run) => {
    if (!onViewEvidence) return;
    onViewEvidence({
      alert_id: `RUN-${run.run_id}`,
      alert_type: 'SHELF_INVENTORY_RUN',
      event_type: 'RUN_EVIDENCE_REPLAY',
      sku_name: 'Shelf Panoramic Scan',
      sku_id: run.run_id,
      severity: run.shelf_health === 'GOOD' ? 'LOW' : 'HIGH',
      frame_index: run.total_frames || 75,
      timestamp_sec: 2.96,
      reason: `Run ${run.run_id} evidence replay. ${run.active_visible_facings} active visible facings, ${run.stable_facings} stable facings.`,
      requires_verification: (run.verification_required || 0) > 0,
      current_stable_facings: run.stable_facings,
      active_facings: run.active_visible_facings,
      _evidenceSource: 'run_history',
    });
  };

  const fmtTimestamp = (iso) => {
    if (!iso) return '—';
    try {
      const d = new Date(iso);
      return d.toLocaleString([], {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    } catch {
      return iso;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-12 text-slate-500">
        <History className="h-5 w-5 animate-spin mr-2" />
        <span>Loading run history...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* ── Banner: Context & Semantics ── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-4 rounded-xl bg-slate-800/40 border border-slate-700/50">
        <div className="flex items-start gap-3">
          <History className="h-5 w-5 text-violet-400 shrink-0 mt-0.5" />
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              Run History & Snapshot Comparison
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Compare current live pipeline results against earlier operational scans.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-[11px] font-mono text-slate-400 bg-slate-900/60 px-3 py-1.5 rounded-lg border border-slate-800">
          <Info className="h-3.5 w-3.5 text-violet-400" />
          <span>Historical scans are demo reference baselines for UI comparison</span>
        </div>
      </div>

      {/* ── Runs List / Timeline ── */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">
            Recorded Inventory Runs
          </h4>
          <span className="text-[11px] text-slate-500">
            Click a previous run to compare against the Current Run
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {runs.map((run) => {
            const isLatest = run.is_latest;
            const isSelected = selectedPrevRun?.run_id === run.run_id;
            const health = HEALTH_STYLES[run.shelf_health] || HEALTH_STYLES.ATTENTION_NEEDED;

            return (
              <div
                key={run.run_id}
                onClick={() => {
                  if (!isLatest) setSelectedPrevRunId(run.run_id);
                }}
                className={`relative rounded-xl border p-4 transition-all flex flex-col justify-between ${
                  isLatest
                    ? 'bg-gradient-to-b from-violet-950/25 to-slate-900/80 border-violet-500/50 shadow-lg shadow-violet-950/20 ring-1 ring-violet-500/30'
                    : isSelected
                    ? 'bg-slate-800/80 border-cyan-500/60 ring-1 ring-cyan-500/30 cursor-pointer'
                    : 'bg-slate-900/60 border-slate-800 hover:border-slate-700 cursor-pointer'
                }`}
              >
                {/* Header: Run ID + Tags */}
                <div>
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-bold text-white font-mono">{run.run_id}</span>
                        {isLatest ? (
                          <span className="px-2 py-0.5 rounded-md text-[9px] font-bold uppercase bg-violet-500/25 text-violet-300 border border-violet-500/40">
                            Current Live
                          </span>
                        ) : (
                          <span className="px-1.5 py-0.5 rounded-md text-[9px] font-semibold uppercase bg-amber-500/15 text-amber-300 border border-amber-500/30">
                            Demo Snapshot
                          </span>
                        )}
                      </div>
                      <p className="text-[11px] text-slate-400 mt-1 flex items-center gap-1.5">
                        <Calendar className="h-3 w-3 text-slate-500" />
                        {fmtTimestamp(run.timestamp)}
                      </p>
                    </div>

                    {/* Radio / Selection Indicator */}
                    {!isLatest && (
                      <div
                        className={`h-5 w-5 rounded-full flex items-center justify-center border transition-colors ${
                          isSelected
                            ? 'bg-cyan-500 border-cyan-400 text-slate-950'
                            : 'border-slate-700 text-transparent'
                        }`}
                      >
                        <Check className="h-3 w-3 stroke-[3]" />
                      </div>
                    )}
                  </div>

                  {/* Explicit Historical Disclaimer Label */}
                  {run.is_demo_snapshot && (
                    <div className="mt-2.5 px-2 py-1 rounded bg-amber-500/10 border border-amber-500/20 text-[10px] text-amber-300 font-mono leading-tight">
                      DEMO HISTORICAL SNAPSHOT — NOT ACTUAL MODEL MEASUREMENT
                    </div>
                  )}

                  {/* Metadata Chips */}
                  <div className="grid grid-cols-2 gap-2 mt-3 pt-3 border-t border-slate-800/80 text-xs">
                    <div>
                      <span className="text-[10px] text-slate-500 block uppercase">Visible Facings</span>
                      <span className="font-semibold text-slate-200">
                        {run.active_visible_facings}{' '}
                        <span className="text-[10px] font-normal text-slate-500">
                          ({run.stable_facings} stable)
                        </span>
                      </span>
                    </div>

                    <div>
                      <span className="text-[10px] text-slate-500 block uppercase">Shelf Health</span>
                      <span className={`inline-flex items-center gap-1 text-[11px] font-semibold ${health.badge} px-2 py-0.5 rounded-full border`}>
                        <span className={`h-1.5 w-1.5 rounded-full ${health.dot}`} />
                        {health.label}
                      </span>
                    </div>

                    <div>
                      <span className="text-[10px] text-slate-500 block uppercase">Alerts Fired</span>
                      <span className="font-semibold text-slate-200">
                        {run.total_alerts}{' '}
                        <span className="text-[10px] font-normal text-rose-400">
                          ({run.high_alerts} high)
                        </span>
                      </span>
                    </div>

                    <div>
                      <span className="text-[10px] text-slate-500 block uppercase">Verification Req</span>
                      <span className="font-semibold text-amber-400">
                        {run.verification_required}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Footer: Source + Evidence Button */}
                <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between gap-2">
                  <span className="text-[10px] text-slate-500 font-mono truncate max-w-[150px]">
                    {run.video_source}
                  </span>

                  {run.evidence_video && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleOpenRunEvidence(run);
                      }}
                      className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-violet-500/15 border border-violet-500/30 text-violet-300 hover:bg-violet-500/25 text-[10px] font-bold transition-colors shrink-0"
                    >
                      <Film className="h-3 w-3" />
                      View Evidence
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Comparison Section: CURRENT RUN vs PREVIOUS RUN ── */}
      {currentRun && selectedPrevRun && (
        <div className="space-y-4 pt-2">
          {/* Comparison Header */}
          <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-3">
            <div>
              <div className="flex items-center gap-2">
                <GitCompare className="h-4 w-4 text-cyan-400" />
                <h4 className="text-sm font-bold text-white tracking-wide">
                  CURRENT RUN vs PREVIOUS RUN
                </h4>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Comparing <span className="text-violet-300 font-mono font-bold">{currentRun.run_id}</span> ({fmtTimestamp(currentRun.timestamp)}) against{' '}
                <span className="text-cyan-300 font-mono font-bold">{selectedPrevRun.run_id}</span> ({fmtTimestamp(selectedPrevRun.timestamp)})
              </p>
            </div>

            <div className="flex items-center gap-2">
              <span className="px-2.5 py-1 rounded-lg text-[10px] font-mono text-amber-300 bg-amber-500/10 border border-amber-500/25">
                {selectedPrevRun.disclaimer || 'DEMO HISTORICAL SNAPSHOT'}
              </span>
              {currentRun.evidence_video && (
                <button
                  onClick={() => handleOpenRunEvidence(currentRun)}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-violet-600/30 border border-violet-500/40 text-violet-200 hover:bg-violet-600/40 text-xs font-semibold transition-colors"
                >
                  <Film className="h-3.5 w-3.5" />
                  View Run Evidence
                </button>
              )}
            </div>
          </div>

          {/* 10-Metric Differential Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            {comparisonMetrics.map((m) => {
              const { diff, isUp, isDown, isNeutral, color } = computeDiff(m.cur, m.prev, m.invertGood);
              return (
                <div
                  key={m.key}
                  className="glass-panel p-3.5 rounded-xl border border-slate-800 flex flex-col justify-between"
                >
                  <p className="text-[10px] uppercase font-bold tracking-wider text-slate-500">
                    {m.label}
                  </p>

                  <div className="my-2 flex items-baseline justify-between">
                    <div>
                      <span className="text-lg font-bold text-white font-mono">{m.cur ?? 0}</span>
                      <span className="text-xs text-slate-500 font-mono ml-1.5">
                        / {m.prev ?? 0}
                      </span>
                    </div>

                    <div className={`flex items-center gap-0.5 text-xs font-mono font-bold ${color}`}>
                      {isUp && <ArrowUpRight className="h-3.5 w-3.5" />}
                      {isDown && <ArrowDownRight className="h-3.5 w-3.5" />}
                      {isNeutral && <Minus className="h-3.5 w-3.5" />}
                      <span>{diff > 0 ? `+${diff}` : diff}</span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-[9px] text-slate-500 pt-1.5 border-t border-slate-800/60 font-mono">
                    <span>Now vs Previous</span>
                    <span>{isNeutral ? 'Unchanged' : diff > 0 ? 'Increased' : 'Decreased'}</span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* ── Dynamic SKU Comparison Table ── */}
          <div className="glass-panel rounded-xl border border-slate-800 overflow-hidden">
            <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
              <div>
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                  <Package className="h-4 w-4 text-violet-400" />
                  SKU-Level Differential Comparison
                </h4>
                <p className="text-[11px] text-slate-500 mt-0.5">
                  Dynamic facing changes and shelf status transitions between {selectedPrevRun.run_id} and {currentRun.run_id}
                </p>
              </div>

              <span className="text-[10px] font-mono text-slate-400">
                {skuComparisons.length} Registered Catalog SKUs
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 bg-slate-900/40 text-[10px] uppercase tracking-wider text-slate-400">
                    <th className="px-4 py-3">SKU / Product</th>
                    <th className="px-3 py-3 font-mono text-center">Previous Stable</th>
                    <th className="px-3 py-3 font-mono text-center">Current Stable</th>
                    <th className="px-3 py-3 font-mono text-center">Change</th>
                    <th className="px-3 py-3">Previous Status</th>
                    <th className="px-3 py-3">Current Status</th>
                    <th className="px-4 py-3">Transition / Notes</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {skuComparisons.map((row) => {
                    const prevStatusStyle = STATUS_STYLES[row.prevStatus] || 'text-slate-400 bg-slate-800 border-slate-700';
                    const curStatusStyle = STATUS_STYLES[row.curStatus] || 'text-slate-400 bg-slate-800 border-slate-700';
                    const diffColor =
                      row.facingDiff > 0
                        ? 'text-emerald-400'
                        : row.facingDiff < 0
                        ? 'text-rose-400'
                        : 'text-slate-400';

                    return (
                      <tr key={row.sku_id} className="hover:bg-slate-800/20 transition-colors">
                        <td className="px-4 py-3">
                          <div className="font-semibold text-slate-200">{row.product_name}</div>
                          <div className="text-[10px] text-slate-500 font-mono">
                            {row.sku_id} · {row.category}
                          </div>
                        </td>

                        <td className="px-3 py-3 text-center font-mono text-slate-300">
                          {row.prevStable}
                        </td>

                        <td className="px-3 py-3 text-center font-mono font-bold text-white">
                          {row.curStable}
                        </td>

                        <td className="px-3 py-3 text-center font-mono font-bold">
                          <span className={diffColor}>
                            {row.facingDiff > 0 ? `+${row.facingDiff}` : row.facingDiff}
                          </span>
                        </td>

                        <td className="px-3 py-3">
                          <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold uppercase border ${prevStatusStyle}`}>
                            {row.prevStatus}
                          </span>
                        </td>

                        <td className="px-3 py-3">
                          <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold uppercase border ${curStatusStyle}`}>
                            {row.curStatus}
                          </span>
                        </td>

                        <td className="px-4 py-3">
                          {row.transitionBadge ? (
                            <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold uppercase border ${row.transitionBadge.style}`}>
                              {row.transitionBadge.text}
                            </span>
                          ) : (
                            <span className="text-[10px] text-slate-500 italic">
                              Status unchanged
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Footer with Mandatory Semantics Statement */}
            <div className="px-4 py-2.5 bg-slate-900/60 border-t border-slate-800 text-[11px] text-slate-500 flex items-center justify-between">
              <span>
                <strong>Note:</strong> Visible facings represent camera-observable front-row products only.
              </span>
              <span className="font-mono text-[10px] text-slate-600">
                Snapshot diff: {selectedPrevRun.run_id} → {currentRun.run_id}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
