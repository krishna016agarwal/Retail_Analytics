import React, { useState } from 'react';
import { ShieldAlert, AlertTriangle, Info, Filter } from 'lucide-react';

// ─── Severity config ──────────────────────────────────────────────────────────
const SEV = {
  HIGH:   { cls: 'border-rose-500/40 bg-rose-500/5',   badge: 'bg-rose-500/15 border-rose-500/30 text-rose-300',   dot: 'bg-rose-500' },
  MEDIUM: { cls: 'border-amber-500/40 bg-amber-500/5', badge: 'bg-amber-500/15 border-amber-500/30 text-amber-300', dot: 'bg-amber-400' },
  LOW:    { cls: 'border-sky-500/40 bg-sky-500/5',     badge: 'bg-sky-500/15 border-sky-500/30 text-sky-300',       dot: 'bg-sky-400'   },
  INFO:   { cls: 'border-slate-700/60 bg-slate-800/20',badge: 'bg-slate-500/15 border-slate-500/30 text-slate-300', dot: 'bg-slate-500' },
};

// ─── Alert type labels ────────────────────────────────────────────────────────
const TYPE_LABELS = {
  LOW_STOCK:                 'Low Stock',
  POSSIBLE_STOCKOUT:         'Possible Stockout',
  RAPID_REMOVAL:             'Rapid Removal',
  PRODUCT_MOVEMENT:          'Product Movement',
  SKU_RECOGNITION_UNCERTAIN: 'SKU Uncertain',
};

// ─── Single alert card ────────────────────────────────────────────────────────
function AlertCard({ alert }) {
  const sev = SEV[alert.severity] || SEV.INFO;
  return (
    <div className={`rounded-xl border p-4 transition-all duration-200 ${sev.cls}`}>
      <div className="flex flex-wrap items-start justify-between gap-2">
        {/* Left: badge + title */}
        <div className="flex items-start gap-2.5">
          <span className={`mt-0.5 h-2 w-2 rounded-full shrink-0 ${sev.dot} ${alert.severity === 'HIGH' ? 'animate-pulse' : ''}`} />
          <div>
            <div className="flex flex-wrap items-center gap-2 mb-0.5">
              <span className={`px-2 py-0.5 rounded-full border text-[10px] font-bold uppercase ${sev.badge}`}>
                {alert.severity}
              </span>
              <span className="text-xs font-semibold text-white">
                {TYPE_LABELS[alert.alert_type] || alert.alert_type?.replace(/_/g, ' ')}
              </span>
              {alert.requires_verification && (
                <span className="inline-flex items-center gap-1 text-[9px] bg-rose-500/10 border border-rose-500/30 text-rose-300 px-1.5 py-0.5 rounded uppercase">
                  <ShieldAlert className="h-2.5 w-2.5" />
                  Verify
                </span>
              )}
            </div>
            <p className="text-sm font-medium text-slate-200">{alert.sku_name}</p>
            <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">{alert.reason}</p>
          </div>
        </div>

        {/* Right: stats */}
        <div className="flex flex-col items-end gap-1 shrink-0">
          <span className="text-[10px] font-mono text-slate-500">@{alert.timestamp_sec?.toFixed(2)}s</span>
          <span className="text-[10px] text-slate-500">Stable: {alert.current_stable_facings} &nbsp;|&nbsp; Active: {alert.active_facings}</span>
          <span className="text-[10px] text-slate-500">Supporting events: {alert.supporting_event_count}</span>
        </div>
      </div>
    </div>
  );
}

// ─── Main panel ───────────────────────────────────────────────────────────────
export default function InventoryAlertsPanel({ alerts, loading }) {
  const [filterSev, setFilterSev] = useState('ALL');
  const [filterType, setFilterType] = useState('ALL');
  const [showVerifOnly, setShowVerifOnly] = useState(false);

  if (loading) {
    return (
      <div className="space-y-3 animate-pulse">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="h-24 bg-slate-800/40 rounded-xl" />
        ))}
      </div>
    );
  }

  if (!alerts?.length) {
    return (
      <div className="glass-panel p-6 flex items-center gap-3 text-slate-400 text-sm">
        <Info className="h-5 w-5 text-emerald-400 shrink-0" />
        No active alerts — shelf appears operational.
      </div>
    );
  }

  const severities = ['ALL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'];
  const types = ['ALL', ...Object.keys(TYPE_LABELS)];

  let displayed = alerts;
  if (filterSev !== 'ALL')   displayed = displayed.filter((a) => a.severity === filterSev);
  if (filterType !== 'ALL')  displayed = displayed.filter((a) => a.alert_type === filterType);
  if (showVerifOnly)         displayed = displayed.filter((a) => a.requires_verification);

  // Sort: HIGH first, then by timestamp desc
  displayed = [...displayed].sort((a, b) => {
    const sevOrder = { HIGH: 0, MEDIUM: 1, LOW: 2, INFO: 3 };
    if (sevOrder[a.severity] !== sevOrder[b.severity]) return sevOrder[a.severity] - sevOrder[b.severity];
    return (b.timestamp_sec ?? 0) - (a.timestamp_sec ?? 0);
  });

  return (
    <div className="space-y-4">
      {/* Filter bar */}
      <div className="glass-panel px-4 py-3 space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <Filter className="h-3.5 w-3.5 text-slate-500" />
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 mr-1">Severity:</span>
          {severities.map((s) => (
            <button
              key={s}
              onClick={() => setFilterSev(s)}
              className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold border transition-colors
                ${filterSev === s ? 'bg-cyan-500/20 border-cyan-500/40 text-cyan-300' : 'border-slate-700/50 text-slate-400 hover:text-slate-200'}`}
            >
              {s}
            </button>
          ))}
          <button
            onClick={() => setShowVerifOnly((v) => !v)}
            className={`ml-auto inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-bold border transition-colors
              ${showVerifOnly ? 'bg-rose-500/20 border-rose-500/40 text-rose-300' : 'border-slate-700/50 text-slate-400 hover:text-slate-200'}`}
          >
            <ShieldAlert className="h-3 w-3" /> Verif. Required Only
          </button>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 mr-1">Type:</span>
          {types.map((t) => (
            <button
              key={t}
              onClick={() => setFilterType(t)}
              className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold border transition-colors
                ${filterType === t ? 'bg-violet-500/20 border-violet-500/40 text-violet-300' : 'border-slate-700/50 text-slate-400 hover:text-slate-200'}`}
            >
              {TYPE_LABELS[t] || t}
            </button>
          ))}
        </div>
      </div>

      {/* Alert count */}
      <div className="flex items-center gap-2">
        <AlertTriangle className="h-4 w-4 text-amber-400" />
        <span className="text-sm font-semibold text-slate-200">
          {displayed.length} alert{displayed.length !== 1 ? 's' : ''} shown
          {filterSev !== 'ALL' || filterType !== 'ALL' || showVerifOnly ? ' (filtered)' : ''}
        </span>
      </div>

      {/* Alert cards */}
      <div className="space-y-2.5 max-h-[60vh] overflow-y-auto pr-1">
        {displayed.map((a) => (
          <AlertCard key={`${a.alert_id}-${a.sku_id}`} alert={a} />
        ))}
        {displayed.length === 0 && (
          <div className="text-center py-8 text-slate-500 text-sm">No alerts match the current filters.</div>
        )}
      </div>

      {/* Verification notice */}
      {alerts.some((a) => a.requires_verification) && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-rose-500/5 border border-rose-500/25 text-[11px] text-rose-300 leading-relaxed">
          <ShieldAlert className="h-4 w-4 mt-0.5 shrink-0" />
          <span>
            <strong>Verification Required:</strong> Alerts marked "Verify" cannot distinguish between a physical stockout and
            the camera panning past the shelf area. Physical or additional-camera verification is needed before taking action.
          </span>
        </div>
      )}
    </div>
  );
}
