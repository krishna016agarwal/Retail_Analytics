/**
 * InventoryActionCenter.jsx — Step 11: Operational Inventory Demo Workflow
 *
 * Translates raw inventory alerts into an actionable manager interface.
 * All state is frontend-only (no auth, no DB persistence).
 * Data source: the existing inventory_report.json / API — no fabricated data.
 */

import React, { useMemo, useState } from 'react';
import {
  AlertTriangle,
  ShieldAlert,
  Package,
  Move,
  Eye,
  CheckCircle2,
  XCircle,
  PackageMinus,
  Activity,
  ScanSearch,
  ClipboardList,
  Info,
  ChevronDown,
  ChevronRight,
  Layers,
  Film,
} from 'lucide-react';
import InventoryEvidenceViewer from './InventoryEvidenceViewer';

// ─── Alert category grouping ───────────────────────────────────────────────────
const CATEGORY_CFG = {
  STOCK_ATTENTION: {
    label:    'Stock Attention',
    Icon:     PackageMinus,
    color:    'rose',
    types:    ['LOW_STOCK', 'POSSIBLE_STOCKOUT'],
    desc:     'SKUs with low visible facings or zero stable front-row products observed.',
  },
  SHELF_ACTIVITY: {
    label:    'Shelf Activity',
    Icon:     Activity,
    color:    'amber',
    types:    ['PRODUCT_MOVEMENT', 'RAPID_REMOVAL'],
    desc:     'Significant product movement or rapid removal events detected.',
  },
  RECOGNITION: {
    label:    'Recognition',
    Icon:     ScanSearch,
    color:    'violet',
    types:    ['SKU_RECOGNITION_UNCERTAIN'],
    desc:     'Products detected but not reliably matched to a catalog SKU.',
  },
};

const TYPE_LABELS = {
  LOW_STOCK:                 'Low Stock',
  POSSIBLE_STOCKOUT:         'Possible Stockout',
  PRODUCT_MOVEMENT:          'Product Movement',
  RAPID_REMOVAL:             'Rapid Removal',
  SKU_RECOGNITION_UNCERTAIN: 'SKU Uncertain',
};

const SEV_ORDER = { HIGH: 0, MEDIUM: 1, LOW: 2, INFO: 3 };

// ─── Severity badge ─────────────────────────────────────────────────────────
function SevBadge({ severity, pulse }) {
  const cls = {
    HIGH:   'bg-rose-500/15 border-rose-500/30 text-rose-300',
    MEDIUM: 'bg-amber-500/15 border-amber-500/30 text-amber-300',
    LOW:    'bg-sky-500/15 border-sky-500/30 text-sky-300',
    INFO:   'bg-slate-500/15 border-slate-500/30 text-slate-400',
  }[severity] || 'bg-slate-500/15 border-slate-500/30 text-slate-400';
  return (
    <span className={`px-2 py-0.5 rounded-full border text-[9px] font-bold uppercase ${cls} ${pulse ? 'animate-pulse' : ''}`}>
      {severity}
    </span>
  );
}

// ─── Verification explanation banner ──────────────────────────────────────────
function VerifExplanation({ alertType }) {
  const msgs = {
    POSSIBLE_STOCKOUT:
      'Product is no longer visible in the camera frame. Camera movement may have caused a field-of-view exit. Verify the shelf physically before marking as stockout.',
    RAPID_REMOVAL:
      'Multiple products disappeared rapidly. This may be customers picking products OR the camera panning away. Physical verification required to distinguish the two.',
    LOW_STOCK:
      'Visible facing count is low but the shelf may have product behind the front row. Camera can only observe front-row facings — verify actual shelf depth.',
  };
  const msg = msgs[alertType] || 'Confirm shelf status physically before taking corrective action.';
  return (
    <div className="mt-2 flex items-start gap-2 p-2.5 rounded-lg bg-amber-500/8 border border-amber-500/25 text-[11px] text-amber-200/80 leading-relaxed">
      <ShieldAlert className="h-3.5 w-3.5 text-amber-400 mt-0.5 shrink-0" />
      <span>{msg}</span>
    </div>
  );
}

// ─── Action buttons ────────────────────────────────────────────────────────────
function ActionButtons({ alertId, state, onVerify, onAck, onViewSku, onViewEvidence, skuId }) {
  const isDone = state === 'acknowledged' || state === 'verified';
  if (isDone) {
    return (
      <div className="flex flex-wrap gap-1.5 items-center">
        <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[10px] font-bold border
          ${state === 'verified'
            ? 'bg-emerald-500/15 border-emerald-500/30 text-emerald-400'
            : 'bg-slate-500/15 border-slate-500/30 text-slate-400'}`}>
          <CheckCircle2 className="h-3 w-3" />
          {state === 'verified' ? 'Verified' : 'Acknowledged'}
        </span>
        <button
          onClick={onViewEvidence}
          className="px-2.5 py-1 rounded-md bg-slate-800/60 border border-slate-700/50 text-slate-400
                     text-[10px] font-bold hover:bg-slate-700 transition-colors inline-flex items-center gap-1"
        >
          <Film className="h-3 w-3" />
          View Evidence
        </button>
      </div>
    );
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      <button
        onClick={() => onVerify(alertId)}
        className="px-2.5 py-1 rounded-md bg-emerald-500/15 border border-emerald-500/30 text-emerald-300
                   text-[10px] font-bold hover:bg-emerald-500/25 transition-colors"
      >
        ✓ Verify
      </button>
      <button
        onClick={() => onAck(alertId)}
        className="px-2.5 py-1 rounded-md bg-slate-700/60 border border-slate-600/50 text-slate-300
                   text-[10px] font-bold hover:bg-slate-600/60 transition-colors"
      >
        Acknowledge
      </button>
      {skuId && skuId !== 'UNKNOWN' && (
        <button
          onClick={() => onViewSku(skuId)}
          className="px-2.5 py-1 rounded-md bg-violet-500/15 border border-violet-500/30 text-violet-300
                     text-[10px] font-bold hover:bg-violet-500/25 transition-colors"
        >
          View SKU
        </button>
      )}
      <button
        onClick={onViewEvidence}
        className="px-2.5 py-1 rounded-md bg-cyan-500/15 border border-cyan-500/30 text-cyan-300
                   text-[10px] font-bold hover:bg-cyan-500/25 transition-colors inline-flex items-center gap-1"
      >
        <Film className="h-3 w-3" />
        View Evidence
      </button>
    </div>
  );
}

// ─── Single alert card ─────────────────────────────────────────────────────────
function AlertCard({ alert, actionState, onVerify, onAck, onViewSku, onViewEvidence }) {
  const [expanded, setExpanded] = useState(false);
  const isDone  = actionState === 'acknowledged' || actionState === 'verified';
  const isVerif = alert.requires_verification;

  return (
    <div className={`rounded-xl border transition-all duration-200 overflow-hidden
      ${isDone ? 'opacity-50 border-slate-800/60 bg-slate-900/20' :
        isVerif ? 'border-amber-500/30 bg-amber-500/4' :
        alert.severity === 'HIGH' ? 'border-rose-500/30 bg-rose-500/4' :
        'border-slate-700/50 bg-slate-900/30'}`}>

      {/* ── Header row ── */}
      <div
        className="flex flex-wrap items-start gap-2 px-4 py-3 cursor-pointer"
        onClick={() => setExpanded(v => !v)}
      >
        {/* Left: dot + info */}
        <div className="flex items-start gap-2.5 flex-1 min-w-0">
          <span className={`mt-1.5 h-2 w-2 rounded-full shrink-0
            ${isDone ? 'bg-slate-600' :
              isVerif ? 'bg-amber-400 animate-pulse' :
              alert.severity === 'HIGH' ? 'bg-rose-500 animate-pulse' :
              alert.severity === 'MEDIUM' ? 'bg-amber-400' : 'bg-sky-400'}`}
          />
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-1.5 mb-0.5">
              <SevBadge severity={alert.severity} pulse={!isDone && alert.severity === 'HIGH'} />
              <span className="text-xs font-semibold text-white truncate">
                {TYPE_LABELS[alert.alert_type] || alert.alert_type}
              </span>
              {isVerif && !isDone && (
                <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-bold
                  bg-amber-500/15 border border-amber-500/30 text-amber-300 uppercase">
                  <ShieldAlert className="h-2.5 w-2.5" />
                  Verify
                </span>
              )}
            </div>
            <p className="text-sm font-medium text-slate-200 truncate">
              {alert.sku_name || alert.sku_id || 'Unknown Product'}
            </p>
          </div>
        </div>

        {/* Right: facings + timestamp + chevron */}
        <div className="flex items-center gap-3 shrink-0">
          <div className="text-right">
            <p className="text-[10px] text-slate-500 font-mono">
              Stable: <span className="text-slate-300 font-bold">{alert.current_stable_facings ?? '—'}</span>
              &nbsp;|&nbsp;
              Active: <span className="text-slate-300 font-bold">{alert.active_facings ?? '—'}</span>
            </p>
            <p className="text-[9px] font-mono text-slate-600">
              @{alert.timestamp_sec?.toFixed(2)}s · f{alert.frame_index}
            </p>
          </div>
          {expanded
            ? <ChevronDown className="h-3.5 w-3.5 text-slate-500 shrink-0" />
            : <ChevronRight className="h-3.5 w-3.5 text-slate-500 shrink-0" />}
        </div>
      </div>

      {/* ── Expanded detail ── */}
      {expanded && (
        <div className="px-4 pb-3 border-t border-slate-800/50 pt-2.5 space-y-2.5">
          {/* Reason */}
          <p className="text-[11px] text-slate-400 leading-relaxed">{alert.reason}</p>

          {/* Verification explanation */}
          {isVerif && !isDone && (
            <VerifExplanation alertType={alert.alert_type} />
          )}

          {/* Actions */}
          <ActionButtons
            alertId={alert.alert_id}
            state={actionState}
            onVerify={onVerify}
            onAck={onAck}
            onViewSku={onViewSku}
            onViewEvidence={() => onViewEvidence({ ...alert, _evidenceSource: 'alert' })}
            skuId={alert.sku_id}
          />
        </div>
      )}
    </div>
  );
}

// ─── Category section ──────────────────────────────────────────────────────────
function CategorySection({ catKey, alerts, actionStates, onVerify, onAck, onViewSku, onViewEvidence }) {
  const cfg = CATEGORY_CFG[catKey];
  const { Icon, label, color, desc } = cfg;
  const [collapsed, setCollapsed] = useState(false);

  const unresolvedCount = alerts.filter(a => {
    const s = actionStates[a.alert_id];
    return s !== 'acknowledged' && s !== 'verified';
  }).length;

  const colorMap = {
    rose:   { badge: 'bg-rose-500/15 border-rose-500/30 text-rose-300',   icon: 'text-rose-400',   header: 'border-rose-500/20' },
    amber:  { badge: 'bg-amber-500/15 border-amber-500/30 text-amber-300', icon: 'text-amber-400',  header: 'border-amber-500/20' },
    violet: { badge: 'bg-violet-500/15 border-violet-500/30 text-violet-300', icon: 'text-violet-400', header: 'border-violet-500/20' },
  };
  const c = colorMap[color] || colorMap.amber;

  if (alerts.length === 0) {
    return (
      <div className="glass-panel p-4">
        <div className="flex items-center gap-2 mb-1">
          <Icon className={`h-4 w-4 ${c.icon}`} />
          <span className="text-sm font-semibold text-slate-300">{label}</span>
        </div>
        <p className="text-xs text-slate-500 italic">No {label.toLowerCase()} alerts detected.</p>
      </div>
    );
  }

  return (
    <div className="glass-panel overflow-hidden">
      {/* Category header */}
      <div
        className={`flex items-center justify-between px-4 py-3 border-b ${c.header} cursor-pointer hover:bg-slate-800/20 transition-colors`}
        onClick={() => setCollapsed(v => !v)}
      >
        <div className="flex items-center gap-2.5">
          <Icon className={`h-4 w-4 ${c.icon}`} />
          <div>
            <span className="text-sm font-semibold text-white">{label}</span>
            <p className="text-[10px] text-slate-500 mt-0">{desc}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {unresolvedCount > 0 && (
            <span className={`px-2 py-0.5 rounded-full border text-[10px] font-bold ${c.badge}`}>
              {unresolvedCount} open
            </span>
          )}
          {unresolvedCount === 0 && (
            <span className="px-2 py-0.5 rounded-full border text-[10px] font-bold bg-emerald-500/10 border-emerald-500/20 text-emerald-400">
              All resolved
            </span>
          )}
          {collapsed
            ? <ChevronRight className="h-4 w-4 text-slate-500" />
            : <ChevronDown className="h-4 w-4 text-slate-500" />}
        </div>
      </div>

      {/* Alert cards */}
      {!collapsed && (
        <div className="p-3 space-y-2">
          {alerts.map((alert) => (
            <AlertCard
              key={`${alert.alert_id}-${alert.sku_id}`}
              alert={alert}
              actionState={actionStates[alert.alert_id]}
              onVerify={onVerify}
              onAck={onAck}
              onViewSku={onViewSku}
              onViewEvidence={onViewEvidence}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Manager Summary ──────────────────────────────────────────────────────────
function ManagerSummary({ report, alerts, actionStates }) {
  const unresolved = alerts.filter(a => {
    const s = actionStates[a.alert_id];
    return s !== 'acknowledged' && s !== 'verified';
  });

  const needingAttention  = new Set(unresolved.filter(a => ['LOW_STOCK','POSSIBLE_STOCKOUT'].includes(a.alert_type)).map(a => a.sku_id)).size;
  const needingVerif      = unresolved.filter(a => a.requires_verification).length;
  const movementAlerts    = unresolved.filter(a => ['PRODUCT_MOVEMENT','RAPID_REMOVAL'].includes(a.alert_type)).length;
  const recognitionIssues = unresolved.filter(a => a.alert_type === 'SKU_RECOGNITION_UNCERTAIN').length;
  const totalFacings      = report?.total_active_visible_facings ?? 0;
  const stableFacings     = report?.stable_facings ?? 0;

  const rows = [
    { label: 'Products needing attention',       value: needingAttention,  color: needingAttention  > 0 ? 'text-rose-300' : 'text-emerald-400',  icon: <PackageMinus className="h-3.5 w-3.5" /> },
    { label: 'Requiring physical verification',  value: needingVerif,      color: needingVerif      > 0 ? 'text-amber-300' : 'text-emerald-400', icon: <ShieldAlert className="h-3.5 w-3.5" /> },
    { label: 'Movement / removal alerts',        value: movementAlerts,    color: movementAlerts    > 0 ? 'text-amber-300' : 'text-emerald-400', icon: <Activity className="h-3.5 w-3.5" /> },
    { label: 'Recognition issues',               value: recognitionIssues, color: recognitionIssues > 0 ? 'text-violet-300' : 'text-emerald-400', icon: <ScanSearch className="h-3.5 w-3.5" /> },
    { label: 'Current visible facings',          value: `${stableFacings} stable / ${totalFacings} total`, color: 'text-cyan-400',  icon: <Eye className="h-3.5 w-3.5" /> },
  ];

  return (
    <div className="glass-panel p-4 space-y-3">
      <div className="flex items-center gap-2 pb-2 border-b border-slate-800/60">
        <ClipboardList className="h-4 w-4 text-slate-400" />
        <h3 className="text-sm font-semibold text-white">Manager Summary</h3>
        <span className="ml-auto text-[10px] font-mono text-slate-500">
          {report?.timestamp_iso?.slice(0,19).replace('T',' ')}
        </span>
      </div>
      <div className="space-y-2">
        {rows.map(({ label, value, color, icon }) => (
          <div key={label} className="flex items-center gap-2.5 py-1 border-b border-slate-800/40 last:border-0">
            <span className={`shrink-0 ${color}`}>{icon}</span>
            <span className="text-xs text-slate-400 flex-1">{label}</span>
            <span className={`text-sm font-bold font-mono ${color}`}>{value}</span>
          </div>
        ))}
      </div>
      {/* Overall notice */}
      <div className="flex items-start gap-2 p-2.5 rounded-lg bg-slate-900/50 border border-slate-800/60 text-[10px] text-slate-500 leading-relaxed">
        <Info className="h-3.5 w-3.5 mt-0.5 shrink-0 text-cyan-500" />
        All facing counts reflect camera-observable front-row products only.
        Do not interpret as total physical shelf inventory or back-stock.
      </div>
    </div>
  );
}

// ─── Priority Queue ─────────────────────────────────────────────────────────
function PriorityQueue({ alerts, actionStates, onVerify, onAck, onViewSku, onViewEvidence }) {
  const unresolved = alerts
    .filter(a => {
      const s = actionStates[a.alert_id];
      return s !== 'acknowledged' && s !== 'verified';
    })
    .sort((a, b) => {
      // 1. Severity order (HIGH first)
      const so = (SEV_ORDER[a.severity] ?? 9) - (SEV_ORDER[b.severity] ?? 9);
      if (so !== 0) return so;
      // 2. Verification-required first within same severity
      if (a.requires_verification !== b.requires_verification)
        return a.requires_verification ? -1 : 1;
      // 3. Latest timestamp last (oldest unresolved first)
      return (a.timestamp_sec ?? 0) - (b.timestamp_sec ?? 0);
    });

  if (unresolved.length === 0) {
    return (
      <div className="glass-panel p-6 flex flex-col items-center gap-2 text-center">
        <CheckCircle2 className="h-8 w-8 text-emerald-400" />
        <p className="text-sm font-semibold text-emerald-400">No unresolved alerts</p>
        <p className="text-xs text-slate-500">All inventory alerts have been acknowledged or verified.</p>
      </div>
    );
  }

  return (
    <div className="glass-panel overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-3 border-b border-slate-800/80">
        <Layers className="h-4 w-4 text-slate-400" />
        <h3 className="text-sm font-semibold text-white">Priority Queue</h3>
        <span className="text-[10px] text-slate-500 ml-1">(HIGH → MEDIUM → verification-required first)</span>
        <span className="ml-auto px-2 py-0.5 rounded-full bg-rose-500/15 border border-rose-500/30 text-rose-300 text-[10px] font-bold">
          {unresolved.length} open
        </span>
      </div>
      <div className="p-3 space-y-2 max-h-[55vh] overflow-y-auto">
        {unresolved.map((alert) => (
          <AlertCard
            key={`pq-${alert.alert_id}-${alert.sku_id}`}
            alert={alert}
            actionState={actionStates[alert.alert_id]}
            onVerify={onVerify}
            onAck={onAck}
            onViewSku={onViewSku}
            onViewEvidence={onViewEvidence}
          />
        ))}
      </div>
    </div>
  );
}

// ─── SKU Detail Modal ──────────────────────────────────────────────────────────
function SkuModal({ skuId, skuSummary, onClose }) {
  const sku = skuSummary?.find(s => s.sku_id === skuId);
  if (!sku) return null;

  const statusColor = {
    IN_STOCK:    'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
    LOW_STOCK:   'text-amber-300 bg-amber-500/10 border-amber-500/20',
    OUT_OF_VIEW: 'text-rose-400 bg-rose-500/10 border-rose-500/20',
    UNKNOWN:     'text-slate-400 bg-slate-500/10 border-slate-500/20',
  }[sku.shelf_status] || 'text-slate-400 bg-slate-500/10 border-slate-500/20';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm"
         onClick={onClose}>
      <div className="glass-panel w-full max-w-md p-5 space-y-4 shadow-2xl"
           onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-base font-bold text-white">{sku.product_name}</h3>
            <p className="text-xs font-mono text-slate-500">{sku.sku_id} · {sku.category}</p>
          </div>
          <button onClick={onClose}
            className="text-slate-500 hover:text-slate-300 transition-colors p-1">
            <XCircle className="h-5 w-5" />
          </button>
        </div>

        {/* Status + Facings */}
        <div className="flex flex-wrap gap-3">
          <span className={`px-3 py-1.5 rounded-lg border text-xs font-bold ${statusColor}`}>
            {sku.shelf_status?.replace(/_/g,' ')}
          </span>
          <div className="flex gap-4 text-xs text-slate-400">
            <span>Stable: <strong className="text-white">{sku.stable_facings}</strong></span>
            <span>Active: <strong className="text-white">{sku.current_active_facings}</strong></span>
            <span>Uncertain: <strong className="text-white">{sku.uncertain_facings}</strong></span>
          </div>
        </div>

        {/* Lifecycle */}
        <div className="grid grid-cols-3 gap-2">
          {[
            { label: 'Appeared', value: sku.appeared_count, color: 'text-emerald-400' },
            { label: 'Removed',  value: sku.removed_count,  color: 'text-rose-400'    },
            { label: 'Moved',    value: sku.moved_count,    color: 'text-sky-400'     },
          ].map(({ label, value, color }) => (
            <div key={label} className="text-center p-2 rounded-lg bg-slate-800/40 border border-slate-800/60">
              <p className={`text-lg font-extrabold font-mono ${color}`}>{value}</p>
              <p className="text-[10px] text-slate-500">{label}</p>
            </div>
          ))}
        </div>

        {/* Alerts for this SKU */}
        {sku.active_alerts?.length > 0 && (
          <div>
            <p className="text-[10px] font-bold uppercase tracking-wider text-slate-500 mb-2">
              Active Alerts ({sku.active_alerts.length})
            </p>
            <div className="space-y-1.5 max-h-40 overflow-y-auto">
              {sku.active_alerts.map((a, i) => (
                <div key={i} className="text-[11px] text-slate-400 p-2 rounded bg-slate-900/60 border border-slate-800/60 leading-relaxed">
                  <span className={`font-bold mr-1.5 ${a.severity === 'HIGH' ? 'text-rose-300' : 'text-amber-300'}`}>
                    [{a.severity}]
                  </span>
                  {TYPE_LABELS[a.alert_type] || a.alert_type}: {a.reason}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Semantics notice */}
        <p className="text-[9px] text-slate-600 leading-relaxed border-t border-slate-800/60 pt-2">
          Facings = camera-observable front-row products only. Not total physical inventory.
        </p>
      </div>
    </div>
  );
}

// ─── View toggle ─────────────────────────────────────────────────────────────
const VIEWS = [
  { id: 'priority',   label: 'Priority Queue' },
  { id: 'categories', label: 'By Category'    },
];

// ─── Main Action Center ───────────────────────────────────────────────────────
export default function InventoryActionCenter({ report, loading }) {
  const [actionStates, setActionStates] = useState({});  // alertId → 'acknowledged' | 'verified'
  const [view,         setView]         = useState('priority');
  const [viewSkuId,    setViewSkuId]    = useState(null);
  const [evidenceItem, setEvidenceItem] = useState(null);

  const alerts     = report?.active_alerts ?? [];
  const skuSummary = report?.sku_inventory_summary ?? [];

  // Handlers
  const handleVerify = (alertId) =>
    setActionStates(prev => ({ ...prev, [alertId]: 'verified' }));
  const handleAck    = (alertId) =>
    setActionStates(prev => ({ ...prev, [alertId]: 'acknowledged' }));
  const handleViewSku = (skuId) => setViewSkuId(skuId);
  const handleViewEvidence = (evidenceItem) => setEvidenceItem(evidenceItem);

  // Group alerts by category
  const grouped = useMemo(() => {
    const result = {};
    Object.keys(CATEGORY_CFG).forEach(catKey => {
      const types = CATEGORY_CFG[catKey].types;
      result[catKey] = alerts.filter(a => types.includes(a.alert_type));
    });
    return result;
  }, [alerts]);

  const totalResolved = Object.values(actionStates).filter(s => s === 'acknowledged' || s === 'verified').length;
  const totalUnresolved = alerts.length - totalResolved;

  if (loading) {
    return (
      <div className="space-y-4 animate-pulse">
        <div className="h-24 bg-slate-800/40 rounded-xl" />
        <div className="h-48 bg-slate-800/40 rounded-xl" />
        <div className="h-48 bg-slate-800/40 rounded-xl" />
      </div>
    );
  }

  if (!report) {
    return (
      <div className="glass-panel p-6 flex items-center gap-3 text-slate-400 text-sm">
        <Info className="h-5 w-5 text-cyan-400 shrink-0" />
        No inventory report available. Load a report to view the Action Center.
      </div>
    );
  }

  return (
    <div className="space-y-5">
      {/* ── Section header ── */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <ClipboardList className="h-4 w-4 text-amber-400" />
            Action Center — Operational Workflow
          </h3>
          <p className="text-[10px] text-slate-500 mt-0.5">
            {totalUnresolved} unresolved · {totalResolved} resolved this session (frontend state only)
          </p>
        </div>
        {/* View toggle */}
        <div className="flex gap-1 bg-slate-900/60 rounded-lg p-0.5 border border-slate-800/60">
          {VIEWS.map(({ id, label }) => (
            <button
              key={id}
              onClick={() => setView(id)}
              className={`px-3 py-1.5 rounded-md text-[11px] font-semibold transition-colors
                ${view === id
                  ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                  : 'text-slate-400 hover:text-slate-200'}`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* ── Manager Summary ── */}
      <ManagerSummary report={report} alerts={alerts} actionStates={actionStates} />

      {/* ── Main view ── */}
      {view === 'priority' && (
        <PriorityQueue
          alerts={alerts}
          actionStates={actionStates}
          onVerify={handleVerify}
          onAck={handleAck}
          onViewSku={handleViewSku}
          onViewEvidence={handleViewEvidence}
        />
      )}

      {view === 'categories' && (
        <div className="space-y-4">
          {Object.keys(CATEGORY_CFG).map(catKey => (
            <CategorySection
              key={catKey}
              catKey={catKey}
              alerts={grouped[catKey] || []}
              actionStates={actionStates}
              onVerify={handleVerify}
              onAck={handleAck}
              onViewSku={handleViewSku}
              onViewEvidence={handleViewEvidence}
            />
          ))}
        </div>
      )}

      {/* ── Resolved summary ── */}
      {totalResolved > 0 && (
        <div className="flex items-center gap-2 p-3 rounded-lg bg-slate-900/40 border border-slate-800/50 text-xs text-slate-400">
          <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
          <span>
            <strong className="text-emerald-400">{totalResolved}</strong> alert{totalResolved > 1 ? 's' : ''} resolved this session.
            Actions are UI-only — no database persistence in this step.
          </span>
        </div>
      )}

      {/* ── SKU Detail Modal ── */}
      {viewSkuId && (
        <SkuModal
          skuId={viewSkuId}
          skuSummary={skuSummary}
          onClose={() => setViewSkuId(null)}
        />
      )}

      {/* ── Evidence Viewer Modal ── */}
      {evidenceItem && (
        <InventoryEvidenceViewer
          evidence={evidenceItem}
          onClose={() => setEvidenceItem(null)}
        />
      )}
    </div>
  );
}
