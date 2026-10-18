import React, { useState } from 'react';
import { ChevronDown, ChevronUp, AlertTriangle, CheckCircle, XCircle, Eye, ShieldAlert } from 'lucide-react';

// ─── Status badge ─────────────────────────────────────────────────────────────
function StatusBadge({ status }) {
  const cfg = {
    IN_STOCK:   { cls: 'bg-emerald-500/15 border-emerald-500/30 text-emerald-400', Icon: CheckCircle },
    LOW_STOCK:  { cls: 'bg-amber-500/15 border-amber-500/30 text-amber-300',       Icon: AlertTriangle },
    OUT_OF_VIEW:{ cls: 'bg-rose-500/15 border-rose-500/30 text-rose-300',          Icon: XCircle },
    UNKNOWN:    { cls: 'bg-slate-500/15 border-slate-500/30 text-slate-400',       Icon: Eye },
  };
  const { cls, Icon } = cfg[status] || cfg['UNKNOWN'];
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-[10px] font-bold uppercase ${cls}`}>
      <Icon className="h-2.5 w-2.5" />
      {status?.replace(/_/g, ' ')}
    </span>
  );
}

// ─── Alert severity badge ─────────────────────────────────────────────────────
function SevBadge({ sev }) {
  const cls = {
    HIGH:   'text-rose-300 bg-rose-500/10 border-rose-500/30',
    MEDIUM: 'text-amber-300 bg-amber-500/10 border-amber-500/30',
    LOW:    'text-sky-300 bg-sky-500/10 border-sky-500/30',
    INFO:   'text-slate-300 bg-slate-500/10 border-slate-500/30',
  }[sev] || 'text-slate-300';
  return (
    <span className={`inline-block px-1.5 py-0.5 rounded text-[9px] font-bold border uppercase ${cls}`}>
      {sev}
    </span>
  );
}

// ─── Expanded alert list for a SKU row ───────────────────────────────────────
function AlertList({ alerts }) {
  if (!alerts?.length) return <p className="text-slate-500 text-xs">No alerts for this SKU.</p>;
  return (
    <div className="space-y-2 mt-1">
      {alerts.map((a) => (
        <div key={a.alert_id} className="flex flex-wrap items-start gap-2 p-2.5 rounded-lg bg-slate-900/70 border border-slate-800/60 text-xs">
          <SevBadge sev={a.severity} />
          <span className="text-slate-300 font-medium">{a.alert_type?.replace(/_/g, ' ')}</span>
          {a.requires_verification && (
            <span className="inline-flex items-center gap-1 text-[9px] bg-rose-500/10 border border-rose-500/30 text-rose-300 px-1.5 py-0.5 rounded uppercase">
              <ShieldAlert className="h-2.5 w-2.5" />verif. req.
            </span>
          )}
          <span className="ml-auto font-mono text-slate-500 text-[10px]">@{a.timestamp_sec?.toFixed(2)}s</span>
          <p className="w-full text-slate-400 mt-0.5">{a.reason}</p>
        </div>
      ))}
    </div>
  );
}

// ─── Single SKU row ───────────────────────────────────────────────────────────
function SKURow({ sku, index }) {
  const [expanded, setExpanded] = useState(false);
  const hasAlerts = sku.active_alerts?.length > 0;

  return (
    <>
      <tr
        className={`border-b border-slate-800/60 hover:bg-slate-800/30 transition-colors cursor-pointer ${index % 2 === 0 ? 'bg-slate-900/20' : ''}`}
        onClick={() => hasAlerts && setExpanded((v) => !v)}
      >
        <td className="px-4 py-3">
          <div className="flex flex-col gap-0.5">
            <span className="text-sm font-semibold text-white">{sku.product_name}</span>
            <span className="text-[10px] font-mono text-slate-500">{sku.sku_id}</span>
          </div>
        </td>
        <td className="px-3 py-3 text-xs text-slate-400">{sku.category}</td>
        <td className="px-3 py-3"><StatusBadge status={sku.shelf_status} /></td>
        <td className="px-3 py-3 text-center font-mono text-sm font-bold text-white">{sku.stable_facings}</td>
        <td className="px-3 py-3 text-center font-mono text-sm text-slate-300">{sku.current_active_facings}</td>
        <td className="px-3 py-3 text-center font-mono text-xs text-slate-400">{sku.uncertain_facings}</td>
        <td className="px-3 py-3 text-center font-mono text-xs text-slate-400">{sku.changing_facings}</td>
        <td className="px-3 py-3 text-center font-mono text-xs text-emerald-400">{sku.appeared_count}</td>
        <td className="px-3 py-3 text-center font-mono text-xs text-rose-400">{sku.removed_count}</td>
        <td className="px-3 py-3">
          <div className="flex items-center justify-center gap-1.5">
            <span className={`text-xs font-bold font-mono ${hasAlerts ? (sku.requires_verification ? 'text-rose-400' : 'text-amber-300') : 'text-slate-500'}`}>
              {sku.active_alerts?.length ?? 0}
            </span>
            {sku.requires_verification && <ShieldAlert className="h-3.5 w-3.5 text-rose-400" />}
            {hasAlerts && (
              expanded
                ? <ChevronUp className="h-3.5 w-3.5 text-slate-400" />
                : <ChevronDown className="h-3.5 w-3.5 text-slate-400" />
            )}
          </div>
        </td>
      </tr>

      {/* Expanded alert detail row */}
      {expanded && hasAlerts && (
        <tr className="bg-slate-900/40">
          <td colSpan={10} className="px-6 py-3">
            <AlertList alerts={sku.active_alerts} />
          </td>
        </tr>
      )}
    </>
  );
}

// ─── Sort helpers ─────────────────────────────────────────────────────────────
const SORT_KEYS = {
  name:     (s) => s.product_name?.toLowerCase() ?? '',
  status:   (s) => s.shelf_status ?? '',
  stable:   (s) => -(s.stable_facings ?? 0),
  active:   (s) => -(s.current_active_facings ?? 0),
  alerts:   (s) => -(s.active_alerts?.length ?? 0),
};

// ─── Main table ───────────────────────────────────────────────────────────────
export default function SKUInventoryTable({ skuSummary, loading }) {
  const [sortKey, setSortKey] = useState('alerts');
  const [filterStatus, setFilterStatus] = useState('ALL');

  if (loading) {
    return (
      <div className="glass-panel p-6 animate-pulse space-y-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="h-10 bg-slate-800/40 rounded" />
        ))}
      </div>
    );
  }

  if (!skuSummary?.length) {
    return (
      <div className="glass-panel p-6 text-sm text-slate-400">No SKU inventory data available.</div>
    );
  }

  const statuses = ['ALL', 'IN_STOCK', 'LOW_STOCK', 'OUT_OF_VIEW', 'UNKNOWN'];
  const filtered = skuSummary.filter((s) => filterStatus === 'ALL' || s.shelf_status === filterStatus);
  const sorted   = [...filtered].sort(SORT_KEYS[sortKey] ? (a, b) => {
    const va = SORT_KEYS[sortKey](a);
    const vb = SORT_KEYS[sortKey](b);
    return va < vb ? -1 : va > vb ? 1 : 0;
  } : () => 0);

  const TH = ({ label, sKey }) => (
    <th
      className={`px-3 py-3 text-[10px] font-bold uppercase tracking-wider cursor-pointer select-none whitespace-nowrap text-center
        ${sortKey === sKey ? 'text-cyan-400' : 'text-slate-500 hover:text-slate-300'}`}
      onClick={() => setSortKey(sKey)}
    >
      {label} {sortKey === sKey ? '↑' : ''}
    </th>
  );

  return (
    <div className="glass-panel overflow-hidden">
      {/* Filter bar */}
      <div className="flex flex-wrap items-center gap-2 px-4 py-3 border-b border-slate-800/80">
        <span className="text-xs text-slate-500 font-semibold uppercase tracking-wider mr-1">Filter:</span>
        {statuses.map((s) => (
          <button
            key={s}
            onClick={() => setFilterStatus(s)}
            className={`px-3 py-1 rounded-full text-xs font-bold border transition-colors
              ${filterStatus === s
                ? 'bg-cyan-500/20 border-cyan-500/40 text-cyan-300'
                : 'border-slate-700/60 text-slate-400 hover:border-slate-600 hover:text-slate-200'}`}
          >
            {s.replace(/_/g, ' ')}
          </button>
        ))}
        <span className="ml-auto text-[10px] text-slate-500 font-mono">{sorted.length} SKUs</span>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead className="border-b border-slate-800/80 bg-slate-900/50">
            <tr>
              <th className="px-4 py-3 text-[10px] font-bold uppercase tracking-wider text-slate-500 cursor-pointer" onClick={() => setSortKey('name')}>
                SKU {sortKey === 'name' ? '↑' : ''}
              </th>
              <th className="px-3 py-3 text-[10px] font-bold uppercase tracking-wider text-slate-500">Category</th>
              <th className="px-3 py-3 text-[10px] font-bold uppercase tracking-wider text-slate-500 cursor-pointer" onClick={() => setSortKey('status')}>
                Status {sortKey === 'status' ? '↑' : ''}
              </th>
              <TH label="Stable" sKey="stable" />
              <TH label="Active" sKey="active" />
              <th className="px-3 py-3 text-[10px] font-bold uppercase tracking-wider text-center text-slate-500">Uncert.</th>
              <th className="px-3 py-3 text-[10px] font-bold uppercase tracking-wider text-center text-slate-500">Changing</th>
              <th className="px-3 py-3 text-[10px] font-bold uppercase tracking-wider text-center text-slate-500">Appeared</th>
              <th className="px-3 py-3 text-[10px] font-bold uppercase tracking-wider text-center text-slate-500">Removed</th>
              <TH label="Alerts" sKey="alerts" />
            </tr>
          </thead>
          <tbody>
            {sorted.map((sku, i) => (
              <SKURow key={sku.sku_id} sku={sku} index={i} />
            ))}
          </tbody>
        </table>
      </div>

      <div className="px-4 py-2 border-t border-slate-800/60 text-[10px] text-slate-500">
        Click any row with alerts to expand alert details. &nbsp;|&nbsp; Stable Facings = camera-confirmed persistent front-row products only.
      </div>
    </div>
  );
}
