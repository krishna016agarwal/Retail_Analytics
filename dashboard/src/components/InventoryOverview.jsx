import React from 'react';
import {
  Package,
  CheckCircle,
  AlertTriangle,
  XCircle,
  Eye,
  Activity,
  ShieldAlert,
  Info,
} from 'lucide-react';

// ─── Health badge ────────────────────────────────────────────────────────────
function HealthBadge({ status }) {
  const cfg = {
    HEALTHY: {
      cls: 'bg-emerald-500/15 border-emerald-500/30 text-emerald-400',
      icon: <CheckCircle className="h-3.5 w-3.5" />,
    },
    ATTENTION_REQUIRED: {
      cls: 'bg-amber-500/15 border-amber-500/30 text-amber-300 animate-pulse',
      icon: <AlertTriangle className="h-3.5 w-3.5" />,
    },
    VERIFICATION_REQUIRED: {
      cls: 'bg-rose-500/15 border-rose-500/30 text-rose-300 animate-pulse',
      icon: <ShieldAlert className="h-3.5 w-3.5" />,
    },
  };
  const { cls, icon } = cfg[status] || cfg['ATTENTION_REQUIRED'];
  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full border text-xs font-bold uppercase tracking-wider ${cls}`}>
      {icon}
      {status?.replace(/_/g, ' ')}
    </span>
  );
}

// ─── Single KPI tile ─────────────────────────────────────────────────────────
function KPI({ label, value, sub, Icon, color, pulse }) {
  const colorMap = {
    cyan:    'text-cyan-400 bg-cyan-500/10 border-cyan-500/20',
    emerald: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
    amber:   'text-amber-300 bg-amber-500/10 border-amber-500/20',
    rose:    'text-rose-400 bg-rose-500/10 border-rose-500/20',
    violet:  'text-violet-400 bg-violet-500/10 border-violet-500/20',
    slate:   'text-slate-300 bg-slate-500/10 border-slate-500/20',
    sky:     'text-sky-400 bg-sky-500/10 border-sky-500/20',
  };
  const c = colorMap[color] || colorMap['slate'];
  return (
    <div className={`glass-card flex flex-col gap-2 p-4 ${pulse ? 'ring-1 ring-rose-500/30' : ''}`}>
      <div className={`inline-flex items-center justify-center h-9 w-9 rounded-lg border ${c}`}>
        <Icon className="h-4 w-4" />
      </div>
      <div>
        <p className="text-2xl font-extrabold font-mono text-white">{value ?? '—'}</p>
        <p className="text-xs font-semibold text-slate-200 mt-0.5">{label}</p>
        {sub && <p className="text-[10px] text-slate-500 mt-0.5">{sub}</p>}
      </div>
    </div>
  );
}

// ─── Facings stacked bar ──────────────────────────────────────────────────────
function FacingsBar({ stable, uncertain, changing, unknown, total }) {
  if (!total || total === 0) return null;
  const pct = (v) => `${Math.round((v / total) * 100)}%`;
  return (
    <div>
      <div className="flex h-3 w-full rounded-full overflow-hidden gap-px bg-slate-800">
        {stable > 0    && <div style={{ width: pct(stable) }}    className="bg-emerald-500 transition-all duration-500" title={`Stable: ${stable}`} />}
        {uncertain > 0 && <div style={{ width: pct(uncertain) }} className="bg-amber-400 transition-all duration-500"  title={`Uncertain: ${uncertain}`} />}
        {changing > 0  && <div style={{ width: pct(changing) }}  className="bg-sky-400 transition-all duration-500"    title={`Possibly Changing: ${changing}`} />}
        {unknown > 0   && <div style={{ width: pct(unknown) }}   className="bg-slate-600 transition-all duration-500"  title={`Unknown: ${unknown}`} />}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-1.5 text-[10px] font-mono text-slate-400">
        <span><span className="inline-block h-1.5 w-2 rounded-sm bg-emerald-500 mr-1" />Stable {stable}</span>
        <span><span className="inline-block h-1.5 w-2 rounded-sm bg-amber-400 mr-1" />Uncertain {uncertain}</span>
        <span><span className="inline-block h-1.5 w-2 rounded-sm bg-sky-400 mr-1" />Changing {changing}</span>
        <span><span className="inline-block h-1.5 w-2 rounded-sm bg-slate-600 mr-1" />Unknown {unknown}</span>
      </div>
    </div>
  );
}

// ─── Alert type distribution mini-bar ────────────────────────────────────────
function AlertTypeBar({ alertsByType }) {
  const entries = Object.entries(alertsByType || {}).filter(([, v]) => v > 0);
  if (!entries.length) return <span className="text-xs text-slate-500">None</span>;
  const total = entries.reduce((s, [, v]) => s + v, 0);
  const colors = {
    LOW_STOCK:                'bg-amber-400',
    POSSIBLE_STOCKOUT:        'bg-rose-500',
    RAPID_REMOVAL:            'bg-rose-400',
    PRODUCT_MOVEMENT:         'bg-sky-400',
    SKU_RECOGNITION_UNCERTAIN:'bg-violet-400',
  };
  return (
    <div className="space-y-1.5">
      {entries.map(([type, count]) => (
        <div key={type} className="flex items-center gap-2 text-xs">
          <span className="text-slate-400 w-44 truncate">{type.replace(/_/g, ' ')}</span>
          <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full ${colors[type] || 'bg-slate-500'}`}
              style={{ width: `${Math.round((count / total) * 100)}%` }}
            />
          </div>
          <span className="font-mono text-white w-5 text-right">{count}</span>
        </div>
      ))}
    </div>
  );
}

// ─── Main ─────────────────────────────────────────────────────────────────────
export default function InventoryOverview({ report, loading, pipelineRunning = false }) {
  if (pipelineRunning) {
    return (
      <div className="glass-panel p-8 rounded-2xl border border-violet-500/30 bg-violet-950/10 flex flex-col items-center justify-center text-center space-y-3 py-16">
        <div className="h-12 w-12 rounded-xl bg-violet-600/20 border border-violet-500/40 flex items-center justify-center">
          <Activity className="h-6 w-6 text-violet-400 animate-spin" />
        </div>
        <h3 className="text-base font-bold text-white">Overview Metrics Updating</h3>
        <p className="text-xs text-slate-400 max-w-md">
          Pipeline is actively analyzing shelf video. Facing counts, health status, and SKU indicators will refresh when analysis completes.
        </p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-4 gap-3 animate-pulse">
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="glass-card h-24 bg-slate-800/40" />
        ))}
      </div>
    );
  }

  if (!report) {
    return (
      <div className="glass-panel p-6 flex items-center gap-3 text-slate-400 text-sm">
        <Info className="h-5 w-5 text-cyan-400 shrink-0" />
        No inventory report loaded. Run <code className="bg-slate-800 px-1 rounded">demo_inventory_report.py</code> to generate one.
      </div>
    );
  }

  const r = report;
  const totalFacings = r.total_active_visible_facings + r.total_unknown_facings;

  return (
    <div className="space-y-5">
      {/* Semantics notice */}
      <div className="flex items-start gap-2.5 p-3 rounded-lg bg-slate-900/70 border border-slate-700/60 text-[11px] text-slate-400 leading-relaxed">
        <Info className="h-4 w-4 text-cyan-400 mt-0.5 shrink-0" />
        <span>
          <strong className="text-slate-200">Visible Facings Only:</strong>{' '}
          {r.semantics_notice}
        </span>
      </div>

      {/* Health badge + reason */}
      <div className="flex flex-wrap items-center gap-3">
        <HealthBadge status={r.shelf_health} />
        <span className="text-xs text-slate-400">{r.health_reason}</span>
        <span className="ml-auto text-[10px] font-mono text-slate-500">
          Frame {(r.frame_index ?? 0) + 1}/{r.total_frames ?? '?'} &nbsp;|&nbsp; {r.timestamp_sec?.toFixed(2)}s &nbsp;|&nbsp; {r.timestamp_iso?.slice(0, 19)}
        </span>
      </div>

      {/* KPI grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-4 gap-3">
        <KPI label="Active Visible Facings"  value={r.total_active_visible_facings} sub="Camera front-row only"    Icon={Eye}         color="cyan"    />
        <KPI label="Stable Facings"          value={r.stable_facings}               sub="Confirmed persistent"     Icon={CheckCircle} color="emerald" />
        <KPI label="Uncertain Facings"       value={r.uncertain_facings}            sub="New / fading tracks"      Icon={Activity}    color="amber"   />
        <KPI label="Unknown Facings"         value={r.total_unknown_facings}        sub="Unrecognized products"    Icon={Package}     color="slate"   />
        <KPI label="Catalog SKUs Visible"    value={`${r.catalog_skus_visible}/${r.catalog_skus_registered}`} sub="SKUs in camera view" Icon={Package} color="violet" />
        <KPI label="IN_STOCK SKUs"           value={r.in_stock_skus_count}          sub="≥ low-stock threshold"    Icon={CheckCircle} color="emerald" />
        <KPI label="LOW_STOCK SKUs"          value={r.low_stock_skus_count}         sub="Below threshold"          Icon={AlertTriangle} color="amber" pulse={r.low_stock_skus_count > 0} />
        <KPI label="OUT_OF_VIEW SKUs"        value={r.out_of_view_skus_count}       sub="0 stable facings visible" Icon={XCircle}     color="rose"    pulse={r.out_of_view_skus_count > 0} />
      </div>

      {/* Facings breakdown bar */}
      <div className="glass-panel p-4 space-y-3">
        <p className="text-xs font-bold uppercase tracking-wider text-slate-400">Facing Stability Breakdown</p>
        <FacingsBar
          stable={r.stable_facings}
          uncertain={r.uncertain_facings}
          changing={r.possibly_changing_facings}
          unknown={r.total_unknown_facings}
          total={totalFacings}
        />
      </div>

      {/* Alerts summary strip */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="glass-panel p-4 space-y-3">
          <p className="text-xs font-bold uppercase tracking-wider text-slate-400">Alert Severity</p>
          <div className="flex flex-wrap gap-3">
            {['HIGH', 'MEDIUM', 'LOW', 'INFO'].map((sev) => {
              const cnt = r.alerts_by_severity?.[sev] ?? 0;
              const cls = { HIGH: 'bg-rose-500/15 text-rose-300 border-rose-500/30', MEDIUM: 'bg-amber-500/15 text-amber-300 border-amber-500/30', LOW: 'bg-sky-500/15 text-sky-300 border-sky-500/30', INFO: 'bg-slate-500/15 text-slate-300 border-slate-500/30' }[sev];
              return (
                <div key={sev} className={`flex flex-col items-center px-4 py-2 rounded-lg border ${cls}`}>
                  <span className="text-2xl font-extrabold font-mono">{cnt}</span>
                  <span className="text-[10px] font-bold uppercase">{sev}</span>
                </div>
              );
            })}
            <div className="flex flex-col items-center px-4 py-2 rounded-lg border border-slate-700/50 bg-slate-800/30">
              <span className="text-2xl font-extrabold font-mono text-slate-200">{r.verification_required_count ?? 0}</span>
              <span className="text-[10px] font-bold uppercase text-slate-400">Verif. Req.</span>
            </div>
          </div>
        </div>

        <div className="glass-panel p-4 space-y-3">
          <p className="text-xs font-bold uppercase tracking-wider text-slate-400">Alerts by Type</p>
          <AlertTypeBar alertsByType={r.alerts_by_type} />
        </div>
      </div>
    </div>
  );
}
