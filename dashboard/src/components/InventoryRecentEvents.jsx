import React, { useState } from 'react';
import { PackagePlus, PackageMinus, Move, Tag, Clock } from 'lucide-react';

// ─── Event type config ────────────────────────────────────────────────────────
const EVENT_CFG = {
  PRODUCT_APPEARED: { Icon: PackagePlus,  cls: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20', label: 'Appeared'  },
  PRODUCT_REMOVED:  { Icon: PackageMinus, cls: 'text-rose-400 bg-rose-500/10 border-rose-500/20',         label: 'Removed'   },
  PRODUCT_MOVED:    { Icon: Move,         cls: 'text-sky-400 bg-sky-500/10 border-sky-500/20',            label: 'Moved'     },
  SKU_CHANGED:      { Icon: Tag,          cls: 'text-violet-400 bg-violet-500/10 border-violet-500/20',   label: 'SKU Changed'},
};

function EventRow({ ev }) {
  const cfg = EVENT_CFG[ev.event_type] || { Icon: Clock, cls: 'text-slate-400 bg-slate-500/10 border-slate-500/20', label: ev.event_type };
  const { Icon, cls, label } = cfg;
  return (
    <div className="flex items-start gap-3 py-2.5 border-b border-slate-800/60 last:border-0">
      {/* Icon */}
      <div className={`mt-0.5 h-7 w-7 rounded-lg border flex items-center justify-center shrink-0 ${cls}`}>
        <Icon className="h-3.5 w-3.5" />
      </div>

      {/* Details */}
      <div className="flex-1 min-w-0">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
          <span className="text-xs font-bold text-white">{label}</span>
          <span className="text-xs text-slate-300 truncate">{ev.sku_name || ev.sku_id || 'Unknown'}</span>
          {ev.track_id !== undefined && ev.track_id !== null && (
            <span className="text-[10px] font-mono text-slate-500">Track #{ev.track_id}</span>
          )}
        </div>
        {ev.description && (
          <p className="text-[11px] text-slate-400 mt-0.5 leading-relaxed">{ev.description}</p>
        )}
      </div>

      {/* Timestamp */}
      <div className="text-right shrink-0">
        <span className="text-[10px] font-mono text-slate-500">
          @{typeof ev.timestamp_sec === 'number' ? ev.timestamp_sec.toFixed(2) : '?'}s
        </span>
        {ev.frame_index !== undefined && (
          <span className="block text-[9px] text-slate-600 font-mono">f{ev.frame_index}</span>
        )}
      </div>
    </div>
  );
}

// ─── Event type counter pills ─────────────────────────────────────────────────
function EventCounter({ events }) {
  const counts = {};
  events.forEach((e) => {
    counts[e.event_type] = (counts[e.event_type] || 0) + 1;
  });
  return (
    <div className="flex flex-wrap gap-2">
      {Object.entries(EVENT_CFG).map(([type, { cls, label }]) => {
        const c = counts[type] || 0;
        return (
          <div key={type} className={`px-3 py-1 rounded-lg border text-[10px] font-bold flex items-center gap-1.5 ${cls}`}>
            <span className="text-lg font-extrabold font-mono leading-none">{c}</span>
            {label}
          </div>
        );
      })}
    </div>
  );
}

// ─── Main panel ───────────────────────────────────────────────────────────────
export default function InventoryRecentEvents({ events, loading }) {
  const [filterType, setFilterType] = useState('ALL');

  if (loading) {
    return (
      <div className="space-y-2 animate-pulse">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="h-12 bg-slate-800/40 rounded-lg" />
        ))}
      </div>
    );
  }

  if (!events?.length) {
    return (
      <div className="glass-panel p-6 text-sm text-slate-400">No recent inventory events recorded.</div>
    );
  }

  const types = ['ALL', ...Object.keys(EVENT_CFG)];
  const displayed = filterType === 'ALL'
    ? events
    : events.filter((e) => e.event_type === filterType);

  return (
    <div className="space-y-4">
      {/* Event counter */}
      <div className="glass-panel p-4 space-y-3">
        <p className="text-xs font-bold uppercase tracking-wider text-slate-400">Event Summary</p>
        <EventCounter events={events} />
      </div>

      {/* Filter */}
      <div className="flex flex-wrap items-center gap-2 px-1">
        <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Filter:</span>
        {types.map((t) => (
          <button
            key={t}
            onClick={() => setFilterType(t)}
            className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold border transition-colors
              ${filterType === t ? 'bg-cyan-500/20 border-cyan-500/40 text-cyan-300' : 'border-slate-700/50 text-slate-400 hover:text-slate-200'}`}
          >
            {EVENT_CFG[t]?.label || t}
          </button>
        ))}
        <span className="ml-auto text-[10px] font-mono text-slate-500">{displayed.length} events</span>
      </div>

      {/* Event list */}
      <div className="glass-panel px-4 py-2 max-h-[55vh] overflow-y-auto">
        {displayed.length === 0 ? (
          <p className="text-slate-500 text-sm py-4 text-center">No events match this filter.</p>
        ) : (
          displayed.map((ev, i) => <EventRow key={i} ev={ev} />)
        )}
      </div>
    </div>
  );
}
