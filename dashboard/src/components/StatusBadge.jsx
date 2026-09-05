import React from 'react';

export default function StatusBadge({ status, label, size = 'md' }) {
  const configs = {
    healthy: {
      bg: 'bg-emerald-500/10',
      border: 'border-emerald-500/30',
      text: 'text-emerald-400',
      dot: 'bg-emerald-400',
      pulse: true,
    },
    online: {
      bg: 'bg-emerald-500/10',
      border: 'border-emerald-500/30',
      text: 'text-emerald-400',
      dot: 'bg-emerald-400',
      pulse: false,
    },
    receiving: {
      bg: 'bg-cyan-500/10',
      border: 'border-cyan-500/30',
      text: 'text-cyan-400',
      dot: 'bg-cyan-400',
      pulse: true,
    },
    warning: {
      bg: 'bg-amber-500/10',
      border: 'border-amber-500/30',
      text: 'text-amber-400',
      dot: 'bg-amber-400',
      pulse: false,
    },
    cold_start: {
      bg: 'bg-amber-500/10',
      border: 'border-amber-500/30',
      text: 'text-amber-400',
      dot: 'bg-amber-400',
      pulse: true,
    },
    unhealthy: {
      bg: 'bg-rose-500/10',
      border: 'border-rose-500/30',
      text: 'text-rose-400',
      dot: 'bg-rose-400',
      pulse: false,
    },
    offline: {
      bg: 'bg-rose-500/10',
      border: 'border-rose-500/30',
      text: 'text-rose-400',
      dot: 'bg-rose-400',
      pulse: false,
    },
    neutral: {
      bg: 'bg-slate-800',
      border: 'border-slate-700',
      text: 'text-slate-300',
      dot: 'bg-slate-400',
      pulse: false,
    },
  };

  const current = configs[status] || configs.neutral;
  const sizeClasses = size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-2.5 py-1 text-xs font-medium';

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border ${current.bg} ${current.border} ${current.text} ${sizeClasses}`}
    >
      <span className="relative flex h-2 w-2">
        {current.pulse && (
          <span
            className={`animate-ping absolute inline-flex h-full w-full rounded-full ${current.dot} opacity-75`}
          />
        )}
        <span className={`relative inline-flex rounded-full h-2 w-2 ${current.dot}`} />
      </span>
      {label || status.toUpperCase()}
    </span>
  );
}
