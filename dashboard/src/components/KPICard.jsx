import React from 'react';

export default function KPICard({
  title,
  subtitle,
  value,
  unit = '',
  icon: Icon,
  accentColor = 'cyan', // 'emerald' | 'cyan' | 'amber' | 'rose' | 'indigo' | 'sky'
  badgeText,
  badgeType = 'neutral', // 'positive' | 'warning' | 'alert' | 'neutral'
  loading = false,
}) {
  const colorMap = {
    emerald: {
      bg: 'bg-emerald-500/10',
      border: 'border-emerald-500/20',
      iconText: 'text-emerald-400',
      accentGlow: 'hover:border-emerald-500/40',
    },
    cyan: {
      bg: 'bg-cyan-500/10',
      border: 'border-cyan-500/20',
      iconText: 'text-cyan-400',
      accentGlow: 'hover:border-cyan-500/40',
    },
    amber: {
      bg: 'bg-amber-500/10',
      border: 'border-amber-500/20',
      iconText: 'text-amber-400',
      accentGlow: 'hover:border-amber-500/40',
    },
    rose: {
      bg: 'bg-rose-500/10',
      border: 'border-rose-500/20',
      iconText: 'text-rose-400',
      accentGlow: 'hover:border-rose-500/40',
    },
    indigo: {
      bg: 'bg-indigo-500/10',
      border: 'border-indigo-500/20',
      iconText: 'text-indigo-400',
      accentGlow: 'hover:border-indigo-500/40',
    },
    sky: {
      bg: 'bg-sky-500/10',
      border: 'border-sky-500/20',
      iconText: 'text-sky-400',
      accentGlow: 'hover:border-sky-500/40',
    },
  };

  const badgeColorMap = {
    positive: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
    warning: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
    alert: 'bg-rose-500/15 text-rose-400 border-rose-500/30',
    neutral: 'bg-slate-800 text-slate-300 border-slate-700',
  };

  const colors = colorMap[accentColor] || colorMap.cyan;
  const badgeStyle = badgeColorMap[badgeType] || badgeColorMap.neutral;

  if (loading) {
    return (
      <div className="glass-card animate-pulse">
        <div className="flex items-center justify-between">
          <div className="h-4 w-24 bg-slate-800 rounded mb-2"></div>
          <div className="h-8 w-8 bg-slate-800 rounded-lg"></div>
        </div>
        <div className="h-8 w-16 bg-slate-800 rounded my-3"></div>
        <div className="h-3 w-32 bg-slate-800/60 rounded"></div>
      </div>
    );
  }

  return (
    <div className={`glass-card ${colors.accentGlow} flex flex-col justify-between`}>
      <div>
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
            {title}
          </span>
          {Icon && (
            <div className={`p-2 rounded-lg ${colors.bg} ${colors.border} ${colors.iconText}`}>
              <Icon className="h-4 w-4" />
            </div>
          )}
        </div>

        <div className="mt-2 flex items-baseline gap-1">
          <span className="text-3xl font-bold tracking-tight text-white">
            {value !== undefined && value !== null ? value : '--'}
          </span>
          {unit && (
            <span className="text-xs font-semibold text-slate-400">
              {unit}
            </span>
          )}
        </div>
      </div>

      <div className="mt-3 pt-2.5 border-t border-slate-800/60 flex items-center justify-between text-xs">
        <span className="text-slate-400 truncate">
          {subtitle}
        </span>
        {badgeText && (
          <span className={`px-2 py-0.5 text-[10px] font-medium rounded-full border ${badgeStyle}`}>
            {badgeText}
          </span>
        )}
      </div>
    </div>
  );
}
