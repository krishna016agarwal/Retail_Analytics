import React from 'react';
import { 
  Users, 
  AlertCircle, 
  CheckCircle2, 
  AlertTriangle, 
  TrendingUp, 
  TrendingDown, 
  Minus, 
  Clock, 
  ArrowRight 
} from 'lucide-react';
import { formatSeconds } from '../utils/formatters';

export default function QueueStatus({ 
  queueData = {}, 
  queueLength = 0, 
  peakQueue = 0, 
  avgWait = 0 
}) {
  // Resolve metrics from queueData or props
  const currentQ = queueData.current_queue ?? queueData.current_length ?? queueLength ?? 0;
  const peakQ = queueData.peak_queue ?? queueData.peak_length ?? peakQueue ?? currentQ;
  const growthRate = queueData.growth_rate_per_min ?? queueData.growth_rate ?? 0.0;
  const trend = (queueData.trend || 'STABLE').toUpperCase();
  const predicted3m = queueData.predicted_queue_3min ?? queueData.predicted_in_3m ?? currentQ;
  const risk = (queueData.congestion_risk || (currentQ >= 8 ? 'HIGH' : currentQ >= 4 ? 'MEDIUM' : 'LOW')).toUpperCase();
  const waitSeconds = queueData.average_wait_time ?? queueData.avg_wait_seconds ?? avgWait ?? 0;

  // Determine severity tier styling
  const styles = {
    LOW: {
      badge: 'bg-emerald-500/15 border-emerald-500/30 text-emerald-400',
      meter: 'bg-emerald-500',
      text: 'text-emerald-400',
      icon: CheckCircle2,
      recommendation: 'Normal flow. No counter adjustments needed.',
    },
    MEDIUM: {
      badge: 'bg-amber-500/15 border-amber-500/30 text-amber-400',
      meter: 'bg-amber-500',
      text: 'text-amber-400',
      icon: AlertTriangle,
      recommendation: 'Moderate queue. Prepare standby billing counter.',
    },
    HIGH: {
      badge: 'bg-rose-500/15 border-rose-500/30 text-rose-300 animate-pulse',
      meter: 'bg-rose-500',
      text: 'text-rose-400',
      icon: AlertCircle,
      recommendation: 'Open additional billing counter immediately.',
    },
  }[risk] || {
    badge: 'bg-emerald-500/15 border-emerald-500/30 text-emerald-400',
    meter: 'bg-emerald-500',
    text: 'text-emerald-400',
    icon: CheckCircle2,
    recommendation: 'Normal flow. No counter adjustments needed.',
  };

  const Icon = styles.icon;

  // Trend icon and color
  const getTrendBadge = () => {
    if (trend === 'GROWING') {
      return (
        <span className="flex items-center gap-1 text-[11px] font-semibold text-rose-400 bg-rose-500/10 border border-rose-500/20 px-2 py-0.5 rounded">
          <TrendingUp className="h-3 w-3" />
          Growing ({growthRate > 0 ? `+${growthRate}` : growthRate}/min)
        </span>
      );
    }
    if (trend === 'SHRINKING') {
      return (
        <span className="flex items-center gap-1 text-[11px] font-semibold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded">
          <TrendingDown className="h-3 w-3" />
          Shrinking ({growthRate}/min)
        </span>
      );
    }
    return (
      <span className="flex items-center gap-1 text-[11px] font-semibold text-cyan-400 bg-cyan-500/10 border border-cyan-500/20 px-2 py-0.5 rounded">
        <Minus className="h-3 w-3" />
        Stable ({growthRate}/min)
      </span>
    );
  };

  // Progress Bar gauge (threshold of 8 = 100%)
  const meterPercent = Math.min(100, Math.round((currentQ / 8) * 100));

  return (
    <div className="glass-panel p-5 flex flex-col justify-between">
      <div>
        {/* Header */}
        <div className="flex items-center justify-between pb-3 border-b border-slate-800/80">
          <div className="flex items-center gap-2">
            <Users className="h-4 w-4 text-amber-400" />
            <h3 className="text-sm font-semibold text-white">Queue Intelligence</h3>
          </div>
          <span className={`px-2.5 py-0.5 text-xs font-bold uppercase rounded-full border ${styles.badge}`}>
            {risk} RISK
          </span>
        </div>

        {/* Current Queue & Peak */}
        <div className="mt-4">
          <div className="flex items-baseline justify-between">
            <div>
              <span className="text-3xl font-extrabold text-white">
                {currentQ}
              </span>
              <span className="text-xs text-slate-400 ml-1.5">waiting shoppers</span>
            </div>
            <span className="text-xs text-slate-400">
              Peak: <strong className="text-slate-200">{peakQ}</strong>
            </span>
          </div>

          {/* Progress Bar Gauge */}
          <div className="w-full bg-slate-800 rounded-full h-2 mt-2.5 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${styles.meter}`}
              style={{ width: `${Math.max(6, meterPercent)}%` }}
            />
          </div>
          <div className="flex justify-between text-[10px] text-slate-500 mt-1 font-mono">
            <span>0-3 Low</span>
            <span>4-7 Moderate</span>
            <span>8+ Congested</span>
          </div>
        </div>

        {/* Predictive & Analytical Metrics Grid */}
        <div className="grid grid-cols-2 gap-2.5 mt-4 pt-3 border-t border-slate-800/60 text-xs">
          {/* Predicted Queue in 3m */}
          <div className="bg-slate-950/50 p-2.5 rounded-lg border border-slate-800/60">
            <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
              Predicted in 3m
            </span>
            <div className="flex items-baseline gap-1.5 mt-1">
              <span className={`text-lg font-bold font-mono ${predicted3m >= 8 ? 'text-rose-400' : 'text-slate-100'}`}>
                {predicted3m}
              </span>
              <span className="text-[11px] text-slate-400">shoppers</span>
            </div>
          </div>

          {/* Average Wait Time */}
          <div className="bg-slate-950/50 p-2.5 rounded-lg border border-slate-800/60">
            <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
              Average Wait
            </span>
            <div className="flex items-baseline gap-1.5 mt-1">
              <Clock className="h-3 w-3 text-cyan-400" />
              <span className="text-sm font-bold font-mono text-slate-100">
                {formatSeconds(waitSeconds)}
              </span>
            </div>
          </div>
        </div>

        {/* Queue Trend & Growth Rate */}
        <div className="flex items-center justify-between mt-3 px-1 text-xs">
          <span className="text-slate-400">Queue Dynamic:</span>
          {getTrendBadge()}
        </div>
      </div>

      {/* Actionable recommendation banner */}
      <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-start gap-2">
        <Icon className={`h-4 w-4 shrink-0 mt-0.5 ${styles.text}`} />
        <p className="text-xs text-slate-200 font-medium leading-relaxed">
          {styles.recommendation}
        </p>
      </div>
    </div>
  );
}
