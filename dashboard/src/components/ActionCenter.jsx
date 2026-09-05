import React, { useState } from 'react';
import { 
  AlertTriangle, 
  AlertOctagon, 
  ShieldAlert, 
  TrendingUp, 
  CheckCircle2, 
  Clock, 
  Layers, 
  ArrowRight,
  ShieldCheck,
  Sparkles
} from 'lucide-react';
import { formatShortDateTime, formatRelativeTime } from '../utils/formatters';

export default function ActionCenter({ alerts = [], loading = false }) {
  const [filter, setFilter] = useState('ALL'); // 'ALL' | 'ACTIVE' | 'RESOLVED'

  // Filter alerts if needed
  const activeAlerts = alerts.filter(a => String(a.status || '').toUpperCase() === 'ACTIVE');
  const resolvedAlerts = alerts.filter(a => String(a.status || '').toUpperCase() === 'RESOLVED');
  
  let displayAlerts = alerts;
  if (filter === 'ACTIVE') displayAlerts = activeAlerts;
  if (filter === 'RESOLVED') displayAlerts = resolvedAlerts;

  const getAlertConfig = (type) => {
    switch (type) {
      case 'QUEUE_CONGESTION':
        return {
          icon: AlertOctagon,
          defaultRec: 'Open additional billing counter',
          badgeColor: 'border-rose-500/40 bg-rose-500/10 text-rose-300',
          accent: 'rose',
        };
      case 'STAFFING':
        return {
          icon: ShieldAlert,
          defaultRec: 'Move 1 staff member to zone',
          badgeColor: 'border-amber-500/40 bg-amber-500/10 text-amber-300',
          accent: 'amber',
        };
      case 'CROWD_SPIKE':
        return {
          icon: TrendingUp,
          defaultRec: 'Monitor zone / deploy floor staff',
          badgeColor: 'border-indigo-500/40 bg-indigo-500/10 text-indigo-300',
          accent: 'indigo',
        };
      default:
        return {
          icon: AlertTriangle,
          defaultRec: 'Review operational alert',
          badgeColor: 'border-cyan-500/40 bg-cyan-500/10 text-cyan-300',
          accent: 'cyan',
        };
    }
  };

  const getSeverityBadge = (severity) => {
    const s = String(severity || '').toUpperCase();
    if (s === 'CRITICAL') {
      return 'bg-rose-600/20 border-rose-500/50 text-rose-300 animate-pulse';
    }
    if (s === 'HIGH') {
      return 'bg-rose-500/20 border-rose-500/40 text-rose-300';
    }
    if (s === 'MEDIUM') {
      return 'bg-amber-500/20 border-amber-500/40 text-amber-300';
    }
    return 'bg-cyan-500/20 border-cyan-500/40 text-cyan-300';
  };

  return (
    <div className="glass-panel p-5 mb-6 border-slate-800">
      {/* Action Center Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800/80">
        <div className="flex items-center gap-2.5">
          <div className="h-9 w-9 rounded-lg bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-400">
            <Sparkles className="h-4 w-4" />
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h2 className="text-base font-bold text-white tracking-tight">
                Retail Action Center
              </h2>
              {activeAlerts.length > 0 ? (
                <span className="px-2 py-0.5 text-[10px] font-bold uppercase rounded-full bg-rose-500/20 border border-rose-500/40 text-rose-300 animate-pulse">
                  Active Alerts: {activeAlerts.length}
                </span>
              ) : (
                <span className="px-2 py-0.5 text-[10px] font-bold uppercase rounded-full bg-emerald-500/20 border border-emerald-500/30 text-emerald-300">
                  Active Alerts: 0 (Nominal)
                </span>
              )}
              <span className="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-slate-800 border border-slate-700 text-slate-300">
                Recent Alerts: {alerts.length}
              </span>
              <span className="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-slate-800/80 border border-slate-700/80 text-slate-400">
                Resolved Alerts: {resolvedAlerts.length}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Deterministic operational intelligence & automated dispatch recommendations
            </p>
          </div>
        </div>

        {/* Filter Tabs */}
        {alerts.length > 0 && (
          <div className="flex items-center gap-1.5 p-1 bg-slate-950/60 rounded-lg border border-slate-800 text-xs self-start sm:self-auto flex-wrap">
            <button
              onClick={() => setFilter('ALL')}
              className={`px-3 py-1 rounded-md transition-colors font-medium ${
                filter === 'ALL'
                  ? 'bg-slate-800 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              All Alerts ({alerts.length})
            </button>
            <button
              onClick={() => setFilter('ACTIVE')}
              className={`px-3 py-1 rounded-md transition-colors font-medium ${
                filter === 'ACTIVE'
                  ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Active ({activeAlerts.length})
            </button>
            <button
              onClick={() => setFilter('RESOLVED')}
              className={`px-3 py-1 rounded-md transition-colors font-medium ${
                filter === 'RESOLVED'
                  ? 'bg-slate-700 text-slate-200 border border-slate-600'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Resolved ({resolvedAlerts.length})
            </button>
          </div>
        )}
      </div>

      {/* Content: Alerts List or Nominal State */}
      <div className="mt-4">
        {displayAlerts.length === 0 ? (
          <div className="p-4 rounded-xl border border-dashed border-slate-800/80 bg-slate-950/40 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-full bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400 shrink-0">
                <CheckCircle2 className="h-5 w-5" />
              </div>
              <div>
                <h3 className="text-xs font-semibold text-slate-200">
                  {filter === 'ACTIVE' && alerts.length > 0
                    ? 'No Active Operational Alerts'
                    : 'All Retail Operations Nominal'}
                </h3>
                <p className="text-[11px] text-slate-400 mt-0.5 leading-relaxed">
                  {filter === 'ACTIVE' && alerts.length > 0
                    ? `All ${alerts.length} historical alerts have been resolved. Switch to 'All Alerts' or 'Recent / Resolved' to review operational history.`
                    : 'No active operational alerts. YOLO tracking, queue monitors, and crowd anomaly detection running in real-time.'}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 text-[10px] font-mono text-slate-400 bg-slate-900/60 px-3 py-1.5 rounded-lg border border-slate-800">
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
              <span>Edge Intelligence Active</span>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {displayAlerts.map((alert) => {
              const cfg = getAlertConfig(alert.type);
              const Icon = cfg.icon;
              const isResolved = String(alert.status || '').toUpperCase() === 'RESOLVED';
              const recommendation = alert.recommendation || cfg.defaultRec;

              return (
                <div
                  key={alert.alert_id || alert.id || Math.random()}
                  className={`p-4 rounded-xl border transition-all ${
                    isResolved
                      ? 'bg-slate-950/40 border-slate-800/60 opacity-70'
                      : 'bg-slate-950/70 border-slate-700/80 shadow-lg'
                  }`}
                >
                  <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3">
                    {/* Left: Icon, Title, Message */}
                    <div className="flex items-start gap-3">
                      <div
                        className={`h-9 w-9 rounded-lg flex items-center justify-center shrink-0 mt-0.5 border ${cfg.badgeColor}`}
                      >
                        <Icon className="h-4 w-4" />
                      </div>
                      <div>
                        <div className="flex flex-wrap items-center gap-2">
                          <span
                            className={`px-2 py-0.5 text-[10px] font-bold uppercase rounded border ${getSeverityBadge(
                              alert.severity
                            )}`}
                          >
                            {alert.severity || 'INFO'}
                          </span>
                          <span className="px-2 py-0.5 text-[10px] font-mono font-medium rounded bg-slate-800 text-slate-300 border border-slate-700">
                            {alert.type || 'SYSTEM'}
                          </span>
                          {alert.zone_id && (
                            <span className="text-[10px] text-slate-400 font-mono">
                              Zone: <span className="text-slate-200">{alert.zone_id}</span>
                            </span>
                          )}
                          <h4 className="text-xs font-bold text-white">
                            {alert.title}
                          </h4>
                        </div>
                        <p className="text-xs text-slate-300 mt-1 leading-relaxed">
                          {alert.message}
                        </p>
                      </div>
                    </div>

                    {/* Right: Recommendation Callout Banner */}
                    <div className="flex flex-col sm:flex-row lg:flex-col items-start lg:items-end justify-between gap-2 shrink-0 bg-slate-900/90 border border-slate-800 p-2.5 rounded-lg">
                      <div className="flex items-center gap-1.5 text-xs font-semibold text-cyan-300">
                        <ArrowRight className="h-3.5 w-3.5 text-cyan-400" />
                        <span>Action: {recommendation}</span>
                      </div>
                      <div className="flex items-center gap-2 text-[10px] text-slate-400">
                        <Clock className="h-3 w-3 text-slate-500" />
                        <span>{formatRelativeTime(alert.timestamp || alert.created_at)}</span>
                        <span>•</span>
                        <span>{formatShortDateTime(alert.timestamp || alert.created_at)}</span>
                      </div>
                    </div>
                  </div>

                  {/* Metrics Row: Current vs Predicted vs Threshold */}
                  <div className="mt-3 pt-2.5 border-t border-slate-800/80 flex flex-wrap items-center gap-4 text-[11px] text-slate-400">
                    <div>
                      Current:{' '}
                      <strong className="text-slate-200 font-mono">
                        {alert.current_value !== undefined && alert.current_value !== null
                          ? alert.current_value
                          : '--'}
                      </strong>
                    </div>
                    {alert.predicted_value !== undefined && alert.predicted_value !== null && (
                      <div>
                        Predicted in 3m:{' '}
                        <strong className="text-amber-300 font-mono">
                          {alert.predicted_value}
                        </strong>
                      </div>
                    )}
                    {alert.threshold !== undefined && alert.threshold !== null && (
                      <div>
                        Threshold:{' '}
                        <strong className="text-slate-300 font-mono">
                          {alert.threshold}
                        </strong>
                      </div>
                    )}
                    <div className="ml-auto">
                      Status:{' '}
                      <span
                        className={`font-semibold ${
                          isResolved ? 'text-slate-400' : 'text-emerald-400'
                        }`}
                      >
                        {alert.status || 'ACTIVE'}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
