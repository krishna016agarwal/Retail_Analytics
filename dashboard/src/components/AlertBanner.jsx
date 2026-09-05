import React from 'react';
import { AlertTriangle, WifiOff, RefreshCw, XCircle, Info } from 'lucide-react';

export default function AlertBanner({ error, onRetry }) {
  if (!error) return null;

  const isColdStart = error.type === 'COLD_START';
  const isNetwork = error.type === 'NETWORK_ERROR';
  const isDbIssue = error.type === 'DATABASE_DISCONNECTED';

  let bannerClass = 'bg-rose-500/10 border-rose-500/30 text-rose-300';
  let Icon = XCircle;
  let title = 'Central Cloud API Error';

  if (isColdStart) {
    bannerClass = 'bg-amber-500/10 border-amber-500/30 text-amber-300';
    Icon = AlertTriangle;
    title = 'Central Cloud Service Waking Up (Render Cold Start)';
  } else if (isNetwork) {
    bannerClass = 'bg-rose-500/10 border-rose-500/30 text-rose-300';
    Icon = WifiOff;
    title = 'Network Connection Issue';
  } else if (isDbIssue) {
    bannerClass = 'bg-amber-500/10 border-amber-500/30 text-amber-300';
    Icon = Info;
    title = 'PostgreSQL Connectivity Notice';
  }

  return (
    <div className={`p-4 rounded-xl border mb-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 ${bannerClass}`}>
      <div className="flex items-start gap-3">
        <Icon className="h-5 w-5 shrink-0 mt-0.5" />
        <div>
          <h4 className="text-xs font-bold uppercase tracking-wider">{title}</h4>
          <p className="text-xs mt-0.5 opacity-90 leading-relaxed">
            {error.message}
          </p>
        </div>
      </div>

      {onRetry && (
        <button
          onClick={onRetry}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900/80 hover:bg-slate-900 border border-slate-700 text-xs font-semibold text-white shrink-0 transition-colors"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          Retry Connection
        </button>
      )}
    </div>
  );
}
