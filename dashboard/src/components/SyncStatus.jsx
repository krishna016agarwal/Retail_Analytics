import React from 'react';
import { Database, Server } from 'lucide-react';
import StatusBadge from './StatusBadge';
import { formatRelativeTime } from '../utils/formatters';

export default function SyncStatus({
  syncData = {},
  healthData = {},
  latestSnapshot = {},
  isRefreshing = false,
}) {
  // Strict status evaluation directly from /health endpoint per requirements
  const isApiHealthy = String(healthData?.status || '').toLowerCase() === 'healthy';
  const isDbConnected = String(healthData?.database || '').toLowerCase() === 'connected';

  const totalSnapshots = syncData?.total_snapshots || 0;
  const pendingSnapshots = syncData?.pending_snapshots ?? 0;
  const syncedSnapshots = syncData?.synced_snapshots ?? Math.max(0, totalSnapshots - pendingSnapshots);
  const storeCount = syncData?.stores || 1;
  const deviceCount = syncData?.devices || 1;
  const latestTs = syncData?.latest_timestamp || latestSnapshot?.created_at;

  return (
    <div className="glass-panel p-5 flex flex-col justify-between">
      <div>
        <div className="flex items-center justify-between pb-3 border-b border-slate-800/80">
          <div className="flex items-center gap-2">
            <Server className="h-4 w-4 text-cyan-400" />
            <h3 className="text-sm font-semibold text-white">Cloud Sync & Telemetry</h3>
          </div>
          <StatusBadge
            status={isApiHealthy ? 'healthy' : 'warning'}
            label={isApiHealthy ? 'Connected / Healthy' : 'Degraded'}
            size="sm"
          />
        </div>

        {/* Sync Telemetry Metrics Grid: Clearly Distinguishing Edge vs Synced vs Pending */}
        <div className="grid grid-cols-3 gap-2 mt-4 text-center">
          <div className="bg-slate-950/50 p-2.5 rounded-lg border border-slate-800/60">
            <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Edge Local</span>
            <span className="text-lg font-bold text-white mt-0.5 block font-mono">
              {totalSnapshots}
            </span>
          </div>

          <div className="bg-slate-950/50 p-2.5 rounded-lg border border-slate-800/60">
            <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Cloud Synced</span>
            <span className="text-lg font-bold text-emerald-400 mt-0.5 block font-mono">
              {syncedSnapshots}
            </span>
          </div>

          <div className="bg-slate-950/50 p-2.5 rounded-lg border border-slate-800/60">
            <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Pending</span>
            <span className={`text-lg font-bold mt-0.5 block font-mono ${pendingSnapshots > 0 ? 'text-amber-400' : 'text-slate-400'}`}>
              {pendingSnapshots}
            </span>
          </div>
        </div>

        {/* Health status breakdown strictly driven by /health response */}
        <div className="mt-3 space-y-2 text-xs">
          {/* Requirement: PostgreSQL Connected when /health.database == "connected" */}
          <div className="flex items-center justify-between p-2 rounded bg-slate-950/30 border border-slate-800/40">
            <span className="text-slate-400 flex items-center gap-1.5">
              <Database className="h-3.5 w-3.5 text-indigo-400" />
              Central PostgreSQL
            </span>
            <span className={`font-semibold ${isDbConnected ? 'text-emerald-400' : 'text-rose-400'}`}>
              {isDbConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>

          {/* Requirement: Central Cloud API Healthy when /health.status == "healthy" */}
          <div className="flex items-center justify-between p-2 rounded bg-slate-950/30 border border-slate-800/40">
            <span className="text-slate-400 flex items-center gap-1.5">
              <Server className="h-3.5 w-3.5 text-cyan-400" />
              Central Cloud API
            </span>
            <span className={`font-semibold ${isApiHealthy ? 'text-emerald-400' : 'text-rose-400'}`}>
              {isApiHealthy ? 'Connected / Healthy' : 'Disconnected'}
            </span>
          </div>
        </div>
      </div>

      {/* Footer / Last Synced Timestamp */}
      <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
        <span>Last Cloud Ingest:</span>
        <span className="text-slate-200 font-medium">
          {latestTs ? formatRelativeTime(latestTs) : 'Pending first sync'}
        </span>
      </div>
    </div>
  );
}
