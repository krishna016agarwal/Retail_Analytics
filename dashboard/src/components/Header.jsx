import React from 'react';
import { 
  Building2, 
  Cpu, 
  RefreshCw, 
  Activity, 
  Cloud, 
  Clock, 
  ShieldCheck,
  Database
} from 'lucide-react';
import StatusBadge from './StatusBadge';
import { formatTime } from '../utils/formatters';

export default function Header({
  storeId = 'store_001',
  deviceId = 'edge_device_01',
  healthStatus = { status: 'unknown', database: 'unknown' },
  connectionStatus = 'healthy', // 'healthy' | 'offline' | 'cold_start' | 'loading'
  lastUpdated,
  isRefreshing,
  onRefresh,
  autoRefreshInterval,
  setAutoRefreshInterval,
  countdown,
}) {
  const isApiHealthy = String(healthStatus?.status || '').toLowerCase() === 'healthy';
  const isDbConnected = String(healthStatus?.database || '').toLowerCase() === 'connected';
  const isColdStart = healthStatus?.status === 'cold_start';

  const apiBadgeStatus = isColdStart ? 'cold_start' : isApiHealthy ? 'healthy' : 'offline';
  const apiBadgeLabel = isColdStart ? 'API: Connecting' : isApiHealthy ? 'API: Connected / Healthy' : 'API: Disconnected';

  return (
    <header className="glass-panel p-4 md:p-6 mb-6 border-slate-800">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        {/* Left branding and context */}
        <div>
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
              <Activity className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl md:text-2xl font-bold tracking-tight text-white">
                  Intelligent Retail Analytics
                </h1>
                <span className="hidden sm:inline-block px-2 py-0.5 text-[10px] uppercase tracking-wider font-semibold rounded bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
                  SIH 179
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Edge AI Telemetry & Footfall Intelligence • YOLO11 + ByteTrack
              </p>
            </div>
          </div>
        </div>

        {/* Middle & Right: Metadata pills & Controls */}
        <div className="flex flex-wrap items-center gap-2.5 sm:gap-3 text-xs">
          {/* Store badge */}
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/80 text-slate-200">
            <Building2 className="h-3.5 w-3.5 text-cyan-400" />
            <span className="text-slate-400">Store:</span>
            <span className="font-semibold text-white">{storeId}</span>
          </div>

          {/* Device badge */}
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/80 text-slate-200">
            <Cpu className="h-3.5 w-3.5 text-indigo-400" />
            <span className="text-slate-400">Device:</span>
            <span className="font-semibold text-white">{deviceId}</span>
          </div>

          {/* Central API status badge */}
          <StatusBadge status={apiBadgeStatus} label={apiBadgeLabel} size="sm" />

          {/* Central PostgreSQL status badge */}
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/80 text-slate-200">
            <Database className="h-3.5 w-3.5 text-indigo-400" />
            <span className="text-slate-400">PostgreSQL:</span>
            <span className={`font-semibold ${isDbConnected ? 'text-emerald-400' : 'text-rose-400'}`}>
              {isDbConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>

          {/* Last updated indicator */}
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/50 border border-slate-700/50 text-slate-300">
            <Clock className="h-3.5 w-3.5 text-slate-400" />
            <span className="text-slate-400">Updated:</span>
            <span className="font-medium text-slate-200">
              {lastUpdated ? formatTime(lastUpdated) : '--:--:--'}
            </span>
          </div>

          {/* Auto-refresh interval selector & countdown */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-800/70 border border-slate-700/70">
            <span className="text-slate-400">Auto:</span>
            <select
              value={autoRefreshInterval}
              onChange={(e) => setAutoRefreshInterval(Number(e.target.value))}
              className="bg-transparent text-slate-200 font-medium focus:outline-none cursor-pointer"
            >
              <option value={5} className="bg-slate-900 text-white">5s</option>
              <option value={10} className="bg-slate-900 text-white">10s</option>
              <option value={30} className="bg-slate-900 text-white">30s</option>
              <option value={0} className="bg-slate-900 text-white">Off</option>
            </select>
            {autoRefreshInterval > 0 && (
              <span className="text-[10px] text-cyan-400 font-mono w-4 text-center">
                {countdown}s
              </span>
            )}
          </div>

          {/* Manual refresh button */}
          <button
            onClick={onRefresh}
            disabled={isRefreshing}
            title="Trigger manual API refresh"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600/20 hover:bg-emerald-600/30 border border-emerald-500/40 text-emerald-300 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
            <span className="font-medium">Refresh</span>
          </button>
        </div>
      </div>
    </header>
  );
}
