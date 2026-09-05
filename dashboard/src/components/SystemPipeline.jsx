import React from 'react';
import { Camera, Cpu, HardDrive, RefreshCw, Cloud, ArrowRight, ShieldCheck } from 'lucide-react';

export default function SystemPipeline({
  isCloudConnected = true,
  isSyncing = false,
  totalSnapshots = 0,
}) {
  const steps = [
    {
      id: 'camera',
      title: 'Camera / Video',
      detail: 'RTSP Stream / MP4',
      icon: Camera,
      status: 'active',
      badge: '1080p / 30fps',
      color: 'emerald',
    },
    {
      id: 'edge_ai',
      title: 'Edge AI Tracking',
      detail: 'YOLO11n + ByteTrack',
      icon: Cpu,
      status: 'active',
      badge: 'CPU Optimized',
      color: 'emerald',
    },
    {
      id: 'sqlite',
      title: 'Local SQLite',
      detail: 'Edge Persistent Cache',
      icon: HardDrive,
      status: 'active',
      badge: 'Offline Buffer',
      color: 'cyan',
    },
    {
      id: 'sync',
      title: 'Batch Sync',
      detail: 'HTTPS Retry Worker',
      icon: RefreshCw,
      status: isSyncing ? 'syncing' : 'active',
      badge: isSyncing ? 'Syncing...' : 'Auto-Sync Active',
      color: isSyncing ? 'amber' : 'indigo',
    },
    {
      id: 'cloud',
      title: 'Central Render API',
      detail: 'PostgreSQL Cloud Store',
      icon: Cloud,
      status: isCloudConnected ? 'active' : 'offline',
      badge: isCloudConnected ? `${totalSnapshots} Ingested` : 'Disconnected',
      color: isCloudConnected ? 'emerald' : 'rose',
    },
  ];

  return (
    <div className="glass-panel p-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-800/80">
        <div>
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-4 w-4 text-emerald-400" />
            <h3 className="text-sm font-semibold text-white">Offline-First Edge AI Architecture</h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Offline-first buffering with reliable retry-based synchronization during network outages
          </p>
        </div>
        <span className="text-[11px] font-medium text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-1 rounded-full w-fit">
          Fault-Tolerant Pipeline
        </span>
      </div>

      {/* Horizontal Flow Pipeline */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-3 mt-4">
        {steps.map((step, idx) => {
          const Icon = step.icon;
          const isLast = idx === steps.length - 1;

          return (
            <div key={step.id} className="relative flex flex-col items-center">
              <div className="w-full bg-slate-950/60 border border-slate-800 p-3 rounded-xl flex flex-col items-center text-center hover:border-slate-700 transition-colors">
                <div className={`h-9 w-9 rounded-lg flex items-center justify-center mb-2 bg-${step.color}-500/10 border border-${step.color}-500/30 text-${step.color}-400`}>
                  <Icon className={`h-4 w-4 ${step.status === 'syncing' ? 'animate-spin' : ''}`} />
                </div>
                <h4 className="text-xs font-semibold text-white">{step.title}</h4>
                <span className="text-[11px] text-slate-400 mt-0.5">{step.detail}</span>
                <span className={`mt-2 text-[10px] px-2 py-0.5 rounded-full border bg-${step.color}-500/10 text-${step.color}-300 border-${step.color}-500/30`}>
                  {step.badge}
                </span>
              </div>

              {/* Connecting Arrow for desktop */}
              {!isLast && (
                <div className="hidden md:flex absolute -right-2.5 top-1/2 -translate-y-1/2 z-10 text-slate-600">
                  <ArrowRight className="h-4 w-4" />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
