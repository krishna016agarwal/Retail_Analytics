import React from 'react';
import { Eye, Shield, Cpu, Sparkles } from 'lucide-react';

export default function HeatmapCard() {
  return (
    <div className="glass-panel p-5 flex flex-col justify-between relative overflow-hidden">
      {/* Background visual grid representing retail store floor layout */}
      <div className="absolute inset-0 opacity-[0.03] pointer-events-none bg-[radial-gradient(#38bdf8_1px,transparent_1px)] [background-size:16px_16px]" />

      <div>
        <div className="flex items-center justify-between pb-3 border-b border-slate-800/80">
          <div className="flex items-center gap-2">
            <Eye className="h-4 w-4 text-cyan-400" />
            <h3 className="text-sm font-semibold text-white">Edge-generated Shopper Heatmap</h3>
          </div>
          <span className="px-2.5 py-0.5 text-[10px] uppercase font-bold tracking-wider rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            Phase 4 Edge CV
          </span>
        </div>

        {/* Heatmap Wireframe Placeholder Graphic */}
        <div className="my-4 p-5 rounded-xl border border-dashed border-slate-800 bg-slate-950/40 relative flex flex-col items-center justify-center text-center min-h-[160px]">
          <div className="h-12 w-12 rounded-full bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 mb-3 shadow-lg shadow-cyan-500/5">
            <Cpu className="h-6 w-6" />
          </div>

          <h4 className="text-xs font-semibold text-slate-200">
            Edge-Processed Computer Vision Heatmap
          </h4>

          <p className="text-[11px] text-slate-400 max-w-sm mt-1.5 leading-relaxed">
            Privacy Note: Raw video is not synchronized. Only anonymous spatial density and aggregated telemetry are transmitted.
          </p>

          <div className="mt-3 flex items-center gap-2 text-[10px] text-slate-400 font-mono bg-slate-900/80 px-3 py-1 rounded-full border border-slate-800">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
            Edge module flag: <code className="text-cyan-300">--heatmap</code>
          </div>
        </div>
      </div>

      {/* Footer Info */}
      <div className="pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
        <span className="flex items-center gap-1 text-[11px]">
          <Shield className="h-3 w-3 text-emerald-400" />
          Raw Video Not Synchronized (Edge Only)
        </span>
        <span className="text-[11px] text-cyan-400 font-medium">
          Ready for Edge Stream Bridge
        </span>
      </div>
    </div>
  );
}
