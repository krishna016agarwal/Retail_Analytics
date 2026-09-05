import React from 'react';
import { 
  Camera, 
  Video, 
  Radio, 
  Cpu, 
  Layers, 
  AlertCircle,
  ShieldCheck,
  CheckCircle2
} from 'lucide-react';

export default function MultiCameraSection() {
  const cameras = [
    {
      id: 'CAM_01',
      name: 'Food Section',
      role: 'Zone Analytics',
      roleColor: 'cyan',
      zoneId: 'zone_01',
      stream: 'videos/test.mp4',
      fps: 30,
      res: '1080p',
      status: 'Active',
      isSimulation: true,
    },
    {
      id: 'CAM_02',
      name: 'Electronics Section',
      role: 'Zone Analytics',
      roleColor: 'cyan',
      zoneId: 'zone_02',
      stream: 'videos/test.mp4',
      fps: 30,
      res: '1080p',
      status: 'Active',
      isSimulation: true,
    },
    {
      id: 'CAM_03',
      name: 'Clothing Section',
      role: 'Zone Analytics',
      roleColor: 'cyan',
      zoneId: 'zone_03',
      stream: 'videos/test.mp4',
      fps: 30,
      res: '1080p',
      status: 'Active',
      isSimulation: true,
    },
    {
      id: 'CAM_04',
      name: 'Store Entrance',
      role: 'Footfall Counting',
      roleColor: 'emerald',
      zoneId: 'entrance_gate',
      stream: 'videos/test.mp4',
      fps: 30,
      res: '1080p',
      status: 'Active',
      isSimulation: true,
    },
    {
      id: 'CAM_05',
      name: 'Checkout Queue',
      role: 'Queue Analytics',
      roleColor: 'amber',
      zoneId: 'checkout_zone',
      stream: 'videos/test.mp4',
      fps: 30,
      res: '1080p',
      status: 'Active',
      isSimulation: true,
    },
  ];

  return (
    <section className="glass-panel p-5 mb-6 border-slate-800">
      {/* Header & Simulation Mode Banner */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 pb-3 border-b border-slate-800/80">
        <div>
          <div className="flex items-center gap-2">
            <Camera className="h-4 w-4 text-emerald-400" />
            <h3 className="text-sm font-semibold text-white">
              Multi-Camera Edge Ingestion (5 Streams)
            </h3>
            <span className="px-2 py-0.5 text-[10px] font-bold uppercase rounded-full bg-amber-500/15 border border-amber-500/30 text-amber-300">
              Simulation Mode
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Concurrent edge ingestion across 5 virtual camera workers aggregating to logical edge device <code className="text-slate-300 font-mono">edge_device_01</code>
          </p>
        </div>

        {/* Simulation note badge */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-amber-950/30 border border-amber-500/30 text-xs text-amber-200 self-start lg:self-auto">
          <Radio className="h-3.5 w-3.5 text-amber-400 animate-pulse" />
          <span className="text-[11px]">
            Simulation Mode — shared test video used to validate multi-camera concurrency and edge aggregation.
          </span>
        </div>
      </div>

      {/* 5 Cameras Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3.5 mt-4">
        {cameras.map((cam) => {
          return (
            <div
              key={cam.id}
              className="bg-slate-950/60 border border-slate-800 hover:border-slate-700 p-3.5 rounded-xl transition-all flex flex-col justify-between"
            >
              <div>
                {/* Top Row: Camera ID & Live Status */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <span className="h-2 w-2 rounded-full bg-emerald-400 animate-ping" />
                    <span className="text-xs font-bold font-mono text-white">
                      {cam.id}
                    </span>
                  </div>
                  <span className="text-[9px] font-bold font-mono px-1.5 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
                    LIVE
                  </span>
                </div>

                {/* Camera Name & Role */}
                <h4 className="text-sm font-semibold text-slate-200 mt-2">
                  {cam.name}
                </h4>

                <div className="mt-2 flex flex-wrap gap-1.5">
                  <span className="text-[10px] font-medium px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                    {cam.role}
                  </span>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20">
                    Sim
                  </span>
                </div>

                {/* Simulated Stream Graphic Placeholder */}
                <div className="my-3 p-2.5 rounded-lg border border-dashed border-slate-800 bg-slate-900/50 flex items-center justify-center gap-2 text-slate-400">
                  <Video className="h-3.5 w-3.5 text-cyan-400" />
                  <span className="text-[10px] font-mono truncate max-w-[120px]">
                    {cam.stream}
                  </span>
                </div>
              </div>

              {/* Footer specs */}
              <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-[10px] font-mono text-slate-400">
                <span>{cam.res} @ {cam.fps}fps</span>
                <span className="text-emerald-400 flex items-center gap-1">
                  <CheckCircle2 className="h-3 w-3" />
                  Synced
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
