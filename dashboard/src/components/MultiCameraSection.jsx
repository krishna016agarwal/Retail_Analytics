import React, { useState } from 'react';
import { 
  Camera, 
  Video, 
  Radio, 
  Cpu, 
  Layers, 
  CheckCircle2,
  Clock,
  Maximize2,
  LayoutGrid
} from 'lucide-react';

export default function MultiCameraSection({ cameras = [], clockInfo = {} }) {
  const [showGridWall, setShowGridWall] = useState(false);
  const apiBase = import.meta.env.VITE_EDGE_API_URL || 'http://127.0.0.1:8000';

  const defaultCameras = [
    {
      camera_id: 'CAM_01',
      name: 'Food',
      role: 'Department Analytics',
      source: 'videos/food/food.mp4',
      fps: 30.0,
      frame_count: 0,
      current_shoppers: 0,
      status: 'Processing',
      is_simulation: true,
    },
    {
      camera_id: 'CAM_02',
      name: 'Electronics',
      role: 'Department Analytics',
      source: 'videos/electronics/electronics.mp4',
      fps: 25.0,
      frame_count: 0,
      current_shoppers: 0,
      status: 'Processing',
      is_simulation: true,
    },
    {
      camera_id: 'CAM_03',
      name: 'Grocery',
      role: 'Department Analytics',
      source: 'videos/grocery/grocery.mp4',
      fps: 59.9,
      frame_count: 0,
      current_shoppers: 0,
      status: 'Processing',
      is_simulation: true,
    },
    {
      camera_id: 'CAM_04',
      name: 'Checkout',
      role: 'Queue Analytics',
      source: 'videos/checkout/checkout.mp4',
      fps: 13.1,
      frame_count: 0,
      current_shoppers: 0,
      status: 'Processing',
      is_simulation: true,
    },
  ];

  const items = cameras && cameras.length > 0 ? cameras : defaultCameras;
  const simTime = clockInfo?.simulated_store_time || '17:00:00';

  return (
    <section className="glass-panel p-5 mb-6 border-slate-800 rounded-xl">
      {/* Header & Simulation Mode Banner */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 pb-3 border-b border-slate-800/80">
        <div>
          <div className="flex items-center gap-2">
            <Camera className="h-4 w-4 text-emerald-400" />
            <h3 className="text-sm font-semibold text-white">
              Multi-Camera Ingestion (4 Concurrent Video Streams)
            </h3>
            <span className="px-2.5 py-0.5 text-[10px] font-bold uppercase rounded-full bg-amber-500/15 border border-amber-500/30 text-amber-300">
              Recorded Video / Live AI Inference
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Real-time YOLO11 + ByteTrack person detection, persistent IDs, and trajectory tracking.
          </p>
        </div>

        {/* Right controls: 2x2 Wall toggle & Simulation Clock */}
        <div className="flex flex-wrap items-center gap-2.5">
          <button
            onClick={() => setShowGridWall(!showGridWall)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all border ${
              showGridWall
                ? 'bg-emerald-500/20 border-emerald-500/40 text-emerald-300 shadow-sm shadow-emerald-500/20'
                : 'bg-slate-900 border-slate-700 text-slate-300 hover:text-white hover:border-slate-600'
            }`}
          >
            <LayoutGrid className="h-3.5 w-3.5 text-emerald-400" />
            <span>{showGridWall ? 'Hide 2x2 Video Wall' : 'View 2x2 Video Wall'}</span>
          </button>

          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/80 border border-slate-800 text-xs text-slate-200 font-mono">
            <Clock className="h-3.5 w-3.5 text-cyan-400" />
            <span>Simulated Store Time:</span>
            <strong className="text-cyan-300 font-bold">{simTime}</strong>
          </div>
        </div>
      </div>

      {/* Optional 2x2 Composite Video Wall */}
      {showGridWall && (
        <div className="mt-4 p-4 rounded-xl bg-slate-950 border border-slate-800">
          <div className="flex items-center justify-between pb-2 mb-3 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              <h4 className="text-xs font-bold text-white font-mono uppercase tracking-wider">
                Composite 2x2 Multi-Stream Wall (All 4 Feeds Synchronized)
              </h4>
            </div>
            <span className="text-[10px] font-mono text-slate-400">
              Live MJPEG Stream • 960x540
            </span>
          </div>
          <div className="relative aspect-video rounded-lg overflow-hidden bg-black border border-slate-800 shadow-2xl">
            <img
              src={`${apiBase}/api/v1/cameras/grid/feed`}
              alt="2x2 Camera Video Wall"
              className="w-full h-full object-contain"
            />
          </div>
        </div>
      )}

      {/* 4 Cameras Individual Stream Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-4">
        {items.map((cam) => {
          const shoppers = cam.current_shoppers ?? cam.current_detections ?? 0;
          const fps = cam.fps || 30.0;
          const frameCount = cam.frame_count || 0;
          const camId = cam.camera_id || cam.id;

          return (
            <div
              key={camId}
              className="bg-slate-950/70 border border-slate-800 hover:border-slate-700 p-3.5 rounded-xl transition-all flex flex-col justify-between"
            >
              <div>
                {/* Top Row: Camera ID & Live Status */}
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-1.5">
                    <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
                    <span className="text-xs font-bold font-mono text-white">
                      {camId}
                    </span>
                  </div>
                  <span className="text-[10px] font-bold font-mono px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 flex items-center gap-1">
                    ● {cam.status || 'Processing'}
                  </span>
                </div>

                {/* Camera Name & Role */}
                <h4 className="text-sm font-bold text-white">
                  {cam.name || cam.department}
                </h4>

                <div className="mt-1 flex flex-wrap gap-1">
                  <span className="text-[9px] font-medium px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                    {cam.role}
                  </span>
                  <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20">
                    Live Stream
                  </span>
                </div>

                {/* Live Camera Video Feed */}
                <div className="my-2.5 relative aspect-video rounded-lg overflow-hidden bg-slate-900 border border-slate-800 group shadow-inner">
                  <img
                    src={`${apiBase}/api/v1/cameras/${camId}/feed`}
                    alt={cam.name}
                    className="w-full h-full object-cover"
                    loading="lazy"
                    onError={(e) => {
                      e.currentTarget.style.display = 'none';
                      const fallback = e.currentTarget.parentElement?.querySelector('.feed-fallback');
                      if (fallback) fallback.style.display = 'flex';
                    }}
                  />
                  <div className="feed-fallback hidden absolute inset-0 flex flex-col items-center justify-center bg-slate-950/90 text-slate-500 text-xs font-mono p-2 text-center">
                    <Video className="h-5 w-5 text-slate-600 mb-1" />
                    <span>Connecting feed...</span>
                    <span className="text-[9px] text-slate-600 truncate max-w-full">{cam.source}</span>
                  </div>
                  <div className="absolute top-1.5 left-1.5 flex items-center gap-1 px-1.5 py-0.5 rounded bg-black/75 backdrop-blur-sm border border-emerald-500/40 text-[9px] font-mono text-emerald-400 font-bold">
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    LIVE CV
                  </div>
                  <div className="absolute bottom-1.5 right-1.5 px-1.5 py-0.5 rounded bg-black/75 text-[8px] font-mono text-slate-300 border border-slate-700">
                    YOLO11n
                  </div>
                </div>

                {/* Detections Counter */}
                <div className="flex items-baseline justify-between py-1 border-t border-slate-800/60">
                  <span className="text-[11px] text-slate-400">Active Shoppers:</span>
                  <span className="text-base font-mono font-extrabold text-white">
                    {shoppers}
                  </span>
                </div>
              </div>

              {/* Footer specs */}
              <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-[10px] font-mono text-slate-400">
                <span>{fps} FPS {frameCount > 0 ? `(#${frameCount})` : ''}</span>
                <span className="text-emerald-400 flex items-center gap-1">
                  <CheckCircle2 className="h-3 w-3" />
                  Real CV
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
