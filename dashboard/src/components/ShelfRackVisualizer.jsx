import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Camera,
  RefreshCw,
  Clock,
  Package,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  TrendingDown,
  Play,
  Pause,
  Cpu,
  Layers,
  Flame,
  ArrowRight,
  Maximize2,
} from 'lucide-react';
import { getShelfStages } from '../api/client';
import InventoryLiveVideos from './InventoryLiveVideos';

const DEFAULT_STAGES = [
  {
    stage_index: 1,
    title: 'Stage 1 — Full Stock',
    source_image: 'videos/1.jpeg',
    image_url: '/evidence/shelf_stage_1.jpg',
    status: 'IN_STOCK',
    status_label: 'IN STOCK / FULLY STOCKED',
    observed_count: 663,
    capacity: 663,
    empty_spaces_count: 0,
    occupancy_pct: 100.0,
    alert_severity: 'NONE',
    alert_message: null,
    color: '#10b981',
  },
  {
    stage_index: 2,
    title: 'Stage 2 — Low Stock',
    source_image: 'videos/2.jpeg',
    image_url: '/evidence/shelf_stage_2.jpg',
    status: 'LOW_STOCK',
    status_label: 'LOW STOCK DETECTED',
    observed_count: 634,
    capacity: 663,
    empty_spaces_count: 29,
    occupancy_pct: 95.6,
    alert_severity: 'MEDIUM',
    alert_message: 'LOW STOCK ALERT: 29 empty shelf spaces detected on Aisle Shelf. Replenishment recommended.',
    color: '#f59e0b',
  },
  {
    stage_index: 3,
    title: 'Stage 3 — Urgent Need of Stock',
    source_image: 'videos/3.jpeg',
    image_url: '/evidence/shelf_stage_3.jpg',
    status: 'URGENT_RESTOCK',
    status_label: 'URGENT NEED OF STOCK HERE',
    observed_count: 332,
    capacity: 663,
    empty_spaces_count: 331,
    occupancy_pct: 50.1,
    alert_severity: 'HIGH',
    alert_message: 'URGENT NEED OF STOCK HERE: Critical out-of-stock condition on Main Shelf! 331 empty slots detected.',
    color: '#ef4444',
  },
];

export default function ShelfRackVisualizer() {
  const [stagesData, setStagesData] = useState(DEFAULT_STAGES);
  const [currentStageIdx, setCurrentStageIdx] = useState(1); // 1, 2, or 3
  const [isPlaying, setIsPlaying] = useState(true);
  const [intervalSec, setIntervalSec] = useState(5);
  const [countdown, setCountdown] = useState(5);
  const [imgTimestamp, setImgTimestamp] = useState(Date.now());

  // Fetch precomputed stages from API / json
  useEffect(() => {
    const loadStages = async () => {
      const data = await getShelfStages();
      if (data && data.stages && data.stages.length > 0) {
        setStagesData(data.stages);
      }
    };
    loadStages();
  }, []);

  // 5-Second Interval Automatic Progression: Stage 1 -> Stage 2 -> Stage 3 -> Stage 1
  useEffect(() => {
    if (!isPlaying) return;

    const timer = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          // Advance to next stage
          setCurrentStageIdx((curr) => {
            const next = curr >= 3 ? 1 : curr + 1;
            return next;
          });
          setImgTimestamp(Date.now());
          return intervalSec;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [isPlaying, intervalSec]);

  // Current stage details
  const activeStage = stagesData.find((s) => s.stage_index === currentStageIdx) || stagesData[0];

  const handleSelectStage = (idx) => {
    setCurrentStageIdx(idx);
    setCountdown(intervalSec);
    setImgTimestamp(Date.now());
  };

  const togglePlayPause = () => {
    setIsPlaying(!isPlaying);
    setCountdown(intervalSec);
  };

  return (
    <div className="space-y-6">
      {/* ─── Top Control & Stage Telemetry Banner ──────────────────────────── */}
      <div className="bg-slate-900/90 border border-slate-800/80 backdrop-blur-md rounded-2xl p-5 shadow-xl">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className={`flex h-2.5 w-2.5 rounded-full ${isPlaying ? 'bg-cyan-400 animate-ping' : 'bg-slate-500'}`} />
              <span className="text-xs font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-1.5 font-mono">
                <Camera className="h-3.5 w-3.5" />
                Shelf Stock Depletion Lifecycle Monitoring (SIH PS 179)
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 font-mono">
                Edge Computer Vision AI
              </span>
            </div>
            <h2 className="text-xl font-extrabold text-white flex items-center gap-2">
              Shelf Stock & Empty Space Detection
            </h2>
            <p className="text-xs text-slate-400 max-w-2xl">
              Monitors shelf product presence across consecutive stages. Detects all products in full stock, 
              flags missing products with <strong className="text-rose-400">Red Bounding Boxes</strong> on empty spaces, 
              and triggers automated <strong>Low Stock</strong> and <strong>Urgent Stock</strong> alerts.
            </p>
          </div>

          {/* 5s Interval & Auto-Play Controls */}
          <div className="flex flex-wrap items-center gap-3">
            {/* 5-Sec Countdown Timer Card */}
            <div className="flex items-center gap-3 px-3.5 py-2 rounded-xl bg-slate-950/80 border border-slate-800">
              <Clock className={`h-4 w-4 ${isPlaying ? 'text-cyan-400 animate-pulse' : 'text-slate-500'}`} />
              <div>
                <div className="text-[10px] uppercase font-bold text-slate-400">
                  {isPlaying ? 'Next Stage Cycle' : 'Cycle Paused'}
                </div>
                <div className="text-sm font-mono font-bold text-white">
                  {isPlaying ? `00:0${countdown}s` : 'PAUSED'}{' '}
                  <span className="text-[10px] text-slate-500 font-sans">({intervalSec}s interval)</span>
                </div>
              </div>
            </div>

            {/* Play / Pause Toggle */}
            <button
              onClick={togglePlayPause}
              className={`inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-bold border transition-all cursor-pointer ${
                isPlaying
                  ? 'bg-amber-500/15 border-amber-500/30 text-amber-300 hover:bg-amber-500/25'
                  : 'bg-emerald-500/15 border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/25'
              }`}
            >
              {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
              {isPlaying ? 'Pause 5s Cycle' : 'Resume 5s Cycle'}
            </button>
          </div>
        </div>

        {/* 5-Second Cycle Progress Bar */}
        <div className="mt-4 pt-3 border-t border-slate-800/80">
          <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono mb-1.5">
            <span className="flex items-center gap-1.5 text-cyan-300 font-bold">
              <Layers className="h-3.5 w-3.5" />
              Automated 5-Second Interval Cycle:
            </span>
            <span className="text-slate-300">
              {activeStage.title} ({activeStage.source_image})
            </span>
          </div>
          <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-cyan-500 via-blue-500 to-emerald-400 transition-all duration-1000 ease-linear"
              style={{ width: `${((intervalSec - countdown + 1) / intervalSec) * 100}%` }}
            />
          </div>
        </div>

        {/* Interactive Stage Selector Tabs */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-4">
          {stagesData.map((stg) => {
            const isSelected = stg.stage_index === currentStageIdx;
            const isStage2 = stg.stage_index === 2;
            const isStage3 = stg.stage_index === 3;

            let borderClass = 'border-slate-800 hover:border-slate-700 bg-slate-950/60';
            if (isSelected) {
              if (isStage3) borderClass = 'border-rose-500 bg-rose-950/30 shadow-lg shadow-rose-950/30';
              else if (isStage2) borderClass = 'border-amber-500 bg-amber-950/30 shadow-lg shadow-amber-950/30';
              else borderClass = 'border-emerald-500 bg-emerald-950/30 shadow-lg shadow-emerald-950/30';
            }

            return (
              <button
                key={stg.stage_index}
                onClick={() => handleSelectStage(stg.stage_index)}
                className={`text-left p-3 rounded-xl border transition-all cursor-pointer flex flex-col justify-between ${borderClass}`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[10px] font-mono uppercase font-bold text-slate-400">
                    STAGE 0{stg.stage_index}
                  </span>
                  <span
                    className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                      isStage3
                        ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                        : isStage2
                        ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                        : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                    }`}
                  >
                    {stg.status_label}
                  </span>
                </div>
                <div className="mt-1 font-bold text-sm text-white">
                  {stg.title}
                </div>
                <div className="mt-2 pt-2 border-t border-slate-800/60 flex items-center justify-between text-[11px] font-mono text-slate-400">
                  <span>{stg.observed_count}/{stg.capacity} units</span>
                  {stg.empty_spaces_count > 0 ? (
                    <span className="text-rose-400 font-bold flex items-center gap-1">
                      ● {stg.empty_spaces_count} empty red boxes
                    </span>
                  ) : (
                    <span className="text-emerald-400 font-semibold">100% Full</span>
                  )}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* ─── Dynamic Alert Banner (Stage 2 & Stage 3) ──────────────────────── */}
      {activeStage.alert_message && (
        <div
          className={`p-4 rounded-2xl border transition-all ${
            activeStage.alert_severity === 'HIGH'
              ? 'bg-rose-500/15 border-rose-500/40 text-rose-200 animate-pulse shadow-xl shadow-rose-950/40'
              : 'bg-amber-500/15 border-amber-500/40 text-amber-200 shadow-xl shadow-amber-950/30'
          }`}
        >
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="flex items-start sm:items-center gap-3">
              {activeStage.alert_severity === 'HIGH' ? (
                <div className="p-2 rounded-xl bg-rose-500/20 border border-rose-500/40 shrink-0">
                  <Flame className="h-6 w-6 text-rose-400 animate-bounce" />
                </div>
              ) : (
                <div className="p-2 rounded-xl bg-amber-500/20 border border-amber-500/40 shrink-0">
                  <AlertTriangle className="h-6 w-6 text-amber-400" />
                </div>
              )}
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono font-bold uppercase px-2 py-0.5 rounded bg-black/40 border border-current">
                    {activeStage.alert_severity === 'HIGH' ? 'CRITICAL DISPATCH' : 'INVENTORY ATTENTION'}
                  </span>
                  <span className="text-xs font-mono opacity-80">
                    Source: {activeStage.source_image}
                  </span>
                </div>
                <h4 className="text-base font-extrabold text-white mt-1">
                  {activeStage.alert_message}
                </h4>
                <p className="text-xs opacity-90 mt-0.5">
                  {activeStage.alert_severity === 'HIGH'
                    ? `Severe product depletion detected! ${activeStage.empty_spaces_count} empty positions flagged with red boxes. Restock immediately.`
                    : `Customers purchased multiple items. ${activeStage.empty_spaces_count} shelf slots are now empty and marked with red boxes.`}
                </p>
              </div>
            </div>

            <div className="shrink-0 flex items-center gap-2">
              <span className="text-xs font-mono px-3 py-1.5 rounded-xl bg-black/50 border border-current font-bold">
                {activeStage.empty_spaces_count} Empty Slots
              </span>
            </div>
          </div>
        </div>
      )}

      {/* ─── Shelf Image Visual Evidence (All Products Marked + Red Boxes) ─── */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Camera className="h-4 w-4 text-cyan-400" />
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              {activeStage.title} — Marked Product & Empty Space Detection Evidence
            </h3>
          </div>
          <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
            <span className="px-2 py-0.5 rounded bg-slate-800 text-cyan-300">
              Source: {activeStage.source_image}
            </span>
            <span className="px-2 py-0.5 rounded bg-slate-800 text-emerald-300">
              {activeStage.occupancy_pct}% Capacity
            </span>
          </div>
        </div>

        {/* Large Main Image Display */}
        <div className="relative aspect-[1024/559] w-full rounded-xl overflow-hidden bg-black border border-slate-800 shadow-2xl flex items-center justify-center">
          <img
            key={`${currentStageIdx}-${imgTimestamp}`}
            src={`${activeStage.image_url}?_t=${imgTimestamp}`}
            alt={activeStage.title}
            className="w-full h-full object-contain"
          />

          {/* Top Left Watermark */}
          <div className="absolute top-2.5 left-2.5 px-3 py-1 rounded-lg bg-slate-950/85 backdrop-blur-md border border-slate-700/60 text-xs font-mono text-cyan-300 flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-ping" />
            <span>{activeStage.title} ({activeStage.source_image})</span>
          </div>

          {/* Bottom Right AI Latency Pill */}
          <div className="absolute bottom-2.5 right-2.5 px-3 py-1 rounded-lg bg-slate-950/85 backdrop-blur-md border border-slate-700/60 text-xs font-mono text-slate-300 flex items-center gap-1.5">
            <Cpu className="h-3.5 w-3.5 text-cyan-400" />
            <span>YOLO11n Edge AI • 35ms</span>
          </div>
        </div>

        {/* Color Legend & Stock Level Metrics */}
        <div className="pt-2 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-1.5 text-slate-300 font-medium">
              <span className="h-3 w-3 rounded bg-emerald-500 border border-emerald-400" />
              <span>Green Box: In-Stock Products ({activeStage.observed_count})</span>
            </div>
            <div className="flex items-center gap-1.5 text-slate-300 font-medium">
              <span className="h-3 w-3 rounded bg-rose-500 border border-rose-400" />
              <span className="text-rose-400 font-bold">
                Red Box: Empty Spaces / Missing Stock ({activeStage.empty_spaces_count})
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3 font-mono text-slate-400 text-[11px]">
            <span>Total Shelf Capacity: <strong className="text-white">{activeStage.capacity}</strong></span>
            <span>•</span>
            <span>Stock Occupancy: <strong className="text-cyan-300">{activeStage.occupancy_pct}%</strong></span>
          </div>
        </div>
      </div>

      {/* ─── Live Video Streams for Judges Demo (inventory.mp4 & inventory2.mp4) ─ */}
      <InventoryLiveVideos />
    </div>
  );
}
