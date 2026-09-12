import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Camera,
  RefreshCw,
  Clock,
  Package,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  TrendingDown,
  Sparkles,
  Play,
  Pause,
  Cpu,
  ShieldCheck,
  Search,
  Filter,
  MapPin,
  Flame,
} from 'lucide-react';
import {
  getShelfSnapshotLatest,
  triggerShelfScanNow,
  setShelfScanInterval,
  startShelfSnapshot,
  stopShelfSnapshot,
} from '../api/client';

export default function ShelfRackVisualizer() {
  const [telemetry, setTelemetry] = useState(null);
  const [isScanning, setIsScanning] = useState(false);
  const [countdown, setCountdown] = useState(10);
  const [intervalSec, setIntervalSec] = useState(10);
  const [showCameraView, setShowCameraView] = useState(true);
  const [statusFilter, setStatusFilter] = useState('ALL'); // 'ALL' | 'SOLD_OUT' | 'LOW_STOCK' | 'IN_STOCK'
  const [searchQuery, setSearchQuery] = useState('');
  const [completedTasks, setCompletedTasks] = useState({});

  const timerRef = useRef(null);

  // ─── Fetch latest telemetry ────────────────────────────────────────────────
  const fetchTelemetry = useCallback(async () => {
    const data = await getShelfSnapshotLatest();
    if (data) {
      setTelemetry(data);
      if (data.interval_seconds) {
        setIntervalSec(data.interval_seconds);
      }
      if (typeof data.seconds_until_next_scan === 'number') {
        setCountdown(Math.max(0, Math.round(data.seconds_until_next_scan)));
      }
    }
  }, []);

  useEffect(() => {
    fetchTelemetry();
    const interval = setInterval(fetchTelemetry, 2500);
    return () => clearInterval(interval);
  }, [fetchTelemetry]);

  // ─── Local 1-second countdown tick ─────────────────────────────────────────
  const isRunning = telemetry?.is_running ?? false;

  useEffect(() => {
    if (!isRunning) return;
    timerRef.current = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          fetchTelemetry();
          return intervalSec;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(timerRef.current);
  }, [intervalSec, fetchTelemetry, isRunning]);

  // ─── Manual "Scan Now" trigger ─────────────────────────────────────────────
  const handleScanNow = async () => {
    setIsScanning(true);
    await triggerShelfScanNow();
    await fetchTelemetry();
    setCountdown(intervalSec);
    setIsScanning(false);
  };

  // ─── Toggle Background Scanning ────────────────────────────────────────────
  const handleToggleRunning = async () => {
    if (isRunning) {
      await stopShelfSnapshot();
    } else {
      await startShelfSnapshot();
    }
    await fetchTelemetry();
  };

  // ─── Change Scan Interval ──────────────────────────────────────────────────
  const handleIntervalChange = async (sec) => {
    setIntervalSec(sec);
    setCountdown(sec);
    await setShelfScanInterval(sec);
    await fetchTelemetry();
  };

  const scan = telemetry?.latest_scan;
  const rawProducts = scan?.products || [];
  const alerts = scan?.alerts || [];

  // Filter products by status and search
  const filteredProducts = rawProducts.filter((p) => {
    if (statusFilter === 'SOLD_OUT' && !p.is_finished && p.status !== 'SOLD_OUT') return false;
    if (statusFilter === 'LOW_STOCK' && p.status !== 'LOW_STOCK') return false;
    if (statusFilter === 'IN_STOCK' && p.status !== 'IN_STOCK') return false;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      return (
        p.product_name.toLowerCase().includes(q) ||
        p.location.toLowerCase().includes(q) ||
        p.category.toLowerCase().includes(q)
      );
    }
    return true;
  });

  const soldOutCount = rawProducts.filter((p) => p.is_finished || p.status === 'SOLD_OUT').length;
  const lowStockCount = rawProducts.filter((p) => p.status === 'LOW_STOCK').length;
  const inStockCount = rawProducts.filter((p) => p.status === 'IN_STOCK').length;

  const getStatusBadge = (product) => {
    if (product.is_finished || product.status === 'SOLD_OUT') {
      return {
        label: 'SOLD OUT / FINISHED',
        bg: 'bg-rose-500/15',
        border: 'border-rose-500/40',
        text: 'text-rose-400',
        dot: 'bg-rose-500 animate-ping',
        progress: 'from-rose-600 to-red-500',
      };
    }
    if (product.status === 'LOW_STOCK') {
      return {
        label: 'LOW STOCK',
        bg: 'bg-amber-500/15',
        border: 'border-amber-500/40',
        text: 'text-amber-400',
        dot: 'bg-amber-400',
        progress: 'from-amber-500 to-yellow-400',
      };
    }
    return {
      label: 'IN STOCK',
      bg: 'bg-emerald-500/15',
      border: 'border-emerald-500/30',
      text: 'text-emerald-400',
      dot: 'bg-emerald-400',
      progress: 'from-emerald-500 to-teal-400',
    };
  };

  return (
    <div className="space-y-6">
      {/* ─── Top Control & Telemetry Banner ─────────────────────────────────── */}
      <div className="bg-slate-900/80 border border-slate-800/80 backdrop-blur-md rounded-2xl p-5 shadow-xl">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className={`flex h-2.5 w-2.5 rounded-full ${isRunning ? 'bg-cyan-400 animate-ping' : 'bg-slate-500'}`} />
              <span className="text-xs font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-1.5">
                <Camera className="h-3.5 w-3.5" />
                Edge Camera Video Monitor: {scan?.source || 'videos/inventory2.mp4'}
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 font-mono">
                {scan?.aisle_name || 'Main Aisle — Beverages, Personal Care & Packaged Goods'}
              </span>
            </div>
            <h2 className="text-xl font-extrabold text-white flex items-center gap-2">
              Product Stock & Sold-Out Detection (SIH PS 179)
            </h2>
            <p className="text-xs text-slate-400 max-w-2xl">
              Monitors individual products on the shelf. Automatically marks products in stock and 
              identifies when any product is <strong>SOLD OUT / FINISHED</strong> from the shelf.
            </p>
          </div>

          {/* Controls */}
          <div className="flex flex-wrap items-center gap-3">
            {/* Auto-Scan Status & Countdown */}
            <div className="flex items-center gap-3 px-3.5 py-2 rounded-xl bg-slate-950/70 border border-slate-800">
              <Clock className={`h-4 w-4 ${isRunning ? 'text-cyan-400 animate-pulse' : 'text-slate-500'}`} />
              <div>
                <div className="text-[10px] uppercase font-bold text-slate-400">
                  {isRunning ? 'Next Auto-Scan' : 'Scan Paused'}
                </div>
                <div className="text-sm font-mono font-bold text-white">
                  {isRunning ? `00:${countdown.toString().padStart(2, '0')}` : 'PAUSED'}{' '}
                  <span className="text-[10px] text-slate-500 font-sans">({intervalSec}s)</span>
                </div>
              </div>
            </div>

            {/* Interval Preset Selector */}
            <div className="flex items-center gap-1 bg-slate-950/70 p-1 rounded-xl border border-slate-800 text-xs">
              <button
                onClick={() => handleIntervalChange(10)}
                className={`px-2.5 py-1 rounded-lg font-medium transition-all ${
                  intervalSec === 10
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold'
                    : 'text-slate-400 hover:text-white'
                }`}
                title="10-second fast demo"
              >
                10s Demo
              </button>
              <button
                onClick={() => handleIntervalChange(30)}
                className={`px-2.5 py-1 rounded-lg font-medium transition-all ${
                  intervalSec === 30
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                30s
              </button>
            </div>

            {/* Pause / Resume Button */}
            <button
              onClick={handleToggleRunning}
              className={`inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-bold border transition-all cursor-pointer ${
                isRunning
                  ? 'bg-amber-500/15 border-amber-500/30 text-amber-300 hover:bg-amber-500/25'
                  : 'bg-emerald-500/15 border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/25'
              }`}
            >
              {isRunning ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
              {isRunning ? 'Pause Auto' : 'Start Auto'}
            </button>

            {/* Manual Scan Button */}
            <button
              onClick={handleScanNow}
              disabled={isScanning}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white text-xs font-bold shadow-lg shadow-cyan-500/20 transition-all disabled:opacity-50 cursor-pointer"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isScanning ? 'animate-spin' : ''}`} />
              {isScanning ? 'Analyzing...' : 'Scan Now'}
            </button>
          </div>
        </div>

        {/* Mini KPI Bar */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 pt-4 border-t border-slate-800/80">
          <div className="bg-slate-950/40 rounded-xl p-2.5 border border-slate-800/50">
            <span className="text-[10px] text-slate-400 uppercase font-semibold">Total Products Monitored</span>
            <div className="text-base font-bold text-white">
              {rawProducts.length}{' '}
              <span className="text-xs text-slate-500">catalog items</span>
            </div>
          </div>
          <div className="bg-slate-950/40 rounded-xl p-2.5 border border-slate-800/50">
            <span className="text-[10px] text-slate-400 uppercase font-semibold">Sold Out / Finished</span>
            <div className={`text-base font-bold flex items-center gap-1.5 ${soldOutCount > 0 ? 'text-rose-400' : 'text-emerald-400'}`}>
              {soldOutCount > 0 ? <XCircle className="h-4 w-4 text-rose-500 animate-pulse" /> : <CheckCircle2 className="h-4 w-4" />}
              {soldOutCount > 0 ? `${soldOutCount} Finished` : 'All Available'}
            </div>
          </div>
          <div className="bg-slate-950/40 rounded-xl p-2.5 border border-slate-800/50">
            <span className="text-[10px] text-slate-400 uppercase font-semibold">Low Stock Attention</span>
            <div className={`text-base font-bold ${lowStockCount > 0 ? 'text-amber-400' : 'text-slate-400'}`}>
              {lowStockCount} items
            </div>
          </div>
          <div className="bg-slate-950/40 rounded-xl p-2.5 border border-slate-800/50">
            <span className="text-[10px] text-slate-400 uppercase font-semibold">Edge AI Latency</span>
            <div className="text-base font-mono font-bold text-cyan-300 flex items-center gap-1">
              <Cpu className="h-3.5 w-3.5" />
              {scan?.inference_time_ms ? `${scan.inference_time_ms} ms` : '80 ms'}
            </div>
          </div>
        </div>
      </div>

      {/* ─── Sold Out Urgent Banner ─────────────────────────────────────────── */}
      {soldOutCount > 0 && (
        <div className="flex items-center justify-between gap-3 p-4 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-300 animate-pulse">
          <div className="flex items-center gap-3">
            <AlertTriangle className="h-5 w-5 text-rose-400 shrink-0" />
            <div>
              <h4 className="text-sm font-bold text-white">
                {soldOutCount} Product Completely Finished from Shelf!
              </h4>
              <p className="text-xs text-rose-300/90">
                The AI detected empty shelf space where products have sold out. Immediate replenishment is recommended.
              </p>
            </div>
          </div>
          <button
            onClick={() => setStatusFilter('SOLD_OUT')}
            className="px-3 py-1.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold shrink-0 cursor-pointer"
          >
            View Sold Out Item
          </button>
        </div>
      )}

      {/* ─── Product Stock Availability Matrix & Camera Snapshot ─────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Product Availability List (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            {/* Filter Tabs */}
            <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-xl border border-slate-800 text-xs">
              <button
                onClick={() => setStatusFilter('ALL')}
                className={`px-3 py-1 rounded-lg font-medium transition-all ${
                  statusFilter === 'ALL'
                    ? 'bg-slate-800 text-white font-bold'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                All ({rawProducts.length})
              </button>
              <button
                onClick={() => setStatusFilter('SOLD_OUT')}
                className={`px-3 py-1 rounded-lg font-medium transition-all flex items-center gap-1.5 ${
                  statusFilter === 'SOLD_OUT'
                    ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 font-bold'
                    : 'text-slate-400 hover:text-rose-400'
                }`}
              >
                <span className="h-2 w-2 rounded-full bg-rose-500" />
                Sold Out ({soldOutCount})
              </button>
              <button
                onClick={() => setStatusFilter('LOW_STOCK')}
                className={`px-3 py-1 rounded-lg font-medium transition-all flex items-center gap-1.5 ${
                  statusFilter === 'LOW_STOCK'
                    ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 font-bold'
                    : 'text-slate-400 hover:text-amber-400'
                }`}
              >
                <span className="h-2 w-2 rounded-full bg-amber-400" />
                Low ({lowStockCount})
              </button>
              <button
                onClick={() => setStatusFilter('IN_STOCK')}
                className={`px-3 py-1 rounded-lg font-medium transition-all flex items-center gap-1.5 ${
                  statusFilter === 'IN_STOCK'
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-bold'
                    : 'text-slate-400 hover:text-emerald-400'
                }`}
              >
                <span className="h-2 w-2 rounded-full bg-emerald-400" />
                In Stock ({inStockCount})
              </button>
            </div>

            {/* Search Box */}
            <div className="relative">
              <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-slate-500" />
              <input
                type="text"
                placeholder="Search products or shelf..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full sm:w-48 pl-8 pr-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
              />
            </div>
          </div>

          {/* Product Cards */}
          <div className="space-y-3">
            {filteredProducts.length === 0 ? (
              <div className="text-center py-8 text-xs text-slate-400 bg-slate-900/50 rounded-2xl border border-slate-800 p-6">
                No products found matching the filter.
              </div>
            ) : (
              filteredProducts.map((p) => {
                const badge = getStatusBadge(p);
                const isTaskDone = completedTasks[p.slot_id];

                return (
                  <div
                    key={p.slot_id}
                    className={`relative overflow-hidden rounded-2xl border ${badge.border} ${
                      p.is_finished ? 'bg-gradient-to-r from-rose-950/40 to-slate-950/90' : 'bg-slate-900/90'
                    } p-4 shadow-lg transition-all hover:border-slate-600`}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="px-2 py-0.5 rounded-md bg-slate-800 text-[10px] font-bold text-cyan-300 uppercase font-mono">
                            {p.slot_id}
                          </span>
                          <span className="text-[10px] text-slate-400 font-medium flex items-center gap-1">
                            <MapPin className="h-3 w-3 text-slate-500" />
                            {p.location}
                          </span>
                        </div>
                        <h4 className="text-base font-bold text-white flex items-center gap-2">
                          {p.product_name}
                        </h4>
                        <span className="text-xs text-slate-400">{p.category}</span>
                      </div>

                      {/* Stock Status Pill */}
                      <div className="flex flex-col items-end gap-1 shrink-0">
                        <span
                          className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold ${badge.bg} ${badge.text} border ${badge.border}`}
                        >
                          <span className={`h-2 w-2 rounded-full ${badge.dot}`} />
                          {badge.label}
                        </span>
                        <span className="text-xs text-slate-400 font-mono">
                          {p.observed_count} / {p.capacity} facings
                        </span>
                      </div>
                    </div>

                    {/* Visual Capacity Bar */}
                    <div className="mt-3.5 space-y-1">
                      <div className="flex justify-between text-[11px] text-slate-400 font-medium">
                        <span>Shelf Stock Level</span>
                        <span className="font-bold text-white">
                          {p.is_finished ? '0% (EMPTY)' : `${p.occupancy_pct}% Capacity`}
                        </span>
                      </div>
                      <div className="w-full h-2 bg-slate-800/80 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full bg-gradient-to-r ${badge.progress} transition-all duration-700`}
                          style={{ width: `${Math.min(100, p.occupancy_pct)}%` }}
                        />
                      </div>
                    </div>

                    {/* Recommendation & Restock Action */}
                    <div className="mt-3 pt-2.5 border-t border-slate-800/60 flex items-center justify-between text-xs gap-2">
                      <div className="flex items-center gap-1.5 text-slate-300 text-[11px]">
                        {p.is_finished ? (
                          <Flame className="h-4 w-4 text-rose-400 shrink-0" />
                        ) : p.deficit > 0 ? (
                          <TrendingDown className="h-4 w-4 text-amber-400 shrink-0" />
                        ) : (
                          <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
                        )}
                        <span>{p.recommendation}</span>
                      </div>

                      {p.deficit > 0 && (
                        <button
                          onClick={() =>
                            setCompletedTasks((prev) => ({
                              ...prev,
                              [p.slot_id]: !prev[p.slot_id],
                            }))
                          }
                          className={`px-3 py-1 rounded-lg text-xs font-bold transition-all shrink-0 cursor-pointer ${
                            isTaskDone
                              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                              : 'bg-slate-800 hover:bg-slate-700 text-slate-200'
                          }`}
                        >
                          {isTaskDone ? '✓ Refilled' : `Restock +${p.deficit}`}
                        </button>
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Right: Camera Snapshot with Marked Products & Empty Slots (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2">
              <Camera className="h-4 w-4 text-cyan-400" />
              Live Camera Detection Evidence
            </h3>
            <button
              onClick={() => setShowCameraView(!showCameraView)}
              className="text-xs text-cyan-400 hover:underline cursor-pointer"
            >
              {showCameraView ? 'Hide Camera' : 'Show Camera'}
            </button>
          </div>

          {showCameraView && (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-2 shadow-xl overflow-hidden">
              <div className="relative aspect-[16/9] w-full rounded-xl overflow-hidden bg-slate-950 flex items-center justify-center">
                <img
                  src={scan?.snapshot_image_url || '/evidence/latest_shelf_snapshot.jpg'}
                  alt="Shelf Camera with Marked In-Stock and Sold-Out Products"
                  className="w-full h-full object-contain rounded-xl"
                  onError={(e) => {
                    e.target.src = '/evidence/latest_shelf_snapshot.jpg';
                  }}
                />
                <div className="absolute top-2 left-2 px-2.5 py-1 rounded-md bg-slate-950/80 backdrop-blur-md border border-slate-700/60 text-[10px] font-mono text-cyan-300 flex items-center gap-1.5">
                  <span className="h-1.5 w-1.5 rounded-full bg-red-500 animate-ping" />
                  {scan?.source || 'videos/inventory2.mp4'}
                </div>
                <div className="absolute bottom-2 right-2 px-2.5 py-1 rounded-md bg-slate-950/80 backdrop-blur-md border border-slate-700/60 text-[10px] font-mono text-slate-300">
                  {scan?.inference_time_ms || 80}ms Edge CPU
                </div>
              </div>
              <div className="p-2 space-y-1 text-[11px] text-slate-400">
                <div className="flex items-center gap-3">
                  <span className="flex items-center gap-1">
                    <span className="h-2.5 w-2.5 rounded bg-emerald-500" />
                    Green: In Stock
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="h-2.5 w-2.5 rounded bg-amber-400" />
                    Amber: Low Stock
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="h-2.5 w-2.5 rounded bg-rose-500" />
                    Red Box: SOLD OUT / FINISHED
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Action Checklist for Sold Out & Low Stock Items */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3">
            <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider flex items-center gap-1.5">
              <Package className="h-3.5 w-3.5 text-cyan-400" />
              Staff Replenishment Dispatch
            </h4>
            <div className="space-y-2">
              {alerts.length === 0 ? (
                <div className="text-center py-4 text-xs text-slate-400 flex flex-col items-center gap-1">
                  <CheckCircle2 className="h-6 w-6 text-emerald-400" />
                  <span>All products are sufficiently stocked.</span>
                </div>
              ) : (
                alerts.map((alt) => (
                  <div
                    key={alt.alert_id}
                    className={`flex items-start justify-between gap-2 p-2.5 rounded-xl border text-xs ${
                      alt.is_finished
                        ? 'bg-rose-950/30 border-rose-500/40 text-rose-200'
                        : 'bg-slate-950/60 border-slate-800/80 text-slate-200'
                    }`}
                  >
                    <div className="space-y-0.5">
                      <div className="font-bold text-white flex items-center gap-1.5">
                        <span
                          className={`h-2 w-2 rounded-full ${
                            alt.is_finished ? 'bg-rose-500 animate-pulse' : 'bg-amber-400'
                          }`}
                        />
                        {alt.product_name}
                      </div>
                      <p className="text-slate-400 text-[11px] flex items-center gap-1">
                        <MapPin className="h-3 w-3 text-slate-500 shrink-0" />
                        {alt.location}
                      </p>
                      <p className="text-slate-300 text-[11px] font-medium">{alt.message}</p>
                    </div>
                    <span
                      className={`text-[10px] font-mono px-2 py-0.5 rounded shrink-0 font-bold ${
                        alt.is_finished
                          ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                          : 'bg-amber-500/15 text-amber-300 border border-amber-500/30'
                      }`}
                    >
                      +{alt.deficit} needed
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
