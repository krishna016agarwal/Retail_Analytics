import React from 'react';
import { 
  Layers, 
  Users, 
  UserCheck, 
  TrendingUp, 
  AlertCircle, 
  CheckCircle2, 
  Flame, 
  Activity,
  ShieldAlert,
  ArrowUpRight
} from 'lucide-react';
import { formatNumber } from '../utils/formatters';

export default function StoreZones({
  zones = [],
  crowdData = {},
  liveOccupancy = 0,
}) {
  // Configured default fallback zones if edge hasn't recorded zone snapshots yet
  const defaultZones = [
    {
      zone_id: 'zone_01',
      zone_name: 'Food Section',
      current_shoppers: 0,
      peak_shoppers: 0,
      expected_staff: 2,
      shopper_load_per_staff: 0.0,
      zone_status: 'OPTIMAL',
      traffic_level: 'NORMAL',
    },
    {
      zone_id: 'zone_02',
      zone_name: 'Electronics Section',
      current_shoppers: 0,
      peak_shoppers: 0,
      expected_staff: 1,
      shopper_load_per_staff: 0.0,
      zone_status: 'OPTIMAL',
      traffic_level: 'NORMAL',
    },
    {
      zone_id: 'zone_03',
      zone_name: 'Clothing Section',
      current_shoppers: 0,
      peak_shoppers: 0,
      expected_staff: 2,
      shopper_load_per_staff: 0.0,
      zone_status: 'OPTIMAL',
      traffic_level: 'NORMAL',
    },
  ];

  const displayZones = zones && zones.length > 0 ? zones : defaultZones;

  // Crowd Intelligence values
  const currentOccupancy = crowdData.current_occupancy ?? liveOccupancy ?? 0;
  const baselineOccupancy = crowdData.baseline_occupancy ?? currentOccupancy ?? 0;
  const increasePercentage = crowdData.increase_percentage ?? 0.0;
  const isSpike = crowdData.is_spike ?? (crowdData.spike_status === 'SPIKE DETECTED') ?? false;

  const getStatusBadge = (status) => {
    const s = String(status || '').toUpperCase();
    if (s === 'CONGESTED') {
      return {
        cls: 'bg-rose-500/15 border-rose-500/30 text-rose-300 animate-pulse',
        label: 'CONGESTED',
      };
    }
    if (s === 'UNDERSTAFFED') {
      return {
        cls: 'bg-amber-500/15 border-amber-500/30 text-amber-300',
        label: 'UNDERSTAFFED',
      };
    }
    return {
      cls: 'bg-emerald-500/15 border-emerald-500/30 text-emerald-400',
      label: 'OPTIMAL',
    };
  };

  return (
    <section className="mb-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Left 2 Cols: Store Zones Section */}
      <div className="lg:col-span-2 glass-panel p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-800/80 gap-2">
          <div className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-cyan-400" />
            <h3 className="text-sm font-semibold text-white">Store Zones & Staff Allocation</h3>
          </div>
          <span className="text-[11px] text-slate-400 font-mono">
            Spatial YOLO11 Shopper Tracking • Configured Staff Capacity
          </span>
        </div>

        {/* Zones Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3.5 mt-4">
          {displayZones.map((zone) => {
            const statusInfo = getStatusBadge(zone.zone_status);
            const shoppers = zone.current_shoppers ?? 0;
            const staff = zone.expected_staff ?? 1;
            const load = zone.shopper_load_per_staff ?? (shoppers / staff).toFixed(1);
            const peak = zone.peak_shoppers ?? shoppers;

            return (
              <div
                key={zone.zone_id || Math.random()}
                className="bg-slate-950/60 border border-slate-800 hover:border-slate-700/80 p-4 rounded-xl transition-all flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <h4 className="text-sm font-bold text-white tracking-tight">
                        {zone.zone_name || zone.name || zone.zone_id}
                      </h4>
                      <span className="text-[10px] text-slate-400 font-mono">
                        ID: {zone.zone_id}
                      </span>
                    </div>
                    <span
                      className={`px-2 py-0.5 text-[10px] font-bold rounded-full border ${statusInfo.cls}`}
                    >
                      {statusInfo.label}
                    </span>
                  </div>

                  {/* Zone Numbers */}
                  <div className="grid grid-cols-2 gap-2 mt-3.5 pt-3 border-t border-slate-800/60">
                    <div>
                      <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
                        Shoppers
                      </span>
                      <span className="text-xl font-extrabold text-white mt-0.5 block font-mono">
                        {formatNumber(shoppers)}
                      </span>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
                        Peak Shoppers
                      </span>
                      <span className="text-xl font-extrabold text-slate-300 mt-0.5 block font-mono">
                        {formatNumber(peak)}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Configured Staffing Capacity & Shopper Load Ratio */}
                <div className="mt-3.5 pt-2.5 border-t border-slate-800/60 flex items-center justify-between text-xs">
                  <div className="flex items-center gap-1.5 text-slate-400">
                    <UserCheck className="h-3.5 w-3.5 text-emerald-400" />
                    <span>Configured Staff: <strong className="text-slate-200">{staff}</strong></span>
                  </div>
                  <div className="text-right">
                    <span className="text-[11px] text-slate-400">Shopper Load / Staff: </span>
                    <span
                      className={`font-mono font-bold ${
                        Number(load) >= 5.0
                          ? 'text-rose-400'
                          : Number(load) >= 3.0
                          ? 'text-amber-400'
                          : 'text-emerald-400'
                      }`}
                    >
                      {load}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Right 1 Col: Crowd Intelligence Card */}
      <div className="glass-panel p-5 flex flex-col justify-between">
        <div>
          {/* Header */}
          <div className="flex items-center justify-between pb-3 border-b border-slate-800/80">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-purple-400" />
              <h3 className="text-sm font-semibold text-white">Crowd Intelligence</h3>
            </div>
            <span
              className={`px-2.5 py-0.5 text-xs font-bold uppercase rounded-full border ${
                isSpike
                  ? 'bg-rose-500/15 border-rose-500/30 text-rose-300 animate-pulse'
                  : 'bg-emerald-500/15 border-emerald-500/30 text-emerald-400'
              }`}
            >
              {isSpike ? 'SPIKE DETECTED' : 'NOMINAL'}
            </span>
          </div>

          {/* Current Occupancy vs Rolling Baseline */}
          <div className="mt-4">
            <div className="flex items-baseline justify-between">
              <div>
                <span className="text-3xl font-extrabold text-white font-mono">
                  {currentOccupancy}
                </span>
                <span className="text-xs text-slate-400 ml-1.5">store occupancy</span>
              </div>
              <div className="text-right">
                <span className="text-[10px] text-slate-400 block uppercase tracking-wider">
                  Rolling Baseline
                </span>
                <span className="text-sm font-bold text-slate-200 font-mono">
                  {typeof baselineOccupancy === 'number' ? baselineOccupancy.toFixed(1) : baselineOccupancy}
                </span>
              </div>
            </div>

            {/* Increase Percentage Banner */}
            <div className="mt-4 p-3 rounded-lg bg-slate-950/50 border border-slate-800/60 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <TrendingUp
                  className={`h-4 w-4 ${
                    increasePercentage > 20
                      ? 'text-rose-400'
                      : increasePercentage > 0
                      ? 'text-amber-400'
                      : 'text-slate-400'
                  }`}
                />
                <span className="text-xs text-slate-300">Increase vs Baseline:</span>
              </div>
              <span
                className={`text-sm font-mono font-bold ${
                  increasePercentage >= 50
                    ? 'text-rose-400'
                    : increasePercentage > 0
                    ? 'text-amber-300'
                    : 'text-slate-400'
                }`}
              >
                {increasePercentage > 0 ? `+${increasePercentage}%` : `${increasePercentage}%`}
              </span>
            </div>
          </div>
        </div>

        {/* Actionable status note */}
        <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-start gap-2">
          {isSpike ? (
            <Flame className="h-4 w-4 text-rose-400 shrink-0 mt-0.5 animate-bounce" />
          ) : (
            <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
          )}
          <p className="text-xs text-slate-300 leading-relaxed">
            {isSpike
              ? 'Crowd concentration exceeds statistical baseline (+50%). Deploy floor staff.'
              : 'Store shopper density is within statistical limits. No crowd spikes detected.'}
          </p>
        </div>
      </div>
    </section>
  );
}
