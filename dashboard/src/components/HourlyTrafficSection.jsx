import React from 'react';
import { Clock, TrendingUp, Award, BarChart3, Info } from 'lucide-react';
import { formatNumber } from '../utils/formatters';

export default function HourlyTrafficSection({ hourlyData = {}, loading = false }) {
  const data = hourlyData || {};
  const isInsufficient = !hourlyData || data.status === 'insufficient_data' || !data.departments || Object.keys(data.departments).length === 0;

  const departments = data.departments || {};
  const peakDept = data.peak_department || 'N/A';
  const peakHour = data.peak_hour || 'N/A';
  const overallTrend = data.overall_trend || 'STABLE';

  // Gather unique hours across all departments
  const allHoursSet = new Set();
  Object.values(departments).forEach((d) => {
    if (d.hourly) {
      Object.keys(d.hourly).forEach((h) => allHoursSet.add(h));
    }
  });
  const allHours = Array.from(allHoursSet);

  return (
    <div className="glass-panel p-5 border border-slate-800/80 rounded-xl mb-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-800/80 gap-2">
        <div className="flex items-center gap-2">
          <Clock className="h-4 w-4 text-emerald-400" />
          <h3 className="text-sm font-semibold text-white tracking-wide">
            Department Traffic by Hour (Historical Observations)
          </h3>
          <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-slate-800 text-slate-300 border border-slate-700">
            Simulated Store Time
          </span>
        </div>

        {/* Derived Summary Badges */}
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800">
            <Award className="h-3.5 w-3.5 text-amber-400" />
            <span className="text-slate-400">Peak Dept:</span>
            <span className="font-bold text-white">{peakDept}</span>
          </div>
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800">
            <Clock className="h-3.5 w-3.5 text-cyan-400" />
            <span className="text-slate-400">Peak Hour:</span>
            <span className="font-bold text-cyan-300">{peakHour}</span>
          </div>
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800">
            <TrendingUp className="h-3.5 w-3.5 text-emerald-400" />
            <span className="text-slate-400">Trend:</span>
            <span className="font-bold text-emerald-400">{overallTrend}</span>
          </div>
        </div>
      </div>

      {/* Content */}
      {isInsufficient ? (
        <div className="py-8 text-center text-slate-400 flex flex-col items-center justify-center gap-2">
          <BarChart3 className="h-8 w-8 text-slate-600 animate-pulse" />
          <p className="text-sm font-medium text-slate-300">
            Accumulating Stored Observations...
          </p>
          <p className="text-xs text-slate-500 max-w-md">
            Hourly traffic curves are derived directly from accumulated computer vision detections. Records will populate as the 4 camera streams process.
          </p>
        </div>
      ) : (
        <div className="mt-4 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(departments).map(([key, dept]) => {
              const hoursObj = dept.hourly || {};
              const deptPeak = dept.peak_hour || 'N/A';
              const maxShoppers = dept.peak_shoppers || 0;

              return (
                <div
                  key={key}
                  className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 hover:border-slate-700/80 transition-all"
                >
                  <div className="flex items-center justify-between border-b border-slate-800/60 pb-2">
                    <span className="text-sm font-bold text-white flex items-center gap-1.5">
                      <span className="h-2 w-2 rounded-full bg-cyan-400" />
                      {dept.name}
                    </span>
                    <span className="text-[10px] font-mono text-slate-400">
                      {dept.camera_id}
                    </span>
                  </div>

                  {/* Hourly Table for this department */}
                  <div className="mt-3 space-y-2">
                    {Object.keys(hoursObj).length === 0 ? (
                      <div className="text-xs text-slate-500 italic py-2">No hourly records yet</div>
                    ) : (
                      Object.entries(hoursObj).map(([hLabel, hMetric]) => (
                        <div
                          key={hLabel}
                          className="flex items-center justify-between text-xs py-1 border-b border-slate-800/40"
                        >
                          <span className="font-mono text-slate-300 font-semibold">{hLabel}</span>
                          <div className="flex items-center gap-3 font-mono">
                            <span className="text-slate-400 text-[11px]">
                              Avg: <strong className="text-white">{hMetric.avg_shoppers}</strong>
                            </span>
                            <span className="text-slate-400 text-[11px]">
                              Peak: <strong className="text-cyan-300">{hMetric.peak_shoppers}</strong>
                            </span>
                          </div>
                        </div>
                      ))
                    )}
                  </div>

                  <div className="mt-3 pt-2 border-t border-slate-800/60 flex items-center justify-between text-[11px] text-slate-400">
                    <span>Peak: <strong className="text-amber-300">{deptPeak}</strong></span>
                    <span>Max: <strong className="text-emerald-400">{maxShoppers} shoppers</strong></span>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="flex items-center gap-2 p-2.5 rounded-lg bg-slate-900/50 border border-slate-800/60 text-xs text-slate-400">
            <Info className="h-4 w-4 text-cyan-400 shrink-0" />
            <span>
              Peak hours and department traffic trends are strictly calculated from stored CV observations, not fixed store-clock assumptions.
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
