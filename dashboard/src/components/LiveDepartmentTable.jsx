import React from 'react';
import { Layers, Users, UserCheck, Clock, TrendingUp, AlertTriangle } from 'lucide-react';
import { formatNumber, formatSeconds } from '../utils/formatters';

export default function LiveDepartmentTable({ departments = [], loading = false }) {
  const defaultDepts = [
    {
      camera_id: 'CAM_01',
      zone_id: 'food',
      department: 'Food',
      current_shoppers: 0,
      footfall: 0,
      avg_dwell: 0,
      max_dwell: 0,
      traffic_level: 'LOW',
      traffic_trend: 'STABLE',
      expected_staff: 2,
      shopper_load_per_staff: 0.0,
      status: 'OPTIMAL',
    },
    {
      camera_id: 'CAM_02',
      zone_id: 'electronics',
      department: 'Electronics',
      current_shoppers: 0,
      footfall: 0,
      avg_dwell: 0,
      max_dwell: 0,
      traffic_level: 'LOW',
      traffic_trend: 'STABLE',
      expected_staff: 1,
      shopper_load_per_staff: 0.0,
      status: 'OPTIMAL',
    },
    {
      camera_id: 'CAM_03',
      zone_id: 'grocery',
      department: 'Grocery',
      current_shoppers: 0,
      footfall: 0,
      avg_dwell: 0,
      max_dwell: 0,
      traffic_level: 'LOW',
      traffic_trend: 'STABLE',
      expected_staff: 2,
      shopper_load_per_staff: 0.0,
      status: 'OPTIMAL',
    },
  ];

  const items = departments && departments.length > 0 ? departments : defaultDepts;

  const getTrafficBadge = (level) => {
    const l = String(level || '').toUpperCase();
    if (l === 'PEAK') {
      return 'bg-rose-500/15 border-rose-500/30 text-rose-300 animate-pulse';
    }
    if (l === 'HIGH') {
      return 'bg-amber-500/15 border-amber-500/30 text-amber-300';
    }
    if (l === 'NORMAL') {
      return 'bg-cyan-500/15 border-cyan-500/30 text-cyan-300';
    }
    return 'bg-slate-700/50 border-slate-600 text-slate-300';
  };

  const getLoadBadge = (load) => {
    const n = Number(load) || 0;
    if (n >= 4.0) return 'text-rose-400 font-bold';
    if (n >= 2.5) return 'text-amber-400 font-bold';
    return 'text-emerald-400 font-semibold';
  };

  return (
    <div className="glass-panel p-5 border border-slate-800/80 rounded-xl mb-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-800/80 gap-2">
        <div className="flex items-center gap-2">
          <Layers className="h-4 w-4 text-cyan-400" />
          <h3 className="text-sm font-semibold text-white tracking-wide">
            Live Department View (Food, Electronics, Grocery)
          </h3>
          <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-emerald-500/15 border border-emerald-500/30 text-emerald-300">
            Real CV Detections
          </span>
        </div>
        <div className="text-[11px] text-slate-400 font-mono flex items-center gap-2">
          <UserCheck className="h-3.5 w-3.5 text-slate-400" />
          <span>Staff Load = Live Shoppers / Configured Staff Capacity</span>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto mt-4">
        <table className="w-full text-left text-xs text-slate-300">
          <thead className="bg-slate-900/60 text-slate-400 font-mono uppercase text-[10px] tracking-wider border-b border-slate-800">
            <tr>
              <th className="py-3 px-3">Camera</th>
              <th className="py-3 px-3">Department</th>
              <th className="py-3 px-3 text-right">Live Shoppers</th>
              <th className="py-3 px-3 text-right">Footfall</th>
              <th className="py-3 px-3 text-right">Avg Dwell</th>
              <th className="py-3 px-3 text-center">Traffic</th>
              <th className="py-3 px-3 text-center">Configured Staff</th>
              <th className="py-3 px-3 text-right">Staff Load</th>
              <th className="py-3 px-3 text-center">Operational Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-mono">
            {items.map((dept) => {
              const shoppers = dept.current_shoppers ?? 0;
              const footfall = dept.footfall ?? shoppers;
              const avgDwell = dept.avg_dwell ?? 0;
              const staff = dept.expected_staff ?? 1;
              const load = dept.shopper_load_per_staff ?? (shoppers / staff).toFixed(1);
              const status = Number(load) >= 4.0 ? 'UNDERSTAFFED' : (shoppers >= 8 ? 'CONGESTED' : 'OPTIMAL');

              return (
                <tr key={dept.camera_id || dept.zone_id} className="hover:bg-slate-900/40 transition-colors">
                  <td className="py-3 px-3 text-slate-400 font-bold">{dept.camera_id}</td>
                  <td className="py-3 px-3 text-white font-semibold flex items-center gap-1.5 font-sans">
                    <span className="h-1.5 w-1.5 rounded-full bg-cyan-400" />
                    {dept.department || dept.zone_name || dept.zone_id}
                  </td>
                  <td className="py-3 px-3 text-right text-white font-bold text-sm">
                    {formatNumber(shoppers)}
                  </td>
                  <td className="py-3 px-3 text-right text-slate-300">
                    {formatNumber(footfall)}
                  </td>
                  <td className="py-3 px-3 text-right text-slate-300">
                    {formatSeconds(avgDwell)}
                  </td>
                  <td className="py-3 px-3 text-center">
                    <span className={`px-2 py-0.5 text-[10px] font-bold rounded-full border ${getTrafficBadge(dept.traffic_level)}`}>
                      {dept.traffic_level || 'LOW'}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-center text-slate-200">
                    {staff} staff
                  </td>
                  <td className={`py-3 px-3 text-right ${getLoadBadge(load)}`}>
                    {load} / staff
                  </td>
                  <td className="py-3 px-3 text-center">
                    <span
                      className={`px-2 py-0.5 text-[10px] font-bold rounded-full border ${
                        status === 'UNDERSTAFFED'
                          ? 'bg-amber-500/15 border-amber-500/30 text-amber-300'
                          : status === 'CONGESTED'
                          ? 'bg-rose-500/15 border-rose-500/30 text-rose-300 animate-pulse'
                          : 'bg-emerald-500/15 border-emerald-500/30 text-emerald-400'
                      }`}
                    >
                      {status}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-3 text-[11px] text-slate-500 italic">
        * Note: Staff counts represent configured staffing capacity per section. YOLO does not attempt automated employee recognition.
      </div>
    </div>
  );
}
