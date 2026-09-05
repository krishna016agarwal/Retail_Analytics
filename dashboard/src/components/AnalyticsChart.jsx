import React, { useState, useMemo } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { TrendingUp, Users, Clock, AlertTriangle, Layers } from 'lucide-react';
import { formatShortDateTime, formatFullDateTime, formatSeconds } from '../utils/formatters';

// Custom dark styled Tooltip preserving original video timestamp and showing ingestion time
function CustomTooltip({ active, payload, label }) {
  if (active && payload && payload.length) {
    const dataPoint = payload[0]?.payload;
    return (
      <div className="bg-slate-900/95 border border-slate-700/80 rounded-lg p-3 shadow-2xl backdrop-blur-md text-xs min-w-[210px]">
        {/* Header: Ingestion time + Video timestamp badge */}
        <div className="flex items-center justify-between gap-3 mb-2 pb-1.5 border-b border-slate-800">
          <span className="font-semibold text-slate-200">
            {dataPoint?.fullTime || label}
          </span>
          <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-cyan-500/15 border border-cyan-500/30 text-cyan-300 font-bold">
            {dataPoint?.videoTimestamp || 'N/A'}
          </span>
        </div>

        {/* Metrics breakdown */}
        <div className="space-y-1.5">
          {payload.map((entry, index) => (
            <div key={`item-${index}`} className="flex items-center justify-between gap-4">
              <span className="flex items-center gap-1.5" style={{ color: entry.color }}>
                <span className="h-2 w-2 rounded-full" style={{ backgroundColor: entry.color }} />
                {entry.name}:
              </span>
              <span className="font-bold text-white">
                {entry.name.toLowerCase().includes('time') ||
                entry.name.toLowerCase().includes('dwell') ||
                entry.name.toLowerCase().includes('wait')
                  ? formatSeconds(entry.value)
                  : entry.value}
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  }
  return null;
}

export default function AnalyticsChart({ snapshots = [], loading = false }) {
  const [activeTab, setActiveTab] = useState('inflow_outflow'); // 'inflow_outflow' | 'occupancy' | 'queue' | 'dwell_wait'

  // Chronological sorting strictly by created_at / ingestion timestamp
  // Ensures timeline progresses forward continuously without resetting or jumping backwards across runs
  const chartData = useMemo(() => {
    const list = Array.isArray(snapshots) 
      ? snapshots 
      : (Array.isArray(snapshots?.items) ? snapshots.items : (Array.isArray(snapshots?.snapshots) ? snapshots.snapshots : []));
    if (!list.length) return [];

    return [...list]
      .filter((s) => s && (s.created_at || s.id !== undefined))
      .sort((a, b) => {
        const timeA = a.created_at ? new Date(a.created_at).getTime() : a.id;
        const timeB = b.created_at ? new Date(b.created_at).getTime() : b.id;
        if (timeA !== timeB) return timeA - timeB;
        return a.id - b.id;
      })
      .map((s) => ({
        id: s.id,
        videoTimestamp: s.timestamp || `T+${s.id}`,
        fullTime: s.created_at ? formatFullDateTime(s.created_at) : `#${s.id}`,
        displayTime: s.created_at ? formatShortDateTime(s.created_at) : `#${s.id}`,
        entries: Number(s.entries || 0),
        exits: Number(s.exits || 0),
        occupancy: Number(s.occupancy || 0),
        peakOccupancy: Number(s.peak_occupancy || 0),
        queueLength: Number(s.queue_length || 0),
        peakQueue: Number(s.peak_queue || 0),
        avgDwell: Number(s.avg_dwell || 0),
        maxDwell: Number(s.max_dwell || 0),
        avgWait: Number(s.avg_wait || 0),
      }));
  }, [snapshots]);

  const tabs = [
    { id: 'inflow_outflow', label: 'Inflow vs Outflow', icon: TrendingUp },
    { id: 'occupancy', label: 'Occupancy Profile', icon: Users },
    { id: 'queue', label: 'Queue Congestion', icon: AlertTriangle },
    { id: 'dwell_wait', label: 'Dwell & Wait Times', icon: Clock },
  ];

  return (
    <div className="glass-panel p-5">
      {/* Chart Header & Tab Navigation */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800/80">
        <div className="flex items-center gap-2">
          <Layers className="h-4 w-4 text-cyan-400" />
          <h2 className="text-base font-semibold text-white">Store Telemetry Analytics</h2>
          <span className="text-xs text-slate-400">({chartData.length} chronological points)</span>
        </div>

        {/* Tab Pills */}
        <div className="flex items-center gap-1 bg-slate-950/60 p-1 rounded-lg border border-slate-800/80 overflow-x-auto">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium transition-all duration-150 whitespace-nowrap ${
                  isActive
                    ? 'bg-slate-800 text-white shadow-sm border border-slate-700/60'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                }`}
              >
                <Icon className={`h-3.5 w-3.5 ${isActive ? 'text-cyan-400' : 'text-slate-500'}`} />
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Chart Container */}
      <div className="h-[320px] w-full pt-4">
        {loading && chartData.length === 0 ? (
          <div className="h-full w-full flex items-center justify-center text-slate-500 text-xs animate-pulse">
            Loading chart data from Central API...
          </div>
        ) : chartData.length === 0 ? (
          <div className="h-full w-full flex flex-col items-center justify-center text-slate-500 text-xs">
            <p>No telemetry records available to plot.</p>
            <span className="text-[11px] text-slate-600 mt-1">
              Start edge sync worker (`python main.py --sync`) to ingest live snapshots.
            </span>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            {activeTab === 'inflow_outflow' && (
              <AreaChart data={chartData} margin={{ top: 10, right: 15, left: -15, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorEntries" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.35} />
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
                  </linearGradient>
                  <linearGradient id="colorExits" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.35} />
                    <stop offset="95%" stopColor="#f43f5e" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="displayTime"
                  stroke="#64748b"
                  tick={{ fill: '#64748b', fontSize: 10 }}
                  minTickGap={35}
                  interval="preserveStartEnd"
                />
                <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 10 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                <Area
                  type="monotone"
                  dataKey="entries"
                  name="Cumulative Entries"
                  stroke="#10b981"
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#colorEntries)"
                />
                <Area
                  type="monotone"
                  dataKey="exits"
                  name="Cumulative Exits"
                  stroke="#f43f5e"
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#colorExits)"
                />
              </AreaChart>
            )}

            {activeTab === 'occupancy' && (
              <AreaChart data={chartData} margin={{ top: 10, right: 15, left: -15, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorOcc" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="displayTime"
                  stroke="#64748b"
                  tick={{ fill: '#64748b', fontSize: 10 }}
                  minTickGap={35}
                  interval="preserveStartEnd"
                />
                <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 10 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                <Area
                  type="monotone"
                  dataKey="occupancy"
                  name="Current Occupancy"
                  stroke="#06b6d4"
                  strokeWidth={2.5}
                  fillOpacity={1}
                  fill="url(#colorOcc)"
                />
                <Line
                  type="monotone"
                  dataKey="peakOccupancy"
                  name="Peak Occupancy"
                  stroke="#f59e0b"
                  strokeWidth={1.5}
                  strokeDasharray="4 4"
                  dot={false}
                />
              </AreaChart>
            )}

            {activeTab === 'queue' && (
              <BarChart data={chartData} margin={{ top: 10, right: 15, left: -15, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="displayTime"
                  stroke="#64748b"
                  tick={{ fill: '#64748b', fontSize: 10 }}
                  minTickGap={35}
                  interval="preserveStartEnd"
                />
                <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 10 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                <Bar dataKey="queueLength" name="Queue Count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                <Line
                  type="monotone"
                  dataKey="peakQueue"
                  name="Peak Queue"
                  stroke="#ef4444"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                />
              </BarChart>
            )}

            {activeTab === 'dwell_wait' && (
              <LineChart data={chartData} margin={{ top: 10, right: 15, left: -15, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="displayTime"
                  stroke="#64748b"
                  tick={{ fill: '#64748b', fontSize: 10 }}
                  minTickGap={35}
                  interval="preserveStartEnd"
                />
                <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 10 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                <Line
                  type="monotone"
                  dataKey="avgDwell"
                  name="Average Dwell (s)"
                  stroke="#6366f1"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                />
                <Line
                  type="monotone"
                  dataKey="avgWait"
                  name="Average Wait (s)"
                  stroke="#0284c7"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                />
              </LineChart>
            )}
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
