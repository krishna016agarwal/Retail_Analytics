import React, { useState, useMemo } from 'react';
import { Table, Download, Search, ChevronLeft, ChevronRight, Filter } from 'lucide-react';
import { formatTime, formatDate, formatSeconds } from '../utils/formatters';

export default function HistoricalTable({ 
  snapshots = [], 
  loading = false, 
  sourceLabel = 'Edge Telemetry Records' 
}) {
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 10;

  const safeSnapshots = useMemo(() => {
    if (Array.isArray(snapshots)) return snapshots;
    if (Array.isArray(snapshots?.items)) return snapshots.items;
    if (Array.isArray(snapshots?.snapshots)) return snapshots.snapshots;
    return [];
  }, [snapshots]);

  // Filter snapshots based on search term
  const filteredSnapshots = useMemo(() => {
    if (!searchTerm.trim()) return safeSnapshots;
    const term = searchTerm.toLowerCase();
    return safeSnapshots.filter(
      (s) =>
        String(s?.id || '').includes(term) ||
        String(s?.timestamp || '').toLowerCase().includes(term) ||
        String(s?.store_id || '').toLowerCase().includes(term) ||
        String(s?.device_id || '').toLowerCase().includes(term)
    );
  }, [safeSnapshots, searchTerm]);

  // Pagination calculation
  const totalPages = Math.max(1, Math.ceil(filteredSnapshots.length / rowsPerPage));
  const paginatedRows = useMemo(() => {
    const start = (currentPage - 1) * rowsPerPage;
    return filteredSnapshots.slice(start, start + rowsPerPage);
  }, [filteredSnapshots, currentPage]);

  // Reset to page 1 on search change
  const handleSearch = (e) => {
    setSearchTerm(e.target.value);
    setCurrentPage(1);
  };

  // CSV Export utility
  const handleExportCSV = () => {
    if (!filteredSnapshots.length) return;
    const headers = [
      'ID',
      'Store ID',
      'Device ID',
      'Timeline Timestamp',
      'UTC Created At',
      'Cumulative Entries',
      'Cumulative Exits',
      'Current Occupancy',
      'Peak Occupancy',
      'Queue Length',
      'Peak Queue',
      'Avg Dwell (s)',
      'Max Dwell (s)',
      'Avg Wait (s)',
    ];

    const rows = filteredSnapshots.map((s) => [
      s.id ?? '',
      s.store_id ?? '',
      s.device_id ?? '',
      `"${s.timestamp || ''}"`,
      `"${s.created_at || ''}"`,
      s.total_entries ?? 0,
      s.total_exits ?? 0,
      s.current_occupancy ?? 0,
      s.peak_occupancy ?? 0,
      s.queue_length ?? 0,
      s.peak_queue_length ?? 0,
      s.average_dwell_time ?? 0,
      s.max_dwell_time ?? 0,
      s.average_wait_time ?? 0,
    ]);

    const csvContent =
      'data:text/csv;charset=utf-8,' +
      [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `retail_analytics_export_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="glass-panel p-5">
      {/* Table Header Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800/80">
        <div>
          <div className="flex items-center gap-2">
            <Table className="h-4 w-4 text-cyan-400" />
            <h3 className="text-sm font-semibold text-white">Historical Telemetry Snapshots</h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            {sourceLabel} ({filteredSnapshots.length} total)
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* Search bar */}
          <div className="relative">
            <Search className="h-3.5 w-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              placeholder="Search timestamp / store..."
              value={searchTerm}
              onChange={handleSearch}
              className="bg-slate-950/60 border border-slate-800 rounded-lg pl-8 pr-3 py-1.5 text-xs text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-cyan-500/50 w-44 sm:w-56"
            />
          </div>

          {/* Export CSV button */}
          <button
            onClick={handleExportCSV}
            disabled={!snapshots.length}
            title="Download records as CSV"
            className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700/80 border border-slate-700/80 text-xs text-slate-200 font-medium transition-colors disabled:opacity-40"
          >
            <Download className="h-3.5 w-3.5 text-slate-400" />
            <span className="hidden sm:inline">Export</span> CSV
          </button>
        </div>
      </div>

      {/* Table Container */}
      <div className="overflow-x-auto mt-4">
        <table className="w-full text-left border-collapse text-xs">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400 font-medium bg-slate-950/40">
              <th className="py-2.5 px-3">#</th>
              <th className="py-2.5 px-3">Timestamp</th>
              <th className="py-2.5 px-3">Store</th>
              <th className="py-2.5 px-3">Device</th>
              <th className="py-2.5 px-3 text-right text-emerald-400">Entries</th>
              <th className="py-2.5 px-3 text-right text-rose-400">Exits</th>
              <th className="py-2.5 px-3 text-right text-cyan-400">Occupancy</th>
              <th className="py-2.5 px-3 text-right text-amber-400">Queue</th>
              <th className="py-2.5 px-3 text-right">Avg Dwell</th>
              <th className="py-2.5 px-3 text-right">Avg Wait</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/50">
            {loading && snapshots.length === 0 ? (
              <tr>
                <td colSpan="10" className="py-8 text-center text-slate-500 animate-pulse">
                  Loading historical telemetry from Central API...
                </td>
              </tr>
            ) : paginatedRows.length === 0 ? (
              <tr>
                <td colSpan="10" className="py-8 text-center text-slate-500">
                  No telemetry snapshots found matching filter.
                </td>
              </tr>
            ) : (
              paginatedRows.map((row) => (
                <tr
                  key={row.id}
                  className="hover:bg-slate-800/40 transition-colors group"
                >
                  <td className="py-2.5 px-3 font-mono text-slate-500 text-[11px]">
                    #{row.id}
                  </td>
                  <td className="py-2.5 px-3 text-slate-200">
                    <span className="font-semibold text-white block">
                      {row.timestamp || 'N/A'}
                    </span>
                    {row.created_at && (
                      <span className="text-[10px] text-slate-500 block">
                        {formatTime(row.created_at)}
                      </span>
                    )}
                  </td>
                  <td className="py-2.5 px-3 text-slate-300 font-mono text-[11px]">
                    {row.store_id || 'store_001'}
                  </td>
                  <td className="py-2.5 px-3 text-slate-300 font-mono text-[11px]">
                    {row.device_id || 'edge_device_01'}
                  </td>
                  <td className="py-2.5 px-3 text-right font-semibold text-emerald-400">
                    {row.entries}
                  </td>
                  <td className="py-2.5 px-3 text-right font-semibold text-rose-400">
                    {row.exits}
                  </td>
                  <td className="py-2.5 px-3 text-right font-bold text-cyan-300">
                    {row.occupancy}
                  </td>
                  <td className="py-2.5 px-3 text-right font-semibold text-amber-400">
                    {row.queue_length}
                  </td>
                  <td className="py-2.5 px-3 text-right text-slate-300">
                    {formatSeconds(row.avg_dwell)}
                  </td>
                  <td className="py-2.5 px-3 text-right text-slate-300">
                    {formatSeconds(row.avg_wait)}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Footer */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between pt-4 mt-3 border-t border-slate-800/80 text-xs text-slate-400">
          <span>
            Showing <strong className="text-slate-200">{(currentPage - 1) * rowsPerPage + 1}</strong> to{' '}
            <strong className="text-slate-200">
              {Math.min(currentPage * rowsPerPage, filteredSnapshots.length)}
            </strong>{' '}
            of <strong className="text-slate-200">{filteredSnapshots.length}</strong>
          </span>

          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="p-1.5 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 disabled:opacity-30 disabled:hover:bg-slate-800"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <span className="px-2 font-medium text-slate-300">
              Page {currentPage} of {totalPages}
            </span>
            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              className="p-1.5 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 disabled:opacity-30 disabled:hover:bg-slate-800"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
