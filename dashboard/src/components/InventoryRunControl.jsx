/**
 * InventoryRunControl.jsx — Step 14: End-to-End Inventory Run Control
 *
 * Dedicated Run Control bar allowing retail operators to:
 * - Select available video demo sources.
 * - Launch the full modular inventory pipeline.
 * - View real-time progress: frames processed, elapsed time, processing rate.
 * - See distinct states: READY -> RUNNING -> COMPLETED -> FAILED.
 * - Automatically trigger dashboard refresh on run completion.
 *
 * Strictly orchestration layer: does not duplicate or alter pipeline logic.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Play,
  Loader2,
  CheckCircle2,
  AlertTriangle,
  Film,
  RotateCcw,
  Clock,
  Layers,
  Activity,
  Check,
} from 'lucide-react';
import {
  getInventoryVideos,
  getInventoryRunStatus,
  triggerInventoryRun,
} from '../api/client';

export default function InventoryRunControl({ onRunComplete, apiAlive }) {
  const [videos, setVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState('shelf_pan_demo.mp4');
  const [maxFrames, setMaxFrames] = useState(75);
  const [runStatus, setRunStatus] = useState({
    state: 'READY',
    run_id: null,
    video_source: 'shelf_pan_demo.mp4',
    frames_processed: 0,
    total_frames: 75,
    progress_percent: 0.0,
    elapsed_sec: 0.0,
    fps: 0.0,
    error_message: null,
    latest_run: null,
  });
  const [launching, setLaunching] = useState(false);
  const [completionBanner, setCompletionBanner] = useState(null);
  const pollTimerRef = useRef(null);

  // Load available demo videos on mount
  useEffect(() => {
    getInventoryVideos().then((vids) => {
      if (vids && vids.length > 0) {
        setVideos(vids);
        setSelectedVideo(vids[0].filename);
      }
    });
  }, []);

  // Poll status while RUNNING
  const pollStatus = useCallback(async () => {
    const status = await getInventoryRunStatus();
    if (status) {
      setRunStatus(status);

      if (status.state === 'COMPLETED') {
        if (pollTimerRef.current) {
          clearInterval(pollTimerRef.current);
          pollTimerRef.current = null;
        }
        setLaunching(false);
        setCompletionBanner({
          run_id: status.run_id || status.latest_run?.run_id || 'COMPLETED',
          elapsed_sec: status.elapsed_sec,
          facings: status.latest_run?.active_visible_facings ?? null,
          alerts: status.latest_run?.total_alerts ?? null,
        });

        // Notify parent dashboard to reload report and run history
        if (onRunComplete) {
          onRunComplete();
        }
      } else if (status.state === 'FAILED') {
        if (pollTimerRef.current) {
          clearInterval(pollTimerRef.current);
          pollTimerRef.current = null;
        }
        setLaunching(false);
      }
    }
  }, [onRunComplete]);

  // Handle start run
  const handleStartRun = async () => {
    setLaunching(true);
    setCompletionBanner(null);
    setRunStatus((prev) => ({
      ...prev,
      state: 'RUNNING',
      frames_processed: 0,
      progress_percent: 0.0,
      elapsed_sec: 0.0,
      error_message: null,
    }));

    const resp = await triggerInventoryRun(maxFrames, selectedVideo);
    if (!resp || resp.status === 'already_running') {
      if (resp?.status === 'already_running') {
        // Run already in progress; attach polling
      } else {
        setRunStatus((prev) => ({
          ...prev,
          state: 'FAILED',
          error_message: 'Could not connect to Inventory API. Ensure run_inventory_api.py is running on port 8001.',
        }));
        setLaunching(false);
        return;
      }
    }

    // Begin fast polling every 800ms
    if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    pollTimerRef.current = setInterval(pollStatus, 800);
  };

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, []);

  const isRunning = runStatus.state === 'RUNNING' || launching;

  return (
    <div className="glass-panel p-4 rounded-xl border border-slate-800 space-y-3">
      {/* Top row: Header, Video Select, Run CTA */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="p-2 rounded-lg bg-violet-600/15 border border-violet-500/30 text-violet-400 shrink-0">
            <Activity className="h-4 w-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-bold text-white tracking-wide">
                Inventory Run Control
              </h3>
              {/* State Badge */}
              {runStatus.state === 'READY' && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-cyan-500/15 border border-cyan-500/30 text-cyan-300 text-[10px] font-bold uppercase">
                  <Check className="h-2.5 w-2.5" />
                  Ready
                </span>
              )}
              {runStatus.state === 'RUNNING' && (
                <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-violet-500/20 border border-violet-500/40 text-violet-300 text-[10px] font-bold uppercase animate-pulse">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Running Pipeline
                </span>
              )}
              {runStatus.state === 'COMPLETED' && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 text-[10px] font-bold uppercase">
                  <CheckCircle2 className="h-2.5 w-2.5" />
                  Completed
                </span>
              )}
              {runStatus.state === 'FAILED' && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-rose-500/15 border border-rose-500/30 text-rose-300 text-[10px] font-bold uppercase">
                  <AlertTriangle className="h-2.5 w-2.5" />
                  Failed
                </span>
              )}
            </div>
            <p className="text-[11px] text-slate-500 mt-0.5">
              Launch live analysis run using trained retail detector &amp; ByteTrack
            </p>
          </div>
        </div>

        {/* Video Selector & Controls */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Video Selector Dropdown */}
          <div className="flex items-center gap-1.5 bg-slate-900/80 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-300">
            <Film className="h-3.5 w-3.5 text-slate-500 shrink-0" />
            <select
              value={selectedVideo}
              onChange={(e) => setSelectedVideo(e.target.value)}
              disabled={isRunning}
              className="bg-transparent text-slate-200 text-xs focus:outline-none cursor-pointer disabled:cursor-not-allowed"
            >
              {videos.map((v) => (
                <option key={v.filename} value={v.filename} className="bg-slate-900 text-slate-200">
                  {v.label || v.filename}
                </option>
              ))}
            </select>
          </div>

          {/* Frames Limit Selector */}
          <div className="flex items-center gap-1 bg-slate-900/80 border border-slate-800 rounded-lg px-2 py-1.5 text-xs text-slate-400">
            <span className="text-[10px] text-slate-500 uppercase font-mono">Frames:</span>
            <select
              value={maxFrames}
              onChange={(e) => setMaxFrames(Number(e.target.value))}
              disabled={isRunning}
              className="bg-transparent text-slate-200 text-xs font-mono focus:outline-none cursor-pointer disabled:cursor-not-allowed"
            >
              <option value={50} className="bg-slate-900">50 frames</option>
              <option value={75} className="bg-slate-900">75 frames (full)</option>
              <option value={100} className="bg-slate-900">100 frames</option>
            </select>
          </div>

          {/* Start Run CTA Button */}
          <button
            onClick={handleStartRun}
            disabled={isRunning || !apiAlive}
            className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold transition-all shadow-lg ${
              isRunning
                ? 'bg-violet-600/30 text-violet-300 border border-violet-500/40 cursor-not-allowed'
                : !apiAlive
                ? 'bg-slate-800 text-slate-500 border border-slate-700 cursor-not-allowed'
                : 'bg-violet-600 hover:bg-violet-500 text-white shadow-violet-600/25 hover:shadow-violet-600/40 active:scale-[0.98]'
            }`}
          >
            {isRunning ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                <span>Processing...</span>
              </>
            ) : (
              <>
                <Play className="h-3.5 w-3.5 fill-current" />
                <span>Start Inventory Run</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Progress Bar (Visible when RUNNING) */}
      {isRunning && (
        <div className="pt-2 border-t border-slate-800/80 space-y-2">
          <div className="flex items-center justify-between text-[11px] font-mono">
            <span className="text-slate-400 flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-violet-400 animate-ping" />
              Processing: <strong className="text-white">{runStatus.frames_processed}</strong> / {runStatus.total_frames || maxFrames} frames
            </span>
            <div className="flex items-center gap-3 text-slate-400">
              <span>Elapsed: <strong className="text-slate-200">{runStatus.elapsed_sec}s</strong></span>
              <span>Speed: <strong className="text-slate-200">{runStatus.fps} fps</strong></span>
              <span className="text-violet-300 font-bold">{runStatus.progress_percent}%</span>
            </div>
          </div>

          <div className="w-full h-2 bg-slate-950 rounded-full overflow-hidden border border-slate-800">
            <div
              className="h-full bg-gradient-to-r from-violet-600 via-indigo-500 to-cyan-400 rounded-full transition-all duration-300 ease-out"
              style={{ width: `${Math.min(runStatus.progress_percent || 0, 100)}%` }}
            />
          </div>
        </div>
      )}

      {/* Success Notification Banner */}
      {completionBanner && !isRunning && (
        <div className="flex items-center justify-between p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-xs text-emerald-300">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
            <span>
              <strong>{completionBanner.run_id}</strong> completed in{' '}
              <strong>{completionBanner.elapsed_sec}s</strong>. Dashboard and Run History refreshed with live measurements!
            </span>
          </div>
          <button
            onClick={() => setCompletionBanner(null)}
            className="text-[10px] text-slate-500 hover:text-slate-300 font-mono ml-3"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Error Banner */}
      {runStatus.state === 'FAILED' && runStatus.error_message && (
        <div className="p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/30 text-xs text-rose-300 flex items-start gap-2">
          <AlertTriangle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <p className="font-bold">Run Execution Failed</p>
            <p className="text-[11px] text-rose-400/80 font-mono mt-0.5">{runStatus.error_message}</p>
          </div>
        </div>
      )}
    </div>
  );
}
