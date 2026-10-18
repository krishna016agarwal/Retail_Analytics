/**
 * InventoryEvidenceViewer.jsx — Step 12: Alert Evidence & Video Replay
 *
 * Modal evidence viewer for inventory alerts and events.
 * - Plays the annotated shelf video (output/inventory_alerts/inventory_alerts_video.mp4)
 *   served from /evidence/inventory_alerts_video.mp4 (Vite public/).
 * - Auto-seeks to the alert/event timestamp when opened.
 * - Displays all alert/event metadata.
 * - Prominent verification-required warning for FOV alerts.
 *
 * No pipeline/detection/tracking logic is touched.
 * All evidence fields (frame_index, timestamp_sec, track_id, etc.)
 * come from the existing inventory_report.json produced by Steps 1-8.
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  X,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  ShieldAlert,
  Film,
  AlertTriangle,
  Clock,
  Hash,
  Tag,
  CheckCircle2,
  Info,
  Crosshair,
} from 'lucide-react';

// ─── Video source path (served from Vite public/) ────────────────────────────
const EVIDENCE_VIDEO_SRC = '/evidence/inventory_alerts_video.mp4';

// ─── Alert type labels (reused from ActionCenter) ─────────────────────────────
const TYPE_LABELS = {
  LOW_STOCK:                 'Low Stock',
  POSSIBLE_STOCKOUT:         'Possible Stockout',
  PRODUCT_MOVEMENT:          'Product Movement',
  RAPID_REMOVAL:             'Rapid Removal',
  SKU_RECOGNITION_UNCERTAIN: 'SKU Uncertain',
  PRODUCT_APPEARED:          'Product Appeared',
  PRODUCT_REMOVED:           'Product Removed',
  PRODUCT_MOVED:             'Product Moved',
  SKU_CHANGED:               'SKU Changed',
};

const SEV_COLORS = {
  HIGH:   'text-rose-300 bg-rose-500/15 border-rose-500/30',
  MEDIUM: 'text-amber-300 bg-amber-500/15 border-amber-500/30',
  LOW:    'text-sky-300 bg-sky-500/15 border-sky-500/30',
  INFO:   'text-slate-400 bg-slate-700/40 border-slate-600/30',
};

// ─── FOV verification explanation ─────────────────────────────────────────────
const FOV_MSGS = {
  POSSIBLE_STOCKOUT:
    'Product is no longer visible in the camera frame. Camera movement may have caused a field-of-view exit. Verify the shelf physically before treating this as a true stockout.',
  RAPID_REMOVAL:
    'Multiple products disappeared rapidly. This may be customers picking products, OR the camera panning away. Physical shelf verification is required to distinguish the two causes.',
  LOW_STOCK:
    'Visible facing count is low but the shelf may have product behind the front row. Camera can only observe front-row facings — verify actual shelf depth before restocking.',
  SKU_RECOGNITION_UNCERTAIN:
    'Several products could not be matched to catalog entries with sufficient confidence. Verify which products are actually present on the shelf.',
  DEFAULT:
    'Confirm shelf status physically before taking corrective action. Camera-based data reflects front-row facings only.',
};

function fovMsg(alertType) {
  return FOV_MSGS[alertType] || FOV_MSGS.DEFAULT;
}

// ─── Metadata chip ─────────────────────────────────────────────────────────────
function Chip({ icon: Icon, label, value, mono }) {
  return (
    <div className="flex items-start gap-1.5 py-1.5 px-2.5 rounded-lg bg-slate-800/50 border border-slate-700/40 min-w-0">
      <Icon className="h-3.5 w-3.5 text-slate-500 shrink-0 mt-0.5" />
      <div className="min-w-0">
        <p className="text-[9px] text-slate-500 uppercase tracking-wide">{label}</p>
        <p className={`text-xs font-semibold text-slate-200 truncate ${mono ? 'font-mono' : ''}`}>
          {value ?? '—'}
        </p>
      </div>
    </div>
  );
}

// ─── Video player controls ────────────────────────────────────────────────────
function VideoControls({ videoRef, duration, currentTime, onSeek }) {
  const [playing, setPlaying] = useState(false);

  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) { v.play(); setPlaying(true); }
    else           { v.pause(); setPlaying(false); }
  };

  const stepBack = () => {
    const v = videoRef.current;
    if (v) v.currentTime = Math.max(0, v.currentTime - 0.04); // ≈1 frame @25fps
  };
  const stepFwd = () => {
    const v = videoRef.current;
    if (v) v.currentTime = Math.min(duration, v.currentTime + 0.04);
  };

  const fmt = (s) => {
    const min = Math.floor(s / 60);
    const sec = (s % 60).toFixed(2).padStart(5, '0');
    return `${min}:${sec}`;
  };

  // Keep playing state in sync with video events
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const onPlay  = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    v.addEventListener('play',  onPlay);
    v.addEventListener('pause', onPause);
    return () => { v.removeEventListener('play', onPlay); v.removeEventListener('pause', onPause); };
  }, [videoRef]);

  return (
    <div className="flex flex-col gap-2 px-4 pb-3 pt-1">
      {/* Seek bar */}
      <div className="flex items-center gap-2">
        <span className="text-[10px] font-mono text-slate-400 w-12 shrink-0">{fmt(currentTime)}</span>
        <input
          type="range"
          min={0}
          max={duration || 100}
          step={0.04}
          value={currentTime}
          onChange={e => onSeek(parseFloat(e.target.value))}
          className="flex-1 h-1.5 accent-violet-500 cursor-pointer"
        />
        <span className="text-[10px] font-mono text-slate-500 w-12 shrink-0 text-right">{fmt(duration || 0)}</span>
      </div>

      {/* Buttons */}
      <div className="flex items-center justify-center gap-3">
        <button onClick={stepBack}
          className="p-2 rounded-lg bg-slate-800 border border-slate-700/50 text-slate-400 hover:text-slate-200 hover:bg-slate-700 transition-colors">
          <SkipBack className="h-4 w-4" />
        </button>
        <button onClick={togglePlay}
          className="p-2.5 rounded-xl bg-violet-600/30 border border-violet-500/40 text-violet-300 hover:bg-violet-600/40 transition-colors">
          {playing ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
        </button>
        <button onClick={stepFwd}
          className="p-2 rounded-lg bg-slate-800 border border-slate-700/50 text-slate-400 hover:text-slate-200 hover:bg-slate-700 transition-colors">
          <SkipForward className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

// ─── Main Evidence Viewer Modal ────────────────────────────────────────────────
export default function InventoryEvidenceViewer({ evidence, onClose }) {
  const videoRef    = useRef(null);
  const [videoReady,   setVideoReady]   = useState(false);
  const [videoError,   setVideoError]   = useState(false);
  const [currentTime,  setCurrentTime]  = useState(0);
  const [duration,     setDuration]     = useState(0);
  const [seekDone,     setSeekDone]     = useState(false);

  if (!evidence) return null;

  const {
    // Alert fields
    alert_id, alert_type, event_type,
    sku_id, sku_name,
    severity,
    frame_index, timestamp_sec,
    track_id,
    reason, details,
    requires_verification,
    current_stable_facings, active_facings,
    bbox,
    // Source label: 'alert' | 'event'
    _evidenceSource,
  } = evidence;

  const displayType  = alert_type || event_type;
  const displayLabel = TYPE_LABELS[displayType] || displayType || 'Unknown';
  const targetSec    = timestamp_sec ?? 0;
  const isVerif      = requires_verification;

  // Seek to alert timestamp once video is loaded
  const handleCanPlay = useCallback(() => {
    const v = videoRef.current;
    if (!v || seekDone) return;
    v.currentTime = targetSec;
    setSeekDone(true);
    setVideoReady(true);
    setDuration(v.duration || 0);
  }, [targetSec, seekDone]);

  const handleTimeUpdate = () => {
    if (videoRef.current) setCurrentTime(videoRef.current.currentTime);
  };

  const handleMetadata = () => {
    if (videoRef.current) setDuration(videoRef.current.duration || 0);
  };

  const handleSeek = (t) => {
    if (videoRef.current) videoRef.current.currentTime = t;
  };

  const handleVideoError = () => setVideoError(true);

  // Jump to alert frame button
  const jumpToAlert = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = targetSec;
      videoRef.current.pause();
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="glass-panel w-full max-w-2xl shadow-2xl overflow-hidden flex flex-col max-h-[92vh]"
        onClick={e => e.stopPropagation()}
      >
        {/* ── Header ── */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800/80">
          <div className="flex items-center gap-2.5">
            <Film className="h-4 w-4 text-violet-400 shrink-0" />
            <div>
              <p className="text-sm font-bold text-white leading-none">Alert Evidence</p>
              <p className="text-[10px] text-slate-500 mt-0.5">
                inventory_alerts_video.mp4 · annotated pipeline output
              </p>
            </div>
          </div>
          <button onClick={onClose}
            className="p-1.5 rounded-lg text-slate-500 hover:text-slate-200 hover:bg-slate-800 transition-colors">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="overflow-y-auto flex-1">
          {/* ── Video section ── */}
          <div className="bg-black">
            {videoError ? (
              <div className="flex flex-col items-center justify-center gap-3 h-48 text-slate-500 text-sm">
                <AlertTriangle className="h-7 w-7 text-amber-500" />
                <p className="text-center">
                  Visual evidence unavailable for this event.
                  <br />
                  <span className="text-[11px] font-mono text-slate-600">
                    Expected: /evidence/inventory_alerts_video.mp4
                  </span>
                </p>
              </div>
            ) : (
              <video
                ref={videoRef}
                src={EVIDENCE_VIDEO_SRC}
                className="w-full max-h-72 object-contain bg-black"
                preload="auto"
                onCanPlay={handleCanPlay}
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={handleMetadata}
                onError={handleVideoError}
                playsInline
              />
            )}
          </div>

          {/* Video controls */}
          {!videoError && (
            <VideoControls
              videoRef={videoRef}
              duration={duration}
              currentTime={currentTime}
              onSeek={handleSeek}
            />
          )}

          <div className="px-4 pb-4 space-y-4">
            {/* ── Alert type header ── */}
            <div className="flex flex-wrap items-center gap-2 pt-1">
              {severity && (
                <span className={`px-2.5 py-1 rounded-full border text-[10px] font-bold uppercase ${SEV_COLORS[severity] || SEV_COLORS.INFO}`}>
                  {severity}
                </span>
              )}
              <span className="text-sm font-bold text-white">{displayLabel}</span>
              {isVerif && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-amber-500/15 border border-amber-500/30 text-amber-300 text-[9px] font-bold uppercase">
                  <ShieldAlert className="h-2.5 w-2.5" />
                  Verification Required
                </span>
              )}
              {/* Jump to alert frame */}
              <button
                onClick={jumpToAlert}
                className="ml-auto inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-violet-500/15 border border-violet-500/30 text-violet-300 text-[10px] font-bold hover:bg-violet-500/25 transition-colors"
              >
                <Crosshair className="h-3 w-3" />
                Jump to Alert
              </button>
            </div>

            {/* ── Metadata chips ── */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
              <Chip icon={Tag}   label="SKU"        value={sku_name || sku_id || '—'} />
              <Chip icon={Hash}  label="Frame"      value={frame_index != null ? `#${frame_index}` : '—'} mono />
              <Chip icon={Clock} label="Timestamp"  value={timestamp_sec != null ? `${timestamp_sec.toFixed(2)}s` : '—'} mono />
              {track_id != null && (
                <Chip icon={Hash} label="Track ID" value={`#${track_id}`} mono />
              )}
              {alert_id != null && (
                <Chip icon={Info} label="Alert ID"  value={`#${alert_id}`} mono />
              )}
              <Chip icon={Tag}   label="Type"       value={displayLabel} />
              {current_stable_facings != null && (
                <Chip icon={Info} label="Stable Facings" value={current_stable_facings} mono />
              )}
              {active_facings != null && (
                <Chip icon={Info} label="Active Facings" value={active_facings} mono />
              )}
            </div>

            {/* ── Reason ── */}
            {(reason || details) && (
              <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800/60">
                <p className="text-[9px] uppercase tracking-wide text-slate-500 mb-1">Reason / Details</p>
                <p className="text-xs text-slate-300 leading-relaxed">{reason || details}</p>
              </div>
            )}

            {/* ── Verification warning ── */}
            {isVerif && (
              <div className="flex items-start gap-2.5 p-3 rounded-xl bg-amber-500/8 border border-amber-500/30">
                <ShieldAlert className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
                <div>
                  <p className="text-[10px] font-bold uppercase tracking-wide text-amber-300 mb-0.5">
                    Camera / FOV Verification Required
                  </p>
                  <p className="text-xs text-amber-200/80 leading-relaxed">
                    {fovMsg(displayType)}
                  </p>
                </div>
              </div>
            )}

            {/* ── BBox (if available) ── */}
            {bbox && bbox.length === 4 && (
              <div className="p-2.5 rounded-lg bg-slate-900/40 border border-slate-800/50">
                <p className="text-[9px] uppercase tracking-wide text-slate-500 mb-1">Bounding Box (x1,y1,x2,y2)</p>
                <p className="text-[10px] font-mono text-slate-400">
                  [{bbox.map(v => Math.round(v)).join(', ')}]
                </p>
              </div>
            )}

            {/* ── Evidence notice ── */}
            <div className="flex items-start gap-2 p-2.5 rounded-lg bg-slate-900/40 border border-slate-800/50">
              <Info className="h-3.5 w-3.5 text-cyan-500 shrink-0 mt-0.5" />
              <p className="text-[10px] text-slate-500 leading-relaxed">
                Evidence is from <strong className="text-slate-400">inventory_alerts_video.mp4</strong> — the annotated output produced by the complete Steps 1–8 pipeline. Bounding boxes, track IDs, and temporal states are drawn by the pipeline visualiser. Camera-observable front-row facings only; not total physical inventory.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
