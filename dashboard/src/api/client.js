import axios from 'axios';

// Central Cloud API base URL and Local Edge API base URL
const CENTRAL_URL = import.meta.env.VITE_CENTRAL_API_URL || 'https://retail-analytics-api-6ni3.onrender.com';
const EDGE_URL = import.meta.env.VITE_EDGE_API_URL || 'http://127.0.0.1:8000';

export const centralClient = axios.create({
  baseURL: CENTRAL_URL,
  timeout: 5000, // Quick timeout so fallback to local Edge API is responsive
  headers: {
    Accept: 'application/json',
    'Content-Type': 'application/json',
  },
});

export const edgeClient = axios.create({
  baseURL: EDGE_URL,
  timeout: 5000,
  headers: {
    Accept: 'application/json',
    'Content-Type': 'application/json',
  },
});

// Track active backend: 'CENTRAL' or 'EDGE'
let activeBackend = 'CENTRAL';

export function getActiveBackendInfo() {
  return {
    source: activeBackend,
    isOnlineCentral: activeBackend === 'CENTRAL',
    label: activeBackend === 'CENTRAL' ? 'ONLINE — Central Analytics' : 'OFFLINE — Edge Analytics',
    badgeClass:
      activeBackend === 'CENTRAL'
        ? 'bg-emerald-500/15 border-emerald-500/30 text-emerald-400'
        : 'bg-amber-500/15 border-amber-500/30 text-amber-300',
  };
}

/**
 * Resilient GET request helper with automatic Central -> Edge fallback:
 * 1. Attempts Central Cloud API (Render + PostgreSQL).
 * 2. If Central is offline, cold-starting, or errors, gracefully falls back to Local Edge API (SQLite).
 */
async function resilientGet(path, config = {}) {
  // Try Central first
  try {
    const response = await centralClient.get(path, {
      ...config,
      validateStatus: (status) => status < 500,
    });
    if (response.status < 400) {
      activeBackend = 'CENTRAL';
      return response.data;
    }
  } catch (centralErr) {
    // Central unreachable or timed out
  }

  // Fallback to local Edge API (SQLite)
  try {
    const edgeResponse = await edgeClient.get(path, {
      ...config,
      validateStatus: (status) => status < 600,
    });
    activeBackend = 'EDGE';
    return edgeResponse.data;
  } catch (edgeErr) {
    // If running in browser with Vite dev proxy
    if (typeof window !== 'undefined' && import.meta.env.DEV) {
      try {
        const proxyResponse = await axios.get(path, {
          ...config,
          timeout: 5000,
          validateStatus: (status) => status < 600,
        });
        return proxyResponse.data;
      } catch (proxyErr) {
        throw edgeErr;
      }
    }
    throw edgeErr;
  }
}

/**
 * Fetch health status.
 */
export async function getHealth() {
  try {
    const data = await centralClient.get('/health', { timeout: 4000 });
    if (data.status === 200 && data.data?.status === 'healthy') {
      activeBackend = 'CENTRAL';
      return data.data;
    }
  } catch {
    // Central unhealthy, check edge
  }

  try {
    const edgeHealth = await edgeClient.get('/health', { timeout: 2000 });
    if (edgeHealth.status === 200) {
      activeBackend = 'EDGE';
      return { status: 'healthy', database: 'connected', source: 'edge' };
    }
  } catch {
    // Both unavailable
  }

  return { status: 'unhealthy', database: 'disconnected' };
}

/**
 * Fetch latest telemetry snapshot.
 */
export async function getLatestAnalytics() {
  return await resilientGet('/api/v1/analytics/latest');
}

/**
 * Fetch historical analytics snapshots.
 */
export async function getAnalytics(params = { limit: 100 }) {
  return await resilientGet('/api/v1/analytics', { params });
}

/**
 * Fetch sync metrics and telemetry status.
 */
export async function getSyncStatus() {
  return await resilientGet('/api/v1/sync/status');
}

/**
 * Fetch latest aggregated retail intelligence (queues, crowds, zones, active alerts).
 */
export async function getLatestIntelligence() {
  return await resilientGet('/api/v1/intelligence/latest');
}

/**
 * Fetch operational alerts (QUEUE_CONGESTION, STAFFING, CROWD_SPIKE).
 */
export async function getAlerts(params = { limit: 50 }) {
  return await resilientGet('/api/v1/alerts', { params });
}

/**
 * Fetch latest status and metrics across store zones.
 */
export async function getZones() {
  return await resilientGet('/api/v1/zones');
}

/**
 * Fetch live department analytics (Food, Electronics, Grocery).
 */
export async function getDepartments() {
  return await resilientGet('/api/v1/departments');
}

/**
 * Fetch 4-camera processing status (CAM_01 to CAM_04).
 */
export async function getCameras() {
  return await resilientGet('/api/v1/cameras');
}

/**
 * Fetch hourly department traffic derived from stored CV observations.
 */
export async function getHourlyPatterns(params = { limit: 2000 }) {
  return await resilientGet('/api/v1/patterns/hourly', { params });
}

/**
 * Fetch simulation clock status.
 */
export async function getSimulationClock() {
  return await resilientGet('/api/v1/simulation/clock');
}

/**
 * Fetch historical traffic patterns.
 */
export async function getPatterns(params = { limit: 200 }) {
  return await resilientGet('/api/v1/patterns', { params });
}

// ─── Inventory API (Step 10 — port 8001, proxied via /inventory/*) ─────────

/**
 * Fetch the latest StoreInventoryReport from the inventory backend.
 * Returns { data, source } where source is 'LIVE' or 'DEMO'.
 * Falls back to the static public/inventory_report.json if API is unavailable.
 */
export async function getInventoryReport() {
  // 1. Try the live inventory API
  try {
    const res = await axios.get('/inventory/report', { timeout: 5000 });
    if (res.status === 200 && res.data) {
      const apiSource = res.data._api_source; // 'DISK' | 'PIPELINE'
      return { data: res.data, source: apiSource === 'PIPELINE' ? 'LIVE' : 'DISK' };
    }
  } catch (_) {
    // Inventory API unavailable — fall through to static fallback
  }

  // 2. Fallback: static JSON in dashboard/public/
  try {
    const res = await axios.get(`/inventory_report.json?_t=${Date.now()}`, { timeout: 4000 });
    if (res.status === 200 && res.data) {
      return { data: res.data, source: 'DEMO' };
    }
  } catch (_) {
    // Both unavailable
  }

  return { data: null, source: 'NONE' };
}

/**
 * Fetch available demo video sources for inventory pipeline.
 */
export async function getInventoryVideos() {
  try {
    const res = await axios.get('/inventory/videos', { timeout: 3000 });
    if (res.status === 200) {
      return res.data?.videos || [];
    }
  } catch (_) {}
  return [
    {
      filename: 'shelf_pan_demo.mp4',
      relative_path: 'inventory_data/demo_videos/shelf_pan_demo.mp4',
      size_bytes: 6697151,
      label: 'shelf_pan_demo.mp4 (Shelf Panoramic Demo)',
    },
  ];
}

/**
 * Fetch live inventory pipeline execution status and progress.
 */
export async function getInventoryRunStatus() {
  try {
    const res = await axios.get('/inventory/run/status', { timeout: 3000 });
    if (res.status === 200) {
      return res.data;
    }
  } catch (_) {}
  return {
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
  };
}

/**
 * Trigger an inventory pipeline run (POST /inventory/run).
 * Returns the API response or null on failure.
 */
export async function triggerInventoryRun(maxFrames = 75, videoSource = null) {
  try {
    let url = `/inventory/run?max_frames=${maxFrames}`;
    if (videoSource) {
      url += `&video_source=${encodeURIComponent(videoSource)}`;
    }
    const res = await axios.post(url, null, { timeout: 5000 });
    return res.data;
  } catch (err) {
    if (err.response?.status === 409) {
      return { status: 'already_running', message: 'Pipeline run already active' };
    }
    return null;
  }
}

/**
 * Fetch inventory API health status.
 * Returns { alive, pipelineRunning, cacheSource, runState } or null if unreachable.
 */
export async function getInventoryHealth() {
  try {
    const res = await axios.get('/inventory/health', { timeout: 3000 });
    if (res.status === 200) {
      return {
        alive:           true,
        pipelineRunning: res.data.pipeline_running ?? false,
        cacheSource:     res.data.cache_source ?? 'NONE',
        cacheTimestamp:  res.data.cache_timestamp ?? null,
        runState:        res.data.run_state ?? 'READY',
      };
    }
  } catch (_) {
    // Inventory API offline
  }
  return { alive: false, pipelineRunning: false, cacheSource: 'NONE', cacheTimestamp: null, runState: 'READY' };
}

/**
 * Fetch latest periodic shelf snapshot telemetry, countdown, and planogram status.
 */
export async function getShelfSnapshotLatest() {
  try {
    const res = await axios.get('/inventory/snapshot/latest', { timeout: 3500 });
    if (res.status === 200 && res.data) {
      return res.data;
    }
  } catch (_) {
    // Fallback to static snapshot JSON if API is temporarily unavailable
  }

  try {
    const res = await axios.get(`/evidence/latest_shelf_snapshot.json?_t=${Date.now()}`, { timeout: 3000 });
    if (res.status === 200 && res.data) {
      return {
        is_running: true,
        interval_seconds: 10,
        seconds_until_next_scan: 5,
        total_scans_completed: 1,
        latest_scan: res.data,
      };
    }
  } catch (_) {}

  return null;
}

/**
 * Trigger immediate edge camera scan on demand.
 */
export async function triggerShelfScanNow() {
  try {
    const res = await axios.post('/inventory/snapshot/scan-now', null, { timeout: 6000 });
    return res.data;
  } catch (err) {
    return null;
  }
}

/**
 * Start/resume periodic background video scanning.
 */
export async function startShelfSnapshot() {
  try {
    const res = await axios.post('/inventory/snapshot/start', null, { timeout: 3000 });
    return res.data;
  } catch (err) {
    return null;
  }
}

/**
 * Stop/pause periodic background video scanning.
 */
export async function stopShelfSnapshot() {
  try {
    const res = await axios.post('/inventory/snapshot/stop', null, { timeout: 3000 });
    return res.data;
  } catch (err) {
    return null;
  }
}

/**
 * Update periodic camera scan interval (e.g. 10s for demo, 300s for prod).
 */
export async function setShelfScanInterval(seconds) {
  try {
    const res = await axios.post(`/inventory/snapshot/interval?interval_sec=${seconds}`, null, { timeout: 3000 });
    return res.data;
  } catch (err) {
    return null;
  }
}


/**
 * Helper to classify API errors.
 */
export function classifyApiError(error) {
  if (error.code === 'ECONNABORTED' || (error.message && error.message.includes('timeout'))) {
    return {
      type: 'COLD_START',
      message: 'Central Render service is waking up (~30-50s). Falling back to Local Edge Analytics.',
    };
  }
  if (!error.response) {
    return {
      type: 'NETWORK_ERROR',
      message: 'Central service unavailable. Displaying local offline edge analytics.',
    };
  }
  if (error.response.status === 404) {
    return {
      type: 'EMPTY_DATA',
      message: 'No snapshots recorded yet. Waiting for camera workers.',
    };
  }
  if (error.response.status === 503) {
    return {
      type: 'DATABASE_DISCONNECTED',
      message: 'Database reconnecting.',
    };
  }
  return {
    type: 'SERVER_ERROR',
    message: error.response?.data?.detail || error.message || 'An unexpected error occurred.',
  };
}
