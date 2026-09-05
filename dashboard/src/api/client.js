import axios from 'axios';

// Central Cloud API base URL
const DIRECT_URL = import.meta.env.VITE_CENTRAL_API_URL || 'https://retail-analytics-api-6ni3.onrender.com';

export const apiClient = axios.create({
  baseURL: DIRECT_URL,
  timeout: 25000,
  headers: {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
  },
});

/**
 * Resilient GET request helper:
 * Tries direct Central Cloud API URL first.
 * If browser encounters a CORS / network error in dev mode, falls back to the Vite dev proxy.
 */
async function resilientGet(path, config = {}) {
  try {
    const response = await apiClient.get(path, {
      ...config,
      validateStatus: (status) => status < 600,
    });
    return response.data;
  } catch (directErr) {
    // If direct request failed in browser dev mode (e.g. CORS block on localhost), fallback to Vite proxy
    if (typeof window !== 'undefined' && import.meta.env.DEV) {
      try {
        const proxyResponse = await axios.get(path, {
          ...config,
          timeout: 25000,
          validateStatus: (status) => status < 600,
        });
        return proxyResponse.data;
      } catch (proxyErr) {
        throw directErr;
      }
    }
    throw directErr;
  }
}

/**
 * Fetch central service and PostgreSQL health status.
 * Expected healthy: { status: 'healthy', database: 'connected' }
 * Expected unhealthy: { status: 'unhealthy', database: 'disconnected' }
 */
export async function getHealth() {
  const data = await resilientGet('/health');
  if (typeof data === 'string') {
    try {
      return JSON.parse(data);
    } catch {
      return { status: 'unknown', database: 'unknown' };
    }
  }
  return data;
}

/**
 * Fetch latest edge telemetry snapshot ingested in central PostgreSQL.
 */
export async function getLatestAnalytics() {
  try {
    return await resilientGet('/api/v1/analytics/latest');
  } catch (err) {
    if (err.response?.status === 404) {
      try {
        return await resilientGet('/api/latest');
      } catch {
        throw err;
      }
    }
    throw err;
  }
}

/**
 * Fetch historical analytics snapshots with optional store/device filter.
 * @param {Object} params - { store_id, device_id, limit }
 */
export async function getAnalytics(params = { limit: 100 }) {
  try {
    return await resilientGet('/api/v1/analytics', { params });
  } catch (err) {
    if (err.response?.status === 404) {
      try {
        return await resilientGet('/api/snapshots', { params });
      } catch {
        throw err;
      }
    }
    throw err;
  }
}

/**
 * Fetch sync metrics and telemetry aggregation status from central database.
 */
export async function getSyncStatus() {
  try {
    return await resilientGet('/api/v1/sync/status');
  } catch (err) {
    if (err.response?.status === 404) {
      try {
        return await resilientGet('/api/sync/status');
      } catch {
        throw err;
      }
    }
    throw err;
  }
}

/**
 * Fetch latest aggregated retail intelligence (queues, crowds, zones, active alerts).
 */
export async function getLatestIntelligence() {
  return await resilientGet('/api/v1/intelligence/latest');
}

/**
 * Fetch operational alerts (QUEUE_CONGESTION, STAFFING, CROWD_SPIKE).
 * @param {Object} params - { severity, type, status, limit }
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
 * Fetch historical traffic patterns or insufficient data status.
 * @param {Object} params - { store_id, limit }
 */
export async function getPatterns(params = { limit: 200 }) {
  return await resilientGet('/api/v1/patterns', { params });
}

/**
 * Helper to classify API errors: Render cold start, network failure, or HTTP error.
 */
export function classifyApiError(error) {
  if (error.code === 'ECONNABORTED' || (error.message && error.message.includes('timeout'))) {
    return {
      type: 'COLD_START',
      message: 'Central Render service is waking up from sleep mode (spin-up takes ~30-50s). Automatic retry in progress...',
    };
  }
  if (!error.response) {
    return {
      type: 'NETWORK_ERROR',
      message: 'Unable to connect to Central API. Please check your internet connection or Render service status.',
    };
  }
  if (error.response.status === 404) {
    return {
      type: 'EMPTY_DATA',
      message: 'No analytics snapshots have been synchronized yet. Edge device has not uploaded telemetry.',
    };
  }
  if (error.response.status === 503) {
    return {
      type: 'DATABASE_DISCONNECTED',
      message: 'Central service is online, but PostgreSQL database is currently reconnecting.',
    };
  }
  return {
    type: 'SERVER_ERROR',
    message: error.response?.data?.detail || error.message || 'An unexpected server error occurred.',
  };
}
