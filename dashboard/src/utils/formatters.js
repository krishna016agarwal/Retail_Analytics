/**
 * Utility formatters for Retail Analytics dashboard telemetry
 * Enforces Indian Standard Time (IST, UTC+5:30) across all charts and tables.
 */

const IST_TZ = 'Asia/Kolkata';
const LOCALE = 'en-IN';

/**
 * Parse date string and enforce Indian Standard Time alignment.
 * Handles legacy simulated store timestamps stored with +00:00/Z.
 */
export function parseISTDate(isoString, storeTimestamp = null) {
  if (!isoString) return null;
  let cleanStr = String(isoString).trim();

  // If this record has a simulated store timestamp (e.g. "17:08:54") and was stored as UTC ("+00:00" or "Z"),
  // align it to IST (+05:30) so the graph and history match the store clock.
  if (
    storeTimestamp &&
    typeof storeTimestamp === 'string' &&
    storeTimestamp.includes(':') &&
    !storeTimestamp.startsWith('T+')
  ) {
    if (cleanStr.endsWith('+00:00') || cleanStr.endsWith('Z')) {
      cleanStr = cleanStr.replace(/(\+00:00|Z)$/, '+05:30');
    }
  }

  const d = new Date(cleanStr);
  return isNaN(d.getTime()) ? null : d;
}

export function formatTime(isoString, storeTimestamp = null) {
  if (!isoString) return '--:--:--';
  try {
    const date = parseISTDate(isoString, storeTimestamp);
    if (!date) return String(isoString);
    return date.toLocaleTimeString(LOCALE, {
      timeZone: IST_TZ,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: true,
    });
  } catch {
    return String(isoString);
  }
}

export function formatDate(isoString, storeTimestamp = null) {
  if (!isoString) return '---';
  try {
    const date = parseISTDate(isoString, storeTimestamp);
    if (!date) return String(isoString);
    return date.toLocaleDateString(LOCALE, {
      timeZone: IST_TZ,
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  } catch {
    return String(isoString);
  }
}

export function formatShortDateTime(isoString, storeTimestamp = null) {
  if (!isoString) return '--';
  try {
    const d = parseISTDate(isoString, storeTimestamp);
    if (!d) return String(isoString);
    const month = d.toLocaleDateString(LOCALE, { timeZone: IST_TZ, month: 'short' });
    const day = d.toLocaleDateString(LOCALE, { timeZone: IST_TZ, day: 'numeric' });
    const time = d.toLocaleTimeString(LOCALE, {
      timeZone: IST_TZ,
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    });
    return `${month} ${day}, ${time}`;
  } catch {
    return String(isoString);
  }
}

export function formatFullDateTime(isoString, storeTimestamp = null) {
  if (!isoString) return '--';
  try {
    const d = parseISTDate(isoString, storeTimestamp);
    if (!d) return String(isoString);
    const datePart = d.toLocaleDateString(LOCALE, {
      timeZone: IST_TZ,
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
    const timePart = d.toLocaleTimeString(LOCALE, {
      timeZone: IST_TZ,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: true,
    });
    return `${datePart}, ${timePart} (IST)`;
  } catch {
    return String(isoString);
  }
}

export function formatSeconds(sec) {
  if (sec === undefined || sec === null || isNaN(sec)) return '0.0s';
  const val = Number(sec);
  if (val < 60) {
    return `${val.toFixed(1)}s`;
  }
  const mins = Math.floor(val / 60);
  const remSec = (val % 60).toFixed(0);
  return `${mins}m ${remSec}s`;
}

export function formatNumber(num) {
  if (num === undefined || num === null || isNaN(num)) return '0';
  return Number(num).toLocaleString();
}

export function formatRelativeTime(isoString) {
  if (!isoString) return 'Never';
  try {
    const date = new Date(isoString);
    const now = new Date();
    const diffSec = Math.floor((now - date) / 1000);
    if (diffSec < 5) return 'Just now';
    if (diffSec < 60) return `${diffSec}s ago`;
    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHours = Math.floor(diffMin / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    return formatDate(isoString);
  } catch {
    return '---';
  }
}
