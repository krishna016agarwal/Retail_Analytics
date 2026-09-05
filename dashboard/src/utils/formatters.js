/**
 * Utility formatters for Retail Analytics dashboard telemetry
 */

export function formatTime(isoString) {
  if (!isoString) return '--:--:--';
  try {
    const date = new Date(isoString);
    if (isNaN(date.getTime())) return String(isoString);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return String(isoString);
  }
}

export function formatDate(isoString) {
  if (!isoString) return '---';
  try {
    const date = new Date(isoString);
    if (isNaN(date.getTime())) return String(isoString);
    return date.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' });
  } catch {
    return String(isoString);
  }
}

export function formatShortDateTime(isoString) {
  if (!isoString) return '--';
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return String(isoString);
    const month = d.toLocaleDateString([], { month: 'short' });
    const day = d.getDate();
    const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
    return `${month} ${day}, ${time}`;
  } catch {
    return String(isoString);
  }
}

export function formatFullDateTime(isoString) {
  if (!isoString) return '--';
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return String(isoString);
    return `${d.toLocaleDateString([], { month: 'short', day: 'numeric' })} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })}`;
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
