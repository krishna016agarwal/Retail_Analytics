# Intelligent Retail Analytics - Web Dashboard (Phase 7)
### SIH Problem Statement 179 | Production React + Vite + Tailwind CSS Frontend

A production-grade retail intelligence dashboard presenting real-time edge telemetry, footfall metrics, queue congestion monitoring, shopper dwell analytics, and offline-first edge AI pipeline visualization.

Directly consumes live telemetry from the Render Central Cloud API:  
**https://retail-analytics-api-6ni3.onrender.com**

---

## Key Features

1. **Top Header**:
   - Live Central API connection status badge
   - Active Store ID (`store_001`) and Device ID (`edge_device_01`) pills
   - Auto-refresh cycle (5s / 10s configurable countdown) with manual refresh trigger
   - Last updated timestamp

2. **7 High-Contrast KPI Cards**:
   - **Current Occupancy**: Real-time shoppers inside store
   - **Latest Entries**: Cumulative footfall inflow
   - **Latest Exits**: Cumulative footfall outflow
   - **Peak Occupancy**: Maximum simultaneous store density recorded
   - **Queue Length**: Shoppers actively in checkout zone
   - **Average Dwell Time**: Shopper store engagement duration in seconds
   - **Average Wait Time**: Checkout queue delay in seconds

3. **Multi-Metric Interactive Recharts**:
   - **Inflow vs. Outflow**: Cumulative entries vs. exits comparative area chart
   - **Occupancy Profile**: Dynamic crowd density area chart with peak occupancy trendline
   - **Queue Dynamics**: Bar chart of checkout line length with peak threshold
   - **Dwell & Wait Analytics**: Timeline of shopper engagement vs. cashier queue wait

4. **Queue Congestion Monitor**:
   - Visual gauge with **LOW** (0–2), **MEDIUM** (3–5), and **HIGH** (6+) status tiers
   - Actionable cashier operational recommendations (e.g. Open counter 2)

5. **Cloud Sync & Telemetry Health**:
   - Central PostgreSQL connection state
   - Total snapshots synchronized
   - Active store and node counts

6. **Offline-First Edge AI Flow**:
   - Visualization: `Camera (RTSP/MP4) ➔ Edge AI (YOLO11 + ByteTrack) ➔ Local SQLite Cache ➔ HTTPS Batch Sync ➔ Render Cloud + PostgreSQL`
   - Demonstrating offline-first buffering with reliable retry-based synchronization during network outages

7. **Spatial Movement Heatmap Panel**:
   - Privacy architecture badge
   - Clearly explains that spatial heatmaps are computed locally at the edge, synchronizing only aggregate telemetry to the cloud to preserve shopper privacy

8. **Historical Telemetry Table**:
   - Complete historical record log
   - Search filter by timestamp / store / device
   - Client-side pagination (10 rows/page)
   - One-click CSV export

---

## Prerequisites

- **Node.js**: v18.0.0 or later (v22+ recommended)
- **npm**: v9.0.0 or later

---

## Quickstart

### 1. Navigate to the dashboard directory
```bash
cd dashboard
```

### 2. Install dependencies
```bash
npm install
```

### 3. Start the development server
```bash
npm run dev
```

The application will launch on **http://localhost:3000**.

> **Note on Port 3000**: The Render Central Web Service CORS policy permits `http://localhost:3000`. The Vite server is configured to bind on port `3000` with an integrated proxy to ensure zero CORS errors out-of-the-box.

### 4. Build for production
```bash
npm run build
```
Generates an optimized static production bundle in `dashboard/dist/`.

---

## Environment Variables

Configure `.env` in the `dashboard/` directory (or copy from `.env.example`):

```env
# Central Cloud API Base URL (Render Web Service)
VITE_CENTRAL_API_URL=https://retail-analytics-api-6ni3.onrender.com
```

---

## API Endpoints Consumed

| Endpoint | HTTP Method | Description |
| :--- | :---: | :--- |
| `/health` | `GET` | Central API server & PostgreSQL database health check |
| `/api/v1/analytics/latest` | `GET` | Most recent telemetry snapshot recorded in PostgreSQL |
| `/api/v1/analytics` | `GET` | Historical telemetry snapshots (supports `limit`, `store_id`, `device_id`) |
| `/api/v1/sync/status` | `GET` | Ingestion totals, connected stores, and devices summary |
