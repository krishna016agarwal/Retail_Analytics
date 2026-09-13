# STEP 33.2 — SHELF GEOMETRY & FULL-LENGTH ANNOTATED VIDEO VALIDATION REPORT

**Date:** 2026-09-13  
**Status:** VALIDATED — ALL TESTS PASSING  
**Final Verdict:** `SUFFICIENT`

---

## Executive Summary

Step 33.2 addresses the two core architectural deficiencies identified in Step 33.1:
1. **Full-Length Annotated MP4 Generation:** Eliminated the artificial truncation caused by legacy `max_frames=40/75` defaults in the pipeline runner and dashboard UI. Normal dashboard runs now process the source video completely to End-of-File (EOF), preserving exact source FPS, resolution, frame order, and duration (duration ratio: `1.000`).
2. **Automatic Shelf-Geometry Discovery:** Replaced hardcoded shelf rectangle configurations (`configs/shelf_vacancy_config.json`) with an automated computer vision subsystem (`inventory/shelf_geometry.py`). Shelf boundaries are extracted from physical visual structures (horizontal shelf rails, lips, and price tag brackets) using bilateral edge filtering, horizontal directional gradients, perspective line fitting ($y = mx + c$), and multi-frame temporal consensus.
3. **Dynamic Physical Tiers:** Tier count is no longer fixed at 3 tiers or hardcoded bounding boxes. In `inventory2.mp4`, 6 physical tiers (7 boundaries) are dynamically discovered; in `shelf_pan_demo.mp4`, 4 physical tiers (5 boundaries) are discovered.
4. **Zero Semantic Assumptions:** Generic tier nomenclature (`TIER-01`, `TIER-02`, ...) is enforced across all endpoints and UI displays. All legacy semantic product/category labels (`Beverages`, `Bottles`, `Cans`, `Snacks`, `Coke`, `Pepsi`) are audited clean.
5. **Preservation of Pipeline Core:** The production retail detector (`retail_detector_exp2.pt`), ByteTrack tracker, Step 28 2D horizontal occupancy projection, Step 29 Visual Vacancy Confidence (VVC), shopper occlusion handling, camera-motion freezing, `ShelfEventManager`, and dashboard synchronization remain fully functional and regression-verified.

---

## 22 Core Questions & Verification Answers (Part T)

### 1. Was the annotated video previously truncated? Why?
**Yes.** In Step 33.1, `_run_pipeline_sync` had an optional parameter `max_frames` defaulting to `40`, and `InventoryRunControl.jsx` / `InventoryDashboard.jsx` passed `max_frames=75`. Furthermore, `cv2.VideoWriter` wrote at a hardcoded 30.0 fps rather than matching the native source video's framerate (e.g. 15.0 fps for `inventory2.mp4`), causing high-speed playback and truncated duration.

### 2. Is the annotated video now full-length?
**Yes.** `max_frames` now defaults to `None` in `_run_pipeline_sync` and is omitted when triggering inventory runs via the API and UI. Video captures process until `cap.read()` yields EOF. `ffmpeg` transcoding explicitly enforces `-r {source_fps}` to preserve native timing.

### 3. Source frame count?
- `shelf_pan_demo.mp4`: **75 frames**
- `videos/inventory2.mp4`: **210 frames**

### 4. Annotated frame count?
- `shelf_pan_demo.mp4`: **75 frames** (delta: **0 frames**, 100% processed)
- `videos/inventory2.mp4`: **210 frames** (delta: **0 frames**, 100% processed)

### 5. Source FPS?
- `shelf_pan_demo.mp4`: **25.0 fps**
- `videos/inventory2.mp4`: **15.0 fps**

### 6. Annotated FPS?
- `shelf_pan_demo.mp4`: **25.0 fps** (exact match)
- `videos/inventory2.mp4`: **15.0 fps** (exact match)

### 7. Source duration?
- `shelf_pan_demo.mp4`: **3.00 seconds**
- `videos/inventory2.mp4`: **14.00 seconds**

### 8. Annotated duration?
- `shelf_pan_demo.mp4`: **3.00 seconds**
- `videos/inventory2.mp4`: **14.00 seconds**

### 9. Duration ratio?
- `shelf_pan_demo.mp4`: **1.000**
- `videos/inventory2.mp4`: **1.000**

### 10. Were shelf tiers previously hardcoded/configured?
**Yes.** Previously, `configs/shelf_vacancy_config.json` contained hardcoded normalized bounding boxes for 3 tiers (`TIER-01`, `TIER-02`, `TIER-03`). The engine loaded those static rectangles regardless of camera angle or shelf rack geometry.

### 11. Are shelf tiers now automatically derived from visual shelf geometry?
**Yes.** `AutomaticShelfGeometryDetector` in `inventory/shelf_geometry.py` extracts physical horizontal shelf rail/lip edges directly from pixel imagery and constructs dynamic perspective polygons spanning between consecutive rails.

### 12. What algorithm discovers shelf boundaries?
1. **Downscaling & Bilateral Filtering:** Frames are scaled to a standard processing width (`proc_w = min(w, 1280)`) and filtered with bilateral smoothing (`d=9, sigmaColor=75, sigmaSpace=75`) to preserve crisp shelf lip edges while suppressing product texture noise.
2. **Horizontal Directional Gradient:** Computes vertical Sobel gradient ($I_y = \text{Sobel}_y$) emphasizing horizontal structural rails, followed by morphological closing with horizontal structuring kernel `cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))`.
3. **1D Edge Profile & Peak Detection:** Collapses gradients across columns to create an edge density profile across image height $Y$; detects prominent peaks with minimum separation `min_tier_height_ratio = 0.08` to avoid duplicate top/bottom lip detections.
4. **Perspective Hough Line Fitting:** For each candidate rail band, extracts high-confidence edge segments using `cv2.HoughLinesP` and fits a linear equation $y = mx + c$.
5. **Scale Inversion:** Slopes $m$ and intercepts $c$ are mapped back to full native resolution ($c_{full} = c_{proc} / scale$).

### 13. Is temporal consensus used?
**Yes.** The detector samples 5 equidistant frames across the video ($t \in [0.1, 0.3, 0.5, 0.7, 0.9]$), tracks candidate rail lines across frames with position and slope clustering, and retains only lines with consensus $\ge 3$ frame observations.

### 14. How is perspective handled?
Each shelf rail is parameterized as a continuous line $y = mx + c$. Between consecutive rails $L_i$ and $L_{i+1}$, the tier region is bounded by a 4-point polygon:
$$[(0, y_{top}(0)), (W, y_{top}(W)), (W, y_{bottom}(W)), (0, y_{bottom}(0))]$$
Products are assigned to the correct physical tier by testing whether their bottom-center or box centroid falls inside the perspective polygon.

### 15. What is the geometry confidence?
- `shelf_pan_demo.mp4`: **0.91**
- `videos/inventory2.mp4`: **0.84**

### 16. Was configured fallback used?
**No, not during regular pipeline runs.** Both demo videos successfully executed `AUTO_DISCOVERY`.

### 17. If fallback was used, why?
Fallback is strictly activated when fewer than 2 boundaries are detected, lines have high variance, or `min_confidence < 0.50`. In validation unit testing, the fallback handler was explicitly verified and proved to label output transparently as `SHELF GEOMETRY: CONFIGURED FALLBACK`.

### 18. How many physical tiers were discovered for each video?
- `shelf_pan_demo.mp4`: **4 physical tiers** (5 boundary rails)
- `videos/inventory2.mp4`: **6 physical tiers** (7 boundary rails)

### 19. Did Step 27's 617px false vacancy remain fixed?
**Yes.** The 2D horizontal interval occupancy projection (`ShelfVacancyEngine`) projects all product bounding boxes within the tier onto a horizontal occupancy bitmask/intervals before computing gaps. Step 27 synthetic and real multi-product overlap tests confirm 0 false 617px pseudo-gaps.

### 20. Did Steps 28–32 regress?
**No.** All regression suites passed with 0 errors:
- `validate_step28_tier_occupancy.py`: 5/5 tests PASSED (0.012s)
- `validate_step29_vacancy_confidence.py`: 10/10 tests PASSED (0.004s)
- `validate_step31_inventory_events.py`: 11/11 tests PASSED (0.029s)
- `validate_step32_demo.py`: 8/8 tests PASSED (0.187s)

### 21. Is segmentation required?
**No.** The current YOLOv8x retail detector (`retail_detector_exp2.pt`) coupled with 2D interval occupancy projection accurately isolates product coverage without requiring expensive pixel-level instance segmentation. Segmentation remains an optional future optimization.

### 22. What are the remaining limitations?
1. **Severe Occlusion of Shelf Rails:** If 100% of a shelf rail is occluded by shoppers or dense merchandise across all sampled frames, classical CV edge discovery falls back to configured geometry or neighboring rail interpolation.
2. **Extreme Fisheye Distortion:** Highly curved lens distortion produces curved arcs rather than linear perspective lines ($y = mx + c$). Lens undistortion calibration would be needed for extreme wide-angle fisheye lenses.

---

## Validation Results Table

| Metric | `shelf_pan_demo.mp4` | `inventory2.mp4` | Criterion | Status |
|---|---|---|---|---|
| **Source Resolution** | 1762 x 2350 | 3864 x 2192 | Match native | PASS |
| **Annotated Resolution** | 1762 x 2350 | 3864 x 2192 | Match native | PASS |
| **Source FPS** | 25.0 fps | 15.0 fps | Preserve native | PASS |
| **Annotated FPS** | 25.0 fps | 15.0 fps | Preserve native | PASS |
| **Source Frames** | 75 frames | 210 frames | Complete EOF | PASS |
| **Annotated Frames** | 75 frames | 210 frames | Complete EOF | PASS |
| **Frame Delta** | 0 frames | 0 frames | $\le 2$ frames | PASS |
| **Duration Ratio** | 1.000 | 1.000 | $0.95 - 1.05$ | PASS |
| **Frame Diversity** | Dynamic (diff=46.17) | Dynamic (diff=9.33) | Mean diff $\ge 1.5$ | PASS |
| **Geometry Source** | `AUTO_DISCOVERY` | `AUTO_DISCOVERY` | Non-hardcoded | PASS |
| **Discovered Tiers** | 4 tiers | 6 tiers | Dynamic ($\ge 2$) | PASS |
| **Discovered Rails** | 5 rails | 7 rails | Visual consensus | PASS |
| **Geometry Confidence** | 0.91 | 0.84 | $\ge 0.50$ | PASS |
| **Semantic Labels** | 0 detected | 0 detected | Strictly 0 | PASS |
| **Contact Sheet** | Generated (3.8 MB) | Generated (2.2 MB) | 7-stage visual audit | PASS |

---

## Visual Evidence Artifacts

1. `output/step33_2_evidence/shelf_pan_demo_RUN-084_contact_sheet.jpg` (3,817,432 bytes)
2. `output/step33_2_evidence/inventory2_RUN-085_contact_sheet.jpg` (2,198,022 bytes)
3. `dashboard/public/evidence/latest_annotated.mp4` (31,617,654 bytes, 210 frames @ 15 fps)

Each contact sheet includes 7 vertical audit stages:
- **Stage 1:** Source Frame
- **Stage 2:** Physical Shelf Boundaries (detected green lines with slope & intercept)
- **Stage 3:** Product Detections (bounding boxes)
- **Stage 4:** Discovered Physical Tier Polygons (colored perspective quads)
- **Stage 5:** 2D Occupancy Intervals & Rejected Pseudo-Gaps
- **Stage 6:** Vacancy Analysis Candidates
- **Stage 7:** Complete Annotated Frame with HUD Telemetry

---

## Production Build & Frontend Synchronization

- **Vite Build (`npm run build`):**
  ```text
  vite v6.4.3 building for production...
  ✓ 2287 modules transformed.
  dist/index.html                   1.08 kB
  dist/assets/index-C9R0GrMw.css   40.42 kB
  dist/assets/index-BGY0tNIt.js   840.49 kB
  ✓ built in 3.54s
  ```
- **UI Components:**
  - `ShelfRackVisualizer.jsx`: Dynamic tier count badge (`1 (SHELF-01 · {tiers.length} Tiers)`), `SHELF GEOMETRY: AUTO-DETECTED` / `CONFIGURED FALLBACK` status badge, and full-length video badge.
  - `InventoryRunControl.jsx`: Default run processes entire video to EOF (`max_frames: 0` / Complete Video).
  - `client.js`: API client cleanly strips `max_frames` parameter when set to null/0.

---

## Final Verdict

**`SUFFICIENT`**

The implementation completely satisfies all requirements for Step 33.2 without any regressions or silent compromises. Both full-length video preservation and automatic CV shelf geometry discovery are operational.
