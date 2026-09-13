# STEP 28: Physical Shelf Tier Partitioning & 2D Occupancy Validation Report

**Evaluation Date**: 2026-09-13  
**Evaluated Systems**:
- Old Architecture (Step 27): Monolithic whole-rack ROI + 1D unguided centroid row clustering + row-isolated gap checking.
- New Architecture (Step 28): Physical Shelf Tier Partitioning (`SHELF-01-TIER-01`, `TIER-02`, etc.) + 2D Horizontal Occupancy Projection & Interval Union.
**Test Footage**:
- `inventory_data/demo_videos/shelf_pan_demo.mp4` (1762×2350, Portrait, 75 frames, 25.0 FPS)
- `videos/inventory2.mp4` (3864×2192, 4K Landscape, 210 frames, 15.0 FPS)
**Validation Suites**:
- [`scripts/validate_step28_tier_occupancy.py`](file:///d:/RetailShop/Retail_Analytics/scripts/validate_step28_tier_occupancy.py) (5 unit/integration tests)
- [`scripts/visualize_step28_tier_occupancy.py`](file:///d:/RetailShop/Retail_Analytics/scripts/visualize_step28_tier_occupancy.py) (Multi-frame visual evidence & contact sheet)
- [`scripts/validate_step26_vacancy_robustness.py`](file:///d:/RetailShop/Retail_Analytics/scripts/validate_step26_vacancy_robustness.py) (7 regression tests)

---

## 1. Problem Discovered in Step 27

In Step 27, real-video validation discovered a critical false vacancy in `shelf_pan_demo.mp4`:
- A **$617\text{px}$ horizontal gap** ($3.91\times$ to $4.58\times$ median product width) was detected on `ROW-02` ($X \in [823, 1440]$) on Frame 0 and persistently confirmed across subsequent frames.
- In physical reality, **the shelf was 100% stocked**: four 2-liter green 7UP bottles physically occupied that exact horizontal space.
- Threshold sweeps ($1.50\times$ to $2.25\times$) proved that changing the gap multiplier could not fix this issue because the artificial gap was $3.9\times$ the median width, dwarfing the thresholds.

---

## 2. Why 1D Row Clustering Failed

1. **Bounding Box Variance**: The retail product detector (`retail_detector_exp2.pt`) localized the bottles on the left ($X \in [292, 823]$) as full-body boxes ($h \approx 300\text{px}, cy \approx 590\text{px}$), but localized the four adjacent bottles in the middle ($X \in [994, 1424]$) by their top necks/caps only ($h \approx 200\text{px}, cy \approx 424\text{px}$).
2. **Unguided Centroid Clustering**: Because $|590 - 424| = 166\text{px} > 0.45 \times 250\text{px}$, the 1D clustering algorithm split a single physical shelf into two artificial pseudo-rows:
   - `ROW-01` ($y_{\text{mean}} \approx 444\text{px}$): captured the neck/cap detections.
   - `ROW-02` ($y_{\text{mean}} \approx 583\text{px}$): captured the left bottles and jumped to the far right bottle.
3. **Row Isolation**: Gap detection was performed independently per row. `ROW-02` calculated an internal distance of $617\text{px}$ between $X=823$ and $X=1440$, completely blind to the fact that four bottles were detected in `ROW-01` inside that exact horizontal span.

---

## 3. New Physical-Tier Architecture

To solve this geometrically, Step 28 replaces 1D centroid clustering with a physical tier hierarchy:
$$\text{Shelf Bay (SHELF-01)} \longrightarrow \text{Physical Tiers (TIER-01, TIER-02, ...)} \longrightarrow \text{2D Horizontal Occupancy Projection} \longrightarrow \text{Merged Occupied Spans} \longrightarrow \text{Vacancy Extraction}$$

All products resting on a physical shelf share that shelf board regardless of whether the detector localized the whole bottle, the neck, or the label.

---

## 4. Configuration Changes

In [`configs/shelf_vacancy_config.json`](file:///d:/RetailShop/Retail_Analytics/configs/shelf_vacancy_config.json), `shelves` and `video_rois` were enhanced with explicit physical shelf tiers:

### `shelf_pan_demo.mp4`:
- **`SHELF-01-TIER-01`**: Top shelf (2-liter bottles), ROI: `[0.02, 0.12, 0.98, 0.35]`
- **`SHELF-01-TIER-02`**: Middle shelf (20oz bottles), ROI: `[0.02, 0.36, 0.98, 0.59]`
- **`SHELF-01-TIER-03`**: Lower shelf (12-pack cans), ROI: `[0.02, 0.60, 0.98, 0.88]`

### `inventory2.mp4`:
- **`SHELF-01-TIER-01`**: Eye-level snacks, ROI: `[0.05, 0.28, 0.95, 0.52]`
- **`SHELF-01-TIER-02`**: Lower shelf, ROI: `[0.05, 0.52, 0.95, 0.75]`

*Backward Compatibility Note*: Single-ROI configurations without explicit tiers gracefully default to a single physical tier spanning the ROI.

---

## 5. 2D Horizontal Occupancy Implementation

In [`inventory/shelf_vacancy.py`](file:///d:/RetailShop/Retail_Analytics/inventory/shelf_vacancy.py):
1. **Tier Assignment**: Product bounding boxes are assigned to a physical tier if their vertical center or substantial vertical overlap ($\ge 30\%$) lies within the tier ROI $[ty_1, ty_2]$.
2. **Horizontal Interval Union**: Inside each physical tier, all product bounding box spans $[bx_1, bx_2]$ are projected onto the horizontal shelf axis and merged:
   - Contiguous or overlapping detections merge into continuous occupied spans.
   - Jitter tolerance of $\pm 3\text{px}$ absorbs detector boundary vibration.
3. **Vacancy Extraction**: Gaps are extracted only between adjacent merged occupied intervals:
   $$G_j = \text{merged}[j+1].x_1 - \text{merged}[j].x_2$$
4. **2D Cross-Row Validation**: If an extracted candidate gap has horizontal overlap $\ge 35\%$ with any product detection within the tier vertical extent, it is rejected and logged as:
   `"REJECTED — PRODUCT OCCUPANCY EXISTS IN OVERLAPPING VERTICAL REGION"`.

---

## 6. Partial Detection Handling

Step 28 specifically tested mixed-height detections:
- Full-body bottle: $h \approx 310\text{px}$, $cy \approx 595\text{px}$
- Neck/cap detection: $h \approx 240\text{px}$, $cy \approx 430\text{px}$
- Because horizontal projection considers the $X$-span $[bx_1, bx_2]$ of all items inside the physical tier, the neck/cap detections $[994, 1128]$, $[1148, 1278]$, and $[1300, 1424]$ directly occupy the horizontal axis.
- Gaps between the adjacent bottles collapse to normal spacing ($10\text{px} - 23\text{px}$), completely eliminating the pseudo-gap.

---

## 7. Investigation of the 617px False-Positive Before and After

| Metric | Old Step 27 Architecture | New Step 28 Architecture |
| :--- | :--- | :--- |
| **Row / Tier Grouping** | Split into 2 pseudo-rows (`ROW-01` and `ROW-02`) | Unified into **`SHELF-01-TIER-01`** (16 bottles) |
| **Median Product Width** | $158.0\text{px}$ | $121.5\text{px}$ |
| **Detected Gap at $X \in [823, 1440]$** | **$617.0\text{px}$ ($3.91\times$ median width)** | **$0\text{px}$ (Merged occupied intervals)** |
| **Tier 1 Status** | `VACANT` (`TEMPORARY_VACANCY` $\to$ `CONFIRMED`) | **`OCCUPIED` (`NORMAL`)** |
| **Alert Fired** | `"SHELF 1 — EMPTY SPACE DETECTED"` | **None (0 alerts)** |

---

## 8. Real-Video Validation Results

1. **`shelf_pan_demo.mp4` (Frame 0 to 75)**:
   - Frame 0: Tier 1 occupancy measured at **$72.2\%$**, Tier 2 at **$84.1\%$**, Tier 3 at **$99.0\%$**.
   - Overall shelf status on Frame 0: **`OCCUPIED` (NORMAL)**.
   - The false alert on the top shelf is completely eliminated across the entire video run.
2. **`inventory2.mp4` (Frame 0 to 75)**:
   - Shopper presence across 74/75 frames correctly held temporal state at **`UNCERTAIN` / `NORMAL`**.
   - **0 false stock-out alerts fired.**

---

## 9. Synthetic & Unit Regression Results

- [`scripts/validate_step28_tier_occupancy.py`](file:///d:/RetailShop/Retail_Analytics/scripts/validate_step28_tier_occupancy.py):
  - `test_01_reproduce_and_solve_step27_failure`: Replicated the exact 7-bottle mixed-height bounding box coordinates from `shelf_pan_demo.mp4`. Verified **0 false vacancies** and `NORMAL` temporal state (**PASSED**).
  - `test_02_genuine_vacancy_in_tier`: Verified that a genuine $330\text{px}$ ($2.2\times$ median) empty void confirms vacancy after 10 frames (**PASSED**).
  - `test_03_shopper_occlusion_freezes_tier_confirmation`: Verified shopper blocking shelf freezes confirmation (**PASSED**).
  - `test_04_camera_panning_boundary_suppression`: Verified boundary margin suppresses edge exits (**PASSED**).
  - `test_05_coordinate_jitter_durability`: Verified Gaussian noise ($\sigma=3\text{px}$) does not create artificial gaps (**PASSED**).
- [`scripts/validate_step26_vacancy_robustness.py`](file:///d:/RetailShop/Retail_Analytics/scripts/validate_step26_vacancy_robustness.py):
  - All 7 tests (**PASSED in 5.445s**).
- [`scripts/test_vacancy_video_pipeline.py`](file:///d:/RetailShop/Retail_Analytics/scripts/test_vacancy_video_pipeline.py):
  - 40 frames executed cleanly across both video streams at ~8.5 FPS (**PASSED**).

---

## 10. Visual Evidence Locations

Visual evidence generated by [`scripts/visualize_step28_tier_occupancy.py`](file:///d:/RetailShop/Retail_Analytics/scripts/visualize_step28_tier_occupancy.py) is saved in `output/step28_evidence/`:

1. **`step28_contact_sheet.jpg`**: 6-panel comparison:
   - *Panel 1*: Step 27 Frame 0 false-positive (yellow $617\text{px}$ gap).
   - *Panel 2*: Step 28 Frame 0 corrected Tier 1 result (cyan tier boundary, green boxes, `OCCUPIED`).
   - *Panel 3*: Step 28 Frame 25 genuine vacancy candidate on Tier 2.
   - *Panel 4*: `inventory2` shopper occlusion (orange overlay, confirmation frozen).
   - *Panel 5*: Step 28 Frame 40 confirmed vacancy alert on Tier 2.
   - *Panel 6*: Step 28 Frame 60 with purple rejected pseudo-gap overlay.
2. **`step28_shelf_pan_f000_corrected.jpg`**: High-resolution annotated evidence of Frame 0.
3. **`step28_metrics.json`**: Numerical metrics.

---

## 11. Dashboard Synchronization & Representation

- Updated [`ShelfRackVisualizer.jsx`](file:///d:/RetailShop/Retail_Analytics/dashboard/src/components/ShelfRackVisualizer.jsx):
  - Added clean **Physical Shelf Tiers** breakdown rendering tier badges (`TIER-01`, `TIER-02`, etc.), active tier status (`OCCUPIED` / `VACANT` / `UNCERTAIN`), items count, and occupancy percentage.
  - Retained honest labeling: **`Visible Shelf Occupancy`**.
  - Zero leakage of SKU numbers, brands, or fake planograms.
  - Production build succeeded (`npm run build` in 2.54s).

---

## 12. Scenario Comparison Matrix

| Test Scenario | Old Architecture (Step 27) | New Architecture (Step 28) | Result |
| :--- | :--- | :--- | :--- |
| **617px pseudo-gap** | **FALSE VACANCY** (split into 2 rows; 7UP bottles ignored) | **0 VACANCIES** (2D interval union merges footprints) | **FIXED** |
| **Normal product spacing** | Gaps $\le 0.40\bar{w}$ ignored | Gaps $\le 0.40\bar{w}$ ignored between merged intervals | **PASS** |
| **Genuine vacancy** | Confirmed after 10 unoccluded frames | Confirmed after 10 unoccluded frames in tier | **PASS** |
| **Partial product detection** | Split into multiple rows; phantom gaps created | Projected onto shelf axis; partial boxes merged | **FIXED** |
| **Shopper occlusion** | Gaps tagged `UNCERTAIN`; counter frozen | Gaps tagged `UNCERTAIN`; counter frozen | **PASS** |
| **Camera panning** | Outer boundary margin ignores edge exits | Outer boundary margin ignores edge exits | **PASS** |
| **Detection coordinate jitter** | Small box jitter absorbs into median width | 3px tolerance + interval merging absorbs jitter | **PASS** |

---

## 13. Computational & Performance Impact

- **Interval Union Complexity**: Sorting $N$ product boxes per tier and merging intervals runs in $O(N \log N)$ time where $N \le 35$ per tier.
- **Execution Time**: The 2D interval projection executes in $< 0.5\text{ms}$ per frame, incurring zero latency overhead compared to the old 1D clustering.
- **Inference Speed**: Sustained ~8.5 FPS on CPU end-to-end (including YOLO detection and optical flow).

---

## 14. Remaining Limitations

1. **Pre-Configured Tier ROIs**: Physical shelf tiers currently require horizontal bounding box ROI definitions in `configs/shelf_vacancy_config.json`. Highly angled perspective shots where shelves appear diagonally require quadrilateral/polygon tier ROIs.
2. **Severe Multi-Depth Stacking**: Products placed deeply behind front-row items are not tracked in depth; the system remains focused on front-facing shelf availability.

---

## 15. Final Decision & Recommendation

### **TIER + 2D OCCUPANCY APPROACH IS SUFFICIENT**

### Evidence-Based Rationale:
1. Physical shelf tier partitioning directly reflects the physical geometry of retail shelf racks.
2. 2D horizontal occupancy projection eliminates the reliance on unstable centroid clustering.
3. The known $617\text{px}$ false vacancy across the four 2-liter 7UP bottles is 100% eliminated on real footage.
4. Genuine empty regions, shopper occlusions, camera panning, and detection jitter are handled with complete stability.
5. All regression tests across Steps 25, 26, and 28 pass cleanly with zero SKU, brand, or catalog assumptions.
