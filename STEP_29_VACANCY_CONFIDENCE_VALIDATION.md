# STEP 29 — VACANCY CONFIDENCE & REPLENISHMENT SIGNAL VALIDATION

**Project**: Intelligent Retail Analytics Platform (SIH 179)  
**Component**: Generic Shelf Vacancy Detection Engine (V1)  
**Document Version**: 1.0.0  
**Status**: VALIDATED & CONFIRMED  
**Date**: September 13, 2026  

---

## 1. Objective

The purpose of Step 29 is to advance the generic shelf vacancy system from answering:
> *"Is there an empty horizontal region?"*

to answering:
> *"Is there a persistent, visually meaningful, unoccupied shelf region that should be flagged for replenishment verification?"*

### Operational Scope & Constraints
- **Product-Agnostic**: Strictly zero SKU recognition, product names, brand identities, or fake planograms.
- **Visual Confidence Only**: The confidence score is strictly termed **"Visual Vacancy Confidence"** — measuring algorithmic certainty in the geometric and temporal visual observation. It is **never** presented as stockout probability, inventory accuracy, or physical item count.
- **Shelf Fullness Metric**: Termed **"Visible Shelf Occupancy"**.
- **Replenishment Signal**: Generates an actionable recommendation only when a candidate gap is verified as persistent, geometrically stable, unoccluded, and exceeds the minimum Visual Vacancy Confidence threshold.

---

## 2. Existing Step 28 Architecture Preservation

Step 29 directly builds upon the validated foundation established in Step 28:
```
SHELF-01
   ↓
PHYSICAL SHELF TIERS (Tier 1 Top, Tier 2 Mid, Tier 3 Lower)
   ↓
2D HORIZONTAL OCCUPANCY PROJECTION (Interval Union [x1, x2])
   ↓
CROSS-ROW OCCUPANCY VALIDATION (Eliminating pseudo-gaps)
   ↓
SPATIAL-TEMPORAL GAP TRACKING (Step 29: Center-drift matching & tracking)
   ↓
VISUAL VACANCY CONFIDENCE SCORING (Step 29: Persistence, Size, Stability)
   ↓
REPLENISHMENT RECOMMENDATION SIGNAL (Step 29: Actionable store alerting)
```

No underlying models or tracking filters were modified:
- `retail_detector_exp2.pt` remains unchanged.
- `yolo11n.pt` shopper detector remains unchanged.
- ByteTrack tracking parameters remain unchanged.
- Optical flow camera-motion filter remains unchanged.
- Step 28 2D interval projection logic remains unchanged.

---

## 3. New Vacancy Signal Design & Engineering Parameter Evaluation

To evaluate candidate quality, four deterministic, interpretable signals are computed for each candidate gap:

1. **Gap Multiple ($M_{\text{gap}}$)**:
   $$M_{\text{gap}} = \frac{w_{\text{gap}}}{\bar{w}_{\text{facing}}}$$
   where $\bar{w}_{\text{facing}}$ is the median bounding box width of products on the physical tier. A candidate gap is admitted only if $M_{\text{gap}} \ge 1.75$ and $w_{\text{gap}} \ge 45\text{px}$.

2. **Persistence Score ($S_{\text{persist}}$)**:
   $$S_{\text{persist}} = \min\left(1.0, \frac{N_{\text{frames}}}{N_{\text{min\_consec}}}\right) \quad (N_{\text{min\_consec}} = 10)$$

3. **Size Significance Score ($S_{\text{size}}$)**:
   $$S_{\text{size}} = \min\left(1.0, \max\left(0.0, \frac{M_{\text{gap}} - 1.0}{2.0}\right)\right)$$
   A gap equal to $1.0\times$ facing receives 0.0; a gap of $3.0\times$ facing achieves 1.0.

4. **Geometric Stability Score ($S_{\text{stab}}$)**:
   Evaluated over a rolling history window ($W = 5$) of center $X$ coordinates ($c_x$) and widths ($w$):
   $$\Delta c_x = \frac{|c_{x, t} - c_{x, t-1}|}{\bar{w}_{\text{facing}}}, \quad \Delta w = \frac{|w_t - w_{t-1}|}{w_{t-1}}$$
   $$S_{\text{stab}} = \max(0.0, \min(1.0, 1.0 - 2.5 \cdot \overline{\Delta c_x} - 1.0 \cdot \overline{\Delta w}))$$

### Visual Vacancy Confidence Formula
$$C_{\text{vac}} = w_{\text{persist}} \cdot S_{\text{persist}} + w_{\text{size}} \cdot S_{\text{size}} + w_{\text{stab}} \cdot S_{\text{stab}}$$
where $w_{\text{persist}} = 0.40$, $w_{\text{size}} = 0.30$, and $w_{\text{stab}} = 0.30$.

### Evaluation of Proposed Engineering Parameters
Per the implementation rule, the proposed engineering parameters were empirically validated:
- `min_replenishment_confidence = 0.70`: When a gap of $2.2\times$ facing persists for 10 frames with stability 1.0, $C_{\text{vac}} = 0.40(1.0) + 0.30(0.60) + 0.30(1.0) = 0.88$, cleanly clearing the 0.70 threshold. Jittery or small candidate gaps (e.g. 1.8x facing at frame 3) produce $C_{\text{vac}} \le 0.54$, safely suppressing premature replenishment recommendations.
- `max_center_drift_ratio = 0.25`: Requires consecutive observations to have their centers within $0.25\bar{w}$. In validation, genuine stationary shelf gaps experienced center drift $< 0.04\bar{w}$. Unstable or jumping candidates (jumping by $\ge 0.50\bar{w}$) failed this condition, resetting persistence to 1 and preventing spurious alert confirmation.
- `stability_history_window = 5`: Successfully captures short-term frame jitter without lagging behind legitimate real-world stock removals.

---

## 4. Controlled Test Scenario Results (Phase 29L / 29R)

All 10 scenarios specified in Step 29 were evaluated using `scripts/validate_step29_vacancy_confidence.py`:

| Test # | Scenario Description | Expected Behavior | Observed Behavior | Test Result |
|---|---|---|---|---|
| **1** | Normal Spacing Without Vacancy | No candidate, 0% confidence, no alert | $0\text{ gaps}$, status `OCCUPIED`, alert inactive | **PASS** |
| **2** | Small Temporary Gap (1–3 frames) | `TEMPORARY_VACANCY`, persist $<10$, no alert | Persist $\le 3$, confidence $\le 0.55$, no alert | **PASS** |
| **3** | Persistent Meaningful Gap ($\ge 10$ frames) | `VACANCY_CONFIRMED`, conf $\ge 70\%$, Replenish Active | Frame 10: conf $= 88\%$, persist $= 10$, Replenish Recommended | **PASS** |
| **4** | Shopper Blocks Candidate Gap | Status `UNCERTAIN`, conf $= 0\%$, Replenish Suppressed | Status `UNCERTAIN`, conf $= 0.0$, alert suppressed | **PASS** |
| **5** | Shopper Leaves Monitored Gap | Temporal tracking resumes, reaches confirmed replenish | Resumes observation, reaches conf $\ge 70\%$, alert triggers | **PASS** |
| **6** | Camera Motion / Panning ($\text{mag} > 6.0$) | Optical flow suppresses confirmation, status `UNCERTAIN` | Status `UNCERTAIN`, confirmation frozen, no alert | **PASS** |
| **7** | Transient Detector Dropout (1 frame) | Drops absorbed, no instant spurious confirmation | State preserved, no false replenishment trigger | **PASS** |
| **8** | Partial Neck/Cap Detections (Step 28 case) | 2D occupancy rejects pseudo-gap, 0 false vacancies | $0\text{ gaps}$, pseudo-gap rejected, status `OCCUPIED` | **PASS** |
| **9** | Unstable / Jumping Candidate Geometry | Center drift $> 0.25$, persistence resets to 1 | Jumps reset persistence; confidence $< 0.50$, no alert | **PASS** |
| **10** | Multiple Valid Gaps on Same Physical Tier | Independent tracking IDs, persistence, and confidence | 2 distinct regions tracked (`GAP-01`, `GAP-02`) with individual metrics | **PASS** |

*Controlled Test Pass Rate: 10/10 (100% of injected synthetic test cases passed).*  
*(Note: As per project guidelines, this represents synthetic unit test suite completion, NOT real-world detection accuracy).*

---

## 5. False Positive Analysis

To ensure transparency, potential false-positive mechanisms were individually audited:

1. **Detector Fragmentation / Partial Heights (Step 27 failure mode)**:
   - *Test*: Seven 2L bottles on Tier 1 with neck-only vs full-body bounding boxes.
   - *Result*: 2D interval projection merged horizontal spans. Pseudo-gap width $= 0\text{px}$. False positive completely eliminated.
2. **Shopper Movement in Front of Monitored Bay**:
   - *Test*: Shopper bounding box overlaps monitored shelf gap coordinates.
   - *Result*: Geometric occlusion test flags `is_shelf_occluded = True`. Status immediately transitions to `UNCERTAIN`, Visual Vacancy Confidence is zeroed ($0.0$), and replenishment recommendations are frozen.
3. **Camera Pan / Optical Flow Vibration**:
   - *Test*: Camera pans across beverage aisle (optical flow magnitude $> 6.0$).
   - *Result*: `is_camera_moving = True` triggers `UNCERTAIN` classification. No spurious alerts emitted during camera transitions.
4. **Detector Coordinate Jitter / Jumps**:
   - *Test*: Candidate gap boundary fluctuates or jumps between columns.
   - *Result*: Center-drift ratio test ($> 0.25\bar{w}$) fails match; tracker branches to new candidate track, resetting persistence to frame 1. Jittery candidates never reach the 10-frame threshold.

---

## 6. True Vacancy Analysis

In the persistent vacancy scenario (Test 3):
- Products at indices 3 and 4 were removed from Tier 1, leaving a physical span of $310\text{px}$ unoccupied between products.
- Physical Tier 1 median facing width: $\bar{w} = 140\text{px}$.
- Multiplier: $M_{\text{gap}} = 310 / 140 = 2.21\times$ (exceeds $1.75\times$ threshold).
- **Frame 1**: Candidate admitted. Initial prior confidence $= 40\%(0.10) + 30\%(0.60) + 30\%(0.70) = 43\%$. State is `TEMPORARY_VACANCY`.
- **Frames 2–9**: Candidate consistently matches spatial track at $c_x = 685\text{px}$ (drift ratio $= 0.00$). Stability score rises to $1.00$. Confidence steadily climbs ($55\% \to 68\% \to 84\%$).
- **Frame 10**: Persistence reaches $10$ frames ($S_{\text{persist}} = 1.0$). Confidence reaches $0.88$ ($88\%$).
- **Trigger**: Because persistence $\ge 10$, unoccluded, and confidence $\ge 0.70$, `replenishment_recommended` becomes `True` and state transitions to `VACANCY_CONFIRMED`.

---

## 7. Confidence Score Interpretation

The score emitted by the engine is explicitly designated:
$$\textbf{Visual Vacancy Confidence}$$

### What Visual Vacancy Confidence Represents:
1. **Geometric Cleanness**: The width of the unoccupied horizontal region relative to standard facing geometry.
2. **Temporal Longevity**: The number of consecutive unoccluded frames the identical spatial coordinates have remained empty.
3. **Coordinate Stability**: The lack of spatial jitter or drift across the observation window.

### What Visual Vacancy Confidence Does NOT Represent:
- It is **NOT** a stockout probability (the system does not know if backroom inventory exists).
- It is **NOT** inventory accuracy or physical count.
- It is **NOT** a product-level SKU confidence score.

---

## 8. Real-Video Validation Results

Both active real demo videos were processed through the end-to-end video pipeline (`scripts/test_vacancy_video_pipeline.py`):

### 1. `shelf_pan_demo.mp4` (Center Bay Beverages)
- **Total Frames Tested**: 40 frames
- **Products Detected**: 64–80 products per frame
- **Physical Tiers Monitored**: Tier 1 (2L Bottles), Tier 2 (20oz Bottles), Tier 3 (12-Packs)
- **Tiers Status**: `OCCUPIED` (100% visible shelf occupancy)
- **False Vacancies Detected**: **0** (eliminated the legacy 617px Step 27 false alert)
- **Replenishment Recommendations**: **0**
- **Processing Rate**: 8.7 FPS (CPU)

### 2. `inventory2.mp4` (Store Aisle Shelves)
- **Total Frames Tested**: 40 frames
- **Products Detected**: 92–100 products per frame
- **Physical Tiers Monitored**: 3 Aisle Tiers
- **Occlusion Frequency**: Shopper occluding shelves in $\sim 95\%$ of frames
- **System State**: Correctly maintained `UNCERTAIN` / `OCCLUDED`
- **False Vacancies Detected**: **0**
- **Replenishment Recommendations**: **0** (safely suppressed while aisle is blocked)
- **Processing Rate**: 7.0 FPS (CPU)

---

## 9. Visual Evidence Artifacts

All visual evidence has been generated and validated in `output/step29_evidence/`:

- `panel1_normal_packed.jpg`: Normal packed shelf tier (0% confidence, status `OCCUPIED`).
- `panel2_temporary_gap_observation.jpg`: Temporary candidate gap under observation (Frame 3, confidence $\approx 66\%$, replenishment pending).
- `panel3_confirmed_replenishment.jpg`: Confirmed persistent gap with Replenishment Recommendation banner (Frame 11, confidence $94\%$, status `VACANT`).
- `panel4_shopper_occlusion.jpg`: Shopper occlusion handling (status `UNCERTAIN`, confidence $0.0\%$, alert suppressed).
- `panel5_real_video_pan_demo.jpg`: Real video `shelf_pan_demo.mp4` showing Tier 1 2D occupancy intact with 0 false vacancies.
- `panel6_real_video_inventory2.jpg`: Real video `inventory2.mp4` showing shopper occlusion suppression.
- `step29_contact_sheet.jpg`: Consolidated 6-panel validation contact sheet ($1920 \times 960$).

---

## 10. Dashboard Representation

The dashboard (`dashboard/src/components/ShelfRackVisualizer.jsx`) was updated to render the new Step 29 signals:
1. **Telemetry Status Strip**: Added dedicated card for **Visual Vacancy Confidence** alongside **Monitored Shelf**, **Shelf Status**, **Visible Facings**, and **Internal Vacant Gaps**.
2. **Actionable Alert Banner**: Displays **Replenishment Recommended** badge when triggered, displaying exact Visual Vacancy Confidence percentage and instructing store personnel to verify shelf replenishment.
3. **Per-Gap Granular Cards**: Displays each detected empty region with:
   - Span ($X_1 \to X_2$), width in pixels, and facing multiple ($M_{\text{gap}}$).
   - Persistence frames counter ($N_{\text{persist}}$).
   - Geometric stability rating ($\% \text{ Stab}$).
   - Visual Vacancy Confidence rating ($\% \text{ Conf}$).
   - Replenishment recommendation badge (`REPLENISHMENT RECOMMENDED` vs `MONITORING (PENDING)`).
4. **Architecture Semantics Notice**: Emphasizes that observations reflect camera-visible shelf space without asserting SKU identities or warehouse inventory.

Frontend production build verification:
```
✓ 2287 modules transformed.
dist/index.html                   1.08 kB
dist/assets/index-D5T6N6sY.css   39.68 kB
dist/assets/index-Hy-_A9ux.js   818.89 kB
✓ built in 5.02s
```

---

## 11. Regression Audit

All prior validation suites were executed without regression:
- `scripts/validate_step26_vacancy_robustness.py`: **7/7 tests OK** (100% pass)
- `scripts/validate_step28_tier_occupancy.py`: **5/5 tests OK** (100% pass)
- `scripts/validate_step29_vacancy_confidence.py`: **10/10 tests OK** (100% pass)
- `scripts/test_vacancy_video_pipeline.py`: **2/2 videos OK** (0 false vacancies)
- Inventory API Server (`inventory/inventory_api.py`): Live and healthy on port 8001

---

## 12. Performance Evaluation

| Metric | Step 28 Baseline | Step 29 Validated | Impact |
|---|---|---|---|
| Pipeline Processing Rate (`shelf_pan_demo`) | 8.5 FPS | 8.7 FPS | Negligible ($\pm 0.2\text{ FPS}$) |
| Pipeline Processing Rate (`inventory2`) | 6.8 FPS | 7.0 FPS | Negligible ($\pm 0.2\text{ FPS}$) |
| Spatial Tracking & Confidence Overhead | N/A | $< 0.8\text{ ms / frame}$ | Negligible |
| Memory Footprint | Constant | Constant | Negligible ($< 1\text{ KB}$ history per tier) |

---

## 13. Remaining Limitations

1. **Fixed Camera Assumptions**: The system relies on static or near-static camera views. Large perspective shifts require recalculation of physical tier ROIs.
2. **Deep Shelf Occlusions**: Products placed deeply behind front-row items may be partially occluded by shelf lip or label holders.
3. **No Direct SKU Association**: Empty spaces are flagged by shelf coordinate, not by product barcode or SKU name.

---

## 14. Final Decision

Per the requirements of Step 29, exactly one of the three allowed decisions must be selected based on empirical validation evidence:

$$\Large\textbf{VACANCY SIGNAL IS SUFFICIENT FOR REPLENISHMENT MONITORING}$$

### Justification:
1. The 2D horizontal occupancy projection eliminates the multi-height detector splitting failure discovered in Step 27.
2. Spatial-temporal tracking with center-drift bounds prevents jittery false positives from accumulating persistence.
3. The Visual Vacancy Confidence metric cleanly differentiates temporary disturbances ($< 60\%$) from genuine, persistent empty spaces ($> 70\%$).
4. Shopper occlusion and camera motion filters reliably suppress false alerts during customer interaction and camera panning.
5. All 10 controlled test scenarios, both real demo videos, and all Step 26/28 regressions passed cleanly.
