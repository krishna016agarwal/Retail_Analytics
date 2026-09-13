# STEP 27: Real-Video Vacancy Validation & Threshold Calibration Report

**Evaluation Date**: 2026-09-12  
**Dataset Footage**:
- `inventory_data/demo_videos/shelf_pan_demo.mp4` (1762×2350, Portrait, 75 frames, 25.0 FPS)
- `videos/inventory2.mp4` (3864×2192, 4K Landscape, 210 frames, 15.0 FPS)
**Evaluation Suite**: [`scripts/visualize_step27_real_vacancies.py`](file:///d:/RetailShop/Retail_Analytics/scripts/visualize_step27_real_vacancies.py)  
**Visual Artifacts**: `output/step27_evidence/`

---

## 1. Objective

Validate whether the generic shelf-vacancy architecture established in Steps 25 and 26 functions reliably on **real demo videos** rather than only synthetic or unit-test scenarios. 

Specifically determine:
> *"Do the gaps detected by the current algorithm correspond to genuine empty shelf regions, or are they artifacts of shelf geometry, camera framing, perspective distortion, or row clustering?"*

---

## 2. Videos Tested

| Video Source | Resolution | Aspect Ratio | Frame Count | Key Visual Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| `shelf_pan_demo.mp4` | 1762 × 2350 | Portrait | 75 frames | Clean, unobstructed horizontal panning across a 4-tier beverage rack (2L bottles, 20oz bottles, 12-pack boxes). Zero shopper occlusions. |
| `inventory2.mp4` | 3864 × 2192 | 4K Landscape | 210 frames | Wide grocery store aisle with multiple shelf racks, merchandise, support pillars, and persistent shopper/worker movements. |

---

## 3. Current Algorithm Description

The V1 Generic Shelf Vacancy Detection pipeline operates as follows:
1. **Shelf ROI Filtering**: Normalized coordinates from `configs/shelf_vacancy_config.json` filter raw product detections to `SHELF-01`.
2. **Shopper Occlusion Check**: Person detections from `yolo11n.pt` are checked for overlap against `SHELF-01` and individual gap regions.
3. **1D Centroid Row Clustering**: Products are sorted vertically and grouped into horizontal rows when vertical centers satisfy:
   $$|cy - \overline{cy}_{\text{row}}| \le \tau_{\text{row}} \times \bar{h}_{\text{row}} \quad (\tau_{\text{row}} = 0.45)$$
4. **Adaptive Gap Detection**: In each row, products are sorted horizontally. An internal gap $G_i = x_{i+1, 1} - x_{i, 2}$ is flagged if:
   $$G_i \ge \max(1.75 \times \bar{w}_{\text{row}}, 45\text{px}) \quad \text{and} \quad G_i > 0.40 \times \bar{w}_{\text{row}}$$
5. **Boundary Exclusion**: Outer spaces ($x=0$ to first product, and last product to $x=W$) are ignored to distinguish camera panning/out-of-view effects from true vacancies.
6. **Temporal Confirmation**: Gaps must persist unoccluded across $N=10$ consecutive frames to trigger:
   $$\text{"SHELF 1 — EMPTY SPACE DETECTED"}$$

---

## 4. Detection Statistics

### A. Frame-by-Frame Summary

| Video | Frames Processed | Logged Gaps | Occluded Frames | Confirmed Vacancy State? | Active Alert Emitted? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `shelf_pan_demo.mp4` | 75 / 75 | 287 | 0 / 75 (0%) | **`VACANCY_CONFIRMED`** (Frames 10–75) | **YES** (`SHELF 1 — EMPTY SPACE DETECTED`) |
| `inventory2.mp4` | 75 / 210 | 1023 | 74 / 75 (98.7%) | **`TEMPORARY_VACANCY`** (Frozen) | **NO** (0 alerts fired) |

### B. Gap Magnitude Distribution in `shelf_pan_demo.mp4`
- Total internal gap instances logged across 75 frames: **287**
- Gaps with width $\ge 300\text{px}$: **193 instances**
- Largest internal gap observed: **1,229.0 px** (ROW-03, Frame 24)
- Average gap width across all frames: **496.8 px**

---

## 5. Investigation of the ~618px Gap in `shelf_pan_demo.mp4`

In Step 25, a large gap of $\approx 618\text{px}$ ($\approx 3.91\times$ to $4.58\times$ median width) was observed on Frame 0 and persistently confirmed through subsequent frames.

### Empirical Dissection:
- **Location**: `ROW-02`, pixel span $X \in [823, 1440]$, width = $617.0\text{px}$, vertical span $Y \in [345, 793]$.
- **Physical Ground Truth** (verified via `raw_shelf_pan_f0.jpg`):
  - In physical reality, this shelf tier holds 2-liter soda bottles.
  - Between $X=823$ and $X=1440$, **four 2-liter green 7UP bottles are physically sitting on the shelf, completely flush and fully stocked**.
  - **There is zero empty space on this shelf.**

### Root Cause Analysis:
1. **Partial Detector Bounding Boxes**: `retail_detector_exp2.pt` detected the bottles on the left ($X \in [292, 823]$) as full-body bottles ($h \approx 300\text{px}, cy \approx 590\text{px}$), but detected the four adjacent bottles in the middle ($X \in [994, 1424]$) by their top necks/caps only ($h \approx 200\text{px}, cy \approx 424\text{px}$).
2. **1D Centroid Clustering Failure**: Because $|590 - 424| = 166\text{px} > 0.45 \times 250\text{px} = 112.5\text{px}$, the clustering algorithm split a single physical shelf into two distinct pseudo-rows:
   - `ROW-01` ($y_{\text{mean}} = 444.3\text{px}$): assigned the four neck/cap detections.
   - `ROW-02` ($y_{\text{mean}} = 583.2\text{px}$): assigned the left bottles ($X=292..823$) and jumped directly to one full-height bottle on the right ($X=1440..1556$).
3. **Row Isolation Artifact**: Gap detection was performed independently per row without cross-row spatial validation. `ROW-02` calculated the distance between $X=823$ and $X=1440$ as an "empty gap" of $617\text{px}$, completely blind to the fact that four bottles were physically detected in `ROW-01` inside that exact horizontal span.

---

## 6. Visual Evidence Locations

Visual evidence generated by [`scripts/visualize_step27_real_vacancies.py`](file:///d:/RetailShop/Retail_Analytics/scripts/visualize_step27_real_vacancies.py) is stored in `output/step27_evidence/`:

1. **Contact Sheet**:
   - `output/step27_evidence/step27_contact_sheet.jpg` (Consolidated 12-panel multi-grid showing temporal progression from `NORMAL` $\to$ `TEMPORARY_VACANCY` $\to$ `VACANCY_CONFIRMED`).
2. **Raw Ground Truth**:
   - `output/step27_evidence/raw_shelf_pan_f0.jpg` (Unannotated frame confirming 100% full stocking of 2L bottles).
3. **Annotated Frame Evidence**:
   - `shelf_pan_demo_f000_TEMPORARY_VACANCY.jpg` (Initial candidate detection).
   - `shelf_pan_demo_f010_VACANCY_CONFIRMED.jpg` (Frame 10 confirmation with red overlay and banner).
   - `inventory2_f000_NORMAL.jpg` (Orange shopper occlusion bounding boxes).
4. **Structured Dataset**:
   - `output/step27_evidence/step27_gap_dataset.json` (Complete JSON log of gap metrics and sweep evaluations).

---

## 7. True Vacancy Examples

- **Controlled Synthetic Validation (Step 26)**: A persistent $150\text{px}$ internal gap ($3.0\times$ median width) was confirmed on Frame 10 with 100% precision.
- **Partial Facing Recess (`shelf_pan_demo.mp4` Shelf 2)**: On the second physical tier (20oz bottles), behind price tag `$4.69`, front facings were pushed back into the shelf cavity ($X \in [446, 693]$, width = $247\text{px}$, $2.08\times$ median width). This represents a valid front-facing void where replenishment could pull stock forward.

---

## 8. False-Positive Examples

1. **Top Shelf 2L Bottles (`shelf_pan_demo.mp4`, ROW-02, $X \in [823, 1440]$)**:
   - Detected as a $617\text{px}$ ($3.91\times$ median) gap.
   - **Ground Truth**: Fully stocked with 4 bottles. Cause: Split-row clustering.
2. **Bottom Shelf 12-Packs (`shelf_pan_demo.mp4`, ROW-08, $X \in [1104, 1577]$)**:
   - Detected as a $473\text{px}$ ($2.61\times$ median) gap.
   - **Ground Truth**: Fully stocked with white Diet 7UP 12-pack boxes. Cause: Lower detector contrast on white packaging combined with vertical row fragmentation.

---

## 9. Perspective & Shelf Geometry Issues

1. **Monolithic ROI vs. Multi-Tier Rack**:
   - In `configs/shelf_vacancy_config.json`, `SHELF-01` is defined as a single ROI covering $Y \in [0.15, 0.85]$.
   - In physical reality, this ROI encompasses **4 separate physical shelf tiers** with vastly different product types and heights ($350\text{px}$ bottles down to $180\text{px}$ boxes).
2. **Unstable Row Partitioning**:
   - The unguided 1D clustering produces between 5 and 8 pseudo-rows depending on small bounding box variations.
   - Because products on the same physical shelf tier vary in height or detector bounding box tightness, they get partitioned into adjacent pseudo-rows, generating artificial cross-row vacancies.

---

## 10. Shopper Occlusion Results

- Tested on `videos/inventory2.mp4` across 75 frames.
- **Results**:
  - Person detections overlapped the shelf region in **74 of 75 frames (98.7%)**.
  - Every detected candidate gap intersecting a person box was marked `is_occluded_by_person = True`.
  - Frame status was held in `UNCERTAIN` / `NORMAL`.
  - Temporal confirmation counter remained frozen at 0.
  - **Zero false stock-out alerts were emitted while shoppers were present.**
  - Confirms 100% adherence to occlusion safety specifications.

---

## 11. Camera-Motion & Panning Results

- Tested on `inventory_data/demo_videos/shelf_pan_demo.mp4` (optical flow motion $\approx 1.5 - 2.5\text{px/frame}$).
- **Boundary Handling**:
  - Outer margins ($3\%$ frame boundary margin) successfully suppressed false gaps at the frame edges as products entered from the right and exited to the left.
  - Panning across frame edges did not trigger "Out of View" false vacancies.
- **Motion Threshold**:
  - Panning optical flow ($2.0\text{px/frame}$) stayed below the severe motion cutoff ($6.0\text{px/frame}$), allowing continuous tracking without false uncertainty aborts.

---

## 12. Threshold Sensitivity Sweep

Evaluated multipliers $\mu \in [1.50, 1.60, 1.75, 1.90, 2.00, 2.25]$ on `shelf_pan_demo.mp4` (40 frames):

| Multiplier ($\times \bar{w}$) | Unoccluded Gaps | Alert Frames (out of 40) | Avg Gap Width (px) | Max Gap Width (px) |
| :--- | :--- | :--- | :--- | :--- |
| **1.50×** | 220 | 31 | 444.6 | 1229.0 |
| **1.60×** | 200 | 31 | 468.1 | 1229.0 |
| **1.75× (Current)** | 179 | 31 | 496.8 | 1229.0 |
| **1.90×** | 169 | 31 | 512.9 | 1229.0 |
| **2.00×** | 161 | 31 | 525.0 | 1229.0 |
| **2.25×** | 160 | 31 | 526.7 | 1229.0 |

### Crucial Insight:
Notice that across ALL multipliers from $1.50\times$ to $2.25\times$, the number of **Alert Frames is invariant at exactly 31 frames** (Frames 10 through 40).
**Threshold tuning cannot fix this false positive**: the geometric pseudo-gap ($617\text{px}$) is $3.9\times$ to $4.5\times$ the median width, dwarfing the $1.75\times$ to $2.25\times$ threshold.

---

## 13. Temporal Confirmation Evaluation

- Current configuration: $N_{\text{confirm}} = 10\text{ frames}$, $N_{\text{recover}} = 5\text{ frames}$.
- **Durability**: 10 frames ($0.40\text{s}$ at $25\text{ FPS}$) effectively absorbs transient 1–3 frame detector dropouts.
- **Occlusion Freeze**: In `inventory2.mp4`, occlusion correctly froze the counter indefinitely.
- **Limitation**: When a geometric pseudo-gap is persistently present across consecutive frames (as in `shelf_pan_demo.mp4`), temporal confirmation reinforces the error rather than filtering it out.

---

## 14. Dashboard Synchronization Verification

- Verified dual-way video execution:
  - `shelf_pan_demo.mp4` $\leftrightarrow$ `inventory2.mp4`.
- Frontend accurately displays `Visible Shelf Occupancy` (never "Inventory Quantity" or "Stock Level").
- Verified `SHELF-01` card renders alert only when backend confirms vacancy.
- Sequential `run_id` allocation and atomic report swapping confirmed.
- Zero legacy SKU/brand fields displayed.

---

## 15. Regression Test Results

- [`scripts/validate_step26_vacancy_robustness.py`](file:///d:/RetailShop/Retail_Analytics/scripts/validate_step26_vacancy_robustness.py): **Ran 7 tests in 5.488s — ALL PASSED (OK)**.
- [`scripts/test_vacancy_video_pipeline.py`](file:///d:/RetailShop/Retail_Analytics/scripts/test_vacancy_video_pipeline.py): **Completed 40 frames across both videos — PASSED**.
- Core YOLO detector, ByteTrack configuration, and crowd/queue modules remain 100% untouched.

---

## 16. Summary Scenario Matrix

| Scenario | Observed Behavior | Correct? | Action |
| :--- | :--- | :--- | :--- |
| **Normal product spacing** | Gaps $\le 0.40\bar{w}$ ignored; spacing between adjacent cans/bottles does not trigger candidates. | **YES** | Retain normal spacing tolerance. |
| **Genuine empty space (synthetic)** | Persistent void $\ge 1.75\bar{w}$ confirmed at frame 10; alert triggered. | **YES** | Retain gap detection mechanics. |
| **Large natural gap / Pseudo-gap (618px)** | Detected $617\text{px}$ gap on top shelf of `shelf_pan_demo.mp4`, but 4 bottles are physically present. Partial bboxes split single shelf into 2 rows, causing cross-row phantom gap. | **NO (False Vacancy)** | **Requires architectural solution (2D spatial validation / physical tier ROIs).** |
| **Shopper occlusion** | Shoppers in 74/75 frames tagged as `OCCLUDED / UNCERTAIN`; confirmation counter frozen; 0 false alerts. | **YES** | Retain shopper occlusion freeze. |
| **Camera panning** | Boundary margin ($3\%$) prevents edge exits from triggering false vacancies; motion flow tracked. | **YES** | Retain boundary margin suppression. |
| **Detector bounding box variance** | Variations in detection heights (neck vs full bottle) split items on same shelf into different rows. | **NO (Vulnerable)** | Implement 2D horizontal span projection or base-aligned clustering. |
| **Perspective distortion** | Multi-tier rack inside single ROI causes vertical perspective tilt and height variation. | **NO (Vulnerable)** | Define distinct physical shelf tier ROIs rather than single monolithic rack ROI. |

---

## 17. Final Decision & Recommendation

### **ARCHITECTURE CHANGE REQUIRED**

### Detailed Architectural Justification:

1. **Why "CURRENT APPROACH IS SUFFICIENT" is rejected**:
   On real footage (`shelf_pan_demo.mp4`), the current algorithm persistently flags a fully stocked 2-liter soda shelf as `VACANCY_CONFIRMED` and emits `"SHELF 1 — EMPTY SPACE DETECTED"`. A production retail system cannot alert store associates to restock a shelf that is physically full.

2. **Why "MINOR TUNING REQUIRED" is rejected**:
   Empirical threshold sweeps from $1.50\times$ to $2.25\times$ demonstrated that the alert triggers on **31 out of 40 frames across every single threshold value**. Increasing `min_gap_multiplier` cannot solve a $3.9\times$ pseudo-gap without destroying sensitivity to real vacancies. Adjusting `row_vertical_tolerance_ratio` merely shifts the row-splitting artifact from the top shelf to lower shelves.

3. **Required Architectural Enhancements for Next Step**:
   - **Physical Shelf Tier Partitioning**: Replace the monolithic whole-rack ROI with separate per-shelf ROIs (`SHELF-01-TIER1`, `SHELF-01-TIER2`, etc.) or base-aligned ($y_2$ contact line) shelf clustering.
   - **2D Cross-Row Spatial Occupancy Validation**: Before an internal gap $[x_1, x_2]$ in row $r$ is declared vacant, verify that no detected product in vertically overlapping rows occupies that horizontal span $[x_1, x_2]$. If a product's horizontal footprint exists within that span, the pseudo-gap is automatically discarded.
