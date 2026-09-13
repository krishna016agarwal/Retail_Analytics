# Inventory Management Architecture — New Direction

## 1. Objective

Redesign the inventory module around a capability that computer vision can reliably demonstrate without requiring prior knowledge of every shop's brands, packaging, or product catalog.

### Primary goal

Detect **empty/vacant shelf spaces** and generate an inventory alert based on the **shelf number**.

Example:

> **Shelf 3 — Empty/Vacant Space Detected**

Do **not** claim the exact product name, quantity, SKU, brand, or product identity unless a later model is explicitly capable of determining it reliably.

---

## 2. Core Principle

The system should separate inventory intelligence into stages:

### Stage 1 — Shelf Vacancy Detection

The model detects:

- shelf regions
- product-present regions
- empty/vacant regions
- persistent empty spaces that are large/relevant enough to indicate possible stock-out

Output:

```text
Shelf 1 → Empty space detected
Shelf 2 → No significant vacancy
Shelf 3 → Empty space detected
```

This is the immediate implementation target.

### Stage 2 — Similar Product Identification (Later)

Once vacancy detection works reliably, add a separate capability for recognizing visually similar/same products.

Only then should the system be able to report something like:

```text
Shelf 3
Product ID: P-014
Status: Low Stock
```

The Product ID should be generated/assigned by the system rather than assuming a universal brand catalog.

### Stage 3 — Advanced Inventory Intelligence (Optional Future)

Only after the above is reliable, consider:

- approximate product/facing counts
- low-stock estimation
- product grouping
- shelf occupancy percentage
- historical vacancy tracking
- restocking priority

These are future extensions, not requirements for the first implementation.

---

## 3. What We MUST NOT Build Now

Remove or avoid features that require information the vision model cannot reliably know.

Do NOT depend on:

- exact brand names
- exact product names
- universal SKU catalogs
- fixed product databases for every shop
- exact product quantity when products are stacked behind one another
- upper/middle/lower shelf assumptions
- fictitious planograms
- hardcoded store-specific product mappings
- claims that a product is definitely out of stock when the camera simply cannot see it
- fake/live-camera terminology for recorded demo footage

The dashboard should never manufacture product identity from a generic detector.

---

## 4. Shelf Numbering Strategy

### Default assumption

For the first version, treat the visible shelf/camera area as:

```text
Shelf 1
```

This keeps the implementation simple and makes the first demo understandable.

### Multiple shelves

The architecture should still be designed so that multiple shelf regions can be supported later.

For example:

```text
Camera 1
 ├── Shelf 1
 ├── Shelf 2
 ├── Shelf 3
 └── Shelf 4
```

A shelf should be represented by a stable region/ID, not by assumptions such as "top shelf" or "bottom shelf".

Recommended future representation:

```json
{
  "shelf_id": "SHELF-01",
  "camera_id": "CAM-01",
  "region": {
    "x1": 0,
    "y1": 0,
    "x2": 1920,
    "y2": 1080
  }
}
```

For the current implementation, one shelf is sufficient.

---

## 5. Vacancy Detection Concept

Do not treat a single empty-looking pixel region or one bad detection as an immediate stock-out alert.

Use temporal confirmation.

Conceptually:

```text
Frame
  ↓
Detect shelf / product occupancy
  ↓
Estimate vacant region
  ↓
Compare across consecutive frames
  ↓
Is vacancy persistent?
  ├── No → Ignore / wait
  └── Yes
       ↓
   Generate alert
       ↓
Shelf 1 — Empty/Vacant Space Detected
```

This prevents alerts caused by:

- a customer's hand
- a person standing in front of the shelf
- temporary occlusion
- motion blur
- detector misses
- lighting changes
- camera movement

The exact temporal threshold should be configurable rather than hardcoded into UI logic.

---

## 6. Important Distinction: Empty Space vs. Product Out of Stock

The first model should report:

> **EMPTY/VACANT SPACE DETECTED**

rather than automatically claiming:

> **PRODUCT OUT OF STOCK**

Reason:

A camera cannot always prove that a product is completely out of stock. It can observe that a shelf region that normally contains products is currently vacant.

Therefore:

```text
Vision observation:
Shelf 1 → Persistent vacant space

Business interpretation:
Potential stock-out / replenishment required
```

The dashboard can phrase the alert as:

> **Shelf 1 — Empty Space Detected**
> Potential stock-out. Replenishment verification required.

This is much safer and technically defensible.

---

## 7. Future Product ID Design

When the system becomes capable of recognizing similar products, do NOT introduce a global catalog such as:

```text
7-Up
Mountain Dew
Coke
Pepsi
...
```

Instead, create store-local product identities.

Example:

```text
Product ID: P-001
Product ID: P-002
Product ID: P-003
```

The system can learn/assign these IDs from the shop's own data.

Then an alert can become:

```text
Shelf 1
Product ID: P-003
Status: Low Stock
```

This allows the same architecture to work across many stores without requiring a manually maintained database of every brand.

---

## 8. Camera-to-Shelf Strategy

A store may have:

### Case A — One camera, one shelf

```text
CAM-01 → SHELF-01
```

Use this for the first demo.

### Case B — One camera, multiple shelves

```text
CAM-01
 ├── SHELF-01
 ├── SHELF-02
 └── SHELF-03
```

Use configured shelf regions/ROIs.

### Case C — Multiple cameras

```text
CAM-01 → SHELF-01, SHELF-02
CAM-02 → SHELF-03, SHELF-04
CAM-03 → SHELF-05
```

The architecture should support this later, but there is no need to implement multi-camera orchestration now.

---

## 9. Recommended Data Model

Keep the inventory output simple.

Example:

```json
{
  "run_id": "RUN-001",
  "camera_id": "CAM-01",
  "video_source": "inventory2.mp4",
  "shelves": [
    {
      "shelf_id": "SHELF-01",
      "vacancy_detected": true,
      "vacancy_score": 0.82,
      "status": "VACANT"
    }
  ],
  "alerts": [
    {
      "alert_id": "ALERT-001",
      "shelf_id": "SHELF-01",
      "type": "SHELF_VACANCY",
      "severity": "HIGH",
      "message": "Empty/vacant shelf space detected."
    }
  ]
}
```

Do not add product names or product quantities to this schema until the model can genuinely provide them.

---

## 10. Dashboard Direction

The dashboard should become much simpler.

### Main view

Show:

- Camera playback
- Shelf number
- Shelf status
- Vacancy indicator
- Alert count
- Timestamp/run information

Example:

```text
EDGE CAMERA — ANALYSIS PLAYBACK

Shelf 1
Status: VACANT
⚠ Empty Space Detected

Potential stock-out — verification required.
```

### Avoid

Remove or redesign:

- 7-SKU catalog cards
- brand/product names
- fake product counts
- fictitious 8-tier planogram
- "stable facings" if the current model cannot reliably support that concept
- product-specific restock buttons
- product-specific alerts
- upper/lower shelf labels
- claims of exact inventory quantity

The UI should reflect what the model actually knows.

---

## 11. Alert Logic

Recommended initial state machine:

```text
NORMAL
  ↓
Vacancy observed
  ↓
TEMPORARY_VACANCY
  ↓
Vacancy persists for N frames / seconds
  ↓
VACANCY_CONFIRMED
  ↓
ALERT GENERATED
```

If the shelf becomes occupied again:

```text
VACANCY_CONFIRMED
  ↓
Products detected again
  ↓
RESTORED
```

This gives us meaningful events instead of noisy frame-by-frame alerts.

---

## 12. What Counts as a Vacancy?

The implementation should define vacancy using visual evidence rather than product identity.

Possible signals:

- shelf/background visible where product occupancy is expected
- continuous empty region between occupied product regions
- occupancy below a configurable threshold
- persistent vacant region over multiple frames

The exact method should be selected based on the current detector and available training/demo data.

Do not assume that every dark/bright region is an empty shelf.

---

## 13. Architecture

Recommended high-level flow:

```text
Camera / Demo Video
        ↓
Frame Extraction
        ↓
Person / Object Detection
        ↓
Occlusion Filtering
        ↓
Shelf Region Detection / ROI
        ↓
Shelf Occupancy Analysis
        ↓
Temporal Confirmation
        ↓
Vacancy Event
        ↓
Inventory Alert Engine
        ↓
API / JSON Report
        ↓
Dashboard
```

The existing detection/tracking pipeline should be reused wherever it is useful. Do not rewrite working model components unnecessarily.

---

## 14. Implementation Priority

### Phase 1 — Foundation

1. Remove dependency on the hardcoded SKU catalog from the inventory UI/report.
2. Remove fictitious shelf tiers.
3. Define a generic shelf representation.
4. Start with `SHELF-01`.
5. Establish one authoritative inventory run/report.

### Phase 2 — Vacancy Detection

6. Detect occupied vs vacant shelf regions.
7. Add temporal confirmation.
8. Generate shelf-level vacancy alerts.
9. Test against occlusions and people crossing the shelf.

### Phase 3 — Dashboard

10. Show camera playback.
11. Show Shelf 1 status.
12. Show vacancy alerts.
13. Show run/video synchronization.
14. Remove unsupported SKU/product metrics.

### Phase 4 — Validation

15. Test with multiple frames.
16. Test with people blocking shelves.
17. Test when products are temporarily hidden.
18. Test when an actually vacant region persists.
19. Test consecutive video runs.
20. Confirm dashboard data exactly matches the active run.

### Phase 5 — Future Product Intelligence

Only after vacancy detection is stable:

21. Add store-local Product IDs.
22. Add similar-product grouping.
23. Add low-stock estimation.
24. Add product-level alerts.

---

## 15. Non-Modification Rule

Unless required by the new vacancy architecture:

- Do not modify the crowd/queue pipeline.
- Do not modify unrelated application features.
- Do not replace detector weights without evidence that it is necessary.
- Do not change working tracking logic unnecessarily.
- Do not introduce a universal product catalog.
- Do not add fake inventory intelligence just to make the dashboard look richer.

Every displayed metric must have a clear source in the actual model/pipeline output.

---

## 16. Definition of Done for the New Inventory Architecture

The first version is successful when:

1. A video is selected.
2. The selected video is actually analyzed by the backend.
3. A unique run ID is created.
4. The model processes the selected frames.
5. The system identifies the configured shelf region.
6. Persistent empty/vacant space is detected.
7. The backend generates a shelf-level vacancy event.
8. The dashboard displays the same run's result.
9. The alert says the shelf number, not an invented product name.
10. Switching videos and rerunning changes the actual analysis output.
11. Previous run data cannot leak into the new run.
12. Temporary occlusion does not immediately create a false stock-out alert.
13. The architecture can later support multiple shelves/cameras and store-local Product IDs.

---

# Immediate Goal

**Do not implement future Product ID/SKU recognition yet.**

First make this work reliably:

```text
VIDEO
  ↓
SHELF 1
  ↓
EMPTY SPACE DETECTION
  ↓
TEMPORAL CONFIRMATION
  ↓
"SHELF 1 — EMPTY SPACE DETECTED"
  ↓
DASHBOARD ALERT
```

Once this is working and validated, we can build the Product ID / similar-product layer on top of it.
