STEP 29 — VACANCY CONFIDENCE & REPLENISHMENT SIGNAL VALIDATION

We have completed Step 28 of the Retail Analytics inventory system.

Step 28 replaced the previous monolithic shelf ROI + 1D row clustering architecture with:

    SHELF
      ↓
    PHYSICAL SHELF TIER
      ↓
    2D HORIZONTAL OCCUPANCY
      ↓
    MERGED OCCUPIED SPANS
      ↓
    VACANCY CANDIDATE
      ↓
    SHOPPER / OCCLUSION FILTER
      ↓
    CAMERA-MOTION FILTER
      ↓
    TEMPORAL CONFIRMATION
      ↓
    OCCUPIED / VACANT / UNCERTAIN

Step 28 successfully eliminated the known ~617px false vacancy caused by detector row fragmentation in:

inventory_data/demo_videos/shelf_pan_demo.mp4

The current architecture uses physical shelf tiers and 2D horizontal occupancy projection.

IMPORTANT:
Do NOT add SKU recognition, product classification, brand recognition, product names, exact inventory counting, or a global product catalog.

The purpose of this step is to make the existing GENERIC SHELF VACANCY signal more meaningful and reliable for replenishment monitoring.

==================================================
STEP 29 OBJECTIVE
==================================================

The current system essentially answers:

    "Is there an empty horizontal region?"

Step 29 should improve this into:

    "Is there a persistent, visually meaningful, unoccupied shelf region
     that should be flagged for replenishment verification?"

We are NOT claiming:

    "Product X is out of stock."

We ARE allowed to claim:

    "Persistent empty shelf space detected on Shelf/Tier X."

The system must remain product-agnostic.

==================================================
CORE DESIGN PRINCIPLE
==================================================

Do NOT replace the current geometric architecture.

Build on:

Physical Tier
    ↓
2D Occupancy
    ↓
Vacancy Candidate
    ↓
Temporal Confirmation

The new layer should evaluate the QUALITY and SIGNIFICANCE of a vacancy candidate.

Conceptually:

    2D Occupancy Candidate
             ↓
       ┌───────────────┐
       │ Is it visible?│
       └───────┬───────┘
               ↓
       Shopper / Occlusion?
               ↓
       Camera Motion?
               ↓
       Persistent?
               ↓
       Meaningful size?
               ↓
       Stable geometry?
               ↓
       Vacancy Confidence
               ↓
    ┌──────────┼──────────┐
    ↓          ↓          ↓
  NORMAL    UNCERTAIN   VACANCY
                         ↓
               REPLENISHMENT ALERT

Do not create a fake "probability that product is out of stock."

Any confidence score must represent confidence in the VISUAL VACANCY OBSERVATION.

==================================================
PHASE 29A — INSPECT CURRENT ARCHITECTURE
==================================================

Before making changes, inspect:

1. inventory/shelf_vacancy.py
2. inventory/inventory_report.py
3. inventory/inventory_api.py
4. configs/shelf_vacancy_config.json
5. scripts/validate_step28_tier_occupancy.py
6. scripts/visualize_step28_tier_occupancy.py
7. scripts/validate_step26_vacancy_robustness.py
8. dashboard/src/components/ShelfRackVisualizer.jsx
9. dashboard/src/components/InventoryDashboard.jsx
10. inventory_management_architecture.md
11. STEP_28_TIER_OCCUPANCY_VALIDATION.md

Understand:

- current vacancy candidate representation
- gap width calculation
- tier occupancy calculation
- temporal state machine
- shopper occlusion handling
- camera-motion handling
- current alert generation
- report schema
- dashboard representation

Do NOT modify anything before understanding the existing flow.

==================================================
PHASE 29B — DEFINE VACANCY SIGNALS
==================================================

Introduce a small number of interpretable signals.

At minimum consider:

1. gap_width_px
2. gap_width_ratio

where:

    gap_width_ratio =
        gap_width / representative_product_width

3. visible_shelf_occupancy
4. temporal_persistence
5. occlusion status
6. camera-motion status
7. geometric stability across frames

Do NOT create dozens of arbitrary features.

Every signal must have an explainable meaning.

==================================================
PHASE 29C — VACANCY SIGNIFICANCE
==================================================

A large gap should generally be more significant than a tiny gap.

However:

DO NOT simply use:

    "large gap = vacancy"

because Step 27 demonstrated that large apparent gaps can be detector/geometry artifacts.

The decision must remain downstream of:

- physical tier partitioning
- 2D occupancy
- occlusion handling
- camera-motion handling

Evaluate whether the current gap threshold:

    max(1.75 × representative width, 45px)

is still appropriate after Step 28.

Do not change it automatically.

Run evidence-based comparisons first.

==================================================
PHASE 29D — TEMPORAL PERSISTENCE
==================================================

Retain the existing temporal state machine:

NORMAL
    ↓
TEMPORARY_VACANCY
    ↓
VACANCY_CONFIRMED
    ↓
RESTORED

Current configuration:

temporal_confirmation_frames = 10
temporal_recovery_frames = 5

Do NOT remove this.

Evaluate whether persistent visibility provides sufficient evidence.

Test:

CASE A:
Gap appears for 1–3 frames.

Expected:
TEMPORARY_VACANCY → NORMAL
No alert.

CASE B:
Gap appears for 5–9 consecutive frames.

Expected:
No confirmed vacancy yet.

CASE C:
Gap persists for >=10 valid, unoccluded frames.

Expected:
VACANCY_CONFIRMED.

CASE D:
Gap disappears after confirmation.

Expected:
RESTORED according to existing recovery logic.

==================================================
PHASE 29E — GEOMETRIC STABILITY
==================================================

A candidate should not be considered highly reliable if its geometry jumps wildly between frames.

Track, where practical:

- gap center X
- gap width
- tier ID
- overlap/occupancy state

Evaluate whether the candidate remains approximately in the same physical location.

Example:

Frame 1:
gap [800, 1000]

Frame 2:
gap [803, 998]

Frame 3:
gap [798, 1005]

This is geometrically stable.

But:

Frame 1:
gap [800, 1000]

Frame 2:
gap [1200, 1500]

Frame 3:
gap [500, 750]

should NOT be treated as one stable physical vacancy.

Do not introduce an unnecessarily complicated tracker.

Use the existing temporal machinery where possible.

==================================================
PHASE 29F — SHOPPER OCCLUSION
==================================================

Preserve the existing shopper protection exactly.

If a person overlaps the candidate vacancy region:

    status = UNCERTAIN

and:

    confirmation counter = FROZEN

Do NOT increase vacancy confidence while the shelf is visually blocked.

Test:

NORMAL
→ shopper arrives
→ candidate gap becomes partially hidden
→ UNCERTAIN
→ counter freezes
→ shopper leaves
→ visibility returns
→ analysis resumes

No false replenishment alert should occur solely because of shopper presence.

==================================================
PHASE 29G — CAMERA MOTION
==================================================

Preserve Step 28 camera-motion handling.

A shelf region leaving the field of view is NOT inventory vacancy.

Test:

- camera pans left
- camera pans right
- product exits frame
- product enters frame
- shelf boundary moves relative to frame

Do not increase vacancy confidence during:

- major camera motion
- boundary/FOV effects
- insufficient shelf visibility

If the existing system already marks such frames UNCERTAIN, preserve that behavior.

==================================================
PHASE 29H — PARTIAL / OCCLUDED PRODUCT DETECTIONS
==================================================

Continue testing the Step 27 failure pattern.

Products may appear as:

- full-body detections
- neck/cap detections
- partial body detections
- different bounding-box heights
- slightly shifted bounding boxes

The system must NOT interpret these detector changes as inventory removal.

The Step 28 2D occupancy layer remains the primary geometric protection.

==================================================
PHASE 29I — VACANCY CONFIDENCE
==================================================

If a confidence score is introduced, it must be:

    VISUAL VACANCY CONFIDENCE

NOT:

    STOCKOUT PROBABILITY

NOT:

    PRODUCT AVAILABILITY PROBABILITY

NOT:

    INVENTORY ACCURACY

A possible interpretable formulation is a bounded score derived from:

- geometric validity
- temporal persistence
- visibility
- occlusion absence
- camera-motion absence
- gap significance
- geometric stability

However:

DO NOT blindly implement this exact formula.

First inspect the current data.

Avoid double-counting the same evidence.

The score should be deterministic and explainable.

Example interpretation:

0.00–0.39:
    weak candidate

0.40–0.69:
    moderate candidate

0.70–1.00:
    strong visual vacancy

These ranges are examples only.

Do NOT choose thresholds just to make the demo produce attractive numbers.

If a confidence score does not materially improve the system, do NOT add it.

==================================================
PHASE 29J — REPLENISHMENT SIGNAL
==================================================

Separate:

VACANCY DETECTION

from:

REPLENISHMENT ALERT

A vacancy candidate should NOT immediately create an alert.

The alert should occur only after:

1. Valid physical tier
2. Valid 2D occupancy gap
3. No shopper occlusion
4. No significant camera motion
5. Sufficient temporal persistence
6. Geometrically stable candidate
7. Meaningful vacancy

Then emit:

    SHELF/TIER — PERSISTENT EMPTY SPACE DETECTED

Suggested wording:

Title:
    "SHELF 1 — PERSISTENT EMPTY SPACE DETECTED"

Message:
    "Persistent unoccupied shelf space detected on Tier 2. Replenishment verification recommended."

Do NOT say:

    "Coca-Cola is out of stock."

Do NOT say:

    "7 units remaining."

Do NOT say:

    "SKU 123 is unavailable."

==================================================
PHASE 29K — MULTIPLE VACANCIES
==================================================

A physical shelf tier may contain multiple independent empty regions.

Example:

[PRODUCT][PRODUCT]    [PRODUCT]       [PRODUCT][PRODUCT]
                    GAP-01
                                  GAP-02

The system should be able to represent multiple vacancy regions if the existing architecture supports this cleanly.

Do NOT merge unrelated gaps merely to simplify the UI.

Each vacancy region should have:

- region ID
- tier ID
- x1
- x2
- width
- persistence
- status

However, do not over-engineer region tracking if it is unnecessary for the current demo.

==================================================
PHASE 29L — IMPORTANT REAL-WORLD SCENARIOS
==================================================

Create controlled validation scenarios for:

TEST 1 — Normal spacing

Expected:
NORMAL
No alert.

TEST 2 — Small temporary gap

Expected:
TEMPORARY_VACANCY
Then NORMAL.
No alert.

TEST 3 — Persistent meaningful gap

Expected:
VACANCY_CONFIRMED.
Replenishment alert.

TEST 4 — Shopper blocks gap

Expected:
UNCERTAIN.
Counter frozen.
No false alert.

TEST 5 — Shopper leaves

Expected:
Visibility restored.
Correct state resumes.

TEST 6 — Camera panning

Expected:
No false vacancy.

TEST 7 — Detector dropout

Expected:
No false persistent vacancy.

TEST 8 — Partial product detection

Expected:
2D occupancy protects against phantom gap.

TEST 9 — Gap geometry jumps

Expected:
No false stable vacancy.

TEST 10 — Multiple valid gaps

Expected:
Each valid region represented independently where supported.

==================================================
PHASE 29M — USE REAL VIDEO
==================================================

Run the new validation on:

1. inventory_data/demo_videos/shelf_pan_demo.mp4
2. videos/inventory2.mp4

Specifically verify:

shelf_pan_demo.mp4:

- previously failing 617px region remains OCCUPIED
- no new false alert appears
- camera panning remains safe
- valid vacancy candidates are distinguishable from geometry artifacts

inventory2.mp4:

- shopper occlusion remains safe
- no vacancy alert is created behind shoppers
- temporal state remains stable
- system resumes correctly after visibility returns

==================================================
PHASE 29N — DO NOT OVERFIT TO CURRENT VIDEOS
==================================================

This is extremely important.

Do NOT introduce rules such as:

    "ignore gaps around X=823"

or:

    "ignore Tier 1 gaps larger than 600px"

or:

    "ignore this exact shelf"

or:

    "if video == shelf_pan_demo.mp4 then..."

No video-specific hacks.

Any threshold or rule must be generic and geometrically explainable.

==================================================
PHASE 29O — CONFIGURATION
==================================================

If new parameters are required, place them in:

configs/shelf_vacancy_config.json

Keep them interpretable.

Possible configuration categories:

- vacancy significance
- temporal persistence
- geometric stability
- minimum visibility
- occlusion handling

Do NOT create dozens of magic numbers.

Every new parameter must have:

- name
- meaning
- default
- reason for value

==================================================
PHASE 29P — DASHBOARD
==================================================

Update the dashboard only where necessary.

Keep the current generic design.

The dashboard should clearly distinguish:

NORMAL
UNCERTAIN
VACANCY DETECTED

If a confidence value is displayed, label it:

    "Visual Vacancy Confidence"

Never:

    "Inventory Confidence"
    "Stock Accuracy"
    "Stockout Probability"

For an alert show:

    SHELF 1
    TIER 2
    PERSISTENT EMPTY SPACE

and optionally:

    Visual Vacancy Confidence: XX%

Do not show:

- brand names
- product names
- SKUs
- exact quantities
- fake planograms

Keep:

    Visible Shelf Occupancy

as the occupancy label.

==================================================
PHASE 29Q — VISUAL EVIDENCE
==================================================

Create:

scripts/visualize_step29_vacancy_confidence.py

Generate annotated evidence for:

1. Normal shelf
2. Temporary gap
3. Confirmed vacancy
4. Shopper occlusion
5. Camera motion
6. Detector dropout
7. Partial detection
8. Multiple vacancy regions if present

Each frame should show:

- physical tier boundary
- occupied spans
- vacancy candidate
- vacancy status
- gap width
- gap ratio
- persistence
- confidence if implemented
- occlusion state
- camera-motion state

Do not display product names.

Store outputs in:

output/step29_evidence/

Create a contact sheet:

output/step29_evidence/step29_contact_sheet.jpg

==================================================
PHASE 29R — VALIDATION SCRIPT
==================================================

Create:

scripts/validate_step29_vacancy_confidence.py

It should test:

1. Normal spacing
2. Temporary gap
3. Persistent vacancy
4. Shopper occlusion
5. Camera motion
6. Detector dropout
7. Partial detections
8. Geometric instability
9. Multiple gaps
10. Step 28 regression

Report:

- total tests
- passed
- failed
- false vacancy alerts
- confirmed vacancies
- uncertain frames
- regression status

Do not report "accuracy %" unless labeled ground truth exists.

==================================================
PHASE 29S — REGRESSION
==================================================

Run:

- Step 26 robustness suite
- Step 28 tier occupancy suite
- Step 29 suite
- existing video pipeline tests
- API/report tests
- frontend build

Verify:

- YOLO unchanged
- ByteTrack unchanged
- shopper detector unchanged
- camera-motion handling preserved
- physical tier architecture preserved
- 2D occupancy preserved
- temporal confirmation preserved
- run synchronization preserved

Do not modify unrelated crowd/queue modules.

==================================================
PHASE 29T — PERFORMANCE
==================================================

Measure:

- CPU processing time
- vacancy computation time
- overall FPS
- memory impact if practical

Compare against Step 28 baseline:

~8.5 FPS end-to-end CPU

Do not optimize prematurely.

If the new logic adds negligible overhead, document it.

If it causes a significant slowdown, identify the bottleneck.

==================================================
PHASE 29U — FINAL REPORT
==================================================

Create:

STEP_29_VACANCY_CONFIDENCE_VALIDATION.md

Include:

1. Objective
2. Existing Step 28 architecture
3. New vacancy signal design
4. Significance evaluation
5. Temporal persistence
6. Geometric stability
7. Shopper occlusion
8. Camera motion
9. Partial detections
10. Real-video results
11. Controlled test results
12. Before/after comparison
13. Dashboard behavior
14. Performance
15. Remaining limitations

Include:

| Scenario | Expected | Observed | Result |
|----------|----------|----------|--------|
| Normal spacing | No alert | ... | PASS/FAIL |
| Temporary gap | No alert | ... | PASS/FAIL |
| Persistent vacancy | Alert | ... | PASS/FAIL |
| Shopper occlusion | UNCERTAIN | ... | PASS/FAIL |
| Camera motion | No false alert | ... | PASS/FAIL |
| Detector dropout | No alert | ... | PASS/FAIL |
| Partial detection | No phantom vacancy | ... | PASS/FAIL |
| Unstable gap geometry | No stable alert | ... | PASS/FAIL |
| Multiple gaps | Correct regions | ... | PASS/FAIL |

Also include:

### False Positive Analysis

Do not simply state "0 false positives."

List the tested scenarios and explain why each potential false positive was rejected.

### True Vacancy Analysis

Document the persistent vacancy scenario and exactly what evidence caused it to become confirmed.

### Confidence Interpretation

If a confidence score exists, document exactly what it means.

Make clear:

    Visual Vacancy Confidence
    ≠
    Probability of a product being out of stock.

==================================================
FINAL DECISION
==================================================

At the end provide EXACTLY ONE:

1. VACANCY SIGNAL IS SUFFICIENT FOR REPLENISHMENT MONITORING

2. MINOR VACANCY-SIGNAL TUNING REQUIRED

3. FURTHER ARCHITECTURAL CHANGE REQUIRED

Choose based on actual evidence.

Do NOT automatically choose option 1.

==================================================
IMPORTANT TERMINOLOGY
==================================================

Use:

"Vacancy"
"Persistent Empty Space"
"Visible Shelf Occupancy"
"Visual Vacancy Confidence"
"Replenishment Verification"
"Replenishment Signal"

Avoid:

"Exact Stock"
"Inventory Count"
"Units Remaining"
"SKU Stockout"
"Product Availability Probability"

unless such information is genuinely supported by a separate validated system.

==================================================
DEFINITION OF DONE
==================================================

Step 29 is complete only when:

[ ] Existing Step 28 architecture is understood and preserved.
[ ] Vacancy candidates are evaluated using interpretable signals.
[ ] Temporal persistence remains active.
[ ] Shopper occlusion remains safe.
[ ] Camera-motion protection remains safe.
[ ] Partial detections remain safe.
[ ] Geometric instability does not create false persistent vacancies.
[ ] Temporary gaps do not create alerts.
[ ] Persistent meaningful gaps generate replenishment signals.
[ ] Multiple vacancy regions are handled where practical.
[ ] No video-specific hacks are introduced.
[ ] No SKU/product/brand intelligence is added.
[ ] No exact inventory counting is added.
[ ] Real demo videos are evaluated.
[ ] Step 26 and Step 28 regressions pass.
[ ] Visual evidence is generated.
[ ] Dashboard remains generic and honest.
[ ] Performance is measured.
[ ] STEP_29_VACANCY_CONFIDENCE_VALIDATION.md is created.
[ ] Final recommendation is exactly one of the three allowed decisions.

DO NOT PROCEED TO PRODUCT IDs, SKU RECOGNITION, PRODUCT CLASSIFICATION, OR EXACT INVENTORY COUNTING AFTER THIS STEP.

First establish that the generic vacancy signal is reliable enough to serve as a legitimate replenishment-monitoring signal.