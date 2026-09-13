STEP 33.2 — FULL-LENGTH ANNOTATED VIDEO + AUTOMATIC SHELF-GEOMETRY DISCOVERY

You are working in:
D:\RetailShop\Retail_Analytics

IMPORTANT:
Inspect the existing implementation before changing anything.

This is a continuation of Steps 28–33.1.

The current canonical inventory pipeline is:
inventory/inventory_api.py::_run_pipeline_sync(...)

The current architecture already contains and must preserve:
- custom retail detector
- ByteTrack
- physical shelf/tier occupancy logic
- 2D horizontal occupancy projection
- temporal vacancy confirmation
- Visual Vacancy Confidence
- shopper occlusion handling
- camera-motion handling
- ShelfEventManager
- dashboard event lifecycle
- run IDs
- annotated/snapshot evidence
- current dashboard synchronization

DO NOT redesign or retrain the product detector.
DO NOT add SKU recognition.
DO NOT add brand recognition.
DO NOT add product catalog assumptions.
DO NOT hardcode product categories.
DO NOT hardcode "5 tiers for inventory2" or "3 tiers for shelf_pan_demo".
DO NOT silently fall back to configured shelf geometry.
DO NOT replace the current vacancy/event architecture.

There are TWO confirmed Step 33.1 problems to solve:

1. Annotated MP4 is truncated because test/demo runs are processing only a limited number of frames.
2. Physical shelf tiers are still ultimately based on configured rectangles instead of actual visual shelf structure.

The goal is to fix both properly.

==================================================
PART A — FIRST INSPECT THE CURRENT CODE
==================================================

Before editing:

1. Inspect:
   - inventory/inventory_api.py
   - inventory/shelf_vacancy.py
   - configs/shelf_vacancy_config.json
   - dashboard/src/api/client.js
   - dashboard/src/components/InventoryRunControl.jsx
   - dashboard/src/components/InventoryDashboard.jsx
   - dashboard/src/components/ShelfRackVisualizer.jsx

2. Identify:
   - where max_frames is currently introduced
   - where source video metadata is read
   - where annotated frames are generated
   - where VideoWriter is initialized/released
   - where tier rectangles are currently obtained
   - where tier labels are rendered
   - where dashboard video URLs are constructed
   - whether stale latest_annotated.mp4 can override a run-specific file

3. Do not assume the previous implementation is correct.
   Trace the actual execution path.

==================================================
PART B — FIX FULL-LENGTH VIDEO PROCESSING
==================================================

Production inventory runs must process the complete source video.

Change:

    max_frames: Optional[int] = None

Default behavior:

    max_frames is None
    OR
    max_frames <= 0

means:

    process until cap.read() reaches EOF.

max_frames remains available ONLY as a debug/test limit.

For example:

    max_frames=40

may still be used by automated tests.

But the dashboard Start Inventory action must NOT automatically pass 40/75/etc.

For every production run:

    source video
        ↓
    read until EOF
        ↓
    process every frame
        ↓
    write every annotated frame
        ↓
    release VideoWriter

Read and store:

- source frame count
- source FPS
- source width
- source height
- source duration

Do not change the original resolution.

Do not drop frames.

Do not duplicate frames.

Do not change frame order.

Annotated video must use the source FPS.

Validate:

    source_frames
    annotated_frames
    source_fps
    annotated_fps
    source_duration
    annotated_duration
    duration_ratio

The frame-count difference should be <= 1 where codec behavior permits.

The duration ratio should be approximately 1.0.

Expected source videos:

    videos/inventory2.mp4
        approximately 210 frames
        15 FPS
        approximately 14 seconds

    inventory_data/demo_videos/shelf_pan_demo.mp4
        approximately 75 frames
        25 FPS
        approximately 3 seconds

Do NOT make these values hardcoded.

Read them from OpenCV/video metadata.

If ffmpeg is available, transcode the generated annotated MP4 to browser-compatible H.264 with:

    -r <source_fps>
    -movflags +faststart

Do not alter duration during transcoding.

After writing, actually reopen the resulting MP4 and validate that it is playable.

==================================================
PART C — REMOVE SEMANTIC / HARDCODED TIER GEOMETRY
==================================================

The current system contains semantic labels such as:

    "Tier 1 — Top (2L Beverages)"
    "Tier 2 — Mid (20oz Bottles)"
    "Tier 3 — Lower (12-Pack Cans)"

These are NOT acceptable.

Remove all semantic product/category labels.

Tier labels must be generic:

    TIER-01
    TIER-02
    TIER-03
    ...

Do not encode product type in a tier name.

More importantly:

RENAMING THE EXISTING CONFIGURED RECTANGLES IS NOT ENOUGH.

The production source of shelf geometry must become visual shelf-structure discovery.

The following must NOT be the primary source:

    hardcoded normalized x1/y1/x2/y2 tier rectangles

Do not simply replace the names with TIER-01 while retaining the exact same geometry as the production detection mechanism.

==================================================
PART D — IMPLEMENT AUTOMATIC SHELF GEOMETRY DISCOVERY
==================================================

Create:

    inventory/shelf_geometry.py

with:

    AutomaticShelfGeometryDetector

and a result structure such as:

    ShelfGeometryResult

The detector should discover PHYSICAL SHELF BOUNDARIES from visual structure.

Important architectural distinction:

    Shelf geometry detection
        =
    physical shelf rails / shelf-row boundaries

NOT:

    product center-Y clustering

Do NOT use detected product Y coordinates as the primary method for discovering shelves.

Do NOT infer physical shelves from "where products happen to be".

==================================================
PART E — SHELF GEOMETRY ALGORITHM
==================================================

Implement a robust classical-CV baseline first.

Do NOT immediately introduce a new trained neural model.

Use a pipeline approximately like:

    sampled video frames
          ↓
    image preprocessing
          ↓
    edge / gradient extraction
          ↓
    horizontal-structure emphasis
          ↓
    morphological filtering
          ↓
    line-segment detection
          ↓
    candidate shelf-boundary lines
          ↓
    line clustering
          ↓
    temporal consensus
          ↓
    perspective-aware boundary fitting
          ↓
    physical shelf tiers

Candidate methods may include:

- bilateral filtering
- Sobel gradients
- horizontal morphology
- Canny where useful
- HoughLinesP / line-segment detection
- weighted least-squares line fitting
- normalized coordinate representation

Do NOT assume that every horizontal edge is a shelf.

Reject or down-weight:

- very short lines
- isolated product edges
- text/price-label edges
- unstable lines
- lines appearing only in one frame
- lines with inconsistent geometry
- obviously non-shelf structures

A shelf boundary should have evidence across multiple sampled frames whenever possible.

==================================================
PART F — TEMPORAL CONSENSUS
==================================================

This is REQUIRED.

Do not detect shelf geometry independently from one frame and immediately accept it.

Sample multiple frames throughout the video.

For each candidate line:

    frame 1
    frame 2
    frame 3
    ...
    frame N

Determine whether the line is temporally stable.

Use temporal support as part of geometry confidence.

The exact number of sampled frames should be configurable, but use enough frames to cover the video.

For moving/panning video, compensate for moderate camera motion where possible or normalize candidate line positions before consensus.

The goal is:

    temporary product edge
        ≠
    persistent physical shelf boundary

==================================================
PART G — PERSPECTIVE-AWARE SHELF BOUNDARIES
==================================================

Do not force every shelf boundary to be perfectly horizontal.

Represent boundaries as line segments / fitted lines:

    y = m*x + c

or equivalent normalized representation.

If the actual video contains perspective:

    left boundary Y
    and
    right boundary Y

may differ.

Generate tier polygons between adjacent stable boundaries.

Example conceptually:

    upper boundary:
        (x_left, y1_left)
        (x_right, y1_right)

    lower boundary:
        (x_left, y2_left)
        (x_right, y2_right)

Then:

    TIER-N =
        [top-left,
         top-right,
         bottom-right,
         bottom-left]

Use polygons rather than pretending every tier is an axis-aligned rectangle.

==================================================
PART H — DYNAMIC TIER COUNT
==================================================

Tier count must be derived from discovered physical boundaries.

Conceptually:

    detected boundaries = N + 1

    generated physical tiers = N

Do NOT hardcode:

    inventory2 = 5 tiers
    shelf_pan_demo = 3 tiers

Those are only expected observations for validation.

The actual detector decides the count.

If the video contains:

    2 tiers → generate 2

If:

    5 tiers → generate 5

If:

    7 tiers → generate 7

etc.

Only create a tier when there is sufficient geometric evidence.

==================================================
PART I — GEOMETRY QUALITY / CONFIDENCE
==================================================

ShelfGeometryResult should include at minimum:

    geometry_source
    geometry_confidence
    boundary_lines
    tier_polygons
    tier_rois
    boundary_count
    tier_count

Recommended internal quality signals:

    temporal_support
    line_fit_error
    line_length_score
    structural_strength
    consistency_score

Do not call geometry_confidence a probability unless it is actually calibrated.

It is an engineering confidence/quality score.

Example:

    geometry_source:
        AUTO_DISCOVERY

    geometry_confidence:
        0.87

==================================================
PART J — EXPLICIT FALLBACK
==================================================

Keep configured shelf geometry ONLY as a fallback.

Fallback must happen only when:

    automatic geometry discovery fails
    OR
    geometry quality is below a defined threshold.

If fallback occurs, explicitly set:

    geometry_source = CONFIGURED_FALLBACK

and record:

    fallback_reason

The UI must display:

    SHELF GEOMETRY: CONFIGURED FALLBACK

Do NOT silently use configured rectangles while claiming:

    AUTO-DETECTED

When automatic detection succeeds, display:

    SHELF GEOMETRY: AUTO-DETECTED

Configured geometry should never be described as detected geometry.

==================================================
PART K — INTEGRATE WITH SHELF VACANCY ENGINE
==================================================

Modify:

    inventory/shelf_vacancy.py

ShelfTier should support:

    polygon
    boundary_top
    boundary_bottom

while preserving compatibility where needed.

The vacancy engine should receive:

    ShelfGeometryResult

and use the discovered physical tier polygons.

Product detections continue to come from the existing retail detector + ByteTrack.

Pipeline:

    video
       ↓
    automatic shelf geometry
       ↓
    physical tier polygons
       ↓
    retail object detector
       ↓
    ByteTrack
       ↓
    assign detections to physical tier
       ↓
    2D horizontal occupancy
       ↓
    vacancy candidate
       ↓
    temporal confirmation
       ↓
    Visual Vacancy Confidence
       ↓
    replenishment recommendation
       ↓
    ShelfEventManager

DO NOT replace the existing Step 28/29/31 logic.

==================================================
PART L — PRESERVE THE 2D OCCUPANCY FIX
==================================================

The Step 28 fix must remain intact.

Do NOT go back to:

    1D row grouping
        ↓
    horizontal gap

The physical tier is now the spatial container.

Within that tier:

    all relevant product bounding-box horizontal spans
        ↓
    interval union
        ↓
    2D vertical-overlap validation
        ↓
    candidate gap

This is what previously eliminated the false 617px pseudo-gap.

Explicitly reproduce the old Step 27 failure case and confirm it remains fixed.

==================================================
PART M — ANNOTATED VIDEO
==================================================

The annotated MP4 must visually prove what the system is doing.

For every frame where applicable, show:

1. Physical shelf boundaries / polygons.

2. Generic labels:

    TIER-01
    TIER-02
    ...

3. Product detections.

4. ByteTrack IDs if already available and useful.

5. Occupancy percentage per tier.

6. Vacancy candidate regions.

7. Persistent vacancy state.

8. Visual Vacancy Confidence.

9. Replenishment Recommended when applicable.

10. Shopper occlusion.

11. Camera motion / uncertainty.

12. Geometry source.

Top HUD should contain something similar to:

    SHELF GEOMETRY: AUTO-DETECTED
    GEOMETRY CONFIDENCE: 87%

or:

    SHELF GEOMETRY: CONFIGURED FALLBACK
    FALLBACK: INSUFFICIENT STRUCTURAL EVIDENCE

Do not render:

    Coca-Cola
    Pepsi
    7UP
    Snacks
    Beverages
    2L
    20oz
    12-Pack
    SKU names
    fake product categories

==================================================
PART N — SEGMENTATION
==================================================

DO NOT replace the current detector with instance segmentation in this step.

Segmentation is NOT the solution to physical shelf geometry.

If useful, you may create a small isolated experiment to determine whether product masks would improve occupancy measurement.

But:

    segmentation experiment
        ≠
    production architecture change

Do not add a new segmentation model to the production pipeline unless existing bounding boxes demonstrably fail after the new geometry engine is implemented.

The production Step 33.2 objective is:

    automatic physical shelf geometry
    +
    existing product detection
    +
    existing 2D occupancy

==================================================
PART O — API / RUN METADATA
==================================================

Modify:

    inventory/inventory_api.py

Store in the report:

    video_metadata:
        source_frames
        source_fps
        source_width
        source_height
        source_duration
        annotated_frames
        annotated_fps
        annotated_duration
        duration_ratio

and:

    shelf_geometry:
        geometry_source
        geometry_confidence
        boundary_count
        tier_count
        fallback_reason
        boundaries
        tiers

Do not fabricate these values.

They must come from the actual pipeline.

==================================================
PART P — DASHBOARD
==================================================

Update:

    dashboard/src/api/client.js
    dashboard/src/components/InventoryRunControl.jsx
    dashboard/src/components/InventoryDashboard.jsx
    dashboard/src/components/ShelfRackVisualizer.jsx

Dashboard run behavior:

    Start Inventory
        ↓
    full source video

Do not pass hardcoded:

    40
    75
    or any other frame limit.

max_frames should only be available to testing/debug functionality.

Display dynamically:

    SHELF 1 (SHELF-01 · N TIERS)

where N comes from the actual report.

Show:

    SHELF GEOMETRY: AUTO-DETECTED

or:

    SHELF GEOMETRY: CONFIGURED FALLBACK

Annotated video should be run-specific:

    /evidence/{run_id}_annotated.mp4

or an equivalent API endpoint.

Do NOT rely solely on:

    latest_annotated.mp4

because that can cause stale-run leakage.

Ensure video source is tied to the current RUN-XXX.

Cache-bust or use a run-specific URL.

Video controls should allow:

    ANNOTATED
    ORIGINAL
    SNAPSHOT

with ANNOTATED as the default.

==================================================
PART Q — CONFIG CLEANUP
==================================================

Update:

    configs/shelf_vacancy_config.json

Remove semantic tier names.

Do not leave things such as:

    Tier 1 — Top (2L Beverages)
    Tier 2 — Mid (20oz Bottles)
    Tier 3 — Lower (12-Pack Cans)

as active production tier definitions.

Generic configured fallback values may remain if necessary, but they must be:

    TIER-01
    TIER-02
    ...

and must be clearly treated as FALLBACK geometry.

Do not remove fallback support if it is useful.

==================================================
PART R — VALIDATION SCRIPT
==================================================

Create:

    scripts/validate_step33_2.py

Run COMPLETE videos, not 40-frame subsets.

Test:

    videos/inventory2.mp4

and:

    inventory_data/demo_videos/shelf_pan_demo.mp4

For each video verify:

1. Source video opens.

2. Source metadata is read.

3. Complete source is processed.

4. Annotated MP4 exists.

5. Annotated MP4 opens.

6. Annotated frame count approximately equals source frame count.

7. FPS is preserved.

8. Duration ratio is approximately 1.0.

9. Video is visually dynamic.

10. Automatic shelf geometry is attempted.

11. Geometry source is reported honestly.

12. Geometry confidence is reported.

13. Tier count is dynamic.

14. No semantic product/category labels exist.

15. Shelf boundaries are visually plausible.

16. Tier polygons align with physical shelf structure.

17. Existing 2D occupancy logic works.

18. Existing vacancy confidence works.

19. Shopper occlusion still suppresses false vacancy.

20. Camera motion still produces uncertainty rather than false vacancy.

21. Existing ShelfEventManager still works.

22. Run IDs remain isolated.

23. Dashboard build succeeds.

Do NOT define success as:

    "100% real-world accuracy"

Instead report:

    tests passed
    frame coverage
    geometry quality
    false-alert observations
    limitations

==================================================
PART S — REQUIRED VISUAL EVIDENCE
==================================================

Create:

    output/step33_2_evidence/

Include evidence showing:

1. Original frame.

2. Raw structural edge/line candidates.

3. Accepted shelf boundary lines.

4. Detected tier polygons.

5. Product detections.

6. 2D occupancy intervals.

7. Vacancy candidate.

8. Confirmed vacancy if present.

9. Final annotated frame.

10. Before/after comparison with the old hardcoded-tier result if possible.

Create a contact sheet.

The evidence must make it obvious that:

    physical shelf boundaries
        came from visual structure

rather than:

    configured rectangles.

==================================================
PART T — REQUIRED REPORT
==================================================

Create:

    STEP_33_2_SHELF_GEOMETRY_AND_ANNOTATED_VIDEO_VALIDATION.md

The report must explicitly answer:

1. Was the annotated video previously truncated? Why?

2. Is the annotated video now full-length?

3. Source frame count?

4. Annotated frame count?

5. Source FPS?

6. Annotated FPS?

7. Source duration?

8. Annotated duration?

9. Duration ratio?

10. Were shelf tiers previously hardcoded/configured?

11. Are shelf tiers now automatically derived from visual shelf geometry?

12. What algorithm discovers shelf boundaries?

13. Is temporal consensus used?

14. How is perspective handled?

15. What is the geometry confidence?

16. Was configured fallback used?

17. If fallback was used, why?

18. How many physical tiers were discovered for each video?

19. Did Step 27's 617px false vacancy remain fixed?

20. Did Steps 28–32 regress?

21. Is segmentation required?

22. What are the remaining limitations?

Be honest.

If something fails, document it instead of manufacturing a passing result.

==================================================
PART U — REGRESSION TESTS
==================================================

After implementation run:

    venv\Scripts\python.exe scripts/validate_step33_2.py

Then run the existing relevant regression tests.

Use the actual filenames currently present in the repository.

At minimum verify the equivalent of:

    Step 28
    Step 29
    Step 31
    Step 32

Do not assume old script filenames exist.

If a referenced validation script is missing, locate the current equivalent before running it.

Also run:

    npm build

for the dashboard.

==================================================
PART V — FINAL ARCHITECTURAL REQUIREMENT
==================================================

The final architecture must clearly separate:

    PHYSICAL SHELF GEOMETRY
            ↓
    PRODUCT/OBJECT DETECTION
            ↓
    2D OCCUPANCY
            ↓
    VACANCY
            ↓
    TEMPORAL CONFIRMATION
            ↓
    VISUAL VACANCY CONFIDENCE
            ↓
    REPLENISHMENT RECOMMENDATION
            ↓
    INVENTORY EVENT

The system must NOT infer shelf geometry from product Y-coordinate clustering.

The system must NOT assume a fixed number of shelves.

The system must NOT assume product categories.

The system must NOT silently use configured rectangles.

The system must NOT claim calibrated probabilities where only heuristic confidence exists.

The system must NOT add SKU/product intelligence.

==================================================
FINAL DECISION CRITERIA
==================================================

Do not declare success merely because the code runs.

Step 33.2 is successful only if:

A. Complete source videos produce complete annotated videos.

B. Annotated FPS/resolution/frame coverage match the source.

C. Physical shelf boundaries are discovered from visual structure.

D. Tier count is dynamically generated.

E. Tier geometry is perspective-aware where necessary.

F. Temporal consensus prevents one-frame false shelf lines.

G. Configured geometry is only a transparent fallback.

H. The old 617px false vacancy remains eliminated.

I. Product detection + ByteTrack remains intact.

J. 2D occupancy remains intact.

K. Visual Vacancy Confidence remains intact.

L. Shopper/camera-motion handling remains intact.

M. Shelf events remain intact.

N. Dashboard uses the current run's full annotated video.

O. No semantic product labels or fake inventory assumptions remain.

IMPORTANT:
Do not stop after implementing the files.

Actually execute the validation.

Inspect the generated annotated videos and evidence.

If the automatic shelf detector produces obviously incorrect boundaries on either real demo video, DO NOT simply mark the test as passed. Diagnose the geometry failure, tune the detector, or explicitly use the configured fallback and report why.

At the end, provide a concise implementation summary containing:

- files changed
- files created
- full-video validation results
- source vs annotated frame counts
- source vs annotated durations
- detected tier counts
- geometry source for each video
- geometry confidence
- regression results
- dashboard build result
- remaining limitations
- final verdict:
  SUFFICIENT
  or
  MINOR TUNING REQUIRED
  or
  ARCHITECTURE CHANGE REQUIRED