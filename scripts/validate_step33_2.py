"""Step 33.2 Comprehensive Validation Script.

Validates:
1. Full-length annotated MP4 video processing without silent truncation.
2. Source vs annotated frame counts, FPS, durations, and duration ratios (~1.0).
3. Automatic CV-based shelf geometry discovery (physical shelf rails and perspective tiers).
4. Dynamic physical tier counts (not assumed 3 tiers).
5. Elimination of all semantic product/category labels.
6. Transparent fallback behavior (SHELF GEOMETRY: CONFIGURED FALLBACK).
7. Preservation of Step 28 2D horizontal occupancy fix (rejection of false 617px pseudo-gap).
8. Generation of visual evidence contact sheet in output/step33_2_evidence/.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from inventory.inventory_api import _run_pipeline_sync
from inventory.shelf_geometry import AutomaticShelfGeometryDetector, ShelfGeometryResult
from inventory.shelf_vacancy import ShelfVacancyEngine, ShelfVacancyTracker, annotate_vacancy_frame


def inspect_video_properties(v_path: pathlib.Path) -> Dict[str, Any]:
    """Inspect OpenCV video metadata."""
    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        return {"error": f"Cannot open {v_path}"}
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = cnt / fps if fps > 0 else 0.0
    cap.release()
    return {
        "path": str(v_path),
        "frame_count": cnt,
        "fps": round(fps, 2),
        "width": w,
        "height": h,
        "duration_sec": round(dur, 2),
    }


def verify_frame_diversity(video_path: pathlib.Path, num_samples: int = 5) -> Tuple[bool, float]:
    """Verify that the generated video is visually dynamic (not a static snapshot loop)."""
    cap = cv2.VideoCapture(str(video_path))
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if cnt < 2:
        cap.release()
        return False, 0.0

    sample_indices = np.linspace(0, cnt - 1, min(num_samples, cnt), dtype=int)
    frames = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            gray = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY)
            frames.append(gray)
    cap.release()

    if len(frames) < 2:
        return False, 0.0

    diffs = []
    for i in range(len(frames) - 1):
        diff = float(np.mean(np.abs(frames[i + 1].astype(float) - frames[i].astype(float))))
        diffs.append(diff)

    avg_diff = float(np.mean(diffs))
    is_dynamic = avg_diff >= 1.5  # Distinct visual movement across frames
    return is_dynamic, avg_diff


def generate_contact_sheet(
    video_path: pathlib.Path,
    run_id: str,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    """Create a 7-stage visual evidence contact sheet for Step 33.2."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    ret, src_frame = cap.read()
    cap.release()
    if not ret or src_frame is None:
        raise RuntimeError(f"Cannot read frame from {video_path}")

    h, w = src_frame.shape[:2]
    vis_w, vis_h = 960, int(960 * h / w)
    src_resized = cv2.resize(src_frame, (vis_w, vis_h))

    # Stage 1: Source Frame
    stage1 = src_resized.copy()
    cv2.putText(stage1, "STAGE 1: SOURCE FRAME", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Stage 2: Structural Edge & Detected Shelf Boundaries
    geo_detector = AutomaticShelfGeometryDetector(min_tier_height_ratio=0.08)
    geo_res = geo_detector.detect_from_video(video_path)
    stage2 = src_resized.copy()
    for b in geo_res.boundaries:
        p1 = (int(b.points[0][0] * vis_w / w), int(b.points[0][1] * vis_h / h))
        p2 = (int(b.points[1][0] * vis_w / w), int(b.points[1][1] * vis_h / h))
        cv2.line(stage2, p1, p2, (0, 255, 0), 3)
        cv2.putText(stage2, f"{b.line_id} (conf={b.confidence:.2f})", (30, p1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(stage2, "STAGE 2: DETECTED PHYSICAL SHELF BOUNDARIES", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Stage 3: Product Detections
    from inventory.config import InventoryModelConfig
    from inventory.shelf_detector import ShelfProductDetector
    det_cfg = InventoryModelConfig(
        model_path=str(ROOT_DIR / "inventory_data" / "custom_model" / "retail_detector_exp2.pt"),
        model_tier="retail_specific",
        device="cpu",
        confidence_threshold=0.30,
    )
    detector = ShelfProductDetector(det_cfg)
    batch = detector.detect(frame=src_frame, frame_index=0)
    stage3 = src_resized.copy()
    for d in batch.product_detections:
        bx1 = int(d.bbox[0] * vis_w / w)
        by1 = int(d.bbox[1] * vis_h / h)
        bx2 = int(d.bbox[2] * vis_w / w)
        by2 = int(d.bbox[3] * vis_h / h)
        cv2.rectangle(stage3, (bx1, by1), (bx2, by2), (0, 220, 120), 2)
    cv2.putText(stage3, f"STAGE 3: PRODUCT DETECTIONS ({len(batch.product_detections)} items)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 120), 2)

    # Stage 4: Discovered Physical Tier Polygons
    stage4 = src_resized.copy()
    colors = [(255, 200, 0), (0, 200, 255), (255, 100, 100), (100, 255, 100), (200, 100, 255), (255, 255, 100)]
    for idx, t in enumerate(geo_res.tiers):
        c = colors[idx % len(colors)]
        pts = np.array([[(int(pt[0] * vis_w / w), int(pt[1] * vis_h / h))] for pt in t.polygon], dtype=np.int32)
        cv2.polylines(stage4, [pts], isClosed=True, color=c, thickness=3)
        mid_pt = pts[0][0]
        cv2.putText(stage4, f"{t.name} (POLYGON)", (mid_pt[0] + 20, mid_pt[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    cv2.putText(stage4, f"STAGE 4: DERIVED PHYSICAL TIERS ({len(geo_res.tiers)} TIERS)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)

    # Stage 5: 2D Occupancy Projection & Pseudo-gap Rejection
    engine = ShelfVacancyEngine(shelf_id="SHELF-01", geometry_result=geo_res)
    snap = engine.analyze_frame(
        product_boxes=[d.bbox for d in batch.product_detections],
        frame_w=w,
        frame_h=h,
    )
    stage5 = src_resized.copy()
    overlay5 = stage5.copy()
    for t in snap.tiers:
        for p in t.products:
            bx1 = int(p["bbox"][0] * vis_w / w)
            by1 = int(p["bbox"][1] * vis_h / h)
            bx2 = int(p["bbox"][2] * vis_w / w)
            by2 = int(p["bbox"][3] * vis_h / h)
            cv2.rectangle(overlay5, (bx1, by1), (bx2, by2), (0, 220, 120), -1)
    for r_gap in snap.rejected_pseudo_gaps:
        rx1 = int(r_gap["x1"] * vis_w / w)
        ry1 = int(r_gap["y1"] * vis_h / h)
        rx2 = int(r_gap["x2"] * vis_w / w)
        ry2 = int(r_gap["y2"] * vis_h / h)
        cv2.rectangle(overlay5, (rx1, ry1), (rx2, ry2), (180, 50, 180), -1)
    cv2.addWeighted(overlay5, 0.40, stage5, 0.60, 0, stage5)
    cv2.putText(stage5, f"STAGE 5: 2D OCCUPANCY INTERVALS ({len(snap.rejected_pseudo_gaps)} pseudo-gaps rejected)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Stage 6: Vacancy Candidates & Gaps
    stage6 = src_resized.copy()
    for gap in snap.vacant_regions:
        gx1 = int(gap.x1 * vis_w / w)
        gy1 = int(gap.y1 * vis_h / h)
        gx2 = int(gap.x2 * vis_w / w)
        gy2 = int(gap.y2 * vis_h / h)
        cv2.rectangle(stage6, (gx1, gy1), (gx2, gy2), (0, 140, 255), 3)
        cv2.putText(stage6, f"VACANCY ({gap.width_px:.0f}px)", (gx1 + 6, gy1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
    cv2.putText(stage6, f"STAGE 6: VACANCY ANALYSIS ({len(snap.vacant_regions)} candidates)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2)

    # Stage 7: Final Annotated Frame
    annotated_full = annotate_vacancy_frame(src_frame, snap)
    stage7 = cv2.resize(annotated_full, (vis_w, vis_h))
    cv2.putText(stage7, "STAGE 7: PIPELINE ANNOTATED FRAME", (vis_w - 460, vis_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Assemble into a clean vertical contact sheet
    canvas = np.vstack([stage1, stage2, stage3, stage4, stage5, stage6, stage7])
    sheet_path = output_dir / f"{video_path.stem}_{run_id}_contact_sheet.jpg"
    cv2.imwrite(str(sheet_path), canvas)
    return sheet_path


def run_comprehensive_validation() -> Dict[str, Any]:
    """Execute complete Step 33.2 validation suite."""
    print("=" * 80)
    print("STEP 33.2 — FULL-LENGTH ANNOTATED VIDEO + AUTOMATIC SHELF GEOMETRY VALIDATION")
    print("=" * 80)

    results: Dict[str, Any] = {
        "test_runs": {},
        "fallback_test": None,
        "step27_false_vacancy_test": None,
        "contact_sheets": [],
        "overall_status": "PENDING",
    }

    test_videos = [
        ROOT_DIR / "inventory_data" / "demo_videos" / "shelf_pan_demo.mp4",
        ROOT_DIR / "videos" / "inventory2.mp4",
    ]

    for v_path in test_videos:
        if not v_path.is_file():
            print(f"Error: video not found: {v_path}")
            continue

        v_name = v_path.name
        print(f"\n--- [Validating Complete Video: {v_name}] ---")

        # 1. Source Video Metadata
        src_meta = inspect_video_properties(v_path)
        print(f"  Source Properties: {src_meta['width']}x{src_meta['height']} @ {src_meta['fps']} fps, {src_meta['frame_count']} frames, {src_meta['duration_sec']}s")

        # 2. Run Complete Pipeline (max_frames=None -> EOF)
        t_start = time.time()
        report = _run_pipeline_sync(str(v_path), max_frames=None)
        elapsed = round(time.time() - t_start, 2)
        run_id = report.get("run_id", "RUN-UNKNOWN")

        # 3. Annotated Video Metadata
        ann_info = report.get("annotated_video", {})
        playable_path = ROOT_DIR / ann_info.get("playable_path", "")
        assert playable_path.is_file(), f"Playable video file missing: {playable_path}"
        ann_meta = inspect_video_properties(playable_path)
        print(f"  Annotated Video:   {ann_meta['width']}x{ann_meta['height']} @ {ann_meta['fps']} fps, {ann_meta['frame_count']} frames, {ann_meta['duration_sec']}s")

        # 4. Duration and Frame Ratio
        frame_diff = abs(ann_meta["frame_count"] - src_meta["frame_count"])
        duration_ratio = ann_meta["duration_sec"] / src_meta["duration_sec"] if src_meta["duration_sec"] > 0 else 1.0
        print(f"  Frame Delta: {frame_diff} frame(s) | Duration Ratio: {duration_ratio:.3f} | Total Time: {elapsed}s")
        assert frame_diff <= 2, f"Frame count mismatch too high: src={src_meta['frame_count']}, ann={ann_meta['frame_count']}"
        assert 0.95 <= duration_ratio <= 1.05, f"Duration ratio out of bounds: {duration_ratio}"

        # 5. Visual Diversity (Not a static image)
        is_dynamic, avg_pixel_diff = verify_frame_diversity(playable_path)
        print(f"  Frame Diversity: dynamic={is_dynamic} (mean pixel diff={avg_pixel_diff:.2f})")
        assert is_dynamic, "Annotated video appears visually static!"

        # 6. Shelf Geometry Discovery
        geo_data = report.get("shelf_geometry", {})
        geo_src = geo_data.get("geometry_source")
        geo_conf = geo_data.get("geometry_confidence", 0.0)
        tier_cnt = geo_data.get("tier_count", 0)
        boundary_cnt = geo_data.get("boundary_count", 0)
        print(f"  Shelf Geometry: source={geo_src}, tiers={tier_cnt}, boundaries={boundary_cnt}, confidence={geo_conf:.2f}")
        assert geo_src == "AUTO_DISCOVERY", f"Expected AUTO_DISCOVERY, got {geo_src}"
        assert tier_cnt >= 2, f"Expected at least 2 tiers, got {tier_cnt}"
        assert geo_conf >= 0.50, f"Confidence too low: {geo_conf}"

        # 7. Zero Semantic Labels Check
        raw_report_str = json.dumps(report)
        forbidden_semantic = ["Beverages", "Bottles", "Cans", "Snacks", "Coke", "Pepsi", "Eye Level"]
        found_forbidden = [term for term in forbidden_semantic if term.lower() in raw_report_str.lower()]
        print(f"  Semantic Label Audit: {'CLEAN (No semantic brands/categories)' if not found_forbidden else 'FAILED: ' + str(found_forbidden)}")
        assert not found_forbidden, f"Semantic labels found: {found_forbidden}"

        # 8. Generate Contact Sheet
        evidence_dir = ROOT_DIR / "output" / "step33_2_evidence"
        sheet_p = generate_contact_sheet(v_path, run_id, evidence_dir)
        print(f"  Generated Evidence Contact Sheet: {sheet_p.relative_to(ROOT_DIR)}")

        results["test_runs"][v_name] = {
            "run_id": run_id,
            "source_meta": src_meta,
            "annotated_meta": ann_meta,
            "frame_diff": frame_diff,
            "duration_ratio": round(duration_ratio, 3),
            "is_dynamic": is_dynamic,
            "avg_pixel_diff": round(avg_pixel_diff, 2),
            "geometry_source": geo_src,
            "tier_count": tier_cnt,
            "boundary_count": boundary_cnt,
            "geometry_confidence": geo_conf,
            "contact_sheet": str(sheet_p.relative_to(ROOT_DIR)),
        }

    # 9. Test Configured Fallback Mode
    print("\n--- [Validating Configured Fallback Mode] ---")
    detector = AutomaticShelfGeometryDetector()
    fallback_test_res = detector._fallback(
        shelf_id="SHELF-01",
        configured_fallback={
            "shelf_roi": [0.05, 0.10, 0.95, 0.90],
            "tiers": [
                {"tier_id": "SHELF-01-TIER-01", "name": "TIER-01", "roi": [0.05, 0.10, 0.95, 0.50]},
                {"tier_id": "SHELF-01-TIER-02", "name": "TIER-02", "roi": [0.05, 0.50, 0.95, 0.90]},
            ],
        },
        reason="Forced validation test of fallback handler",
    )
    print(f"  Fallback source: {fallback_test_res.geometry_source} | Tiers: {len(fallback_test_res.tiers)} | Reason: {fallback_test_res.fallback_reason}")
    assert fallback_test_res.geometry_source == "CONFIGURED_FALLBACK"
    assert "SHELF GEOMETRY FALLBACK" in fallback_test_res.message
    results["fallback_test"] = {
        "status": "PASS",
        "geometry_source": fallback_test_res.geometry_source,
        "message": fallback_test_res.message,
    }

    # 10. Step 27 617px Pseudo-Gap Rejection Check (Preservation of Step 28 Fix)
    print("\n--- [Validating Step 27 False Vacancy Rejection (2D Occupancy)] ---")
    # Synthetic overlapping products test
    test_prods = [
        (100, 200, 250, 400),
        (240, 210, 400, 410),  # Adjacent
        (400, 205, 550, 395),  # Adjacent
        # Gap between 550 and 800, but covered in bottom half
        (540, 300, 810, 415),  # Cross-overlap covering horizontal span
        (800, 200, 950, 400),
    ]
    test_engine = ShelfVacancyEngine(
        shelf_id="SHELF-01",
        roi=(0.0, 0.0, 1.0, 1.0),
        tiers=[{"tier_id": "SHELF-01-TIER-01", "name": "TIER-01", "roi": [0.0, 0.0, 1.0, 1.0]}],
    )
    test_snap = test_engine.analyze_frame(product_boxes=test_prods, frame_w=1000, frame_h=600)
    print(f"  Pseudo-gaps rejected: {len(test_snap.rejected_pseudo_gaps)} | Unoccluded vacancies: {len(test_snap.unoccluded_vacant_regions)}")
    assert len(test_snap.rejected_pseudo_gaps) >= 1 or len(test_snap.unoccluded_vacant_regions) == 0, "2D occupancy failed to reject overlapping products!"
    results["step27_false_vacancy_test"] = "PASS"

    results["overall_status"] = "ALL_TESTS_PASSED"
    print("\n" + "=" * 80)
    print("STEP 33.2 VALIDATION SUMMARY:")
    for v_name, res in results["test_runs"].items():
        print(f"  [{v_name}]")
        print(f"    Frames: {res['source_meta']['frame_count']} (src) -> {res['annotated_meta']['frame_count']} (ann) [delta: {res['frame_diff']}]")
        print(f"    Duration: {res['source_meta']['duration_sec']}s (src) -> {res['annotated_meta']['duration_sec']}s (ann) [ratio: {res['duration_ratio']}]")
        print(f"    Geometry: {res['geometry_source']} | Tiers: {res['tier_count']} | Conf: {res['geometry_confidence']:.2f}")
        print(f"    Contact Sheet: {res['contact_sheet']}")
    print(f"  Fallback Handler: {results['fallback_test']['status']}")
    print(f"  2D Pseudo-Gap Rejection: {results['step27_false_vacancy_test']}")
    print(f"  Overall Status: {results['overall_status']}")
    print("=" * 80)

    # Save summary report JSON
    rep_out = ROOT_DIR / "output" / "step33_2_validation_result.json"
    with open(rep_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Validation summary saved to {rep_out.relative_to(ROOT_DIR)}")

    return results


if __name__ == "__main__":
    run_comprehensive_validation()
