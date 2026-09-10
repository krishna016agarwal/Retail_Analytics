"""Shelf analysis pipeline orchestrator.

Handles image, video, and folder-of-images input. Coordinates:
    ShelfProductDetector  ->  ShelfStateTracker  ->  ShelfVisualizer  ->  ShelfReport

Completely isolated from src/. Zero imports from the crowd/queue pipeline.
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from inventory.config import InventoryConfig, InventoryModelConfig
from inventory.report import PerFrameResult, ShelfReport
from inventory.shelf_detector import ShelfDetectionBatch, ShelfProductDetector
from inventory.shelf_state import ObservationStatus, ShelfStateSnapshot, ShelfStateTracker
from inventory.shelf_visualizer import ShelfVisualizer


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".ts"}


class ShelfAnalysisPipeline:
    """Top-level orchestrator for retail shelf analysis.

    Routes image, video, or folder inputs through the full pipeline and
    produces annotated output files and JSON reports.

    Quick-start::

        config   = InventoryConfig()
        pipeline = ShelfAnalysisPipeline(config)
        report   = pipeline.run("shelf.jpg")
        report.print_summary()

    To swap the detection model, change only InventoryConfig.model — the
    pipeline, state tracker, visualiser, and report system are model-agnostic.
    """

    def __init__(self, config: Optional[InventoryConfig] = None) -> None:
        self.config = config or InventoryConfig()
        self.detector = ShelfProductDetector(self.config.model)
        self.state_tracker = ShelfStateTracker(self.config.temporal)
        self.visualizer = ShelfVisualizer()
        self._output_dir = Path(self.config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        source: str,
        output_path: Optional[str] = None,
        show_display: bool = True,
        max_frames: Optional[int] = None,
    ) -> ShelfReport:
        """Auto-detect source type and run the appropriate sub-pipeline.

        Args:
            source: Path to image file, video file, or folder of images.
            output_path: Explicit output path (auto-named if None).
            show_display: Show a live OpenCV window (video mode only).
            max_frames: Maximum frames / images to process (None = all).

        Returns:
            ShelfReport with complete analysis results.
        """
        src = Path(source)
        if not src.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        if src.is_dir():
            return self._run_folder(src, output_path, max_frames)
        if src.suffix.lower() in _IMAGE_EXTS:
            return self._run_image(src, output_path)
        if src.suffix.lower() in _VIDEO_EXTS:
            return self._run_video(src, output_path, show_display, max_frames)
        # Fallback: try video
        return self._run_video(src, output_path, show_display, max_frames)

    # ------------------------------------------------------------------
    # Image mode
    # ------------------------------------------------------------------

    def _run_image(
        self, image_path: Path, output_path: Optional[str]
    ) -> ShelfReport:
        print(f"\n[ShelfPipeline] Image mode  ->  {image_path}")

        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")

        t0 = time.perf_counter()
        batch = self.detector.detect(frame, frame_index=0)

        # Single image has no temporal history — build a minimal snapshot
        state = ShelfStateSnapshot(
            observation_status=(
                ObservationStatus.UNCERTAIN
                if batch.person_present
                else ObservationStatus.STABLE
            ),
            stable_visible_count=batch.visible_count,
            visible_count_this_frame=batch.visible_count,
            uncertainty_reason=(
                "Person detected in frame" if batch.person_present else ""
            ),
            shelf_occupancy_pct=100.0 if batch.visible_count > 0 else 0.0,
        )

        annotated = self.visualizer.annotate_frame(
            frame, batch, state,
            fps=0.0,
            shelf_roi=self.config.shelf_roi,
            model_tier=self.detector.model_tier,
        )
        total_ms = (time.perf_counter() - t0) * 1000.0

        out_path = self._resolve_output(image_path, output_path, "_annotated", ".jpg")
        cv2.imwrite(str(out_path), annotated)
        print(f"[ShelfPipeline] Annotated image ->  {out_path}")

        per_frame = [
            PerFrameResult(
                frame_index=0,
                timestamp_sec=0.0,
                visible_count=batch.visible_count,
                person_detected=batch.person_present,
                observation_status=state.observation_status.value,
                stable_visible_count=state.stable_visible_count,
                state_change_detected=False,
                shelf_occupancy_pct=state.shelf_occupancy_pct,
                inference_time_ms=round(batch.inference_time_ms, 2),
                detections=self._serialise_dets(batch),
            )
        ]

        report = ShelfReport(
            source_path=str(image_path),
            model_path=self.config.model.model_path,
            model_tier=self.config.model.model_tier,
            processing_time_ms=round(total_ms, 2),
            total_frames=1,
            avg_visible_facings=float(batch.visible_count),
            peak_visible_facings=batch.visible_count,
            min_visible_facings=batch.visible_count,
            state_changes_detected=0,
            uncertain_frames=1 if batch.person_present else 0,
            avg_fps=0.0,
            avg_inference_ms=round(batch.inference_time_ms, 2),
            per_frame_results=per_frame,
        )

        json_path = out_path.with_suffix(".json")
        report.save_json(str(json_path))
        report.print_summary()
        return report

    # ------------------------------------------------------------------
    # Video mode
    # ------------------------------------------------------------------

    def _run_video(
        self,
        video_path: Path,
        output_path: Optional[str],
        show_display: bool,
        max_frames: Optional[int],
    ) -> ShelfReport:
        print(f"\n[ShelfPipeline] Video mode  ->  {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = self._resolve_output(video_path, output_path, "_annotated", ".mp4")
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_src,
            (w, h),
        )

        self.state_tracker.reset()

        per_frame_results: List[PerFrameResult] = []
        visible_counts: List[int] = []
        inference_times: List[float] = []
        state_changes = 0
        uncertain_count = 0
        frame_idx = 0
        fps_smooth = fps_src
        run_start = time.perf_counter()

        print(f"[ShelfPipeline] Source : {n_frames} frames  {fps_src:.1f} FPS  {w}x{h}")
        print(f"[ShelfPipeline] Model  : {self.config.model.model_path}  [{self.config.model.model_tier}]")
        if self.config.model.model_tier == "coco_baseline":
            print("[ShelfPipeline] NOTE   : COCO baseline -- generic object detection, NOT retail SKU recognition.")
        print("[ShelfPipeline] Press 'q' to quit early.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_idx >= max_frames:
                break

            t_frame = time.perf_counter()
            ts = frame_idx / fps_src

            batch = self.detector.detect(frame, frame_index=frame_idx)
            state = self.state_tracker.update(batch, timestamp_sec=ts)

            annotated = self.visualizer.annotate_frame(
                frame, batch, state,
                fps=fps_smooth,
                shelf_roi=self.config.shelf_roi,
                model_tier=self.detector.model_tier,
            )

            writer.write(annotated)

            if show_display:
                cv2.imshow("Retail Shelf Monitor - Inventory Foundation", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            elapsed = time.perf_counter() - t_frame
            fps_smooth = 0.85 * fps_smooth + 0.15 * (1.0 / max(elapsed, 1e-6))

            visible_counts.append(batch.visible_count)
            inference_times.append(batch.inference_time_ms)
            if state.state_change_detected:
                state_changes += 1
            if state.observation_status == ObservationStatus.UNCERTAIN:
                uncertain_count += 1

            if frame_idx % 30 == 0:
                print(
                    f"  Frame {frame_idx:5d} | Facings: {batch.visible_count:3d} | "
                    f"State: {state.observation_status.value:<18s} | FPS: {fps_smooth:.1f}"
                )

            per_frame_results.append(
                PerFrameResult(
                    frame_index=frame_idx,
                    timestamp_sec=round(ts, 3),
                    visible_count=batch.visible_count,
                    person_detected=batch.person_present,
                    observation_status=state.observation_status.value,
                    stable_visible_count=state.stable_visible_count,
                    state_change_detected=state.state_change_detected,
                    shelf_occupancy_pct=round(state.shelf_occupancy_pct, 1),
                    inference_time_ms=round(batch.inference_time_ms, 2),
                    detections=self._serialise_dets(batch),
                )
            )
            frame_idx += 1

        cap.release()
        writer.release()
        if show_display:
            cv2.destroyAllWindows()

        total_ms = (time.perf_counter() - run_start) * 1000.0
        avg_fps = frame_idx / max(total_ms / 1000.0, 1e-6)
        print(f"\n[ShelfPipeline] Annotated video  ->  {out_path}")

        report = ShelfReport(
            source_path=str(video_path),
            model_path=self.config.model.model_path,
            model_tier=self.config.model.model_tier,
            processing_time_ms=round(total_ms, 2),
            total_frames=frame_idx,
            avg_visible_facings=round(
                sum(visible_counts) / max(len(visible_counts), 1), 2
            ),
            peak_visible_facings=max(visible_counts) if visible_counts else 0,
            min_visible_facings=min(visible_counts) if visible_counts else 0,
            state_changes_detected=state_changes,
            uncertain_frames=uncertain_count,
            avg_fps=round(avg_fps, 2),
            avg_inference_ms=round(
                sum(inference_times) / max(len(inference_times), 1), 2
            ),
            per_frame_results=per_frame_results,
        )
        json_path = out_path.with_suffix(".json")
        report.save_json(str(json_path))
        report.print_summary()
        return report

    # ------------------------------------------------------------------
    # Folder mode
    # ------------------------------------------------------------------

    def _run_folder(
        self,
        folder: Path,
        output_path: Optional[str],
        max_frames: Optional[int],
    ) -> ShelfReport:
        images = sorted(
            f for f in folder.iterdir() if f.suffix.lower() in _IMAGE_EXTS
        )
        if not images:
            raise ValueError(f"No supported images found in folder: {folder}")
        if max_frames:
            images = images[:max_frames]

        print(f"\n[ShelfPipeline] Folder mode  ->  {folder}  ({len(images)} images)")
        print(f"[ShelfPipeline] Model  : {self.config.model.model_path}  [{self.config.model.model_tier}]")

        out_dir = Path(output_path) if output_path else self._output_dir / folder.name
        out_dir.mkdir(parents=True, exist_ok=True)

        self.state_tracker.reset()
        per_frame_results: List[PerFrameResult] = []
        visible_counts: List[int] = []
        inference_times: List[float] = []
        state_changes = 0
        uncertain_count = 0
        run_start = time.perf_counter()

        for idx, img_path in enumerate(images):
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"  [SKIP] {img_path.name}")
                continue

            batch = self.detector.detect(frame, frame_index=idx)
            state = self.state_tracker.update(batch, timestamp_sec=float(idx))

            annotated = self.visualizer.annotate_frame(
                frame, batch, state,
                fps=0.0,
                shelf_roi=self.config.shelf_roi,
                model_tier=self.detector.model_tier,
            )

            save_path = out_dir / f"{img_path.stem}_annotated.jpg"
            cv2.imwrite(str(save_path), annotated)

            visible_counts.append(batch.visible_count)
            inference_times.append(batch.inference_time_ms)
            if state.state_change_detected:
                state_changes += 1
            if state.observation_status == ObservationStatus.UNCERTAIN:
                uncertain_count += 1

            print(
                f"  [{idx + 1:3d}/{len(images)}] {img_path.name:<30s} | "
                f"Facings: {batch.visible_count:3d} | "
                f"State: {state.observation_status.value}"
            )

            per_frame_results.append(
                PerFrameResult(
                    frame_index=idx,
                    timestamp_sec=float(idx),
                    visible_count=batch.visible_count,
                    person_detected=batch.person_present,
                    observation_status=state.observation_status.value,
                    stable_visible_count=state.stable_visible_count,
                    state_change_detected=state.state_change_detected,
                    shelf_occupancy_pct=round(state.shelf_occupancy_pct, 1),
                    inference_time_ms=round(batch.inference_time_ms, 2),
                    detections=self._serialise_dets(batch),
                )
            )

        total_ms = (time.perf_counter() - run_start) * 1000.0
        report = ShelfReport(
            source_path=str(folder),
            model_path=self.config.model.model_path,
            model_tier=self.config.model.model_tier,
            processing_time_ms=round(total_ms, 2),
            total_frames=len(per_frame_results),
            avg_visible_facings=round(
                sum(visible_counts) / max(len(visible_counts), 1), 2
            ),
            peak_visible_facings=max(visible_counts) if visible_counts else 0,
            min_visible_facings=min(visible_counts) if visible_counts else 0,
            state_changes_detected=state_changes,
            uncertain_frames=uncertain_count,
            avg_fps=0.0,
            avg_inference_ms=round(
                sum(inference_times) / max(len(inference_times), 1), 2
            ),
            per_frame_results=per_frame_results,
        )
        json_path = out_dir / "shelf_analysis_report.json"
        report.save_json(str(json_path))
        report.print_summary()
        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_output(
        self,
        src: Path,
        explicit: Optional[str],
        suffix: str,
        ext: str,
    ) -> Path:
        if explicit:
            p = Path(explicit)
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        return self._output_dir / f"{src.stem}{suffix}{ext}"

    @staticmethod
    def _serialise_dets(batch: "ShelfDetectionBatch") -> List[dict]:
        return [
            {
                "bbox": list(d.bbox),
                "confidence": round(d.confidence, 4),
                "class_id": d.class_id,
                "class_name": d.class_name,
                "is_person": d.is_person,
            }
            for d in batch.all_detections
        ]
