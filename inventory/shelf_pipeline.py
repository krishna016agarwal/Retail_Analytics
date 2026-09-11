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
        display_scale: Optional[float] = None,
        display_max_h: int = 700,
        display_max_w: int = 960,
    ) -> ShelfReport:
        """Auto-detect source type and run the appropriate sub-pipeline.

        Args:
            source: Path to image file, video file, or folder of images.
            output_path: Explicit output path (auto-named if None).
            show_display: Show a live OpenCV window (video mode only).
            max_frames: Maximum frames / images to process (None = all).
            display_scale: Manual scale factor for OpenCV window (e.g. 0.4).
            display_max_h: Maximum window height in pixels (default: 700).
            display_max_w: Maximum window width in pixels (default: 960).

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
            return self._run_video(
                src, output_path, show_display, max_frames,
                display_scale=display_scale,
                display_max_h=display_max_h,
                display_max_w=display_max_w,
            )
        # Fallback: try video
        return self._run_video(
            src, output_path, show_display, max_frames,
            display_scale=display_scale,
            display_max_h=display_max_h,
            display_max_w=display_max_w,
        )

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
        display_scale: Optional[float] = None,
        display_max_h: int = 700,
        display_max_w: int = 960,
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

        win_name = "Retail Shelf Monitor - Inventory Foundation"
        scale = 1.0
        disp_w, disp_h = w, h
        if show_display:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            if display_scale is not None and display_scale > 0:
                scale = float(display_scale)
            else:
                scale = min(display_max_w / max(w, 1), display_max_h / max(h, 1), 1.0)
            disp_w = max(1, int(w * scale))
            disp_h = max(1, int(h * scale))
            cv2.resizeWindow(win_name, disp_w, disp_h)
            if scale < 1.0:
                print(f"[ShelfPipeline] Display auto-scaled to {disp_w}x{disp_h} ({scale * 100:.0f}%) to fit screen.")

        self.state_tracker.reset()

        product_slots = self._load_planogram_slots()
        prev_slot_counts = {}
        active_alert_message = None
        evidence_dir = Path("dashboard/public/evidence")
        evidence_dir.mkdir(parents=True, exist_ok=True)

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
        print(f"[ShelfPipeline] Slots  : {len(product_slots)} monitored shelf columns/zones")
        print(f"[ShelfPipeline] Mode   : Active products = GREEN  |  Finished/Empty space = RED COLUMN")
        print("[ShelfPipeline] Press 'q' to quit early.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_idx >= max_frames:
                break

            t_frame = time.perf_counter()
            ts = frame_idx / fps_src

            batch = self.detector.detect(frame, frame_index=frame_idx, track=False)
            state = self.state_tracker.update(batch, timestamp_sec=ts)

            # 1. Map product detections to monitored shelf columns/slots
            slot_counts = {s["slot_id"]: 0 for s in product_slots}
            for det in batch.product_detections:
                bx1, by1, bx2, by2 = det.bbox
                cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                for s in product_slots:
                    x1_pct, y1_pct, x2_pct, y2_pct = s["zone_bbox_pct"]
                    zx1, zy1 = int(x1_pct * w), int(y1_pct * h)
                    zx2, zy2 = int(x2_pct * w), int(y2_pct * h)
                    if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                        slot_counts[s["slot_id"]] += 1

            # 2. Check for empty shelf space / finished products (count == 0)
            finished_slots = [
                s for s in product_slots
                if slot_counts.get(s["slot_id"], 0) == 0
            ]

            # 3. Detect customer pick interactions (count decrease while person present)
            if batch.person_present and prev_slot_counts:
                for s in product_slots:
                    sid = s["slot_id"]
                    curr = slot_counts.get(sid, 0)
                    prev = prev_slot_counts.get(sid, curr)
                    if prev - curr >= 3:
                        active_alert_message = f"ITEM PICKED: Customer took products from {s['product_name']} (Remaining: {curr})"

            prev_slot_counts = dict(slot_counts)

            if finished_slots:
                p_first = finished_slots[0]["product_name"]
                active_alert_message = f"EMPTY SPACE DETECTED: {p_first} is FINISHED! Alert sent to Dashboard."

            # 4. Annotate frame: green boxes for present products, red columns for finished/empty slots
            annotated = self.visualizer.annotate_frame(
                frame,
                batch,
                state,
                fps=fps_smooth,
                shelf_roi=self.config.shelf_roi,
                model_tier=self.detector.model_tier,
                product_slots=product_slots,
                slot_counts=slot_counts,
                finished_slots=finished_slots,
                active_alert_message=active_alert_message,
            )

            # 5. Live Dashboard Sync (telemetry & evidence snapshot)
            if frame_idx % 25 == 0 or finished_slots:
                self._sync_dashboard_telemetry(
                    annotated,
                    frame_idx,
                    product_slots,
                    slot_counts,
                    finished_slots,
                    evidence_dir,
                    batch.inference_time_ms,
                )

            writer.write(annotated)

            if show_display:
                if scale < 0.99 or scale > 1.01:
                    disp_frame = cv2.resize(
                        annotated, (disp_w, disp_h), interpolation=cv2.INTER_AREA
                    )
                else:
                    disp_frame = annotated
                cv2.imshow(win_name, disp_frame)
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
                fin_str = f" | Finished: {len(finished_slots)}" if finished_slots else ""
                print(
                    f"  Frame {frame_idx:5d} | Facings: {batch.visible_count:3d}{fin_str} | "
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

    def _load_planogram_slots(self) -> List[dict]:
        """Load product slot definitions from config file."""
        planogram_file = Path("configs/shelf_planogram_config.json")
        if planogram_file.is_file():
            try:
                import json
                with open(planogram_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("product_slots", [])
            except Exception as e:
                print(f"[ShelfPipeline] Warning reading planogram: {e}")
        return []

    def _sync_dashboard_telemetry(
        self,
        frame: np.ndarray,
        frame_idx: int,
        product_slots: List[dict],
        slot_counts: dict,
        finished_slots: List[dict],
        evidence_dir: Path,
        inference_ms: float,
    ) -> None:
        """Sync live shelf slot telemetry and visual snapshot to dashboard public evidence folder."""
        from datetime import datetime, timezone
        import json

        finished_ids = {s["slot_id"] for s in finished_slots}
        products_data = []
        alerts = []
        total_observed = 0
        total_capacity = 0
        total_deficit = 0
        finished_count = len(finished_slots)
        low_count = 0
        in_stock_count = 0

        for slot in product_slots:
            sid = slot["slot_id"]
            pname = slot["product_name"]
            cat = slot.get("category", "General")
            loc = slot.get("location", "Main Aisle")
            cap = slot.get("capacity", 10)
            low_th = slot.get("low_stock_threshold", 3)
            cnt = slot_counts.get(sid, 0)
            deficit = max(0, cap - cnt)
            occupancy = round((cnt / max(cap, 1)) * 100.0, 1)

            total_observed += cnt
            total_capacity += cap
            total_deficit += deficit

            if sid in finished_ids or cnt == 0:
                st = "SOLD_OUT"
                st_label = "SOLD OUT / FINISHED"
                is_fin = True
                sev = "HIGH"
                rec = f"URGENT: {pname} is completely SOLD OUT / FINISHED from shelf at {loc}. Restock {cap} units immediately."
                alerts.append({
                    "alert_id": f"ALT-{sid}-{int(time.time())}",
                    "slot_id": sid,
                    "product_name": pname,
                    "category": cat,
                    "location": loc,
                    "severity": sev,
                    "status": st,
                    "is_finished": True,
                    "message": rec,
                    "observed": cnt,
                    "capacity": cap,
                    "deficit": deficit,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            elif cnt <= low_th:
                st = "LOW_STOCK"
                st_label = "LOW STOCK"
                is_fin = False
                low_count += 1
                sev = "MEDIUM"
                rec = f"LOW STOCK: Only {cnt} units of {pname} left at {loc}. Restock {deficit} units."
                alerts.append({
                    "alert_id": f"ALT-{sid}-{int(time.time())}",
                    "slot_id": sid,
                    "product_name": pname,
                    "category": cat,
                    "location": loc,
                    "severity": sev,
                    "status": st,
                    "is_finished": False,
                    "message": rec,
                    "observed": cnt,
                    "capacity": cap,
                    "deficit": deficit,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            else:
                st = "IN_STOCK"
                st_label = "IN STOCK"
                is_fin = False
                in_stock_count += 1
                sev = "NONE"
                rec = f"Normal stock. {cnt} units available at {loc}."

            products_data.append({
                "slot_id": sid,
                "product_name": pname,
                "category": cat,
                "location": loc,
                "capacity": cap,
                "observed_count": cnt,
                "deficit": deficit,
                "occupancy_pct": occupancy,
                "status": st,
                "status_label": st_label,
                "is_finished": is_fin,
                "severity": sev,
                "recommendation": rec,
            })

        payload = {
            "scan_id": f"LIVE-{frame_idx}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "aisle_name": "Main Aisle — Snacks & Packaged Goods",
            "camera_id": "CAM_CONVENIENCE_AISLE_1",
            "source": "inventory.mp4",
            "frame_index": frame_idx,
            "inference_time_ms": round(inference_ms, 1),
            "total_products_monitored": len(products_data),
            "in_stock_products_count": in_stock_count,
            "low_stock_products_count": low_count,
            "finished_products_count": finished_count,
            "total_observed_facings": total_observed,
            "total_capacity": total_capacity,
            "total_deficit": total_deficit,
            "overall_occupancy_pct": round((total_observed / max(total_capacity, 1)) * 100.0, 1),
            "overall_status": "SOLD_OUT" if finished_count > 0 else ("LOW_STOCK" if low_count > 0 else "HEALTHY"),
            "products": products_data,
            "alerts": alerts,
            "snapshot_image_url": f"/evidence/latest_shelf_snapshot.jpg?t={int(time.time())}",
        }

        try:
            with open(evidence_dir / "latest_shelf_snapshot.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            cv2.imwrite(str(evidence_dir / "latest_shelf_snapshot.jpg"), frame)
        except Exception:
            pass
