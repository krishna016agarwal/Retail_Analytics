"""Modular Person Detector utilizing Ultralytics YOLO.

This module isolates object detection from downstream modules like tracking,
re-identification, heatmaps, and retail shopper analytics.
"""

from dataclasses import dataclass, field
import time
from typing import List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from configs.config import DetectorConfig


@dataclass
class Detection:
    """Individual person detection representation.

    Attributes:
        bbox: Bounding box coordinates (x1, y1, x2, y2) in integer pixels.
        confidence: Model prediction confidence score in range [0.0, 1.0].
        class_id: Target class ID (0 for COCO 'person').
        class_name: Human-readable label ('person').
    """

    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int = 0
    class_name: str = "person"


@dataclass
class DetectionBatch:
    """Container for all person detections in a single frame.

    Structured for direct compatibility with upcoming tracking algorithms
    such as ByteTrack or DeepSORT.
    """

    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    raw_boxes: Optional[object] = None

    @property
    def count(self) -> int:
        """Return the number of people detected in this frame."""
        return len(self.detections)

    @property
    def xyxy_conf(self) -> np.ndarray:
        """Return detections in (N, 5) format [x1, y1, x2, y2, conf].

        This array layout matches standard ByteTrack/SORT tracking input.
        """
        if not self.detections:
            return np.empty((0, 5), dtype=np.float32)

        data = [
            [d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence]
            for d in self.detections
        ]
        return np.array(data, dtype=np.float32)


class PersonDetector:
    """YOLO-based person detector optimized for CPU execution."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        """Initialize detector with configuration and load YOLO weights.

        Args:
            config: DetectorConfig instance. If None, default settings are used.
        """
        self.config = config or DetectorConfig()
        self.device = self.config.device
        self.model_path = self.config.model_path

        # Load YOLO model
        self.model = YOLO(self.model_path)

        # Force CPU device execution
        if hasattr(self.model, "to"):
            self.model.to(self.device)

        # Perform warm-up inference to eliminate initial JIT/allocation latency
        self._warmup()

    def _warmup(self) -> None:
        """Run a dummy frame through the network to prepare CPU runtime caches."""
        dummy_frame = np.zeros((self.config.imgsz, self.config.imgsz, 3), dtype=np.uint8)
        try:
            self.model.predict(
                source=dummy_frame,
                device=self.device,
                classes=self.config.target_classes,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.imgsz,
                verbose=False,
            )
        except Exception:
            # Non-fatal if warm-up encounters minor issue
            pass

    def detect(self, frame: np.ndarray) -> DetectionBatch:
        """Detect people in an input BGR frame.

        Args:
            frame: Input video frame as a NumPy array (H, W, 3) in BGR format.

        Returns:
            DetectionBatch containing structured Detection objects.
        """
        if frame is None or frame.size == 0:
            return DetectionBatch()

        start_time = time.perf_counter()

        # Run inference strictly restricted to class 0 (person)
        results = self.model.predict(
            source=frame,
            device=self.device,
            classes=self.config.target_classes,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            verbose=False,
        )

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        detections: List[Detection] = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy_coords = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)

                for xyxy, conf, cls_id in zip(xyxy_coords, confs, cls_ids):
                    # Ensure strictly person class
                    if cls_id == 0:
                        x1, y1, x2, y2 = map(int, xyxy)
                        detections.append(
                            Detection(
                                bbox=(x1, y1, x2, y2),
                                confidence=float(conf),
                                class_id=0,
                                class_name="person",
                            )
                        )

        raw_boxes = results[0].boxes if (results and len(results) > 0) else None

        return DetectionBatch(
            detections=detections,
            inference_time_ms=inference_time_ms,
            raw_boxes=raw_boxes,
        )
