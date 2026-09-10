"""Shelf product detection module.

Provides a model-agnostic detection interface for retail shelf analysis.
This module is completely isolated from the crowd/queue pipeline (src/).

Model Tiers
-----------
coco_baseline
    Generic COCO object classes (bottle, cup, remote, etc.).
    Used ONLY as a development/demo baseline.
    Does NOT identify retail brands or SKUs.
    Detecting a "bottle" does NOT mean identifying Coca-Cola, Pepsi, or Bisleri.

retail_specific
    Custom YOLO trained on a retail shelf dataset (e.g. SKU-110K fine-tuned).
    Better dense-product localisation. Plug-in via config — zero code changes.

sku_recognition
    Future: model predicts specific brand/SKU labels.
    Requires per-SKU annotated training data.

Plug-in Interface
-----------------
BaseShelfModel is an abstract base. Subclass it to plug in any backend
(ONNX, TFLite, cloud API) without rewriting the pipeline.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from inventory.config import COCO_PERSON_CLASS_ID, InventoryModelConfig


# ---------------------------------------------------------------------------
# Detection output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ShelfDetection:
    """A single detected product or person region on a shelf frame.

    Attributes:
        bbox: Bounding box (x1, y1, x2, y2) in integer pixels.
        confidence: Model confidence score in [0.0, 1.0].
        class_id: Model class index.
        class_name: Human-readable class label from the model.
                    For COCO baseline this is a generic category label
                    (e.g. 'bottle'). It does NOT identify any retail brand
                    or SKU. SKU recognition requires a custom trained model.
        is_person: True if this detection is a person — used as an occlusion
                   signal; excluded from the visible_facings count.
    """

    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
    is_person: bool = False
    sku_id: Optional[str] = None
    sku_name: Optional[str] = None
    sku_confidence: Optional[float] = None
    is_known_sku: bool = False
    track_id: Optional[int] = None
    temporal_state: str = "UNCERTAIN"

    def crop_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """Crop product bounding box from the frame with boundary clamping."""
        if frame is None or frame.size == 0:
            return np.empty((0, 0, 3), dtype=np.uint8)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.bbox
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))
        return frame[y1:y2, x1:x2]


@dataclass
class ShelfDetectionBatch:
    """Frame-level detection results for shelf analysis.

    Attributes:
        all_detections: All detections (products + persons combined).
        product_detections: Detected product regions (persons excluded).
        person_detections: Detected persons (occlusion signal only).
        inference_time_ms: Model inference duration in milliseconds.
        model_tier: Capability tier of the model that produced these results.
        frame_index: Frame number within the video sequence (0 for images).
    """

    all_detections: List[ShelfDetection] = field(default_factory=list)
    product_detections: List[ShelfDetection] = field(default_factory=list)
    person_detections: List[ShelfDetection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    model_tier: str = "coco_baseline"
    frame_index: int = 0

    @property
    def visible_count(self) -> int:
        """Number of visible product facings detected in this frame.

        IMPORTANT: camera-observable front-row count ONLY.
        This is NOT total physical inventory quantity.
        Products stacked behind the front row are invisible to the camera
        and are not counted here.
        Example: 5 visible bottle detections does NOT mean the store has
        exactly 5 bottles in stock.
        """
        return len(self.product_detections)

    @property
    def person_present(self) -> bool:
        """True if at least one person is detected (potential shelf occlusion)."""
        return len(self.person_detections) > 0


# ---------------------------------------------------------------------------
# Abstract model interface (plug-in protocol)
# ---------------------------------------------------------------------------


class BaseShelfModel:
    """Abstract base class for shelf product detection model backends.

    Subclass this to implement any inference backend without modifying
    the downstream pipeline.

    Provided implementations:
        YOLOShelfModel  Ultralytics YOLO (COCO or custom retail weights).

    Future backends (not yet implemented):
        ONNXShelfModel      ONNX Runtime (optimised CPU/GPU).
        TFLiteShelfModel    TensorFlow Lite for embedded edge devices.
        CloudShelfModel     Proxy to a cloud vision API endpoint.

    Example custom backend::

        class MyRetailModel(BaseShelfModel):
            @property
            def model_tier(self): return "retail_specific"
            def predict_frame(self, frame):
                detections = run_my_inference(frame)
                return detections, inference_ms

        detector = ShelfProductDetector(config, model=MyRetailModel())
    """

    @property
    def model_tier(self) -> str:
        """Return tier: 'coco_baseline', 'retail_specific', or 'sku_recognition'."""
        raise NotImplementedError

    @property
    def supports_sku(self) -> bool:
        """True only if this model predicts specific SKU/brand labels."""
        return False

    def predict_frame(
        self,
        frame: np.ndarray,
        track: bool = False,
        persist: bool = True,
        tracker: str = "bytetrack.yaml",
    ) -> Tuple[List[Tuple[Tuple[int, int, int, int], float, int, str, Optional[int]]], float]:
        """Run inference on a single BGR frame.

        Args:
            frame: Input image (H, W, 3) BGR numpy array.
            track: If True, execute multi-object tracking (e.g. ByteTrack).
            persist: If True, persist tracks across consecutive video frames.
            tracker: Tracker configuration name or file ('bytetrack.yaml' or 'botsort.yaml').

        Returns:
            Tuple of:
                raw_detections: list of (bbox_xyxy, confidence, class_id, class_name, track_id)
                inference_time_ms: float
        """
        raise NotImplementedError

    def warmup(self) -> None:
        """Pre-warm model runtime. Called once at init. Override if needed."""
        pass


# ---------------------------------------------------------------------------
# YOLO backend implementation
# ---------------------------------------------------------------------------


class YOLOShelfModel(BaseShelfModel):
    """Ultralytics YOLO-based shelf model.

    Compatible with:
    - COCO pretrained weights (yolo11n.pt, yolov8n.pt, ...) for generic baseline.
    - Custom retail-trained weights (SKU-110K fine-tune, proprietary) for better
      dense product localisation.
    - Future SKU-recognition weights.

    To swap to a custom model (zero pipeline changes required)::

        config.model_path     = "inventory_data/custom_model/retail_sku.pt"
        config.model_tier     = "retail_specific"
        config.target_classes = None   # use all classes the custom model knows
        detector = ShelfProductDetector(config)
    """

    def __init__(self, config: InventoryModelConfig) -> None:
        from ultralytics import YOLO  # type: ignore

        self._config = config
        self._tier = config.model_tier
        self._model = YOLO(config.model_path)
        if hasattr(self._model, "to"):
            self._model.to(config.device)
        self.warmup()

    @property
    def model_tier(self) -> str:
        return self._tier

    @property
    def supports_sku(self) -> bool:
        return self._tier == "sku_recognition"

    def warmup(self) -> None:
        """Run a dummy inference pass to eliminate JIT / cold-start latency."""
        dummy = np.zeros(
            (self._config.imgsz, self._config.imgsz, 3), dtype=np.uint8
        )
        try:
            self._model.predict(
                source=dummy,
                device=self._config.device,
                classes=self._config.target_classes,
                conf=self._config.confidence_threshold,
                iou=self._config.iou_threshold,
                imgsz=self._config.imgsz,
                verbose=False,
            )
        except Exception:
            pass  # Non-fatal warmup failure

    def predict_frame(
        self,
        frame: np.ndarray,
        track: bool = False,
        persist: bool = True,
        tracker: str = "bytetrack.yaml",
    ) -> Tuple[List[Tuple[Tuple[int, int, int, int], float, int, str, Optional[int]]], float]:
        """Run YOLO inference or tracking and return raw detections."""
        t0 = time.perf_counter()
        if track:
            results = self._model.track(
                source=frame,
                persist=persist,
                tracker=tracker,
                device=self._config.device,
                classes=self._config.target_classes,
                conf=self._config.confidence_threshold,
                iou=self._config.iou_threshold,
                imgsz=self._config.imgsz,
                verbose=False,
            )
        else:
            results = self._model.predict(
                source=frame,
                device=self._config.device,
                classes=self._config.target_classes,
                conf=self._config.confidence_threshold,
                iou=self._config.iou_threshold,
                imgsz=self._config.imgsz,
                verbose=False,
            )
        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        raw: List[Tuple[Tuple[int, int, int, int], float, int, str, Optional[int]]] = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            names = results[0].names
            if boxes is not None and len(boxes) > 0:
                track_ids = (
                    boxes.id.cpu().numpy().astype(int)
                    if (track and boxes.id is not None)
                    else [None] * len(boxes)
                )
                for xyxy, conf, cls_id, trk_id in zip(
                    boxes.xyxy.cpu().numpy(),
                    boxes.conf.cpu().numpy(),
                    boxes.cls.cpu().numpy().astype(int),
                    track_ids,
                ):
                    x1, y1, x2, y2 = map(int, xyxy)
                    cls_name = names.get(int(cls_id), f"class_{cls_id}")
                    raw.append((
                        (x1, y1, x2, y2),
                        float(conf),
                        int(cls_id),
                        cls_name,
                        int(trk_id) if trk_id is not None else None,
                    ))

        return raw, inference_time_ms


# ---------------------------------------------------------------------------
# Main detector (model-agnostic orchestrator)
# ---------------------------------------------------------------------------


class ShelfProductDetector:
    """Model-agnostic shelf product detector.

    Accepts any BaseShelfModel backend and handles person-vs-product
    separation so downstream modules need not care about model internals.

    Quick-start::

        config   = InventoryModelConfig()
        detector = ShelfProductDetector(config)   # default YOLO backend
        batch    = detector.detect(frame)
        print(f"{batch.visible_count} visible facings")

    Swap the model without touching any other code::

        detector = ShelfProductDetector(config, model=MyCustomModel(config))
    """

    def __init__(
        self,
        config: InventoryModelConfig,
        model: Optional[BaseShelfModel] = None,
    ) -> None:
        self.config = config
        self._model: BaseShelfModel = (
            model if model is not None else YOLOShelfModel(config)
        )

    @property
    def model_tier(self) -> str:
        return self._model.model_tier

    @property
    def supports_sku(self) -> bool:
        return self._model.supports_sku

    def detect(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        track: bool = False,
        persist: bool = True,
        tracker: str = "bytetrack.yaml",
    ) -> ShelfDetectionBatch:
        """Detect visible products and persons in a shelf frame.

        Args:
            frame: BGR image numpy array.
            frame_index: Frame number in video sequence (0 for images).
            track: If True, executes multi-object tracking across frames.
            persist: If True, persists tracks across consecutive frames.
            tracker: Tracker configuration name or file ('bytetrack.yaml').

        Returns:
            ShelfDetectionBatch with separate product and person lists.

        Key output properties:
            batch.visible_count  -- camera-observable facings ONLY.
                                    NOT total physical inventory.
            batch.person_present -- True if a person may be blocking the shelf.
        """
        if frame is None or frame.size == 0:
            return ShelfDetectionBatch(model_tier=self._model.model_tier)

        raw_detections, inference_time_ms = self._model.predict_frame(
            frame, track=track, persist=persist, tracker=tracker
        )

        all_dets: List[ShelfDetection] = []
        products: List[ShelfDetection] = []
        persons: List[ShelfDetection] = []

        for item in raw_detections:
            bbox, conf, cls_id, cls_name = item[0], item[1], item[2], item[3]
            trk_id = item[4] if len(item) > 4 else None

            if self._model.model_tier == "coco_baseline":
                is_person = cls_id == COCO_PERSON_CLASS_ID or cls_name.lower() == "person"
            else:
                is_person = cls_name.lower() == "person"
            det = ShelfDetection(
                bbox=bbox,
                confidence=conf,
                class_id=cls_id,
                class_name=cls_name,
                is_person=is_person,
                track_id=trk_id,
            )
            all_dets.append(det)
            (persons if is_person else products).append(det)

        return ShelfDetectionBatch(
            all_detections=all_dets,
            product_detections=products,
            person_detections=persons,
            inference_time_ms=inference_time_ms,
            model_tier=self._model.model_tier,
            frame_index=frame_index,
        )
