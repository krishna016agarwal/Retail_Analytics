"""Configuration for the Retail Shelf Inventory Analysis Module.

This module is completely isolated from the crowd/queue pipeline (src/).
No imports from src.* are present here.

Model Tier Hierarchy
--------------------
Stage 1  coco_baseline    Generic COCO objects (bottle, cup, etc.)
                          Used ONLY as a development/demo baseline.
                          Does NOT identify retail brands or SKUs.

Stage 2  retail_specific  Custom YOLO trained on a retail shelf dataset
                          (e.g., fine-tuned on SKU-110K or proprietary data).
                          Better product localisation without SKU labels.
                          Swap in via config -- zero pipeline code changes.

Stage 3  sku_recognition  Future: model predicts specific brand / SKU labels.
                          Requires annotated SKU training data.

To upgrade the model, only change InventoryModelConfig.model_path and
InventoryModelConfig.model_tier. The detection pipeline, temporal analysis,
visualizer and report system are model-agnostic.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# COCO class IDs -- retail-plausible subset
# ---------------------------------------------------------------------------
# These are used ONLY when model_tier == "coco_baseline".
# Class 0 (person) is included to serve as an occlusion signal;
# it is NOT counted in visible_facings.
# A custom retail model should use target_classes = None (all model classes).

COCO_PERSON_CLASS_ID: int = 0

COCO_RETAIL_CLASS_IDS: List[int] = [
    0,   # person   -> occlusion signal; excluded from product count
    39,  # bottle
    40,  # wine glass
    41,  # cup
    45,  # bowl
    46,  # banana
    47,  # apple
    48,  # sandwich
    49,  # orange
    50,  # broccoli
    51,  # carrot
    52,  # hot dog
    53,  # pizza
    54,  # donut
    55,  # cake
    63,  # laptop
    64,  # mouse
    65,  # remote
    66,  # keyboard
    67,  # cell phone
    73,  # book
    74,  # clock
    75,  # vase
    76,  # scissors
    77,  # teddy bear
    78,  # hair drier
    79,  # toothbrush
]


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class InventoryModelConfig:
    """Configuration for the shelf product detection model.

    The model interface is fully replaceable: set model_path and model_tier,
    and the entire downstream pipeline (detection, state, visualizer, report)
    requires zero code changes.

    Attributes:
        model_path: Path to YOLO .pt weights.
                    Default reuses yolo11n.pt already present in the project
                    (COCO baseline -- generic objects only).
                    Set to 'inventory_data/custom_model/retail_yolo.pt' for
                    a custom-trained retail model.
        model_tier: Declared capability of the model weights.
                    'coco_baseline'   -> generic COCO objects. NOT brand/SKU.
                    'retail_specific' -> retail shelf trained. Better localisation.
                    'sku_recognition' -> future; predicts brand/SKU labels.
        device: Hardware target ('cpu', '0' for CUDA GPU).
        confidence_threshold: Detection confidence cutoff. Lower than the
                    person-detection default (0.40) to handle dense, partially
                    occluded retail products.
        iou_threshold: NMS IoU threshold.
        imgsz: Inference frame resolution.
        target_classes: YOLO class IDs to detect.
                    For coco_baseline: use COCO_RETAIL_CLASS_IDS.
                    For retail_specific / sku_recognition: set to None
                    (detect all classes the model knows).
    """

    model_path: str = "yolo11n.pt"
    model_tier: str = "coco_baseline"  # "coco_baseline" | "retail_specific" | "sku_recognition"
    device: str = "cpu"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.40
    imgsz: int = 640
    target_classes: Optional[List[int]] = field(
        default_factory=lambda: list(COCO_RETAIL_CLASS_IDS)
    )


# ---------------------------------------------------------------------------
# Temporal stability configuration
# ---------------------------------------------------------------------------

@dataclass
class InventoryTemporalConfig:
    """Temporal stability parameters for shelf-state analysis.

    Prevents single-frame anomalies (customer blocking the shelf, motion blur,
    lighting transients) from being misinterpreted as real inventory changes.

    Attributes:
        window_size: Number of frames in the rolling observation window.
        min_stable_frames: Consecutive reliable frames required before an
                    observation is promoted to STABLE or a state change is
                    confirmed as STATE_CHANGED.
        occlusion_drop_threshold: If visible_count / stable_baseline drops
                    below this ratio AND the absolute drop is > 1, the frame
                    is classified as UNCERTAIN rather than a real change.
                    Example: 0.50 means a >50 % drop triggers uncertainty.
        change_threshold: Minimum absolute delta in visible_count (across
                    stable clear frames) needed to raise POSSIBLY_CHANGING.
    """

    window_size: int = 15
    min_stable_frames: int = 8
    occlusion_drop_threshold: float = 0.50
    change_threshold: int = 3


# ---------------------------------------------------------------------------
# SKU Recognition configuration
# ---------------------------------------------------------------------------

@dataclass
class SKURecognitionConfig:
    """Configuration for matching detected product crops against a catalog.

    Attributes:
        enabled: Whether to run SKU recognition downstream of product detection.
        catalog_path: Path to store catalog JSON file.
        match_threshold: Minimum cosine similarity score required to accept an SKU match.
                         Scores below this threshold are classified as UNKNOWN.
        unknown_label: Label assigned to unrecognized product crops.
    """

    enabled: bool = False
    catalog_path: Optional[str] = None
    match_threshold: float = 0.60
    unknown_label: str = "UNKNOWN"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class InventoryConfig:
    """Top-level configuration for the Inventory Foundation module.

    Attributes:
        model: Model configuration (path, tier, thresholds).
        temporal: Temporal stability window settings.
        sku_recognition: SKU recognition and store catalog settings.
        shelf_roi: Optional shelf Region of Interest in pixel coordinates
                   (x1, y1, x2, y2). If None, the entire frame is the shelf.
                   Occupancy % is computed relative to this area.
        output_dir: Directory for annotated outputs and JSON reports.
    """

    model: InventoryModelConfig = field(default_factory=InventoryModelConfig)
    temporal: InventoryTemporalConfig = field(default_factory=InventoryTemporalConfig)
    sku_recognition: SKURecognitionConfig = field(default_factory=SKURecognitionConfig)
    shelf_roi: Optional[Tuple[int, int, int, int]] = None
    output_dir: str = "output/shelf_analysis"
