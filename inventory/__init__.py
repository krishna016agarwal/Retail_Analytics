"""Retail Shelf Inventory Analysis Package — Foundation (Stage 1).

Architecture
------------
This package is completely isolated from the crowd/queue pipeline (src/).
No imports from src.* exist anywhere in this package.

Stage 1 (this package)
    Product Detection  ->  Shelf State  ->  Visualisation  ->  JSON Report

Future stages (not yet implemented)
    SKU Recognition  ->  Planogram Compliance  ->  POS Reconciliation
    ->  Low-Stock / OOS Alert  ->  Replenishment Recommendation

Model tiers
-----------
coco_baseline     Generic COCO objects. NOT retail brand / SKU recognition.
retail_specific   Custom model (e.g. SKU-110K fine-tuned). Plug-in via config.
sku_recognition   Future.
"""

from inventory.config import (
    COCO_PERSON_CLASS_ID,
    COCO_RETAIL_CLASS_IDS,
    InventoryConfig,
    InventoryModelConfig,
    InventoryTemporalConfig,
)
from inventory.shelf_detector import (
    BaseShelfModel,
    ShelfDetection,
    ShelfDetectionBatch,
    ShelfProductDetector,
    YOLOShelfModel,
)
from inventory.shelf_state import (
    ObservationStatus,
    ShelfObservation,
    ShelfStateSnapshot,
    ShelfStateTracker,
)
from inventory.shelf_visualizer import ShelfVisualizer
from inventory.shelf_pipeline import ShelfAnalysisPipeline
from inventory.report import PerFrameResult, ShelfReport

__all__ = [
    # Config
    "COCO_PERSON_CLASS_ID",
    "COCO_RETAIL_CLASS_IDS",
    "InventoryConfig",
    "InventoryModelConfig",
    "InventoryTemporalConfig",
    # Detection
    "BaseShelfModel",
    "ShelfDetection",
    "ShelfDetectionBatch",
    "ShelfProductDetector",
    "YOLOShelfModel",
    # State
    "ObservationStatus",
    "ShelfObservation",
    "ShelfStateSnapshot",
    "ShelfStateTracker",
    # Visualisation
    "ShelfVisualizer",
    # Pipeline
    "ShelfAnalysisPipeline",
    # Report
    "PerFrameResult",
    "ShelfReport",
]
