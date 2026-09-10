"""Shelf analysis report generation.

Produces structured JSON output and human-readable summaries per run.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Contextual documentation embedded in every report
# ---------------------------------------------------------------------------

LIMITATIONS: List[str] = [
    (
        "COCO baseline model detects generic COCO object categories (bottle, cup, etc.), "
        "NOT retail brands or SKUs. Detecting 'bottle' does NOT identify Coca-Cola, Pepsi, "
        "Bisleri, or any other specific product."
    ),
    (
        "visible_facings reflects only camera-observable front-row products. "
        "Products stacked front-to-back are not visible and are NOT counted. "
        "Do NOT interpret visible_facings as total physical inventory quantity."
    ),
    (
        "Dense shelf products may partially overlap, causing merged or missed "
        "bounding boxes, especially with a generic COCO model."
    ),
    (
        "Temporal stability requires several frames to initialise the baseline. "
        "Single-image mode has no temporal analysis and reports STABLE or UNCERTAIN only."
    ),
    (
        "Person-as-occlusion detection is heuristic. The system flags UNCERTAIN "
        "when a person overlaps the shelf; it cannot identify whether the person "
        "is a customer, a stocker, or someone adjusting products."
    ),
    (
        "No POS/ERP reconciliation. Visible facings are NOT validated against "
        "actual stock records. This module is Stage 1 of a multi-stage pipeline."
    ),
    (
        "The model interface supports plug-in of retail-specific or SKU-trained "
        "model weights without any pipeline code changes."
    ),
]

NEXT_STEPS_FOR_CUSTOM_MODEL: List[str] = [
    (
        "Collect annotated retail shelf images with dense bounding boxes. "
        "SKU-110K is a good starting point for product localisation training "
        "(dense detection; no per-SKU class labels)."
    ),
    (
        "For brand/SKU classification, build a custom labelled dataset with "
        "per-SKU class annotations and fine-tune a YOLO classification head."
    ),
    (
        "Set InventoryModelConfig.model_path to the fine-tuned weights file and "
        "model_tier to 'retail_specific'. Set target_classes = None so the model "
        "uses its own full class set."
    ),
    (
        "For two-stage SKU recognition: run a dense-detection model for "
        "product localisation, then a crop-level classifier for SKU identification."
    ),
    (
        "Connect visible_facings with planogram data for spatial compliance "
        "checking (future pipeline stage)."
    ),
    (
        "Integrate with POS/ERP for reconciliation and alert generation "
        "(future pipeline stage — NOT implemented in Stage 1)."
    ),
]


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PerFrameResult:
    """Per-frame analysis record included in the JSON report."""

    frame_index: int
    timestamp_sec: float
    visible_count: int
    person_detected: bool
    observation_status: str
    stable_visible_count: int
    state_change_detected: bool
    shelf_occupancy_pct: float
    inference_time_ms: float
    detections: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ShelfReport:
    """Comprehensive shelf analysis report for a single run.

    IMPORTANT TERMINOLOGY
    ---------------------
    visible_facings / visible_count:
        Number of product bounding boxes the camera can directly observe
        in the front row of the shelf. This is NOT total physical stock.
        Products stacked behind the front row are not visible to the camera.

    stable_visible_count:
        Median visible count across reliable (non-occluded) frames in the
        temporal window. More robust than single-frame raw count.
    """

    source_path: str
    model_path: str
    model_tier: str
    processing_time_ms: float
    total_frames: int
    avg_visible_facings: float
    peak_visible_facings: int
    min_visible_facings: int
    state_changes_detected: int
    uncertain_frames: int
    avg_fps: float
    avg_inference_ms: float
    per_frame_results: List[PerFrameResult] = field(default_factory=list)
    limitations: List[str] = field(default_factory=lambda: list(LIMITATIONS))
    next_steps_for_custom_model: List[str] = field(
        default_factory=lambda: list(NEXT_STEPS_FOR_CUSTOM_MODEL)
    )
    notes: str = (
        "visible_facings reflects camera-observable product instances only. "
        "It does NOT represent total physical stock quantity. "
        "Do not use for inventory management without POS reconciliation."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return asdict(self)

    def save_json(self, path: str) -> None:
        """Write a formatted JSON report file.

        Args:
            path: Destination file path. Parent directories are created if needed.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        print(f"[Report] JSON saved  -> {path}")

    def print_summary(self) -> None:
        """Print a concise human-readable summary to stdout."""
        print()
        print("=" * 68)
        print("  RETAIL SHELF ANALYSIS  -  SUMMARY REPORT")
        print("=" * 68)
        print(f"  Source              : {self.source_path}")
        print(f"  Model               : {self.model_path}  [{self.model_tier}]")
        if self.model_tier == "coco_baseline":
            print("  [!] COCO baseline: detects generic objects, NOT retail SKUs.")
        print(f"  Total Frames        : {self.total_frames}")
        print(f"  Avg FPS             : {self.avg_fps:.1f}")
        print(f"  Avg Inference       : {self.avg_inference_ms:.1f} ms / frame")
        print(f"  Avg Visible Facings : {self.avg_visible_facings:.1f}")
        print(f"  Peak Visible        : {self.peak_visible_facings}")
        print(f"  Min  Visible        : {self.min_visible_facings}")
        print(f"  State Changes       : {self.state_changes_detected}")
        print(f"  Uncertain Frames    : {self.uncertain_frames}")
        print("-" * 68)
        print(f"  NOTE: {self.notes}")
        print("=" * 68)
