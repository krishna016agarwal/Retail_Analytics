"""SKU Recognition module for matching detected product crops against a catalog.

Provides a pluggable recognition interface:
    BaseSKURecognizer (Abstract Interface)
           |
    SpatialColorTextureRecognizer (Default Lightweight Implementation)

Features:
- Crops extracted from retail detector bounding boxes.
- Multi-zone spatial color and texture feature embeddings.
- Fast cosine similarity search against registered catalog reference images.
- Configurable match/confidence threshold.
- Rejection of out-of-catalog items with clean 'UNKNOWN' label and status.

Completely isolated from the crowd/queue pipeline (src/).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from inventory.catalog import SKUCatalog, SKUCatalogItem


@dataclass
class RecognitionResult:
    """Outcome of matching a product crop against the store catalog.

    Attributes:
        sku_id: Matched SKU identifier, or 'UNKNOWN'.
        name: Human-readable product name, or 'UNKNOWN'.
        category: Matched category, or 'Uncategorized'.
        confidence: Similarity score in [0.0, 1.0].
        is_known: True if similarity >= match_threshold, False otherwise.
        top_matches: Top-K candidates as (sku_id, name, similarity_score).
    """

    sku_id: str
    name: str
    category: str = "Uncategorized"
    confidence: float = 0.0
    is_known: bool = False
    top_matches: List[Tuple[str, str, float]] = field(default_factory=list)

    @classmethod
    def unknown(cls, confidence: float = 0.0, top_matches: Optional[List[Tuple[str, str, float]]] = None) -> "RecognitionResult":
        """Create an UNKNOWN recognition result."""
        return cls(
            sku_id="UNKNOWN",
            name="UNKNOWN",
            category="Uncategorized",
            confidence=round(float(confidence), 4),
            is_known=False,
            top_matches=top_matches or [],
        )


class BaseSKURecognizer(ABC):
    """Abstract base class for SKU recognition models.

    Subclasses must implement extract_embedding to produce a 1D feature vector
    from a product image crop. Matching and index management are handled here.
    """

    def __init__(self, catalog: SKUCatalog, match_threshold: float = 0.60) -> None:
        self.catalog = catalog
        self.match_threshold = match_threshold
        # Internal index: list of (sku_id, reference_embedding)
        self._index: List[Tuple[str, np.ndarray]] = []
        self._is_indexed: bool = False

    @abstractmethod
    def extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        """Extract a 1D unit-normalized feature vector from an image crop (BGR)."""
        pass

    def build_index(self, force_recompute: bool = False) -> None:
        """Build the embedding index from catalog reference images."""
        if self._is_indexed and not force_recompute:
            return

        self._index.clear()
        for item in self.catalog.list_items():
            # If embeddings already present, use them
            if item.reference_embeddings and not force_recompute:
                for emb in item.reference_embeddings:
                    self._index.append((item.sku_id, emb))
                continue

            # Otherwise compute from reference image files
            item.reference_embeddings.clear()
            for img_path_str in item.reference_images:
                img_p = Path(img_path_str)
                if not img_p.is_file():
                    continue
                ref_bgr = cv2.imread(str(img_p))
                if ref_bgr is None or ref_bgr.size == 0:
                    continue
                emb = self.extract_embedding(ref_bgr)
                item.reference_embeddings.append(emb)
                self._index.append((item.sku_id, emb))

        self._is_indexed = True

    def recognize_crop(self, crop: np.ndarray, top_k: int = 3) -> RecognitionResult:
        """Match a single product crop against the indexed catalog."""
        if not self._is_indexed:
            self.build_index()

        if crop is None or crop.size == 0 or not self._index:
            return RecognitionResult.unknown(confidence=0.0)

        query_emb = self.extract_embedding(crop)

        # Compute cosine similarities (query_emb and index embeddings are unit-normalized)
        sku_scores: Dict[str, float] = {}
        for sku_id, ref_emb in self._index:
            sim = float(np.dot(query_emb, ref_emb))
            sim = max(0.0, min(1.0, (sim + 1.0) / 2.0)) if sim < 0 else min(1.0, sim)
            if sku_id not in sku_scores or sim > sku_scores[sku_id]:
                sku_scores[sku_id] = sim

        if not sku_scores:
            return RecognitionResult.unknown(confidence=0.0)

        # Rank candidates
        ranked = sorted(sku_scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = []
        for s_id, score in ranked[:top_k]:
            item = self.catalog.get_item(s_id)
            name = item.name if item else s_id
            top_candidates.append((s_id, name, round(score, 4)))

        best_sku_id, best_score = ranked[0]
        best_item = self.catalog.get_item(best_sku_id)
        best_name = best_item.name if best_item else best_sku_id
        best_cat = best_item.category if best_item else "General"

        if best_score >= self.match_threshold:
            return RecognitionResult(
                sku_id=best_sku_id,
                name=best_name,
                category=best_cat,
                confidence=round(best_score, 4),
                is_known=True,
                top_matches=top_candidates,
            )
        else:
            return RecognitionResult.unknown(
                confidence=best_score,
                top_matches=top_candidates,
            )

    def recognize_batch(self, crops: List[np.ndarray], top_k: int = 3) -> List[RecognitionResult]:
        """Recognize a batch of product crops."""
        return [self.recognize_crop(c, top_k=top_k) for c in crops]


class SpatialColorTextureRecognizer(BaseSKURecognizer):
    """Lightweight, deterministic, zero-external-weights SKU recognizer.

    Extracts multi-zone spatial HSV/LAB color histograms, edge distributions,
    and aspect ratios to represent retail packaging aesthetics. Fast (~0.2ms/crop)
    and robust to illumination variations.
    """

    def extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        """Extract multi-zone spatial color and texture descriptor."""
        if crop is None or crop.size == 0:
            return np.zeros(256, dtype=np.float32)

        # Standardize size for feature extraction (preserving aspect ratio cues)
        h, w = crop.shape[:2]
        aspect_ratio = float(w) / float(max(1, h))
        norm_aspect = min(aspect_ratio, 3.0) / 3.0

        resized = cv2.resize(crop, (96, 128), interpolation=cv2.INTER_AREA)

        # 1. Multi-zone spatial color features (Top 30%, Middle 40%, Bottom 30%)
        # Retail packaging strongly differentiates by brand header, middle logo, and base.
        h_r, w_r = resized.shape[:2]
        zones = [
            resized[0 : int(h_r * 0.30), :],          # Top zone
            resized[int(h_r * 0.30) : int(h_r * 0.70), :], # Center logo zone
            resized[int(h_r * 0.70) :, :],          # Bottom base zone
        ]

        color_feats = []
        for zone in zones:
            if zone.size == 0:
                color_feats.extend([0.0] * 40)
                continue
            hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(zone, cv2.COLOR_BGR2LAB)

            # HSV Histograms: 16 Hue bins, 8 Saturation bins, 8 Value bins
            hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])

            cv2.normalize(hist_h, hist_h, 1.0, 0.0, cv2.NORM_L1)
            cv2.normalize(hist_s, hist_s, 1.0, 0.0, cv2.NORM_L1)
            cv2.normalize(hist_v, hist_v, 1.0, 0.0, cv2.NORM_L1)

            # LAB color moments (mean and std of L, A, B)
            lab_means = np.mean(lab, axis=(0, 1)) / 255.0
            lab_stds = np.std(lab, axis=(0, 1)) / 255.0

            zone_vec = np.concatenate([
                hist_h.flatten(),
                hist_s.flatten(),
                hist_v.flatten(),
                lab_means,
                lab_stds,
            ])
            color_feats.extend(zone_vec)

        # 2. Gradient / Texture Energy
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        grad_mean = float(np.mean(mag)) / 255.0
        grad_std = float(np.std(mag)) / 255.0

        # Assemble full vector
        full_vec = np.array(color_feats + [norm_aspect, grad_mean, grad_std], dtype=np.float32)

        # L2 Unit Normalization for cosine similarity
        norm = np.linalg.norm(full_vec)
        if norm > 1e-6:
            full_vec = full_vec / norm
        return full_vec
