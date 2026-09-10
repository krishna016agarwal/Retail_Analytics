"""Product catalog definition for retail SKU recognition.

Provides structured representation of store products (SKUs) including:
- SKU ID, human-readable name, category, and metadata
- Reference images or reference crops
- Precomputed or on-demand feature embeddings for fast similarity search
- Serialization to / from JSON catalogs

Completely isolated from the crowd/queue pipeline (src/).
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np


@dataclass
class SKUCatalogItem:
    """A registered product SKU in the store catalog.

    Attributes:
        sku_id: Unique SKU identifier (e.g., 'BEV_7UP_2L', 'MED_SUDAFED_16').
        name: Human-readable product name (e.g., '7-Up Lemon Lime 2L Bottle').
        category: Product category (e.g., 'Beverages', 'Pharmacy', 'Personal Care').
        reference_images: List of paths to reference images or crops of this SKU.
        reference_embeddings: List of feature embedding vectors corresponding to reference images.
        metadata: Optional extra details (e.g. brand, size, unit_price, expected_facing_count).
    """

    sku_id: str
    name: str
    category: str = "General"
    reference_images: List[str] = field(default_factory=list)
    reference_embeddings: List[np.ndarray] = field(default_factory=list, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_reference_image(self, img_path: Union[str, Path]) -> None:
        """Add a reference image path to this item."""
        p_str = str(Path(img_path).resolve())
        if p_str not in self.reference_images:
            self.reference_images.append(p_str)

    def to_dict(self) -> Dict[str, Any]:
        """Convert item to a JSON-serializable dictionary."""
        return {
            "sku_id": self.sku_id,
            "name": self.name,
            "category": self.category,
            "reference_images": self.reference_images,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SKUCatalogItem":
        """Reconstruct catalog item from dictionary."""
        return cls(
            sku_id=data["sku_id"],
            name=data["name"],
            category=data.get("category", "General"),
            reference_images=data.get("reference_images", []),
            metadata=data.get("metadata", {}),
        )


class SKUCatalog:
    """Store-specific product catalog holding registered SKUs and reference data.

    Manages catalog items, indexing, persistence, and reference embeddings.
    """

    def __init__(self, name: str = "DefaultStoreCatalog") -> None:
        self.name = name
        self._items: Dict[str, SKUCatalogItem] = {}

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, sku_id: str) -> bool:
        return sku_id in self._items

    def add_item(self, item: SKUCatalogItem) -> None:
        """Add or update a catalog item."""
        self._items[item.sku_id] = item

    def get_item(self, sku_id: str) -> Optional[SKUCatalogItem]:
        """Retrieve a catalog item by ID."""
        return self._items.get(sku_id)

    def list_items(self) -> List[SKUCatalogItem]:
        """Return all catalog items."""
        return list(self._items.values())

    def list_sku_ids(self) -> List[str]:
        """Return all registered SKU IDs."""
        return list(self._items.keys())

    def save_json(self, path: Union[str, Path]) -> None:
        """Serialize catalog to a JSON file."""
        import json

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "catalog_name": self.name,
            "item_count": len(self._items),
            "items": [item.to_dict() for item in self._items.values()],
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "SKUCatalog":
        """Load a catalog from a JSON file."""
        import json

        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Catalog file not found: {path}")

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        catalog = cls(name=data.get("catalog_name", "LoadedCatalog"))
        for item_data in data.get("items", []):
            item = SKUCatalogItem.from_dict(item_data)
            catalog.add_item(item)
        return catalog
