"""Helper script to construct a demo store catalog with reference crops.

Extracts representative product reference crops from test shelf images:
- Beverages: 7-Up 2L, Mountain Dew 2L, A&W Root Beer 2L, Squirt 12-Pack (from test_1095.jpg)
- Pharmacy: Sudafed Congestion Relief, Clarityn Allergy, Piriteze (from test_109.jpg)

Saves crops to inventory_data/catalogs/reference_crops/
Saves catalog to inventory_data/catalogs/demo_store_catalog.json
"""

import json
import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
from inventory.catalog import SKUCatalog, SKUCatalogItem


def main():
    crops_dir = ROOT_DIR / "inventory_data" / "catalogs" / "reference_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = ROOT_DIR / "inventory_data" / "catalogs" / "demo_store_catalog.json"

    # Images to crop from
    img1095_path = ROOT_DIR / "inventory_data" / "demo_images" / "SKU110K_fixed" / "images" / "test_1095.jpg"
    img109_path = ROOT_DIR / "inventory_data" / "demo_images" / "SKU110K_fixed" / "images" / "test_109.jpg"

    catalog = SKUCatalog(name="Store_Supermarket_Catalog")

    # 1. Beverages from test_1095.jpg
    if img1095_path.is_file():
        img1095 = cv2.imread(str(img1095_path))
        h, w = img1095.shape[:2]

        # Crop reference 1: 7-Up 2L Bottle (shelf 2, green bottle with red 7-up circle)
        # Bounding box approximately: x: 180-250, y: 350-580
        # Let's crop clean exemplars
        crop_7up = img1095[355:575, 175:240]
        p_7up = crops_dir / "bev_7up_2l_ref1.jpg"
        cv2.imwrite(str(p_7up), crop_7up)

        item_7up = SKUCatalogItem(
            sku_id="BEV_7UP_2L",
            name="7-Up Lemon Lime 2L",
            category="Beverages",
            reference_images=[str(p_7up)],
            metadata={"brand": "7-Up", "size": "2 Liters", "price": "$2.49"},
        )
        catalog.add_item(item_7up)

        # Crop reference 2: Mountain Dew 2L Bottle (shelf 2, dark green bottle, left side)
        crop_mdew = img1095[355:575, 95:170]
        p_mdew = crops_dir / "bev_mtndew_2l_ref1.jpg"
        cv2.imwrite(str(p_mdew), crop_mdew)

        item_mdew = SKUCatalogItem(
            sku_id="BEV_MTNDEW_2L",
            name="Mountain Dew 2L",
            category="Beverages",
            reference_images=[str(p_mdew)],
            metadata={"brand": "Mountain Dew", "size": "2 Liters", "price": "$2.49"},
        )
        catalog.add_item(item_mdew)

        # Crop reference 3: A&W Root Beer / Cream Soda (shelf 2, brown bottle, amber label)
        crop_aw = img1095[380:580, 830:900]
        p_aw = crops_dir / "bev_aw_rootbeer_ref1.jpg"
        cv2.imwrite(str(p_aw), crop_aw)

        item_aw = SKUCatalogItem(
            sku_id="BEV_AW_ROOTBEER",
            name="A&W Root Beer 2L",
            category="Beverages",
            reference_images=[str(p_aw)],
            metadata={"brand": "A&W", "size": "2 Liters", "price": "$2.29"},
        )
        catalog.add_item(item_aw)

        # Crop reference 4: Squirt 12-Pack Carton (shelf 4, bright yellow box)
        crop_squirt = img1095[810:960, 770:860]
        p_squirt = crops_dir / "bev_squirt_12pk_ref1.jpg"
        cv2.imwrite(str(p_squirt), crop_squirt)

        item_squirt = SKUCatalogItem(
            sku_id="BEV_SQUIRT_12PK",
            name="Squirt Citrus Soda 12-Pack",
            category="Beverages",
            reference_images=[str(p_squirt)],
            metadata={"brand": "Squirt", "size": "12 Cans", "price": "$5.99"},
        )
        catalog.add_item(item_squirt)

    # 2. Pharmacy from test_109.jpg
    if img109_path.is_file():
        img109 = cv2.imread(str(img109_path))
        h, w = img109.shape[:2]

        # Crop reference 5: Sudafed Congestion Relief (shelf 3, red/yellow box)
        crop_sudafed = img109[650:780, 20:150]
        p_sudafed = crops_dir / "med_sudafed_ref1.jpg"
        cv2.imwrite(str(p_sudafed), crop_sudafed)

        item_sudafed = SKUCatalogItem(
            sku_id="MED_SUDAFED_16",
            name="Sudafed Congestion Relief",
            category="Pharmacy",
            reference_images=[str(p_sudafed)],
            metadata={"brand": "Sudafed", "size": "16 Tablets", "price": "$8.49"},
        )
        catalog.add_item(item_sudafed)

        # Crop reference 6: Clarityn Allergy Relief (shelf 4, dark blue box)
        crop_clarityn = img109[980:1100, 320:440]
        p_clarityn = crops_dir / "med_clarityn_ref1.jpg"
        cv2.imwrite(str(p_clarityn), crop_clarityn)

        item_clarityn = SKUCatalogItem(
            sku_id="MED_CLARITYN_ALLERGY",
            name="Clarityn Allergy Relief",
            category="Pharmacy",
            reference_images=[str(p_clarityn)],
            metadata={"brand": "Clarityn", "size": "30 Tablets", "price": "$11.99"},
        )
        catalog.add_item(item_clarityn)

    catalog.save_json(catalog_path)
    print(f"Constructed demo catalog with {len(catalog)} SKUs -> {catalog_path}")
    for item in catalog.list_items():
        print(f"  [{item.sku_id}] {item.name} ({item.category}) - {len(item.reference_images)} ref crops")


if __name__ == "__main__":
    main()
