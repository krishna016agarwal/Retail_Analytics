# inventory_data/

Local dataset and model folder for the Retail Shelf Inventory Module.

## Folder Structure

```
inventory_data/
├── demo_images/          Drop your own shelf JPEG/PNG images here
│   └── sample_shelf.jpg  Auto-generated synthetic demo image (run --demo)
├── demo_videos/          Drop shelf video files (.mp4, .avi, etc.) here
├── custom_model/         Drop custom YOLO .pt weights here for plug-in
└── README.md             This file
```

## Running the Demo

```powershell
# Generate a synthetic shelf image and analyse it:
python run_inventory.py --demo --no-show

# Analyse your own shelf image:
python run_inventory.py --source inventory_data/demo_images/my_shelf.jpg

# Analyse a shelf video:
python run_inventory.py --source inventory_data/demo_videos/shelf.mp4
```

## Using SKU-110K for Experimentation

SKU-110K is a large-scale dense retail product detection dataset.
Download: https://github.com/eg4000/SKU110K_CVPR19

**Important**: SKU-110K provides bounding box annotations only — it does NOT
provide per-SKU class labels. Use it for:
- Training a dense product localisation / detection model.
- Validating whether a model can localise densely packed products.

It is NOT suitable for training a brand or SKU classifier without additional
per-SKU label annotation.

### Suggested fine-tuning workflow:

1. Download SKU-110K annotations and images.
2. Convert annotations to YOLO format (SKU-110K has its own CSV format).
3. Fine-tune a YOLO model:
   ```
   yolo train model=yolov8n.pt data=sku110k.yaml epochs=50 imgsz=640
   ```
4. Copy the resulting weights to `inventory_data/custom_model/retail_yolo.pt`.
5. Run with the custom model:
   ```
   python run_inventory.py \
       --source inventory_data/demo_images/ \
       --model inventory_data/custom_model/retail_yolo.pt \
       --model-tier retail_specific \
       --all-classes
   ```

## Custom Model Plug-in

Any YOLO-compatible `.pt` weights can be used. The pipeline is model-agnostic:

```python
from inventory.config import InventoryConfig, InventoryModelConfig

config = InventoryConfig(
    model=InventoryModelConfig(
        model_path="inventory_data/custom_model/retail_yolo.pt",
        model_tier="retail_specific",   # or "sku_recognition" for future
        target_classes=None,            # use all model classes
    )
)
```

## Model Tier Reference

| Tier             | Description                                      | SKU recognition? |
|------------------|--------------------------------------------------|------------------|
| `coco_baseline`  | Generic COCO objects (bottle, cup, etc.)         | ❌ No             |
| `retail_specific`| Custom model trained on retail shelf data        | ❌ No (localization)|
| `sku_recognition`| Custom model with per-SKU class labels (future) | ✅ Yes            |

## Limitations

- **visible_facings** = camera-observable front-row product count only.
  It is NOT total physical inventory. Products stacked front-to-back are invisible.
- A COCO baseline model detecting "bottle" does NOT identify Coca-Cola vs Pepsi vs Bisleri.
- For brand/SKU recognition, a custom annotated dataset and custom model are required.
