"""Prepare reproducible staged subsets of SKU-110K for retail product detection.

Features:
- Reproducible random sampling with fixed seed (default: 42)
- Converts bounding boxes to YOLO single-class format (class 0: product)
- Generates Experiment 1 config: 100 train, 25 val
- Generates Experiment 2 config: 500 train, 75 val (prepared, not auto-run)
- Preserves full dataset paths and test set annotations for evaluation
"""

import argparse
import csv
import pathlib
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare SKU-110K subsets for YOLO training")
    parser.add_argument("--dataset-dir", type=str, default="inventory_data/demo_images/SKU110K_fixed")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--exp1-train", type=int, default=100, help="Train image count for Exp 1")
    parser.add_argument("--exp1-val", type=int, default=25, help="Val image count for Exp 1")
    parser.add_argument("--exp2-train", type=int, default=500, help="Train image count for Exp 2")
    parser.add_argument("--exp2-val", type=int, default=100, help="Val image count for Exp 2")
    parser.add_argument("--exp2-test", type=int, default=150, help="Test image count for Exp 2 evaluation")
    return parser.parse_args()


def load_annotations(csv_path: pathlib.Path) -> Dict[str, List[Tuple[float, float, float, float, float, float]]]:
    """Load annotations: image_name -> list of (x1, y1, x2, y2, img_w, img_h)."""
    boxes_by_image = defaultdict(list)
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 8:
                continue
            img_name = row[0]
            try:
                x1, y1, x2, y2 = float(row[1]), float(row[2]), float(row[3]), float(row[4])
                img_w, img_h = float(row[6]), float(row[7])
                boxes_by_image[img_name].append((x1, y1, x2, y2, img_w, img_h))
            except ValueError:
                continue
    return boxes_by_image


def convert_boxes_to_yolo(boxes: List[Tuple[float, float, float, float, float, float]]) -> List[str]:
    """Convert absolute pixel boxes to YOLO normalized strings (class 0: product)."""
    lines = []
    for x1, y1, x2, y2, img_w, img_h in boxes:
        if img_w <= 0 or img_h <= 0:
            continue
        # Clamp coordinates to image boundaries
        x1 = max(0.0, min(img_w, x1))
        x2 = max(0.0, min(img_w, x2))
        y1 = max(0.0, min(img_h, y1))
        y2 = max(0.0, min(img_h, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        bw = x2 - x1
        bh = y2 - y1
        xc = x1 + (bw / 2.0)
        yc = y1 + (bh / 2.0)

        xc_norm = max(0.0, min(1.0, xc / img_w))
        yc_norm = max(0.0, min(1.0, yc / img_h))
        bw_norm = max(0.0, min(1.0, bw / img_w))
        bh_norm = max(0.0, min(1.0, bh / img_h))

        lines.append(f"0 {xc_norm:.6f} {yc_norm:.6f} {bw_norm:.6f} {bh_norm:.6f}")
    return lines


def main():
    args = parse_args()
    base_dir = pathlib.Path(args.dataset_dir).resolve()
    images_dir = base_dir / "images"
    ann_dir = base_dir / "annotations"
    labels_dir = base_dir / "labels"
    subsets_dir = base_dir / "subsets"

    labels_dir.mkdir(parents=True, exist_ok=True)
    subsets_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SKU-110K STAGED SUBSET PREPARATION")
    print("=" * 70)
    print(f"  Dataset Root: {base_dir}")
    print(f"  Random Seed : {args.seed}")
    print(f"  Exp 1 Split : {args.exp1_train} train / {args.exp1_val} val")
    print(f"  Exp 2 Split : {args.exp2_train} train / {args.exp2_val} val (prepared only)")
    print("=" * 70 + "\n")

    # 1. Load CSVs
    print("[1/4] Loading CSV annotations...")
    train_csv = ann_dir / "annotations_train.csv"
    val_csv = ann_dir / "annotations_val.csv"
    test_csv = ann_dir / "annotations_test.csv"

    train_ann = load_annotations(train_csv)
    val_ann = load_annotations(val_csv)
    test_ann = load_annotations(test_csv)

    print(f"      Train images in CSV: {len(train_ann)}")
    print(f"      Val images in CSV  : {len(val_ann)}")
    print(f"      Test images in CSV : {len(test_ann)}")

    # 2. Reproducible Sampling
    print("\n[2/4] Sampling reproducible subsets...")
    rng = random.Random(args.seed)

    all_train_imgs = sorted(train_ann.keys())
    all_val_imgs = sorted(val_ann.keys())
    all_test_imgs = sorted(test_ann.keys())

    # Exp 2 pool (500 train, 100 val)
    exp2_train_imgs = sorted(rng.sample(all_train_imgs, min(args.exp2_train, len(all_train_imgs))))
    exp2_val_imgs = sorted(rng.sample(all_val_imgs, min(args.exp2_val, len(all_val_imgs))))

    # Exp 1 is a strict, reproducible subset of Exp 2
    rng_sub = random.Random(args.seed)
    exp1_train_imgs = sorted(rng_sub.sample(exp2_train_imgs, min(args.exp1_train, len(exp2_train_imgs))))
    exp1_val_imgs = sorted(rng_sub.sample(exp2_val_imgs, min(args.exp1_val, len(exp2_val_imgs))))

    # Test evaluation set: 150 unseen test images from the original SKU-110K test split
    rng_test = random.Random(args.seed)
    exp2_test_imgs = sorted(rng_test.sample(all_test_imgs, min(args.exp2_test, len(all_test_imgs))))
    eval_test_imgs = exp2_test_imgs[:50]

    # Combine all images that need label files written
    images_to_convert: Set[str] = set(exp2_train_imgs) | set(exp2_val_imgs) | set(exp2_test_imgs)
    print(f"      Total unique images to label: {len(images_to_convert)}")

    # 3. Write YOLO label .txt files
    print("\n[3/4] Writing YOLO normalized label files to labels/...")
    written_count = 0
    total_box_count = 0

    all_ann_lookup = {**train_ann, **val_ann, **test_ann}

    for img_name in images_to_convert:
        boxes = all_ann_lookup.get(img_name, [])
        yolo_lines = convert_boxes_to_yolo(boxes)
        stem = pathlib.Path(img_name).stem
        label_file = labels_dir / f"{stem}.txt"
        label_file.write_text("\n".join(yolo_lines), encoding="utf-8")
        written_count += 1
        total_box_count += len(yolo_lines)

    print(f"      Wrote {written_count} label files ({total_box_count} total product boxes converted).")

    # 4. Write image path list text files
    print("\n[4/4] Writing dataset list files & YAML configs...")
    def write_path_list(file_path: pathlib.Path, img_names: List[str]):
        paths = [str((images_dir / name).resolve()) for name in img_names]
        file_path.write_text("\n".join(paths), encoding="utf-8")

    write_path_list(subsets_dir / "exp1_train.txt", exp1_train_imgs)
    write_path_list(subsets_dir / "exp1_val.txt", exp1_val_imgs)
    write_path_list(subsets_dir / "exp2_train.txt", exp2_train_imgs)
    write_path_list(subsets_dir / "exp2_val.txt", exp2_val_imgs)
    write_path_list(subsets_dir / "exp2_test.txt", exp2_test_imgs)
    write_path_list(subsets_dir / "test_eval.txt", eval_test_imgs)

    # Helper to format YAML with forward slashes
    def create_yaml(yaml_path: pathlib.Path, train_txt: str, val_txt: str, test_txt: str):
        content = f"""# SKU-110K Dataset Config for Retail Product Detection
# Generated reproducibly by scripts/prepare_sku110k_subsets.py (seed={args.seed})
path: {base_dir.as_posix()}
train: {train_txt}
val: {val_txt}
test: {test_txt}

# Single detection class for product instance localization
names:
  0: product
"""
        yaml_path.write_text(content, encoding="utf-8")

    yaml_exp1 = base_dir / "sku110k_exp1.yaml"
    yaml_exp2 = base_dir / "sku110k_exp2.yaml"
    yaml_full = base_dir / "sku110k_full.yaml"

    create_yaml(yaml_exp1, "subsets/exp1_train.txt", "subsets/exp1_val.txt", "subsets/test_eval.txt")
    create_yaml(yaml_exp2, "subsets/exp2_train.txt", "subsets/exp2_val.txt", "subsets/exp2_test.txt")
    create_yaml(yaml_full, "annotations/annotations_train.csv", "annotations/annotations_val.csv", "annotations/annotations_test.csv")

    print(f"      Exp 1 YAML created -> {yaml_exp1}")
    print(f"      Exp 2 YAML created -> {yaml_exp2}")
    print(f"      Full   YAML saved   -> {yaml_full}")
    print("\n[OK] Staged subset preparation completed successfully.")


if __name__ == "__main__":
    main()
