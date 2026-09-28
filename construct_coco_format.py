#!/usr/bin/env python3
"""
Convert the VegAnn Hugging Face dataset to COCO segmentation format.

Default output layout:

    coco_vegann/
      train/
        *.png
        _annotations.coco.json
      valid/
        *.png
        _annotations.coco.json

Install runtime dependencies first:

    pip install datasets pillow numpy tqdm pycocotools

Example:

    python construct_coco_format.py \
      --output-dir coco_vegann \
      --fold 1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DATASET_NAME = "simonMadec/VegAnn"
IMAGE_FIELD = "image"
MASK_FIELD = "mask"
CATEGORY_ID = 1
CATEGORY_NAME = "vegetation"


def require_dependencies():
    missing = []
    try:
        import datasets  # noqa: F401
    except ImportError:
        missing.append("datasets")
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")
    try:
        import PIL  # noqa: F401
    except ImportError:
        missing.append("pillow")
    try:
        import tqdm  # noqa: F401
    except ImportError:
        missing.append("tqdm")
    try:
        import pycocotools  # noqa: F401
    except ImportError:
        missing.append("pycocotools")

    if missing:
        packages = " ".join(missing)
        raise SystemExit(
            "Missing dependencies. Install them with:\n\n"
            f"    pip install {packages}\n"
        )


def normalize_split_value(value: Any) -> str:
    return str(value).strip().lower().replace("_", "").replace("-", "")


def split_row(
    row: Dict[str, Any],
    split_column: str,
    train_values: Iterable[str],
    valid_values: Iterable[str],
    test_values: Iterable[str],
    eval_split_name: str,
    include_test: bool,
) -> Optional[str]:
    value = normalize_split_value(row.get(split_column, ""))
    train_set = {normalize_split_value(v) for v in train_values}
    valid_set = {normalize_split_value(v) for v in valid_values}
    test_set = {normalize_split_value(v) for v in test_values}

    if value in train_set:
        return "train"
    if value in valid_set:
        return eval_split_name
    if value in test_set:
        return "test" if include_test else eval_split_name
    return None


def make_binary_mask(mask_image: Any, threshold: int, invert: bool):
    import numpy as np

    mask = mask_image.convert("L")
    array = np.asarray(mask)
    binary = array > threshold
    if invert:
        binary = ~binary
    return binary


def binary_mask_to_rle(binary_mask) -> Dict[str, Any]:
    """Return pycocotools-compatible compressed RLE."""
    import numpy as np
    from pycocotools import mask as mask_utils

    encoded = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    encoded["size"] = [int(encoded["size"][0]), int(encoded["size"][1])]
    return encoded


def mask_bbox_and_area(binary_mask) -> Tuple[Optional[List[float]], int]:
    import numpy as np

    ys, xs = np.where(binary_mask)
    area = int(xs.size)
    if area == 0:
        return None, 0

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    bbox = [
        float(x_min),
        float(y_min),
        float(x_max - x_min + 1),
        float(y_max - y_min + 1),
    ]
    return bbox, area


def save_image(image: Any, path: Path) -> Tuple[int, int]:
    image = image.convert("RGB")
    width, height = image.size
    image.save(path)
    return width, height


def base_coco(description: str) -> Dict[str, Any]:
    return {
        "info": {
            "description": description,
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(timespec="seconds"),
        },
        "licenses": [],
        "categories": [
            {
                "id": CATEGORY_ID,
                "name": CATEGORY_NAME,
                "supercategory": "plant",
            }
        ],
        "images": [],
        "annotations": [],
    }


def convert_vegann_to_coco(args: argparse.Namespace) -> None:
    require_dependencies()

    from datasets import load_dataset
    from tqdm import tqdm

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_column = args.split_column or f"TVT-split{args.fold}"
    train_values = args.train_values.split(",")
    valid_values = args.valid_values.split(",")
    test_values = args.test_values.split(",")

    dataset = load_dataset(
        args.dataset,
        split=args.hf_split,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )

    coco_by_split = {
        "train": base_coco(f"VegAnn COCO train dataset from {args.dataset}"),
        args.eval_split_name: base_coco(
            f"VegAnn COCO {args.eval_split_name} dataset from {args.dataset}"
        ),
    }
    if args.include_test:
        coco_by_split["test"] = base_coco(f"VegAnn COCO test dataset from {args.dataset}")

    next_image_id = {name: 1 for name in coco_by_split}
    next_annotation_id = {name: 1 for name in coco_by_split}
    skipped = Counter()

    for index, row in enumerate(tqdm(dataset, desc="Converting VegAnn")):
        out_split = split_row(
            row=row,
            split_column=split_column,
            train_values=train_values,
            valid_values=valid_values,
            test_values=test_values,
            eval_split_name=args.eval_split_name,
            include_test=args.include_test,
        )
        if out_split is None:
            skipped["unknown_split"] += 1
            continue
        if args.max_images_per_split and len(coco_by_split[out_split]["images"]) >= args.max_images_per_split:
            skipped[f"{out_split}_limit"] += 1
            continue

        image = row.get(IMAGE_FIELD)
        mask = row.get(MASK_FIELD)
        if image is None or mask is None:
            skipped["missing_image_or_mask"] += 1
            continue

        split_dir = output_dir / out_split
        split_dir.mkdir(parents=True, exist_ok=True)

        image_id = next_image_id[out_split]
        file_name = f"vegann_{image_id:06d}.png"
        image_path = split_dir / file_name
        width, height = save_image(image, image_path)

        binary_mask = make_binary_mask(mask, threshold=args.mask_threshold, invert=args.invert_mask)
        bbox, area = mask_bbox_and_area(binary_mask)

        coco = coco_by_split[out_split]
        coco["images"].append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )

        if bbox is None:
            skipped["empty_mask"] += 1
            if not args.keep_empty_images:
                image_path.unlink(missing_ok=True)
                coco["images"].pop()
                continue
        else:
            annotation = {
                "id": next_annotation_id[out_split],
                "image_id": image_id,
                "category_id": CATEGORY_ID,
                "segmentation": binary_mask_to_rle(binary_mask),
                "bbox": bbox,
                "area": float(area),
                "iscrowd": 1,
            }
            coco["annotations"].append(annotation)
            next_annotation_id[out_split] += 1

        next_image_id[out_split] += 1

        if args.limit and sum(len(c["images"]) for c in coco_by_split.values()) >= args.limit:
            break
        if args.max_images_per_split and all(
            len(coco["images"]) >= args.max_images_per_split for coco in coco_by_split.values()
        ):
            break

    for split_name, coco in coco_by_split.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        annotation_path = split_dir / "_annotations.coco.json"
        with annotation_path.open("w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False, indent=2)

        print(
            f"{split_name}: {len(coco['images'])} images, "
            f"{len(coco['annotations'])} annotations -> {annotation_path}"
        )

    if skipped:
        print("Skipped:", dict(skipped))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VegAnn to COCO segmentation format.")
    parser.add_argument("--dataset", default=DATASET_NAME, help="Hugging Face dataset name/path.")
    parser.add_argument("--hf-split", default="train", help="Hugging Face split to read.")
    parser.add_argument(
        "--output-dir",
        default="/data/Data/ft_sam3/coco_vegann",
        help="Output COCO dataset directory.",
    )
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--fold", type=int, default=1, choices=range(1, 6), help="VegAnn TVT fold id.")
    parser.add_argument("--split-column", default=None, help="Override split column, e.g. TVT-split1.")
    parser.add_argument("--train-values", default="Training,Train", help="Comma-separated train split labels.")
    parser.add_argument("--valid-values", default="Validation,Valid,Val", help="Comma-separated valid split labels.")
    parser.add_argument("--test-values", default="Test,Testing", help="Comma-separated test split labels.")
    parser.add_argument(
        "--eval-split-name",
        default="valid",
        choices=["valid", "test"],
        help="Directory name for non-training rows when --include-test is not used.",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Write Test rows to test/. By default Test rows are merged into valid/.",
    )
    parser.add_argument("--mask-threshold", type=int, default=0, help="Foreground threshold for mask pixels.")
    parser.add_argument("--invert-mask", action="store_true", help="Invert mask foreground/background.")
    parser.add_argument("--keep-empty-images", action="store_true", help="Keep images with empty masks.")
    parser.add_argument("--limit", type=int, default=0, help="Convert only the first N selected rows for debugging.")
    parser.add_argument(
        "--max-images-per-split",
        type=int,
        default=0,
        help="Convert at most N images for each output split.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to datasets.load_dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_vegann_to_coco(args)


if __name__ == "__main__":
    main()
