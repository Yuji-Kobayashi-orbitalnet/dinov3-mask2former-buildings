"""Convert Waterloo binary rooftop masks into connected-component COCO RLE."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask


def tiff_files(folder):
    if not folder.is_dir():
        raise ValueError(f"Directory not found: {folder}")
    files = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        key = path.stem.casefold()
        if key in files:
            raise ValueError(f"Duplicate TIFF stem in {folder}: {path.name}")
        files[key] = path
    if not files:
        raise ValueError(f"No TIFF files in {folder}")
    return files


def binary_mask(path):
    with Image.open(path) as image:
        # Check colors rather than palette indices for paletted TIFFs.
        pixels = np.array(image.convert("RGB") if image.mode == "P" else image)
    if pixels.ndim == 3 and pixels.shape[2] == 3 and np.all(pixels == pixels[..., :1]):
        pixels = pixels[..., 0]
    if pixels.ndim != 2:
        raise ValueError(f"Expected a grayscale or monochrome RGB TIFF: {path} (shape {pixels.shape})")
    values = np.unique(pixels)
    if not np.isin(values, [0, 1, 255, 65535]).all() or np.count_nonzero(values) > 1:
        raise ValueError(f"Expected binary labels (0 plus one of 1, 255, 65535): {path}; values {values[:12]}")
    return (pixels != 0).astype(np.uint8)


def convert(images_dir, labels_dir, output, limit, *, allowed_sizes=(512,)):
    images, labels = tiff_files(images_dir), tiff_files(labels_dir)
    missing, extra = sorted(images.keys() - labels.keys()), sorted(labels.keys() - images.keys())
    if missing or extra:
        raise ValueError(f"Unpaired TIFFs: {len(missing)} without labels {missing[:5]}, "
                         f"{len(extra)} without images {extra[:5]}")
    if output.exists():
        raise FileExistsError(f"Choose a new JSON output path: {output}")

    keys = list(images)[:limit]
    dataset = {
        "info": {"description": "Waterloo Building Dataset: instances derived from binary roof masks",
                 "source_doi": "10.7910/DVN/EXRA2V", "instance_source": "connected_components",
                 "connectivity": 8},
        "images": [], "annotations": [], "categories": [{"id": 1, "name": "building"}],
    }
    annotations = dataset["annotations"]
    empty_tiles, max_instances = 0, 0
    cv2.setNumThreads(1)
    for image_id, key in enumerate(keys, start=1):
        image_path, label_path = images[key], labels[key]
        with Image.open(image_path) as image:
            width, height = image.size
        mask = binary_mask(label_path)
        if width != height or width not in allowed_sizes or mask.shape != (height, width):
            raise ValueError(f"Expected a square image with side in {allowed_sizes} and a matching label: "
                             f"{image_path} ({width}x{height}), {label_path} ({mask.shape})")
        dataset["images"].append({"id": image_id, "file_name": image_path.name,
                                  "width": width, "height": height})
        count, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        empty_tiles += count == 1
        max_instances = max(max_instances, count - 1)
        instance_mask = np.zeros((height, width), dtype=np.uint8, order="F")
        for component in range(1, count):
            x, y, w, h, area = map(int, stats[component])
            region = np.s_[y:y+h, x:x+w]
            instance_mask[region] = components[region] == component
            rle = coco_mask.encode(instance_mask)
            rle["counts"] = rle["counts"].decode("ascii")
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": 1,
                                "segmentation": rle, "bbox": [x, y, w, h], "area": area, "iscrowd": 0})
            instance_mask[region] = 0
        if image_id % 1000 == 0:
            print(f"{image_id}/{len(keys)} images, {len(annotations)} components", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(dataset, stream, separators=(",", ":"))
        except BaseException:
            stream.close()
            temporary.unlink()
            raise
    try:
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{output}: {len(keys)} images, {len(annotations)} components, "
          f"{empty_tiles} empty tiles, max {max_instances} components/image")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Directory containing the image TIFFs")
    parser.add_argument("--labels", type=Path, required=True, help="Directory containing the binary TIFF masks")
    parser.add_argument("--output", type=Path, required=True, help="New COCO JSON file")
    parser.add_argument("--limit", type=int, help="Process the first N sorted image names for a trial")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        convert(args.images, args.labels, args.output, args.limit)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Conversion stopped: {exc}\n")


if __name__ == "__main__":
    main()
