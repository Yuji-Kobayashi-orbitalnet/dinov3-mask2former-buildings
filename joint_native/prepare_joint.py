"""Inspect and combine WHU/Waterloo annotations for native 512 and 1024 inputs."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
from waterloo_masks_to_coco import convert


IMAGE_DIRS = {"image", "images", "img", "imgs"}
LABEL_DIRS = {"label", "labels", "mask", "masks"}
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
SPLITS = {"train": {"train", "training"}, "validation": {"val", "valid", "validation"}}
NATIVE_SIDES = (512, 1024)


def split_candidates(root):
    if not root.is_dir():
        raise ValueError(f"Dataset root is missing: {root}")
    candidates = {name: [] for name in SPLITS}
    for path, dirs, _ in os.walk(root):
        folder = Path(path)
        parts = {part.lower() for part in folder.relative_to(root).parts}
        images = [folder / name for name in dirs if name.lower() in IMAGE_DIRS]
        dirs[:] = sorted(name for name in dirs if name.lower() not in IMAGE_DIRS | LABEL_DIRS)
        if len(folder.relative_to(root).parts) >= 5:
            dirs.clear()
        matches = [name for name, aliases in SPLITS.items() if aliases & parts]
        if len(matches) == 1 and not {"test", "testing"} & parts:
            candidates[matches[0]].extend(images)
    return candidates


def read_coco(path):
    with path.open(encoding="utf-8-sig") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or not {"images", "annotations", "categories"} <= data.keys():
        raise ValueError(f"Expected COCO instance JSON: {path}")
    return data


def discover(root):
    inputs = {}
    for split, candidates in split_candidates(root).items():
        if len(candidates) != 1:
            raise ValueError(f"Expected one {split} image directory in {root}; found {candidates}. "
                             "Use --config to specify the directories explicitly.")
        images = candidates[0]
        folder = images.parent
        preferred = folder / "annotations.json"
        if preferred.is_file():
            inputs[split] = {"images": str(images), "annotations": str(preferred)}
            continue
        json_files = []
        for path in sorted(folder.glob("*.json")):
            try:
                read_coco(path)
            except ValueError:
                continue
            json_files.append(path)
        if len(json_files) == 1:
            inputs[split] = {"images": str(images), "annotations": str(json_files[0])}
            continue
        label_folders = sorted(p for p in folder.iterdir() if p.is_dir() and p.name.lower() in LABEL_DIRS)
        if not json_files and len(label_folders) == 1:
            inputs[split] = {"images": str(images), "labels": str(label_folders[0])}
            continue
        raise ValueError(f"Cannot select one COCO JSON or TIFF label folder for {images}. "
                         "Use --config to specify the labels explicitly.")
    return inputs


def inspect_roots(roots):
    for name, root in roots.items():
        print(f"[{name}] {root}")
        for split, candidates in split_candidates(root).items():
            if not candidates:
                print(f"  {split}: no image directory recognized")
            for folder in candidates:
                images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
                sample = [images[i] for i in sorted({0, len(images) // 2, len(images) - 1})] if images else []
                sizes = []
                for path in sample:
                    with Image.open(path) as image:
                        sizes.append(f"{path.name}={image.width}x{image.height}")
                print(f"  {split}: {folder} | {len(images)} images | sample {', '.join(sizes)}")
                for sidecar in sorted(folder.parent.iterdir()):
                    if (sidecar.is_dir() and sidecar.name.lower() in LABEL_DIRS) or sidecar.suffix.lower() == ".json":
                        print(f"    {sidecar}")


def category(data, explicit):
    if explicit is not None and explicit in {c["id"] for c in data["categories"]}:
        return explicit
    if explicit is None:
        buildings = [c for c in data["categories"] if c["name"].strip().lower() in {"building", "buildings"}]
        selected = buildings or data["categories"]
        if len(selected) == 1:
            return selected[0]["id"]
    raise ValueError("Specify category_id in the input configuration for this COCO file")


def merge_split(inputs, split, staging):
    merged = {"info": {"description": "WHU + Waterloo, native-pixel annotations"},
              "images": [], "annotations": [], "categories": [{"id": 1, "name": "building"}]}
    seen = set()
    for source, splits in inputs.items():
        item = splits[split]
        images = Path(item["images"]).expanduser().resolve()
        if not images.is_dir():
            raise ValueError(f"Image directory expected: {images}")
        if ("annotations" in item) == ("labels" in item):
            raise ValueError(f"Supply exactly one of annotations or labels for {source}/{split}")
        has_coco = "annotations" in item
        if has_coco:
            data = read_coco(Path(item["annotations"]).expanduser())
        else:
            intermediate = staging / f"{source}_{split}.json"
            convert(images, Path(item["labels"]).expanduser(), intermediate, None, allowed_sizes=NATIVE_SIDES)
            data = read_coco(intermediate)
        category_id = category(data, item.get("category_id"))
        sizes = Counter()
        id_map = {}
        for image in data["images"]:
            old_id = image["id"]
            if old_id in id_map:
                raise ValueError(f"Duplicate image ID {old_id} in {source}/{split}")
            filename = image.get("file_name") or image.get("path") or image.get("name")
            if not filename:
                raise ValueError(f"Missing image name for ID {old_id} in {source}/{split}")
            path = images / filename
            if not path.is_file():
                path = images / Path(filename).name
            if not path.is_file():
                raise ValueError(f"Image listed in {source}/{split} does not exist: {path}")
            path = path.resolve()
            if path in seen:
                raise ValueError(f"Repeated image file in {split}: {path}")
            seen.add(path)
            width, height = image.get("width"), image.get("height")
            if has_coco:
                with Image.open(path) as actual:
                    if (width not in (None, actual.width)) or (height not in (None, actual.height)):
                        raise ValueError(f"COCO dimensions {width}x{height} disagree with image {actual.size}: {path}")
                    width, height = actual.size
            if width != height or width not in NATIVE_SIDES:
                raise ValueError(f"{source}/{split}: {path} is {width}x{height}; expected a square with side in "
                                 f"{NATIVE_SIDES}. Use --inspect to review native sizes.")
            sizes[width] += 1
            new_id = len(merged["images"]) + 1
            id_map[old_id] = new_id
            merged["images"].append({"id": new_id, "file_name": str(path), "width": width,
                                     "height": height, "source": source})
        instances = Counter()
        for annotation in data["annotations"]:
            if annotation["category_id"] != category_id:
                continue
            new_image_id = id_map[annotation["image_id"]]
            merged["annotations"].append({**annotation, "id": len(merged["annotations"]) + 1,
                                          "category_id": 1, "image_id": new_image_id})
            if not annotation.get("iscrowd", 0):
                instances[new_image_id] += 1
        if not sizes or not instances:
            raise ValueError(f"No images with building labels for {source}/{split}")
        print(f"{source}/{split}: native sides {dict(sizes)}, {sum(instances.values())} objects, "
              f"max {max(instances.values())} objects/image", flush=True)
    return merged, seen


def prepare(inputs, output):
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix="joint-native-") as directory:
        staging = Path(directory)
        result = staging / "prepared"
        result.mkdir()
        train, train_paths = merge_split(inputs, "train", staging)
        with (result / "train.json").open("w", encoding="utf-8") as stream:
            json.dump(train, stream, separators=(",", ":"))
        del train
        validation, val_paths = merge_split(inputs, "validation", staging)
        overlap = train_paths & val_paths
        if overlap:
            raise ValueError(f"Train/validation share {len(overlap)} image files, e.g. {min(overlap)}")
        with (result / "validation.json").open("w", encoding="utf-8") as stream:
            json.dump(validation, stream, separators=(",", ":"))
        (result / "sources.json").write_text(json.dumps({"inputs": inputs, "native_sizes": NATIVE_SIDES},
                                                         indent=2), encoding="utf-8")
        result.rename(output)
    print(f"Prepared: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whu-root", type=Path, default=Path("/mnt/nas02/WHU_Building_Dataset"))
    parser.add_argument("--waterloo-root", type=Path, default=Path("/mnt/nas02/Waterloo_Building_Dataset"))
    parser.add_argument("--config", type=Path, help="Explicit source/split directories and annotation paths")
    parser.add_argument("--inspect", action="store_true", help="List split directories and sample native sizes")
    parser.add_argument("--output", type=Path, default=Path("prepared_joint_native"))
    args = parser.parse_args()
    roots = {"whu": args.whu_root, "waterloo": args.waterloo_root}
    try:
        if args.inspect:
            inspect_roots(roots)
            return
        inputs = (json.loads(args.config.read_text(encoding="utf-8-sig")) if args.config else
                  {name: discover(root) for name, root in roots.items()})
        if set(inputs) != set(roots) or any(set(splits) != set(SPLITS) for splits in inputs.values()):
            raise ValueError("The configuration needs whu/waterloo, each with train and validation")
        prepare(inputs, args.output)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Preparation stopped: {exc}\n")


if __name__ == "__main__":
    main()
