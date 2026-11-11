"""Train the Transformers 4.57.1 Mask2Former with frozen DINOv3-B/16 on WHU COCO instances."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, SwinConfig


class WhuInstances(Dataset):
    def __init__(self, images, annotation_file, size, num_queries, augment, category_id):
        if not images.is_dir() or not annotation_file.is_file():
            raise ValueError(f"Missing WHU images or COCO JSON: {images} / {annotation_file}")
        self.coco = COCO(str(annotation_file))
        categories = self.coco.loadCats(self.coco.getCatIds())
        if category_id is None:
            buildings = [c for c in categories if c["name"].strip().lower() in {"building", "buildings"}]
            selected = buildings or categories
            if len(selected) != 1:
                raise ValueError(f"Select the building class with --category-id: {categories}")
            category_id = selected[0]["id"]
        if category_id not in self.coco.cats:
            raise ValueError(f"Category {category_id} not found in {annotation_file}")
        # COCOeval uses zero as the unmatched sentinel in its annotation ID arrays.
        all_annotations = self.coco.dataset.get("annotations", [])
        if any(ann["id"] <= 0 for ann in all_annotations):
            for new_id, ann in enumerate(all_annotations, start=1):
                ann["id"] = new_id
            self.coco.createIndex()
        self.category_id = category_id
        allowed_sizes = (size,) if isinstance(size, int) else tuple(size)
        self.augment = augment
        self.ids = sorted(self.coco.imgs)
        self.paths = []
        self.annotations = []
        self.native_sizes = []
        empty_masks = {}
        for image_id in self.ids:
            row = self.coco.imgs[image_id]
            filename = row.get("file_name") or row.get("path") or row.get("name")
            if not filename:
                raise ValueError(f"Missing file name for image {image_id} in {annotation_file}")
            path = images / filename
            if not path.is_file():
                path = images / Path(filename).name
            if not path.is_file():
                raise ValueError(f"Image listed in {annotation_file} does not exist: {path}")
            if "width" not in row or "height" not in row:
                with Image.open(path) as image:
                    row.setdefault("width", image.width)
                    row.setdefault("height", image.height)
            shape = (row["height"], row["width"])
            if shape[0] != shape[1] or shape[0] not in allowed_sizes:
                raise ValueError(f"{path} is {row['width']}x{row['height']} in COCO. "
                                 f"Expected a square with side in {allowed_sizes}. Check the input settings.")
            selected_annotations = [a for a in self.coco.imgToAnns[image_id] if a["category_id"] == category_id]
            if any(not a.get("segmentation") for a in selected_annotations):
                raise ValueError(f"Missing instance segmentation for image {image_id}: {path}")
            anns = []
            for ann in selected_annotations:
                ann.setdefault("iscrowd", 0)
                # Use raster area, since a polygon can have positive geometric
                # area without covering any pixels on this image grid.
                raster_area = int(coco_mask.area(self.coco.annToRLE(ann)))
                if raster_area == 0:
                    empty_masks[ann["id"]] = path
                    continue
                ann.setdefault("area", float(raster_area))
                if not ann["iscrowd"]:
                    anns.append(ann)
            if len(anns) > num_queries:
                raise ValueError(f"{path} has {len(anns)} buildings. Increase --num-queries above {num_queries}.")
            self.paths.append(path)
            self.annotations.append(anns)
            self.native_sizes.append(shape[0])
        if empty_masks:
            # Keep the COCO ground truth used for evaluation in sync with the
            # instance targets. Images with no remaining annotations stay in the split.
            self.coco.dataset["annotations"] = [ann for ann in all_annotations if ann["id"] not in empty_masks]
            self.coco.createIndex()
            print(f"{annotation_file}: excluded {len(empty_masks)} zero-pixel annotations", flush=True)
            for ann_id, path in list(empty_masks.items())[:10]:
                print(f"  annotation {ann_id}: {path}", flush=True)
        if not self.ids or not any(self.annotations):
            raise ValueError(f"No labeled buildings in {annotation_file}")
        self.coco.dataset.setdefault("info", {})
        print(f"{annotation_file}: {len(self.ids)} images, "
              f"{sum(not a for a in self.annotations)} without a non-crowd building, "
              f"max {max(map(len, self.annotations))} buildings/image, category {category_id}", flush=True)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        path = self.paths[index]
        row = self.coco.imgs[self.ids[index]]
        width, height = row["width"], row["height"]
        with Image.open(path) as image:
            if image.size != (width, height):
                raise ValueError(f"COCO dimensions and image pixels disagree: {path} ({image.size})")
            pixels = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)
        masks = []
        for ann in self.annotations[index]:
            mask = self.coco.annToMask(ann)
            if not mask.any():
                raise ValueError(f"Empty rasterized mask for annotation {ann['id']} in {path}")
            masks.append(mask)
        targets = (torch.from_numpy(np.stack(masks)).bool() if masks else
                   torch.empty((0, height, width), dtype=torch.bool))
        if self.augment:
            rotation = int(torch.randint(4, ()))
            pixels, targets = pixels.rot90(rotation, (-2, -1)), targets.rot90(rotation, (-2, -1))
            if torch.rand(()) < 0.5:
                pixels, targets = pixels.flip((-1,)), targets.flip((-1,))
        return pixels, targets, self.ids[index]


def collate(items):
    pixels, masks, ids = zip(*items)
    return torch.stack(pixels), masks, ids


def limit_dataset(dataset, limit):
    if limit is None or limit >= len(dataset):
        return dataset
    first = {}
    for i, (image_id, anns) in enumerate(zip(dataset.ids, dataset.annotations)):
        source = dataset.coco.imgs[image_id].get("source", "")
        first.setdefault((source, dataset.native_sizes[i], bool(anns)), i)
    positives = [i for (_, _, has_labels), i in first.items() if has_labels]
    empties = [i for (_, _, has_labels), i in first.items() if not has_labels]
    indices = list(dict.fromkeys(positives + empties + list(range(limit))))
    return Subset(dataset, indices[:limit])


class NativeSizeBatches:
    """Group samples by native side length so collate never pads or resizes."""

    def __init__(self, dataset, batch_size, shuffle):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.groups = defaultdict(list)
        if isinstance(dataset, Subset):
            sizes = (dataset.dataset.native_sizes[i] for i in dataset.indices)
        else:
            sizes = dataset.native_sizes
        for index, size in enumerate(sizes):
            self.groups[size].append(index)

    def __len__(self):
        return sum((len(indices) + self.batch_size - 1) // self.batch_size for indices in self.groups.values())

    def __iter__(self):
        batches = []
        for group in self.groups.values():
            indices = [group[i] for i in torch.randperm(len(group)).tolist()] if self.shuffle else group
            batches.extend(indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size))
        if self.shuffle:
            return (batches[i] for i in torch.randperm(len(batches)).tolist())
        return iter(batches)


class DinoPyramid(nn.Module):
    """Convert four stride-16 ViT taps into the F4/F8/F16/F32 contract."""

    def __init__(self, dino, channels):
        super().__init__()
        self.dino = dino.requires_grad_(False).eval()
        self.projections = nn.ModuleList(
            nn.Sequential(nn.Conv2d(768, dim, 1, bias=False), nn.GroupNorm(8, dim)) for dim in channels
        )
        self.rgb_f4 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, channels[0], 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, channels[0]),
        )

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()
        return self

    def forward(self, pixel_values):
        with torch.no_grad():
            taps = self.dino.get_intermediate_layers(pixel_values, n=[2, 5, 8, 11], reshape=True)
        height, width = pixel_values.shape[-2:]
        features = []
        for tap, projection, stride in zip(taps, self.projections, (4, 8, 16, 32), strict=True):
            feature = projection(tap)
            size = (height // stride, width // stride)
            if stride == 32:
                feature = F.avg_pool2d(feature, 2)
            elif feature.shape[-2:] != size:
                feature = F.interpolate(feature, size=size, mode="bilinear", align_corners=False)
            features.append(feature)
        features[0] = features[0] + self.rgb_f4(pixel_values)
        return SimpleNamespace(feature_maps=tuple(features))


def build_model(dino, num_queries):
    # The constructor uses this config to size the standard pixel decoder.
    # Replace its temporary encoder with DINO before training or moving to CUDA.
    shape_config = SwinConfig(embed_dim=32, depths=[1, 1, 1, 1], num_heads=[1, 2, 4, 8],
        out_features=["stage1", "stage2", "stage3", "stage4"])
    config = Mask2FormerConfig(backbone_config=shape_config.to_dict(), num_labels=1,
        num_queries=num_queries, id2label={0: "building"}, label2id={"building": 0})
    model = Mask2FormerForUniversalSegmentation(config)
    pixel_level = model.model.pixel_level_module
    pixel_level.encoder = DinoPyramid(dino, pixel_level.encoder.channels)
    return model


def normalize(pixels, device):
    pixels = pixels.to(device=device, dtype=torch.float32, non_blocking=True).div_(255)
    mean = pixels.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = pixels.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (pixels - mean) / std


def train_epoch(model, loader, optimizer, device, precision, accumulation):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for step, (pixels, masks, _) in enumerate(loader):
        pixels = normalize(pixels, device)
        labels = [torch.zeros(len(mask), device=device, dtype=torch.long) for mask in masks]
        masks = [mask.to(device=device, dtype=torch.float32, non_blocking=True) for mask in masks]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            output = model(pixel_values=pixels, output_auxiliary_logits=True)
        # SciPy matching and point-sampled losses operate on FP32 logits.
        auxiliary = [{key: value.float() for key, value in item.items()} for item in output.auxiliary_logits]
        loss = model.get_loss(model.get_loss_dict(output.masks_queries_logits.float(),
            output.class_queries_logits.float(), masks, labels, auxiliary))
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at batch {step + 1}. Try --precision fp32.")
        group_size = min(accumulation, len(loader) - step // accumulation * accumulation)
        (loss / group_size).backward()
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0,
                                     error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == len(loader):
            print(f"  batch {step + 1}/{len(loader)} loss={losses[-1]:.4f}", flush=True)
    return float(np.mean(losses))


def inner_boundary(mask, radius):
    kernel = np.ones((3, 3), dtype=np.uint8)
    interior = cv2.erode(mask, kernel, iterations=radius, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return mask.astype(bool) & ~interior.astype(bool)


def score_predictions(dataset, predictions, ids, boundary_radius):
    summary = {"mask_ap": 0.0, "mask_ap50": 0.0, "mask_ap75": 0.0,
               "boundary_iou_matched_at_mask_iou50": None, "boundary_matched_instances": 0,
               "evaluated_images": len(ids), "evaluated_gt_instances": sum(
                   len(anns) for image_id, anns in zip(dataset.ids, dataset.annotations) if image_id in ids),
               "boundary_radius_px": boundary_radius}
    if not predictions:
        return summary
    # pycocotools adds IDs and boxes in place. Keep the original RLE records
    # intact for the subsequent per-source evaluations.
    detections = dataset.coco.loadRes([prediction.copy() for prediction in predictions])
    evaluator = COCOeval(dataset.coco, detections, "segm")
    evaluator.params.imgIds = sorted(ids)
    evaluator.params.catIds = [dataset.category_id]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    summary.update(zip(("mask_ap", "mask_ap50", "mask_ap75"), map(float, evaluator.stats[:3])))
    boundary_scores = []
    for item in evaluator.evalImgs:
        if item is None or item["aRng"] != evaluator.params.areaRng[0]:
            continue
        for dt_id, gt_id, ignore in zip(item["dtIds"], item["dtMatches"][0], item["dtIgnore"][0]):
            if gt_id == 0 or ignore:
                continue
            predicted = inner_boundary(detections.annToMask(detections.anns[dt_id]), boundary_radius)
            target = inner_boundary(dataset.coco.annToMask(dataset.coco.anns[int(gt_id)]), boundary_radius)
            boundary_scores.append(np.count_nonzero(predicted & target) / np.count_nonzero(predicted | target))
    if boundary_scores:
        summary["boundary_iou_matched_at_mask_iou50"] = float(np.mean(boundary_scores))
        summary["boundary_matched_instances"] = len(boundary_scores)
    return summary


@torch.no_grad()
def evaluate(model, loader, dataset, device, boundary_radius):
    model.eval()
    predictions, ids = [], set()
    for pixels, _, image_ids in loader:
        output = model(pixel_values=normalize(pixels, device))
        foreground = output.class_queries_logits.float().softmax(-1)[..., 0]
        for image_id, logits, scores in zip(image_ids, output.masks_queries_logits, foreground):
            row = dataset.coco.imgs[image_id]
            mask_probs = F.interpolate(logits[None].float(), size=(row["height"], row["width"]),
                                       mode="bilinear", align_corners=False)[0].sigmoid()
            image_masks = mask_probs > 0.5
            image_areas = image_masks.sum((-2, -1))
            image_scores = scores * (mask_probs * image_masks).sum((-2, -1)) / image_areas.clamp_min(1)
            ids.add(image_id)
            keep = image_areas > 0
            order = image_scores.masked_fill(~keep, -1).argsort(descending=True)[:100]
            order = order[keep[order]]
            for mask, score in zip(image_masks[order].cpu().numpy(), image_scores[order].tolist()):
                predictions.append({"image_id": image_id, "category_id": dataset.category_id,
                    "segmentation": coco_mask.encode(np.asfortranarray(mask, dtype=np.uint8)), "score": score})
    metrics = score_predictions(dataset, predictions, ids, boundary_radius)
    source_ids = defaultdict(set)
    for image_id in ids:
        source = dataset.coco.imgs[image_id].get("source")
        if source:
            source_ids[source].add(image_id)
    if source_ids:
        source_predictions = defaultdict(list)
        for prediction in predictions:
            source = dataset.coco.imgs[prediction["image_id"]].get("source")
            source_predictions[source].append(prediction)
        metrics["by_source"] = {
            source: score_predictions(dataset, source_predictions[source], members, boundary_radius)
            for source, members in sorted(source_ids.items())
        }
    return metrics


FROZEN_PREFIX = "model.pixel_level_module.encoder.dino."


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=home / "WHU_Building_Dataset")
    for split in ("train", "val"):
        parser.add_argument(f"--{split}-images", type=Path)
        parser.add_argument(f"--{split}-annotations", type=Path)
    parser.add_argument("--dinov3-repo", type=Path, default=home / "repos/dinov3")
    parser.add_argument("--weights", type=Path, default=home / "weights/dinov3_vitb16_pretrain_lvd1689m.pth")
    parser.add_argument("--output", type=Path, required=True)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume", type=Path)
    checkpoint_group.add_argument("--init-weights", type=Path, help="Start a new run with trained head weights")
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument("--image-size", type=int, default=512)
    size_group.add_argument("--mixed-native-sizes", action="store_true", help="Train on native 512 and 1024 tiles without resizing")
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--category-id", type=int)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.image_size < 64 or args.image_size % 32:
        parser.error("--image-size must be at least 64 and divisible by 32")
    if min(args.num_queries, args.epochs, args.batch_size, args.accumulation, args.boundary_radius) < 1:
        parser.error("query count, epochs, batch size, accumulation and boundary radius must be positive")
    if args.workers < 0 or args.lr <= 0 or any(v is not None and v < 1 for v in (args.limit_train, args.limit_val)):
        parser.error("workers must be nonnegative; learning rate and dataset limits must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Activate dino-whu-m2f and run verify_env.py.")
    if args.precision == "bf16" and (args.device != "cuda" or not torch.cuda.is_bf16_supported()):
        parser.error("bf16 mode requires a CUDA device with BF16 support")
    repo, weights = args.dinov3_repo.expanduser().resolve(), args.weights.expanduser().resolve()
    if not (repo / "hubconf.py").is_file() or not weights.is_file():
        parser.error(f"DINOv3 source or weights not found: {repo} / {weights}")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    root = args.data_root.expanduser()
    val_dir = root / "validation"
    if not val_dir.is_dir() and (root / "val").is_dir():
        val_dir = root / "val"
    image_size = [512, 1024] if args.mixed_native_sizes else args.image_size
    train = WhuInstances(args.train_images or root / "train/images",
        args.train_annotations or root / "train/annotations.json", image_size,
        args.num_queries, True, args.category_id)
    val = WhuInstances(args.val_images or val_dir / "images",
        args.val_annotations or val_dir / "annotations.json", image_size,
        args.num_queries, False, args.category_id)
    if {path.resolve() for path in train.paths} & {path.resolve() for path in val.paths}:
        parser.error("train and validation reference some of the same image files")
    loader_options = dict(num_workers=args.workers,
                          pin_memory=device.type == "cuda", collate_fn=collate)
    train_subset = limit_dataset(train, args.limit_train)
    val_subset = limit_dataset(val, args.limit_val)
    if args.mixed_native_sizes:
        train_loader = DataLoader(train_subset, batch_sampler=NativeSizeBatches(train_subset, args.batch_size, True),
                                  **loader_options)
        val_loader = DataLoader(val_subset, batch_sampler=NativeSizeBatches(val_subset, args.batch_size, False),
                                **loader_options)
    else:
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, **loader_options)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, **loader_options)
    with weights.open("rb") as stream:
        weights_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    identity = {"num_queries": args.num_queries, "image_size": image_size,
                "dinov3_weights_sha256": weights_hash, "dinov3_source_commit": commit}
    dino = torch.hub.load(str(repo), "dinov3_vitb16", source="local", weights=str(weights))
    model = build_model(dino, args.num_queries).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.05)
    start_epoch, best_ap = 1, -1.0
    if args.resume or args.init_weights:
        checkpoint = torch.load(args.resume or args.init_weights, map_location="cpu", weights_only=True)
        if args.resume and checkpoint["identity"] != identity:
            parser.error("Resume requires the same DINO source, weights, input size settings and query count")
        init_keys = ("num_queries", "dinov3_weights_sha256", "dinov3_source_commit")
        if args.init_weights and any(checkpoint["identity"].get(k) != identity[k] for k in init_keys):
            parser.error("Initialization requires the same DINO source, weights and query count")
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        if unexpected or any(not name.startswith(FROZEN_PREFIX) for name in missing):
            parser.error(f"Incompatible checkpoint keys: {missing}, {unexpected}")
        if args.resume:
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch, best_ap = checkpoint["epoch"] + 1, checkpoint["best_ap"]
        else:
            print(f"Initialized trainable weights from {args.init_weights}", flush=True)
    if start_epoch > args.epochs:
        parser.error("--epochs is the final epoch number and must exceed the resume checkpoint epoch")
    args.output.mkdir(parents=True, exist_ok=True)
    last_path = args.output / "last.pt"
    if last_path.exists() and (args.resume is None or args.resume.resolve() != last_path.resolve()):
        parser.error("Choose a fresh --output directory, or resume from its last.pt")
    run_info = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "identity": identity, "mask2former_config": model.config.to_dict()}
    (args.output / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        torch.manual_seed(args.seed + epoch)
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        loss = train_epoch(model, train_loader, optimizer, device, args.precision, args.accumulation)
        metrics = evaluate(model, val_loader, val, device, args.boundary_radius)
        metrics.update(epoch=epoch, train_loss=loss, elapsed_seconds=time.perf_counter() - start)
        if device.type == "cuda":
            metrics["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        improved = metrics["mask_ap"] > best_ap
        best_ap = max(best_ap, metrics["mask_ap"])
        state = {"model": {k: v for k, v in model.state_dict().items() if not k.startswith(FROZEN_PREFIX)},
                 "optimizer": optimizer.state_dict(), "epoch": epoch, "best_ap": best_ap,
                 "identity": identity, "metrics": metrics}
        save_checkpoint(last_path, state)
        if improved:
            save_checkpoint(args.output / "best.pt", state)
        with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics, indent=2), flush=True)
    print(f"Training finished. Checkpoints and metrics: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
