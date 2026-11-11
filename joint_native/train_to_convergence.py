"""Continue joint native-size training with loss monitoring until a plateau."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch
from torch.utils.data import DataLoader

from train_whu import (
    FROZEN_PREFIX, NativeSizeBatches, WhuInstances, build_model, collate,
    evaluate, normalize, save_checkpoint, train_epoch,
)


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_counts(dataset):
    return dict(Counter(dataset.coco.imgs[image_id].get("source", "dataset") for image_id in dataset.ids))


def load_trainable_weights(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not name.startswith(FROZEN_PREFIX) for name in missing):
        raise ValueError(f"Incompatible checkpoint keys: {missing}, {unexpected}")


@torch.no_grad()
def validation_loss(model, loader, dataset, device, seed):
    """Use reproducible point sampling and restore the training RNG state."""
    model.eval()
    totals, counts = defaultdict(float), Counter()
    devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        for step, (pixels, masks, image_ids) in enumerate(loader, start=1):
            if len(image_ids) != 1:
                raise ValueError("Validation loss requires one image per batch")
            pixels = normalize(pixels, device)
            labels = [torch.zeros(len(mask), device=device, dtype=torch.long) for mask in masks]
            masks = [mask.to(device=device, dtype=torch.float32) for mask in masks]
            output = model(pixel_values=pixels, output_auxiliary_logits=True)
            auxiliary = [{key: value.float() for key, value in item.items()} for item in output.auxiliary_logits]
            loss = model.get_loss(model.get_loss_dict(output.masks_queries_logits.float(),
                output.class_queries_logits.float(), masks, labels, auxiliary))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite validation loss at batch {step}")
            source = dataset.coco.imgs[image_ids[0]].get("source", "dataset")
            totals[source] += loss.item()
            counts[source] += 1
            if step == 1 or step % 500 == 0 or step == len(loader):
                print(f"  validation loss batch {step}/{len(loader)}", flush=True)
    means = {source: totals[source] / counts[source] for source in sorted(counts)}
    return sum(means.values()) / len(means), means


def main():
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--init-weights", type=Path)
    checkpoint_group.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50, help="Maximum total epoch number, including resumed epochs")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--train-annotations", type=Path, default=Path("prepared_joint_native/train.json"))
    parser.add_argument("--val-annotations", type=Path, default=Path("prepared_joint_native/validation.json"))
    parser.add_argument("--dinov3-repo", type=Path, default=home / "repos/dinov3")
    parser.add_argument("--weights", type=Path, default=home / "weights/dinov3_vitb16_pretrain_lvd1689m.pth")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.accumulation) < 1 or args.workers < 0 or args.seed < 0:
        parser.error("Epochs, batch size and accumulation must be positive; workers and seed must be nonnegative")
    if not 1e-7 <= args.lr < float("inf"):
        parser.error("--lr must be finite and at least 1e-7")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Activate dino-whu-m2f on the GPU server.")
    output_dir = args.output.expanduser().resolve()
    last_path, best_path = output_dir / "last.pt", output_dir / "best_loss.pt"
    checkpoint_path = (args.resume or args.init_weights).expanduser().resolve()
    if args.resume:
        if checkpoint_path != last_path or not best_path.is_file():
            parser.error("Resume from this output directory's last.pt, with best_loss.pt present")
    elif output_dir.exists():
        parser.error("Initialization requires a new output directory")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    identity = checkpoint["identity"]
    repo, weights = args.dinov3_repo.expanduser().resolve(), args.weights.expanduser().resolve()
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if file_digest(weights) != identity["dinov3_weights_sha256"] or commit != identity["dinov3_source_commit"]:
        parser.error("DINOv3 source and weights must match the checkpoint")
    annotation_paths = {"train": args.train_annotations.expanduser().resolve(),
                        "validation": args.val_annotations.expanduser().resolve()}
    settings = {
        "seed": args.seed, "batch_size": args.batch_size, "accumulation": args.accumulation,
        "initial_lr": args.lr, "optimizer": "AdamW", "weight_decay": 0.05, "precision": "fp32",
        "monitor": "macro_average_of_source_validation_losses", "validation_batch_size": 1,
        "validation_seed": args.seed, "relative_improvement": 0.001,
        "scheduler_factor": 0.5, "scheduler_patience": 2, "minimum_lr": 1e-7,
        "stop_patience": 8, "epochs_after_lr_change": 2,
        "annotations": {name: {"path": str(path), "sha256": file_digest(path)}
                        for name, path in annotation_paths.items()},
    }
    if args.resume:
        if checkpoint.get("mode") != "convergence_loss" or checkpoint.get("settings") != settings:
            parser.error("Resume requires a convergence checkpoint with the same data and training settings")
        saved_best = torch.load(best_path, map_location="cpu", weights_only=True)
        if saved_best.get("settings") != settings or saved_best.get("identity") != identity:
            parser.error("best_loss.pt does not belong to this run's data and model")
        # A completed best save can precede an interrupted last save.
        if saved_best["epoch"] > checkpoint["epoch"]:
            checkpoint = saved_best
            print(f"Recovering the newer completed checkpoint at epoch {checkpoint['epoch']}", flush=True)
        if (saved_best["epoch"] != checkpoint["best_epoch"]
                or saved_best["best_val_loss"] != checkpoint["best_val_loss"]):
            parser.error("Best checkpoint and resume metadata disagree")
        del saved_best
        if checkpoint.get("converged"):
            status_path = output_dir / "training_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
            if status.get("final_evaluation_complete") and (output_dir / "final_metrics.json").is_file():
                print(f"Training and final evaluation are complete: {output_dir / 'final_metrics.json'}")
                return
        elif checkpoint["epoch"] >= args.epochs:
            parser.error("--epochs must exceed the last saved epoch")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dino = torch.hub.load(str(repo), "dinov3_vitb16", source="local", weights=str(weights))
    model = build_model(dino, identity["num_queries"]).to(device)
    load_trainable_weights(model, checkpoint["model"])
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, threshold=0.001, threshold_mode="rel", min_lr=1e-7)
    train = WhuInstances(Path.cwd(), annotation_paths["train"], identity["image_size"],
                         identity["num_queries"], True, None)
    val = WhuInstances(Path.cwd(), annotation_paths["validation"], identity["image_size"],
                       identity["num_queries"], False, None)
    if {path.resolve() for path in train.paths} & {path.resolve() for path in val.paths}:
        parser.error("Train and validation reference some of the same image files")
    counts = {"train": source_counts(train), "validation": source_counts(val)}
    parameters = {"trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad)}
    print(json.dumps({"source_images": counts, "parameters": parameters, "dinov3_frozen": True}, indent=2), flush=True)
    options = dict(num_workers=args.workers, pin_memory=device.type == "cuda", collate_fn=collate)
    train_loader = DataLoader(train, batch_sampler=NativeSizeBatches(train, args.batch_size, True), **options)
    val_loader = DataLoader(val, batch_sampler=NativeSizeBatches(val, 1, False), **options)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "loss_history.jsonl"

    def snapshot(epoch, metrics, converged):
        return {"mode": "convergence_loss", "model": {name: value for name, value in model.state_dict().items()
                                                    if not name.startswith(FROZEN_PREFIX)},
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "identity": identity, "settings": settings, "source_counts": counts, "parameters": parameters,
                "dinov3_frozen": True, "epoch": epoch, "metrics": metrics, "converged": converged,
                "best_val_loss": best_loss, "best_epoch": best_epoch, "reference_loss": reference_loss,
                "bad_epochs": bad_epochs, "last_lr_change_epoch": last_lr_change_epoch,
                "optimizer_steps": optimizer_steps, "initialization": initialization}

    def append_history(metrics):
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics, indent=2), flush=True)

    def finalize(epoch):
        status = {"converged": True, "reason": "validation_loss_plateau", "last_epoch": epoch,
                  "best_epoch": best_epoch, "best_val_loss": best_loss, "final_evaluation_complete": False}
        write_json(output_dir / "training_status.json", status)
        print(f"Validation loss plateau reached. Evaluating best_loss.pt from epoch {best_epoch}.", flush=True)
        best = torch.load(best_path, map_location="cpu", weights_only=True)
        load_trainable_weights(model, best["model"])
        del best
        final_metrics = evaluate(model, val_loader, val, device, 2)
        final_metrics.update(selected_epoch=best_epoch, best_val_loss=best_loss)
        write_json(output_dir / "final_metrics.json", final_metrics)
        status["final_evaluation_complete"] = True
        write_json(output_dir / "training_status.json", status)
        print(json.dumps(final_metrics, indent=2), flush=True)

    resume_final_only = args.resume is not None and checkpoint.get("converged", False)
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_loss, best_epoch = checkpoint["best_val_loss"], checkpoint["best_epoch"]
        reference_loss, bad_epochs = checkpoint["reference_loss"], checkpoint["bad_epochs"]
        last_lr_change_epoch = checkpoint["last_lr_change_epoch"]
        optimizer_steps, initialization = checkpoint["optimizer_steps"], checkpoint["initialization"]
        start_epoch = checkpoint["epoch"] + 1
    else:
        initialization = {"checkpoint": str(checkpoint_path), "checkpoint_epoch": checkpoint["epoch"]}
        print("Measuring the initial checkpoint's validation loss", flush=True)
        best_loss, means = validation_loss(model, val_loader, val, device, args.seed)
        scheduler.step(best_loss)
        best_epoch, bad_epochs, last_lr_change_epoch, optimizer_steps = 0, 0, 0, 0
        reference_loss, start_epoch = best_loss, 1
        metrics = {"epoch": 0, "phase": "initial", "train_loss": None, "val_loss_by_source": means,
                   "val_loss_macro": best_loss, "lr_used": args.lr, "lr_next": args.lr,
                   "bad_epochs": 0, "optimizer_steps": 0}
        state = snapshot(0, metrics, False)
        save_checkpoint(best_path, state)
        save_checkpoint(last_path, state)
        append_history(metrics)
    del checkpoint
    if resume_final_only:
        finalize(start_epoch - 1)
        return
    write_json(output_dir / "training_status.json",
               {"converged": False, "reason": "running", "last_epoch": start_epoch - 1, "best_epoch": best_epoch})

    for epoch in range(start_epoch, args.epochs + 1):
        torch.manual_seed(args.seed + epoch)
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        start = time.perf_counter()
        lr_used = optimizer.param_groups[0]["lr"]
        train_loss = train_epoch(model, train_loader, optimizer, device, "fp32", args.accumulation)
        steps_this_epoch = (len(train_loader) + args.accumulation - 1) // args.accumulation
        optimizer_steps += steps_this_epoch
        monitored, means = validation_loss(model, val_loader, val, device, args.seed)
        improved = monitored < best_loss
        if improved:
            best_loss, best_epoch = monitored, epoch
        if monitored < reference_loss * (1 - settings["relative_improvement"]):
            reference_loss, bad_epochs = monitored, 0
        else:
            bad_epochs += 1
        scheduler.step(monitored)
        lr_next = optimizer.param_groups[0]["lr"]
        if lr_next < lr_used:
            last_lr_change_epoch = epoch
        converged = (bad_epochs >= settings["stop_patience"]
                     and epoch - last_lr_change_epoch >= settings["epochs_after_lr_change"])
        metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss_by_source": means,
                   "val_loss_macro": monitored, "lr_used": lr_used, "lr_next": lr_next,
                   "bad_epochs": bad_epochs, "last_lr_change_epoch": last_lr_change_epoch,
                   "optimizer_steps_this_epoch": steps_this_epoch, "optimizer_steps": optimizer_steps,
                   "elapsed_seconds": time.perf_counter() - start}
        state = snapshot(epoch, metrics, converged)
        if improved:
            save_checkpoint(best_path, state)
        save_checkpoint(last_path, state)
        append_history(metrics)
        if converged:
            finalize(epoch)
            return

    write_json(output_dir / "training_status.json",
               {"converged": False, "reason": "epoch_limit", "last_epoch": args.epochs,
                "best_epoch": best_epoch, "best_val_loss": best_loss})
    print("Epoch limit reached before the validation-loss plateau criterion. Resume last.pt with a higher --epochs.",
          flush=True)


if __name__ == "__main__":
    main()
