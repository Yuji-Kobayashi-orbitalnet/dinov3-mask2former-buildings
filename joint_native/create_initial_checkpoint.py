"""Create fresh trainable weights on CPU for native 512/1024 convergence training.

Only DINOv3 uses pretrained weights. No images, training, or evaluation are run.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch
import transformers

from train_whu import FROZEN_PREFIX, build_model


def main():
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New checkpoint file; existing files are never overwritten")
    parser.add_argument("--num-queries", type=int, default=158,
                        help="Query capacity (default: 158). Dataset object counts are checked when training starts")
    parser.add_argument("--seed", type=int, default=42,
                        help="CPU random initialization seed (default: 42)")
    parser.add_argument("--dinov3-repo", type=Path, default=home / "repos/dinov3")
    parser.add_argument("--weights", type=Path,
                        default=home / "weights/dinov3_vitb16_pretrain_lvd1689m.pth")
    args = parser.parse_args()
    if args.num_queries < 1 or not 0 <= args.seed < 2**64:
        parser.error("--num-queries must be positive; --seed must be between 0 and 2**64 - 1")
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        parser.error("Choose a new --output file; existing checkpoints are never overwritten")
    repo = args.dinov3_repo.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    if not (repo / "hubconf.py").is_file() or not weights.is_file():
        parser.error("A local DINOv3 repository with hubconf.py and its pretrained weights are required")
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    with weights.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    identity = {"image_size": [512, 1024], "num_queries": args.num_queries,
                "dinov3_source_commit": commit, "dinov3_weights_sha256": digest}

    torch.random.default_generator.manual_seed(args.seed)
    print("Creating fresh projection, RGB helper, and Mask2Former weights on CPU...", flush=True)
    with torch.device("cpu"):
        dino = torch.hub.load(str(repo), "dinov3_vitb16", source="local", weights=str(weights))
        model = build_model(dino, args.num_queries).cpu()
    initialization = {
        "kind": "random_trainable_weights_with_pretrained_frozen_dinov3",
        "seed": args.seed, "device": "cpu", "training_updates": 0,
        "dinov3_repo": str(repo), "dinov3_weights": str(weights),
        "torch": str(torch.__version__), "transformers": transformers.__version__,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }
    checkpoint = {
        "mode": "random_trainable_initialization", "epoch": 0, "identity": identity,
        "initialization": initialization,
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()
                  if not name.startswith(FROZEN_PREFIX)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also rejects a file created while the model was loading.
    try:
        stream = output.open("xb")
    except FileExistsError:
        parser.error("Output appeared during initialization; it has not been overwritten")
    try:
        with stream:
            torch.save(checkpoint, stream)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    print(json.dumps({"checkpoint": str(output), "epoch": 0, "identity": identity,
                      "initialization": initialization}, indent=2), flush=True)
    print("Ready for train_to_convergence.py --init-weights. No training or evaluation has run.")


if __name__ == "__main__":
    main()
