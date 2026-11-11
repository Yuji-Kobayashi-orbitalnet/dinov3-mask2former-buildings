"""Exercise real GPU kernels and Mask2Former loss/backward, without downloads.

The small Swin backbone isolates Mask2Former dependency checks. This is NOT
the DINOv3 + adapter + Mask2Former training model.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-blackwell", action="store_true")
    parser.add_argument("--dinov3-repo", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parent / "reports" / "gpu-check.json")
    args = parser.parse_args()
    if bool(args.dinov3_repo) != bool(args.weights):
        parser.error("Provide both --dinov3-repo and --weights, or neither.")
    import torch
    import torchvision
    import transformers
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, SwinConfig

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Check driver and CUDA 12.8 PyTorch wheel.")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    if args.require_blackwell and (props.major, props.minor) != (12, 0):
        raise SystemExit(f"Expected RTX PRO Blackwell CC 12.0; got {props.name}, {props.major}.{props.minor}")
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    # Actual FP32 and BF16 operations, rather than just is_available().
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(128, 128, device=device, dtype=dtype, requires_grad=True)
        loss = (x @ x.T).float().square().mean()
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(x.grad).all()

    backbone = SwinConfig(image_size=128, embed_dim=32, depths=[1, 1, 1, 1],
        num_heads=[1, 2, 4, 8], window_size=4,
        out_features=["stage1", "stage2", "stage3", "stage4"])
    config = Mask2FormerConfig(backbone_config=backbone.to_dict(), num_labels=1,
        num_queries=8, feature_size=64, mask_feature_size=64, hidden_dim=64,
        encoder_layers=1, decoder_layers=2, num_attention_heads=4,
        encoder_feedforward_dim=128, dim_feedforward=128, train_num_points=256)
    model = Mask2FormerForUniversalSegmentation(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    image = torch.randn(1, 3, 128, 128, device=device)
    masks = torch.zeros(1, 128, 128, device=device)
    masks[:, 24:100, 30:104] = 1
    start = time.perf_counter()
    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(pixel_values=image, mask_labels=[masks],
            class_labels=[torch.zeros(1, dtype=torch.long, device=device)])
        assert torch.isfinite(output.loss), "Non-finite Mask2Former loss"
        assert output.masks_queries_logits.shape == (1, 8, 32, 32)
        output.loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        losses.append(output.loss.item())
    torch.cuda.synchronize()
    report = dict(torch=torch.__version__, torchvision=torchvision.__version__,
        transformers=transformers.__version__, cuda_runtime=torch.version.cuda,
        device=props.name, capability=f"{props.major}.{props.minor}",
        total_vram_gib=props.total_memory / 2**30,
        torch_arch_list=torch.cuda.get_arch_list(), mask2former_smoke_losses=losses,
        smoke_seconds=time.perf_counter() - start,
        smoke_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        scope="tiny Swin + standard Mask2Former, FP32 forward/loss/backward/optimizer; not WHU training",
        dinov3_checked=False)

    if args.dinov3_repo:
        if not (args.dinov3_repo / "hubconf.py").is_file() or not args.weights.is_file():
            raise SystemExit("DINOv3 repository or checkpoint path does not exist.")
        dino = torch.hub.load(str(args.dinov3_repo.resolve()), "dinov3_vitb16",
            source="local", weights=str(args.weights.resolve())).to(device).eval()
        dino.requires_grad_(False)
        with torch.no_grad():
            taps = dino.get_intermediate_layers(torch.randn(1, 3, 512, 512, device=device),
                n=[2, 5, 8, 11], reshape=True)
        assert len(taps) == 4
        assert all(tuple(t.shape) == (1, 768, 32, 32) for t in taps)
        assert all(torch.isfinite(t).all() and not t.requires_grad for t in taps)
        report["dinov3_checked"] = True
        report["dinov3_shapes"] = [list(t.shape) for t in taps]
        report["dinov3_source_commit"] = subprocess.check_output(
            ["git", "-C", str(args.dinov3_repo), "rev-parse", "HEAD"], text=True).strip()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("PASS: environment smoke check. Integrated DINO/WHU trainer must be checked separately.")


if __name__ == "__main__":
    main()
