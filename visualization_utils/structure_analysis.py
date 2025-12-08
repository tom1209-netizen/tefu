"""
Layer similarity and effective receptive field (ERF) analysis utilities.

This script demonstrates:
  1) Raw ViT stages suffer from oversmoothing (high similarity) and lack hierarchy.
  2) Structural distillation + Adapters recover spatial hierarchy (sharper ERF, distinct stages).
  3) SegFormer (Teacher) provides the target hierarchical structure.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Ensure root is in path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.conch_adapter import ConchAdapter
from model.model import DistilledConch
from utils.pyutils import set_seed
from utils.trainutils import get_cls_dataset


def build_clip_adapter(cfg, device: torch.device) -> ConchAdapter:
    clip_cfg = OmegaConf.to_container(getattr(cfg, "clip", None) or {}, resolve=True)
    return ConchAdapter(
        model_name=clip_cfg.get("model_name", "conch_ViT-B-16"),
        checkpoint_path=clip_cfg.get("checkpoint_path"),
        device=device,
        force_image_size=clip_cfg.get("force_image_size"),
        cache_dir=clip_cfg.get("cache_dir"),
        hf_hub=clip_cfg.get("hf_hub"),
        hf_token=clip_cfg.get("hf_token"),
        proj_contrast=clip_cfg.get("proj_contrast", False),
        freeze=clip_cfg.get("freeze", True),
    )


def build_model(
    cfg, checkpoint_path: Optional[str], device: torch.device
) -> DistilledConch:
    clip_adapter = build_clip_adapter(cfg, device)
    guidance_cfg = OmegaConf.to_container(
        getattr(cfg.model, "segformer_guidance", {}) or {}, resolve=True
    )
    model = DistilledConch(
        cls_num_classes=cfg.dataset.cls_num_classes,
        num_prototypes_per_class=cfg.model.num_prototypes_per_class,
        prototype_feature_dim=cfg.model.prototype_feature_dim,
        clip_adapter=clip_adapter,
        enable_segformer_guidance=guidance_cfg.get("enable", True),
        segformer_backbone=guidance_cfg.get("backbone", "mit_b1"),
        segformer_checkpoint=guidance_cfg.get("checkpoint"),
        guidance_layers=tuple(guidance_cfg.get("layers", (2,))),
        text_prompts=getattr(cfg.model, "text_prompts", None),
        n_ratio=cfg.model.n_ratio,
        pretrained=False,
    )
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}...")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state.get("model", state), strict=False)

    model.to(device)
    model.eval()

    # CRITICAL FIX: Force teacher to train mode (but keep weights frozen)
    # so that gradients can flow through the graph for ERF calculation.
    if model.segformer_teacher is not None:
        model.segformer_teacher.train()
        for p in model.segformer_teacher.parameters():
            p.requires_grad_(False)

    return model


def build_loader(cfg, split: str, batch_size: int, num_workers: int) -> DataLoader:
    _, dataset = get_cls_dataset(cfg, split=split, enable_rotation=False, p=0.0)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(num_workers, os.cpu_count() or 1),
        pin_memory=True,
    )


# --- Feature Extractors (Updated for Raw vs Refined) ---
def _student_feats_grad(
    model: DistilledConch, x: torch.Tensor, use_raw: bool = False
) -> List[torch.Tensor]:
    # use_grad=True is required in visual_intermediates to allow backprop to input
    vis = model.clip_adapter.visual_intermediates(x, use_grad=True)

    # Pad to 4 levels if necessary (standard behavior in your model)
    while len(vis) < 4:
        vis.append(F.avg_pool2d(vis[-1], kernel_size=2, stride=2))

    if use_raw:
        # Return raw backbone features (Grid artifacts, Oversmoothing)
        return vis
    else:
        # Return refined features (Adapters applied)
        return [adapter(f) for adapter, f in zip(model.structure_adapters, vis)]


def _student_feats_no_grad(
    model: DistilledConch, x: torch.Tensor, use_raw: bool = False
) -> List[torch.Tensor]:
    with torch.no_grad():
        vis = model.clip_adapter.visual_intermediates(x, use_grad=False)
        while len(vis) < 4:
            vis.append(F.avg_pool2d(vis[-1], kernel_size=2, stride=2))

        if use_raw:
            return vis
        else:
            # We must detach because model.structure_adapters might track grad otherwise
            return [
                adapter(f.detach()) for adapter, f in zip(model.structure_adapters, vis)
            ]


def _teacher_feats_grad(model: DistilledConch, x: torch.Tensor) -> List[torch.Tensor]:
    if model.segformer_teacher is None:
        raise RuntimeError("SegFormer teacher is not enabled for this model.")

    # Standardize input for teacher
    student_mean = getattr(model, "student_mean", None)
    student_std = getattr(model, "student_std", None)
    teacher_mean = getattr(model, "teacher_mean", None)
    teacher_std = getattr(model, "teacher_std", None)

    x_raw = x
    if student_mean is not None and student_std is not None:
        x_raw = x * student_std + student_mean
    if teacher_mean is not None and teacher_std is not None:
        x_teacher = (x_raw - teacher_mean) / teacher_std
    else:
        x_teacher = x_raw

    feats, _ = model.segformer_teacher(x_teacher)
    return feats


def _teacher_feats_no_grad(
    model: DistilledConch, x: torch.Tensor
) -> Optional[List[torch.Tensor]]:
    if model.segformer_teacher is None:
        return None
    with torch.no_grad():
        return model._get_teacher_feats(x)


# --- Analysis Functions ---
def compute_layer_similarity(
    model: DistilledConch,
    dataloader: DataLoader,
    feat_fn: Callable[[torch.Tensor], List[torch.Tensor]],
    num_batches: int,
    device: torch.device,
    pool_size: int = 16,
) -> Optional[np.ndarray]:
    sim_matrix = None
    count = 0

    for batch_idx, (_, inputs, _, _) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        inputs = inputs.to(device)
        feats = feat_fn(inputs)

        if not feats:
            continue

        # [Crash Fix] Check channel consistency
        current_dims = [f.shape[1] for f in feats]
        if len(set(current_dims)) > 1:
            print(
                f"  [Skipping Similarity] Detected hierarchical dimensions: {current_dims}."
            )
            print(
                "  Cannot compute standard cosine similarity between different channel widths."
            )
            return None

        num_layers = len(feats)
        flat_feats = []
        for f in feats:
            pooled = F.adaptive_avg_pool2d(f, (pool_size, pool_size))
            b, c, h, w = pooled.shape
            flat = pooled.view(b, c, -1).permute(0, 2, 1)  # [B, HW, C]
            flat = F.normalize(flat, dim=2)
            flat_feats.append(flat)

        if sim_matrix is None:
            sim_matrix = torch.zeros(num_layers, num_layers, device=device)

        for r in range(num_layers):
            for c in range(num_layers):
                sim = (flat_feats[r] * flat_feats[c]).sum(dim=2).mean()
                sim_matrix[r, c] += sim
        count += 1

    if sim_matrix is None or count == 0:
        return None
    sim_matrix /= count
    return sim_matrix.detach().cpu().numpy()


def compute_erf(
    model: DistilledConch,
    dataloader: DataLoader,
    feat_fn: Callable[[torch.Tensor], List[torch.Tensor]],
    layer_idx: int,
    num_images: int,
    device: torch.device,
) -> Optional[np.ndarray]:
    erf_accum = None
    seen = 0

    for _, inputs, _, _ in dataloader:
        if seen >= num_images:
            break
        batch = inputs.to(device)[: max(1, min(inputs.size(0), num_images - seen))]
        batch.requires_grad_(True)

        feats = feat_fn(batch)
        if layer_idx >= len(feats):
            # Fallback to last layer if requested index is out of bounds
            target = feats[-1]
        else:
            target = feats[layer_idx]

        h, w = target.shape[-2:]
        grad_mask = torch.zeros_like(target)
        grad_mask[:, :, h // 2, w // 2] = 1.0

        target.backward(gradient=grad_mask)

        grad_map = batch.grad.detach().abs().sum(dim=1)  # [B, H, W]
        if erf_accum is None:
            erf_accum = torch.zeros_like(grad_map[0])

        erf_accum += grad_map.sum(dim=0)
        seen += batch.size(0)
        model.zero_grad(set_to_none=True)

    if erf_accum is None or seen == 0:
        return None

    erf = (erf_accum / seen).cpu().numpy()

    # Log scaling / Sqrt scaling helps visualize the "halo" better
    # Standard min-max norm
    erf = (erf - erf.min()) / (erf.max() - erf.min() + 1e-8)
    return erf


# --- Plotting ---
def plot_similarity(matrix: np.ndarray, title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="magma")
    num_layers = matrix.shape[0]
    labels = [f"Stage {i+1}" for i in range(num_layers)]
    ax.set_xticks(range(num_layers))
    ax.set_yticks(range(num_layers))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    # Add text annotations
    for i in range(num_layers):
        for j in range(num_layers):
            color = "white" if matrix[i, j] < 0.7 else "black"  # dynamic text color
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )

    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_erf(erf_map: np.ndarray, title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.imshow(erf_map, cmap="inferno")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# --- Main Analysis Loop ---
def analyze_model(
    tag: str,
    model: DistilledConch,
    dataloader: DataLoader,
    args,
    output_dir: Path,
) -> None:
    # Raw ViT Features (The "Before" / "Problem" Story)
    print(f"\n[{tag}] Analyzing RAW ViT features (Bypassing Adapters)...")

    # Similarity (Should be high -> Oversmoothing)
    raw_sim = compute_layer_similarity(
        model=model,
        dataloader=dataloader,
        feat_fn=lambda x: _student_feats_no_grad(model, x, use_raw=True),
        num_batches=args.num_batches,
        device=args.device,
    )
    if raw_sim is not None:
        plot_similarity(
            raw_sim,
            f"{tag} - Raw ViT Similarity",
            output_dir / f"{tag}_raw_similarity.png",
        )
        print(f"[{tag}] Saved raw similarity heatmap.")

    # ERF (Should be grid-like / diffuse)
    raw_erf = compute_erf(
        model=model,
        dataloader=dataloader,
        feat_fn=lambda x: _student_feats_grad(model, x, use_raw=True),
        layer_idx=args.erf_layer,
        num_images=args.num_erf_images,
        device=args.device,
    )
    if raw_erf is not None:
        raw_erf = np.power(raw_erf, 0.5)  # Square root scaling
        plot_erf(raw_erf, f"{tag} - Raw ViT ERF", output_dir / f"{tag}_raw_erf.png")
        print(f"[{tag}] Saved raw ERF.")

    # Refined Features (The "After" / "Solution" Story)
    print(f"\n[{tag}] Analyzing REFINED features (With Adapters)...")

    # Similarity (Should be diagonal -> Hierarchical)
    refined_sim = compute_layer_similarity(
        model=model,
        dataloader=dataloader,
        feat_fn=lambda x: _student_feats_no_grad(model, x, use_raw=False),
        num_batches=args.num_batches,
        device=args.device,
    )
    if refined_sim is not None:
        plot_similarity(
            refined_sim,
            f"{tag} - Refined Student Similarity",
            output_dir / f"{tag}_refined_similarity.png",
        )
        print(f"[{tag}] Saved refined similarity heatmap.")

    # ERF (Should be sharper, Gaussian-like)
    refined_erf = compute_erf(
        model=model,
        dataloader=dataloader,
        feat_fn=lambda x: _student_feats_grad(model, x, use_raw=False),
        layer_idx=args.erf_layer,
        num_images=args.num_erf_images,
        device=args.device,
    )
    if refined_erf is not None:
        refined_erf = np.power(refined_erf, 0.5)  # Square root scaling
        plot_erf(
            refined_erf,
            f"{tag} - Refined Student ERF",
            output_dir / f"{tag}_refined_erf.png",
        )
        print(f"[{tag}] Saved refined ERF.")

    # Teacher Features (The Target)
    if not args.skip_teacher and model.segformer_teacher is not None:
        print(f"\n[{tag}] Analyzing TEACHER features (SegFormer)...")

        # Similarity (Will likely skip due to dimension mismatch, verifying hierarchy)
        teacher_sim = compute_layer_similarity(
            model=model,
            dataloader=dataloader,
            feat_fn=lambda x: _teacher_feats_no_grad(model, x),
            num_batches=args.num_batches,
            device=args.device,
        )
        if teacher_sim is not None:
            plot_similarity(
                teacher_sim,
                f"{tag} - Teacher Similarity",
                output_dir / f"{tag}_teacher_similarity.png",
            )

        # ERF (The Ground Truth Gaussian)
        print(f"[{tag}] Computing Teacher ERF...")
        # Note: SegFormer has 4 stages (0-3). Ensure idx is valid.
        teacher_layer_idx = min(args.erf_layer, 3)
        erf_teacher = compute_erf(
            model=model,
            dataloader=dataloader,
            feat_fn=lambda x: _teacher_feats_grad(model, x),
            layer_idx=teacher_layer_idx,
            num_images=args.num_erf_images,
            device=args.device,
        )
        if erf_teacher is not None:
            erf_teacher = np.power(erf_teacher, 0.5)
            plot_erf(
                erf_teacher,
                f"{tag} - Teacher ERF",
                output_dir / f"{tag}_teacher_erf.png",
            )
            print(f"[{tag}] Saved teacher ERF.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer similarity and ERF analysis for CONCH + SegFormer distillation."
    )
    parser.add_argument(
        "--config", required=True, type=str, help="Path to config.yaml."
    )
    parser.add_argument(
        "--checkpoint-distilled",
        type=str,
        required=True,
        help="Checkpoint for the distilled student.",
    )
    parser.add_argument(
        "--checkpoint-baseline",
        type=str,
        default=None,
        help="Optional checkpoint for baseline student.",
    )
    parser.add_argument(
        "--split", choices=["valid", "test"], default="valid", help="Dataset split."
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Num workers.")
    parser.add_argument(
        "--num-batches", type=int, default=10, help="Batches for similarity."
    )
    parser.add_argument(
        "--num-erf-images", type=int, default=20, help="Images for ERF."
    )
    parser.add_argument(
        "--erf-layer",
        type=int,
        default=3,
        help="Layer index (0-3) for ERF visualization.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU id.")
    parser.add_argument(
        "--output-dir", type=str, default="figures/analysis", help="Output directory."
    )
    parser.add_argument(
        "--skip-teacher", action="store_true", help="Skip teacher analysis."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)
    use_cuda = torch.cuda.is_available()
    args.device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    print(f"Using device: {args.device}")

    cfg = OmegaConf.load(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = build_loader(
        cfg, split=args.split, batch_size=args.batch_size, num_workers=args.num_workers
    )

    print("\n=== Analyzing Distilled Model ===")
    distilled = build_model(cfg, args.checkpoint_distilled, args.device)
    analyze_model("distilled", distilled, loader, args, output_dir)

    if args.checkpoint_baseline:
        print("\n=== Analyzing Baseline Model ===")
        baseline = build_model(cfg, args.checkpoint_baseline, args.device)
        analyze_model("baseline", baseline, loader, args, output_dir)

    print(f"\nDone. Results saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
