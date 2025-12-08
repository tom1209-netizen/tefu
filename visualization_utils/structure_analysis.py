"""
Layer similarity and effective receptive field (ERF) analysis utilities.

This script is intended to back the narrative that:
  1) ViT stages over-smooth (high inter-layer cosine similarity).
  2) CNN/SegFormer-style hierarchies keep spatially local receptive fields.
  3) Structural distillation sharpens the ViT receptive field.
"""

import argparse
import os
from pathlib import Path
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

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
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state.get("model", state), strict=False)
    model.to(device)
    model.eval()
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


def _student_feats_grad(model: DistilledConch, x: torch.Tensor) -> List[torch.Tensor]:
    # Preserve gradients to trace ERF back to the input.
    vis = model.clip_adapter.visual_intermediates(x, use_grad=True)
    while len(vis) < 4:
        vis.append(F.avg_pool2d(vis[-1], kernel_size=2, stride=2))
    return [adapter(f) for adapter, f in zip(model.structure_adapters, vis)]


def _student_feats_no_grad(
    model: DistilledConch, x: torch.Tensor
) -> List[torch.Tensor]:
    with torch.no_grad():
        return model._get_student_feats(x)


def _teacher_feats_grad(model: DistilledConch, x: torch.Tensor) -> List[torch.Tensor]:
    if model.segformer_teacher is None:
        raise RuntimeError("SegFormer teacher is not enabled for this model.")
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
            raise ValueError(
                f"Requested layer_idx {layer_idx} but only {len(feats)} layers are available."
            )
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
    erf = (erf - erf.min()) / (erf.max() - erf.min() + 1e-8)
    return erf


def plot_similarity(matrix: np.ndarray, title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="magma")
    num_layers = matrix.shape[0]
    labels = [f"Stage {i+1}" for i in range(num_layers)]
    ax.set_xticks(range(num_layers))
    ax.set_yticks(range(num_layers))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    for i in range(num_layers):
        for j in range(num_layers):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
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


def analyze_model(
    tag: str,
    model: DistilledConch,
    dataloader: DataLoader,
    args,
    output_dir: Path,
) -> None:
    print(f"\n[{tag}] Computing layer-wise cosine similarity for student (ViT)...")
    student_sim = compute_layer_similarity(
        model=model,
        dataloader=dataloader,
        feat_fn=lambda x: _student_feats_no_grad(model, x),
        num_batches=args.num_batches,
        device=args.device,
    )
    if student_sim is not None:
        plot_similarity(
            student_sim,
            f"{tag} - Student similarity",
            output_dir / f"{tag}_student_similarity.png",
        )
        print(f"[{tag}] Saved student similarity heatmap.")

    if not args.skip_teacher and model.segformer_teacher is not None:
        print(
            f"[{tag}] Computing layer-wise cosine similarity for teacher (SegFormer)..."
        )
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
                f"{tag} - Teacher similarity",
                output_dir / f"{tag}_teacher_similarity.png",
            )
            print(f"[{tag}] Saved teacher similarity heatmap.")

    print(f"[{tag}] Computing ERF for student layer {args.erf_layer}...")
    erf_student = compute_erf(
        model=model,
        dataloader=dataloader,
        feat_fn=lambda x: _student_feats_grad(model, x),
        layer_idx=args.erf_layer,
        num_images=args.num_erf_images,
        device=args.device,
    )
    if erf_student is not None:
        plot_erf(
            erf_student,
            f"{tag} - Student ERF (layer {args.erf_layer})",
            output_dir / f"{tag}_student_erf.png",
        )
        print(f"[{tag}] Saved student ERF heatmap.")

    if not args.skip_teacher and model.segformer_teacher is not None:
        print(f"[{tag}] Computing ERF for teacher layer {args.erf_layer}...")
        erf_teacher = compute_erf(
            model=model,
            dataloader=dataloader,
            feat_fn=lambda x: _teacher_feats_grad(model, x),
            layer_idx=min(args.erf_layer, 3),
            num_images=args.num_erf_images,
            device=args.device,
        )
        if erf_teacher is not None:
            plot_erf(
                erf_teacher,
                f"{tag} - Teacher ERF (layer {args.erf_layer})",
                output_dir / f"{tag}_teacher_erf.png",
            )
            print(f"[{tag}] Saved teacher ERF heatmap.")


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
        help="Optional checkpoint for a baseline / pre-distillation student.",
    )
    parser.add_argument(
        "--split",
        choices=["valid", "test"],
        default="valid",
        help="Dataset split to sample from.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size for the analysis loader."
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of dataloader workers."
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=6,
        help="How many mini-batches to average for the similarity matrix.",
    )
    parser.add_argument(
        "--num-erf-images",
        type=int,
        default=12,
        help="How many images to accumulate for the ERF map.",
    )
    parser.add_argument(
        "--erf-layer",
        type=int,
        default=3,
        help="Index (0-3) of the feature level to backprop for ERF visualizations.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU id to use; falls back to CPU if unavailable.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures/analysis",
        help="Where to save the plots.",
    )
    parser.add_argument(
        "--skip-teacher",
        action="store_true",
        help="Skip teacher comparisons (useful if the checkpoint was trained without SegFormer).",
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

    print("\nLoading distilled model...")
    distilled = build_model(cfg, args.checkpoint_distilled, args.device)
    analyze_model("distilled", distilled, loader, args, output_dir)

    if args.checkpoint_baseline:
        print("\nLoading baseline model...")
        baseline = build_model(cfg, args.checkpoint_baseline, args.device)
        analyze_model("baseline", baseline, loader, args, output_dir)

    print(f"\nDone. Results saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
