import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# Ensure root is in path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.conch_adapter import ConchAdapter
from model.model import DistilledConch
from utils.trainutils import get_cls_dataset


def build_model(cfg, checkpoint_path, device):
    clip_cfg = OmegaConf.to_container(getattr(cfg, "clip", None) or {}, resolve=True)
    clip_adapter = ConchAdapter(
        model_name=clip_cfg.get("model_name", "conch_ViT-B-16"),
        checkpoint_path=clip_cfg.get("checkpoint_path"),
        device=device,
        force_image_size=clip_cfg.get("force_image_size"),
    )

    guidance_cfg = OmegaConf.to_container(
        getattr(cfg.model, "segformer_guidance", {}) or {}, resolve=True
    )
    model = DistilledConch(
        cls_num_classes=cfg.dataset.cls_num_classes,
        num_prototypes_per_class=cfg.model.num_prototypes_per_class,
        prototype_feature_dim=cfg.model.prototype_feature_dim,
        clip_adapter=clip_adapter,
        enable_segformer_guidance=True,  # Force True to load teacher
        segformer_backbone=guidance_cfg.get("backbone", "mit_b1"),
        segformer_checkpoint=guidance_cfg.get("checkpoint"),
        guidance_layers=(0, 1, 2, 3),  # We want all layers for visualization
    )

    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state.get("model", state), strict=False)

    model.to(device)
    model.eval()

    # Force teacher to train mode for connectivity (frozen weights)
    if model.segformer_teacher is not None:
        model.segformer_teacher.train()
        for p in model.segformer_teacher.parameters():
            p.requires_grad_(False)

    return model


def process_feat(feat):
    """
    Turn a [1, C, H, W] feature map into a [H, W] heatmap.
    Method: Average across channels -> Normalize.
    """
    # 1. Average across channels to get spatial intensity
    heatmap = feat.mean(dim=1).detach()  # [1, H, W]

    # 2. Normalize to [0, 1] per image
    min_v = heatmap.min()
    max_v = heatmap.max()
    heatmap = (heatmap - min_v) / (max_v - min_v + 1e-8)

    return heatmap.cpu().squeeze().numpy()


def visualize_single_image(model, image_tensor, image_name, output_dir, device):
    """
    Extracts and plots (Raw ViT, Refined Student, Teacher) x (Stage 1-4)
    """
    x = image_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]

    # 1. Get Teacher Features (Target)
    # We need to manually handle normalization as in model.py
    student_mean = model.student_mean
    student_std = model.student_std
    teacher_mean = model.teacher_mean
    teacher_std = model.teacher_std

    x_raw = x * student_std + student_mean
    x_teacher = (x_raw - teacher_mean) / teacher_std
    with torch.no_grad():
        teacher_feats, _ = model.segformer_teacher(x_teacher)

    # 2. Get Raw ViT Features (Before)
    with torch.no_grad():
        raw_vis = model.clip_adapter.visual_intermediates(x, use_grad=False)
        while len(raw_vis) < 4:
            raw_vis.append(F.avg_pool2d(raw_vis[-1], kernel_size=2, stride=2))

    # 3. Get Refined Student Features (After)
    with torch.no_grad():
        refined_feats = [
            adapter(f) for adapter, f in zip(model.structure_adapters, raw_vis)
        ]

    # --- PLOTTING ---
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    rows = ["Raw ViT (Before)", "Distilled (After)", "SegFormer (Target)"]
    stages = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]

    # Data structure: [List of 4 tensors]
    all_data = [raw_vis, refined_feats, teacher_feats]

    for row_idx, feats in enumerate(all_data):
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]

            if col_idx < len(feats):
                # Process heatmap
                heatmap = process_feat(feats[col_idx])

                # Resize to common large size for visualization (e.g. 224x224)
                heatmap = cv2.resize(
                    heatmap, (224, 224), interpolation=cv2.INTER_NEAREST
                )

                im = ax.imshow(heatmap, cmap="jet")
                ax.axis("off")

                if row_idx == 0:
                    ax.set_title(stages[col_idx], fontsize=14, fontweight="bold")
                if col_idx == 0:
                    ax.text(
                        -0.2,
                        0.5,
                        rows[row_idx],
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        fontsize=14,
                        fontweight="bold",
                    )
            else:
                ax.axis("off")

    plt.tight_layout()
    save_path = output_dir / f"feat_map_{image_name}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="figures/feature_maps")
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Model
    model = build_model(cfg, args.checkpoint, device)

    # Load Data (Use Test/Val set)
    _, dataset = get_cls_dataset(cfg, split="valid", enable_rotation=False, p=0.0)

    indices = np.random.choice(len(dataset), args.num_images, replace=False)

    print(f"Visualizing {args.num_images} random images...")

    for idx in indices:
        name, img, _, _ = dataset[
            idx
        ]  # Assuming dataset returns (name, img, label, mask)
        # Handle cases where dataset returns different tuples
        # Adjust based on your specific Dataset __getitem__ return

        visualize_single_image(model, img, name, output_dir, device)


if __name__ == "__main__":
    main()
