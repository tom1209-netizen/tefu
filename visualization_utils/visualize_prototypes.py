import argparse
import os
import sys
from pathlib import Path
import warnings

import albumentations as A
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model import ClsNetwork
from utils.hierarchical_utils import merge_subclass_cams_to_parent
from utils.validate import fuse_cams_with_weights, get_seg_label

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

DEFAULT_PALETTE = [
    [255, 0, 0],    # 0: TUM (Red)
    [0, 255, 0],    # 1: STR (Green)
    [0, 0, 255],    # 2: LYM (Blue)
    [153, 0, 255],  # 3: NEC (Purple)
    [255, 255, 255] # Background
]
DEFAULT_CLASS_NAMES = ["TUM", "STR", "LYM", "NEC"]


def get_validation_transform():
    mean = [0.66791496, 0.47791372, 0.70623304]
    std = [0.1736589, 0.22564577, 0.19820057]
    return A.Compose([
        A.Normalize(mean, std),
        ToTensorV2(transpose_mask=True),
    ])


def load_color_mask(mask_path, palette):
    try:
        mask_np = np.array(Image.open(mask_path))
    except FileNotFoundError:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    color_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    for class_idx, color in enumerate(palette):
        color_mask[mask_np == class_idx] = color
    return color_mask


def generate_heatmap(image_bgr, activation_map):
    heatmap = cv2.applyColorMap(np.uint8(255 * activation_map), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    image_float = np.float32(image_bgr) / 255
    overlay = cv2.addWeighted(image_float, 0.5, heatmap, 0.5, 0)
    return np.uint8(overlay * 255)


def get_activation_map(feature_map, projected_prototypes, proto_idx, img_size):
    """
    Calculates cosine-similarity activation map for one prototype.
    """
    b, c, h, w = feature_map.shape
    assert b == 1, "Batch size > 1 not supported in visualization pipeline."

    feature_map_flat = feature_map.permute(0, 2, 3, 1).reshape(-1, c)
    feature_map_flat_norm = F.normalize(feature_map_flat, p=2, dim=1)

    proto_vec = projected_prototypes[proto_idx]
    proto_norm = F.normalize(proto_vec, p=2, dim=0)
    cos_sim = torch.matmul(feature_map_flat_norm, proto_norm)

    act_map = F.relu(cos_sim.view(h, w))
    act_map_up = F.interpolate(
        act_map.unsqueeze(0).unsqueeze(0),
        size=img_size,
        mode="bilinear",
        align_corners=False
    ).squeeze()

    map_min, map_max = act_map_up.min(), act_map_up.max()
    if map_max > map_min:
        act_map_norm = (act_map_up - map_min) / (map_max - map_min + 1e-8)
    else:
        act_map_norm = torch.zeros_like(act_map_up)
    return act_map_norm.cpu().numpy()


def resolve_class_names(cfg, num_classes):
    if "class_names" in cfg.dataset:
        names = list(cfg.dataset.class_names)
        if len(names) == num_classes:
            return names
    if getattr(cfg.dataset, "name", "").lower() == "bcss":
        return DEFAULT_CLASS_NAMES[:num_classes]
    return [f"Class {i}" for i in range(num_classes)]


def get_prototype_slices(k_list):
    slices = []
    start = 0
    for count in k_list:
        end = start + count
        slices.append(list(range(start, end)))
        start = end
    return slices


def build_model(cfg, checkpoint_path, device):
    model = ClsNetwork(
        backbone=cfg.model.backbone.config,
        stride=cfg.model.backbone.stride,
        cls_num_classes=cfg.dataset.cls_num_classes,
        num_prototypes_per_class=cfg.model.num_prototypes_per_class,
        prototype_feature_dim=cfg.model.prototype_feature_dim,
        n_ratio=cfg.model.n_ratio,
        pretrained=False,
        enable_text_fusion=getattr(cfg.model, "enable_text_fusion", True),
        text_prompts=getattr(cfg.model, "text_prompts", None),
        fusion_dim=getattr(cfg.model, "fusion_dim", None),
        learnable_text_prompt=getattr(cfg.model, "learnable_text_prompt", False),
        prompt_init_scale=getattr(cfg.model, "prompt_init_scale", 0.02),
        prototype_init_mode=getattr(cfg.model, "prototype_init_mode", "text_learnable"),
        prototype_text_noise_std=getattr(cfg.model, "prototype_text_noise_std", 0.02),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def extract_features_and_prototypes(model, image_tensor):
    with torch.no_grad():
        feature_maps, _ = model.encoder(image_tensor)
        feature_map = feature_maps[3].detach()
        prototypes = model.prototypes.detach()
        projected = model.l_fc4(prototypes).detach()
    return feature_map, projected


def predict_mask(model, image_tensor, cls_label, cfg, palette):
    """
    Run the model to obtain fused CAM prediction and return a colorized mask.
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        (_, cam1, _, cam2, _, cam3, _, cam4, _, k_list, _, cam_weights) = outputs

        merge_method = getattr(cfg.train, "merge_test", "max")
        cam2 = merge_subclass_cams_to_parent(cam2, k_list, method=merge_method)
        cam3 = merge_subclass_cams_to_parent(cam3, k_list, method=merge_method)
        cam4 = merge_subclass_cams_to_parent(cam4, k_list, method=merge_method)

        cam2 = get_seg_label(cam2, image_tensor, cls_label).to(image_tensor.device)
        cam3 = get_seg_label(cam3, image_tensor, cls_label).to(image_tensor.device)
        cam4 = get_seg_label(cam4, image_tensor, cls_label).to(image_tensor.device)

        fuse234 = fuse_cams_with_weights(cam2, cam3, cam4, cam_weights)
        cam_max = torch.max(fuse234, dim=1, keepdim=True)[0]
        bg_cam = (1 - cam_max) ** 10
        full_probs_tensor = torch.cat([fuse234, bg_cam], dim=1)
        probs = torch.softmax(full_probs_tensor, dim=1)
        pred_mask = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    color_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(palette):
        color_mask[pred_mask == idx] = color
    return color_mask


def save_activation_grid(
    image_rgb,
    image_bgr,
    mask_color,
    pred_mask_color,
    class_names,
    model_label,
    feature_map,
    projected_prototypes,
    proto_slices,
    out_dir,
    base_name,
):
    num_classes = min(len(class_names), len(proto_slices))
    if num_classes == 0:
        return

    max_protos = max(len(indices) for indices in proto_slices[:num_classes])
    num_cols = max(max_protos, 2) + 1  # one column for labels
    num_rows = 1 + num_classes

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")

    axes[0, 0].text(
        0.5, 0.5,
        model_label,
        ha="center", va="center", fontsize=20, fontweight="bold"
    )
    axes[0, 1].imshow(image_rgb)
    axes[0, 2].imshow(mask_color)
    axes[0, 3].imshow(pred_mask_color)

    for class_idx in range(num_classes):
        row_idx = 1 + class_idx
        class_name = class_names[class_idx]
        axes[row_idx, 0].text(
            0.5, 0.5,
            class_name,
            ha="center", va="center", fontsize=18, fontweight="bold"
        )

        indices = proto_slices[class_idx]
        for rank, proto_idx in enumerate(indices):
            col_idx = 1 + rank
            act_map = get_activation_map(
                feature_map, projected_prototypes, proto_idx, (image_rgb.shape[0], image_rgb.shape[1])
            )
            heatmap = generate_heatmap(image_bgr, act_map)
            axes[row_idx, col_idx].imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
            
    plt.tight_layout(pad=1.0, h_pad=1.0, w_pad=1.0)
    out_path = os.path.join(out_dir, f"{base_name}_{model_label}_prototypes.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize where each prototype focuses for a set of images.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training config.yaml file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained best_cam.pth checkpoint.")
    parser.add_argument("--images", type=str, nargs="+", required=True, help="List of image filenames to visualize.")
    parser.add_argument("--split", type=str, default="test", choices=["test", "valid"], help="Dataset split to load.")
    parser.add_argument("--out-dir", type=str, default="./prototype_visualizations", help="Where to save outputs.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for prototype visualization; no GPU was detected.")

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    cfg = OmegaConf.load(args.config)
    model = build_model(cfg, args.checkpoint, device)
    k_list = list(getattr(model, "k_list", [cfg.model.num_prototypes_per_class] * cfg.dataset.cls_num_classes))
    proto_slices = get_prototype_slices(k_list)
    class_names = resolve_class_names(cfg, cfg.dataset.cls_num_classes)
    transform = get_validation_transform()

    os.makedirs(args.out_dir, exist_ok=True)
    model_label = Path(args.checkpoint).stem

    print(f"Processing {len(args.images)} images...")
    for img_name in tqdm(args.images, desc="Images"):
        img_path = os.path.join(cfg.dataset.val_root, args.split, "img", img_name)
        mask_path = os.path.join(cfg.dataset.val_root, args.split, "mask", img_name)

        if not os.path.exists(img_path):
            print(f"  Skipping {img_name}: not found at {img_path}")
            continue

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"  Skipping {img_name}: failed to read image.")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask_np = None
        if os.path.exists(mask_path):
            try:
                mask_np = np.array(Image.open(mask_path))
            except Exception:
                mask_np = None
        mask_color = load_color_mask(mask_path, DEFAULT_PALETTE)
        image_tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)

        cls_label = torch.ones(1, cfg.dataset.cls_num_classes, device=device)
        if mask_np is not None:
            cls_label_np = np.zeros(cfg.dataset.cls_num_classes, dtype=np.float32)
            present = np.unique(mask_np)
            present = present[present < cfg.dataset.cls_num_classes]
            cls_label_np[present] = 1
            cls_label = torch.from_numpy(cls_label_np).unsqueeze(0).to(device)

        feature_map, projected = extract_features_and_prototypes(model, image_tensor)
        pred_mask_color = predict_mask(model, image_tensor, cls_label, cfg, DEFAULT_PALETTE)
        base_name = Path(img_name).stem
        save_activation_grid(
            image_rgb=image_rgb,
            image_bgr=image_bgr,
            mask_color=mask_color,
            pred_mask_color=pred_mask_color,
            class_names=class_names,
            model_label=model_label,
            feature_map=feature_map,
            projected_prototypes=projected,
            proto_slices=proto_slices,
            out_dir=args.out_dir,
            base_name=base_name,
        )

    print(f"\nDone. Visualizations saved to {args.out_dir}")


if __name__ == "__main__":
    main()
