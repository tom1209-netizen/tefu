import os
import argparse
from pathlib import Path
import warnings

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

from model.model import ClsNetwork as LearnableClsNetwork
from model.pbip_model import ClsNetwork as OriginalPbipNetwork

# Reduce matplotlib warning noise for headless usage.
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


DEFAULT_PALETTE = [
    [255, 0, 0],    # 0: TUM (Red)
    [0, 255, 0],    # 1: STR (Green)
    [0, 0, 255],    # 2: LYM (Blue)
    [153, 0, 255],  # 3: NEC (Purple)
    [255, 255, 255]
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
        mask_pil = Image.open(mask_path)
        mask_np = np.array(mask_pil)
        color_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
        for class_idx, color in enumerate(palette):
            color_mask[mask_np == class_idx] = color
        return color_mask
    except FileNotFoundError:
        return np.zeros((224, 224, 3), dtype=np.uint8)


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


def instantiate_model(cfg, ckpt_path, variant, device):
    num_classes = cfg.dataset.cls_num_classes
    stride = cfg.model.backbone.stride
    backbone = cfg.model.backbone.config
    pretrained = bool(cfg.train.get("pretrained", True))
    n_ratio = cfg.model.n_ratio

    if variant == "learnable":
        model = LearnableClsNetwork(
            backbone=backbone,
            cls_num_classes=num_classes,
            num_prototypes_per_class=cfg.model.num_prototypes_per_class,
            prototype_feature_dim=cfg.model.prototype_feature_dim,
            stride=stride,
            pretrained=pretrained,
            n_ratio=n_ratio
        )
    elif variant == "pbip":
        model = OriginalPbipNetwork(
            backbone=backbone,
            cls_num_classes=num_classes,
            stride=stride,
            pretrained=pretrained,
            n_ratio=n_ratio,
            l_fea_path=cfg.model.label_feature_path
        )
    else:
        raise ValueError(f"Unsupported model variant '{variant}'.")

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def get_prototype_slices(k_list):
    """
    Returns a list of prototype index ranges per class.
    """
    slices = []
    start = 0
    for count in k_list:
        end = start + count
        slices.append(list(range(start, end)))
        start = end
    return slices


class PrototypeModel:
    def __init__(self, label, config_path, ckpt_path, variant, device):
        self.label = label
        self.cfg = OmegaConf.load(config_path)
        self.variant = variant
        self.device = device
        self.model = instantiate_model(self.cfg, ckpt_path, variant, device)

        if not hasattr(self.model, "k_list"):
            raise AttributeError("Model does not expose k_list for prototype counts.")
        self.k_list = list(self.model.k_list)
        self.proto_slices = get_prototype_slices(self.k_list)

    def extract(self, image_tensor):
        with torch.no_grad():
            feature_maps, _ = self.model.encoder(image_tensor)
            feature_map = feature_maps[3].detach()

            if self.variant == "learnable":
                prototypes = self.model.prototypes.detach()
            else:
                outputs = self.model(image_tensor)
                prototypes = outputs[8].detach().to(self.device)

            projected = self.model.l_fc4(prototypes).detach()
        return feature_map, projected


def save_model_activation_grid(
    image_rgb,
    image_bgr,
    mask_color,
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
    num_cols = max(max_protos + 1, 2)  # column 0 for text, column 1+ for activations
    num_rows = 1 + num_classes

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3.2, num_rows * 3.2))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")

    axes[0, 0].text(
        0.5, 0.5,
        "Class",
        ha="center", va="center", fontsize=30, fontweight="bold"
    )

    axes[0, 1].imshow(image_rgb)
    axes[0, 1].set_title("Input Image", fontsize=20)

    if num_cols > 1:
        axes[0, 2].imshow(mask_color)
        axes[0, 2].set_title("Mask", fontsize=20)

    for class_idx in range(num_classes):
        row_idx = 1 + class_idx
        class_name = class_names[class_idx]
        axes[row_idx, 0].text(
            0.5, 0.5,
            f"{class_name}",
            ha="center", va="center", fontsize=30, fontweight="bold"
        )

        indices = proto_slices[class_idx]
        for rank in range(max_protos):
            col_idx = 1 + rank
            if rank < len(indices):
                proto_idx = indices[rank]
                act_map = get_activation_map(
                    feature_map, projected_prototypes, proto_idx, (image_rgb.shape[0], image_rgb.shape[1])
                )
                heatmap = generate_heatmap(image_bgr, act_map)
                axes[row_idx, col_idx].imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
                axes[row_idx, col_idx].set_title(f"Proto {rank + 1}", fontsize=20)
            else:
                axes[row_idx, col_idx].text(0.5, 0.5, "N/A", ha="center", va="center")

    fig.suptitle(f"{model_label} Prototype Activations", fontsize=24)
    plt.tight_layout(pad=2, h_pad=2.0, w_pad=2.0)
    out_path = os.path.join(out_dir, f"{base_name}_{model_label}_activations.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def format_label(path_like, fallback):
    if fallback:
        return fallback
    return Path(path_like).stem


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    model_a_label = format_label(args.ckpt_a, args.label_a)
    model_b_label = format_label(args.ckpt_b, args.label_b)

    print(f"Loading configuration A from {args.config_a}")
    model_a = PrototypeModel(model_a_label, args.config_a, args.ckpt_a, args.model_variant, device)
    print(f"✓ Model A ready: {model_a_label}")

    print(f"Loading configuration B from {args.config_b}")
    model_b = PrototypeModel(model_b_label, args.config_b, args.ckpt_b, args.model_variant, device)
    print(f"✓ Model B ready: {model_b_label}")

    cfg_a = model_a.cfg
    cfg_b = model_b.cfg
    if cfg_a.dataset.val_root != cfg_b.dataset.val_root:
        print("⚠ Warning: Validation roots differ between configs. Using config A for data paths.")

    transform = get_validation_transform()
    class_names = resolve_class_names(cfg_a, cfg_a.dataset.cls_num_classes)

    palette = DEFAULT_PALETTE
    max_classes = min(len(model_a.k_list), len(model_b.k_list), len(class_names))

    print(f"\nProcessing {len(args.images)} images...")
    for img_name in tqdm(args.images, desc="Images"):
        img_path = os.path.join(cfg_a.dataset.val_root, "test", "img", img_name)
        mask_path = os.path.join(cfg_a.dataset.val_root, "test", "mask", img_name)

        if not os.path.exists(img_path):
            print(f"  Skipping {img_name}: not found at {img_path}")
            continue

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"  Skipping {img_name}: failed to read image.")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask_color = load_color_mask(mask_path, palette)
        image_tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)

        feature_map_a, projected_a = model_a.extract(image_tensor)
        feature_map_b, projected_b = model_b.extract(image_tensor)

        base_name = Path(img_name).stem
        save_model_activation_grid(
            image_rgb=image_rgb,
            image_bgr=image_bgr,
            mask_color=mask_color,
            class_names=class_names[:max_classes],
            model_label=model_a.label,
            feature_map=feature_map_a,
            projected_prototypes=projected_a,
            proto_slices=model_a.proto_slices,
            out_dir=args.out_dir,
            base_name=base_name,
        )
        save_model_activation_grid(
            image_rgb=image_rgb,
            image_bgr=image_bgr,
            mask_color=mask_color,
            class_names=class_names[:max_classes],
            model_label=model_b.label,
            feature_map=feature_map_b,
            projected_prototypes=projected_b,
            proto_slices=model_b.proto_slices,
            out_dir=args.out_dir,
            base_name=base_name,
        )

    print(f"\nDone. Results saved to {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare prototype activation maps for two configurations of the same model."
    )
    parser.add_argument("--config_a", type=str, required=True, help="Config path for model A.")
    parser.add_argument("--ckpt_a", type=str, required=True, help="Checkpoint path for model A.")
    parser.add_argument("--config_b", type=str, required=True, help="Config path for model B.")
    parser.add_argument("--ckpt_b", type=str, required=True, help="Checkpoint path for model B.")
    parser.add_argument("--images", type=str, nargs="+", required=True, help="List of image filenames.")
    parser.add_argument("--out_dir", type=str, default="./prototype_comparisons", help="Output directory.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use if available.")
    parser.add_argument(
        "--model_variant",
        type=str,
        default="learnable",
        choices=["pbip", "learnable"],
        help="Model family to instantiate for both configs."
    )
    parser.add_argument("--label_a", type=str, default=None, help="Optional label override for model A.")
    parser.add_argument("--label_b", type=str, default=None, help="Optional label override for model B.")

    main(parser.parse_args())
