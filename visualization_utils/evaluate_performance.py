import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model import ClsNetwork
from utils.pyutils import set_seed
from utils.trainutils import get_cls_dataset
from utils.validate import validate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a split and report segmentation metrics.")
    parser.add_argument("--config", required=True, type=str, help="Path to the config file.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to the model checkpoint (best_cam.pth).")
    parser.add_argument("--split", default="test", choices=["test", "valid"], help="Dataset split to evaluate.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation loader.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of DataLoader workers.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use.")
    return parser.parse_args()


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


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation; no GPU was detected.")

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    set_seed(42)
    cfg = OmegaConf.load(args.config)

    print("Building model and loading weights...")
    model = build_model(cfg, args.checkpoint, device)

    print(f"Preparing {args.split} dataset...")
    _, eval_dataset = get_cls_dataset(cfg, split=args.split, enable_rotation=False, p=0.0)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(args.num_workers, os.cpu_count() or 1),
        pin_memory=True,
    )
    print(f"Dataset size: {len(eval_dataset)} samples")

    loss_function = nn.BCEWithLogitsLoss().to(device)

    print("\nRunning evaluation...")
    mIoU, mean_dice, fw_iu, iou_per_class, dice_per_class = validate(
        model=model,
        data_loader=eval_loader,
        cfg=cfg,
        cls_loss_func=loss_function,
    )

    class_names = getattr(eval_dataset, "CLASSES", [])
    if class_names:
        class_names = class_names[:-1]  # drop background if present

    print("\n" + "=" * 24 + " TEST RESULTS " + "=" * 24)
    print(f"Mean IoU (mIoU): {mIoU:.4f}%")
    print(f"Mean Dice:      {mean_dice:.4f}%")
    print(f"FreqW IoU:      {fw_iu:.4f}%")
    print("-" * 68)
    if class_names:
        print("Per-Class Scores:")
        print(f"{'Class':<12} | {'IoU':<10} | {'Dice':<10}")
        print("-" * 38)
        for i, name in enumerate(class_names):
            iou = iou_per_class[i].item() * 100
            dice = dice_per_class[i].item() * 100
            print(f"{name:<12} | {iou:<10.4f} | {dice:<10.4f}")
    else:
        print("Per-class metrics:")
        for i, (iou, dice) in enumerate(zip(iou_per_class, dice_per_class)):
            if i == len(iou_per_class) - 1:
                continue  # background
            print(f"Class {i}: IoU={iou.item() * 100:.4f}%  Dice={dice.item() * 100:.4f}%")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
