from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import torch.nn as nn
import torch
from model.model import ClsNetwork
from model.conch_adapter import ConchAdapter
from utils.pyutils import set_seed
from utils.trainutils import get_cls_dataset, get_mean_std
from utils.validate import validate
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on a split and report segmentation metrics.")
    parser.add_argument("--config", required=True, type=str,
                        help="Path to the config file.")
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to the model checkpoint (best_cam.pth).")
    parser.add_argument("--split", default="test",
                        choices=["test", "valid"], help="Dataset split to evaluate.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for evaluation loader.")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of DataLoader workers.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use.")
    return parser.parse_args()


def build_clip_adapter(cfg, device):
    clip_cfg = getattr(cfg, "clip", None)
    clip_cfg = OmegaConf.to_container(
        clip_cfg, resolve=True) if clip_cfg is not None else {}
    model_name = clip_cfg.get("model_name", "conch_ViT-B-16")
    checkpoint_path = clip_cfg.get("checkpoint_path")
    cache_dir = clip_cfg.get("cache_dir")
    hf_hub = clip_cfg.get("hf_hub")
    hf_token = clip_cfg.get("hf_token")
    force_image_size = clip_cfg.get("force_image_size")
    proj_contrast = clip_cfg.get("proj_contrast", False)
    freeze = clip_cfg.get("freeze", True)
    return ConchAdapter(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        device=device,
        force_image_size=force_image_size,
        cache_dir=cache_dir,
        hf_hub=hf_hub,
        hf_token=hf_token,
        proj_contrast=proj_contrast,
        freeze=freeze,
    )


def build_model(cfg, checkpoint_path, device):
    clip_adapter = build_clip_adapter(cfg, device)
    guidance_cfg = OmegaConf.to_container(
        getattr(cfg.model, "segformer_guidance", {}), resolve=True)
    input_mean, input_std = get_mean_std(cfg.dataset.name)
    model = ClsNetwork(
        backbone=cfg.model.backbone.config,
        stride=cfg.model.backbone.stride,
        cls_num_classes=cfg.dataset.cls_num_classes,
        num_prototypes_per_class=cfg.model.num_prototypes_per_class,
        prototype_feature_dim=cfg.model.prototype_feature_dim,
        clip_adapter=clip_adapter,
        n_ratio=cfg.model.n_ratio,
        pretrained=False,
        enable_text_fusion=getattr(cfg.model, "enable_text_fusion", True),
        text_prompts=getattr(cfg.model, "text_prompts", None),
        fusion_dim=getattr(cfg.model, "fusion_dim", None),
        learnable_text_prompt=getattr(
            cfg.model, "learnable_text_prompt", False),
        prompt_init_scale=getattr(cfg.model, "prompt_init_scale", 0.02),
        prototype_init_mode=getattr(
            cfg.model, "prototype_init_mode", "text_learnable"),
        prototype_text_noise_std=getattr(
            cfg.model, "prototype_text_noise_std", 0.02),
        use_ctx_prompt=getattr(cfg.model, "use_ctx_prompt", False),
        ctx_prompt_len=getattr(cfg.model, "ctx_prompt_len", 8),
        ctx_class_specific=getattr(cfg.model, "ctx_class_specific", False),
        enable_segformer_guidance=guidance_cfg.get("enable", False),
        segformer_backbone=guidance_cfg.get("backbone", "mit_b1"),
        segformer_checkpoint=guidance_cfg.get("checkpoint", None),
        guidance_layers=tuple(guidance_cfg.get("layers", (2,))),
        train_clip_visual=guidance_cfg.get("train_clip_visual", None),
        input_mean=input_mean,
        input_std=input_std,
        use_structure_adapter=guidance_cfg.get("use_structure_adapter", False),
        enable_cocoop=getattr(cfg.model, "enable_cocoop", False),
        cocoop_n_ctx=getattr(cfg.model, "cocoop_n_ctx", 4),
        cocoop_ctx_init=getattr(cfg.model, "cocoop_ctx_init", "a photo of a"),
        cocoop_class_names=getattr(cfg.model, "cocoop_class_names", None),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Allow soft-prompt params to be absent/present depending on checkpoint/config
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for evaluation; no GPU was detected.")

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    set_seed(42)
    cfg = OmegaConf.load(args.config)

    print("Building model and loading weights...")
    model = build_model(cfg, args.checkpoint, device)

    print(f"Preparing {args.split} dataset...")
    _, eval_dataset = get_cls_dataset(
        cfg, split=args.split, enable_rotation=False, p=0.0)
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
            print(
                f"Class {i}: IoU={iou.item() * 100:.4f}%  Dice={dice.item() * 100:.4f}%")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
