from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import torch
from model.model import DistilledConch
from model.conch_adapter import ConchAdapter
from utils.pyutils import set_seed
from utils.trainutils import get_cls_dataset
from utils.validate import generate_cam
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CAM predictions for a dataset split.")
    parser.add_argument("--config", required=True, type=str,
                        help="Path to the config file.")
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to the model checkpoint (best_cam.pth).")
    parser.add_argument("--split", default="test",
                        choices=["test", "valid"], help="Dataset split to run on.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory to save CAM outputs.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for CAM generation.")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of DataLoader workers.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use.")
    return parser.parse_args()


def build_model(cfg, checkpoint_path, device):
    clip_adapter = None
    if cfg.model.backbone.config.startswith("conch"):
        clip_cfg = OmegaConf.to_container(
            getattr(cfg, "clip", None) or {}, resolve=True)
        clip_adapter = ConchAdapter(
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
    guidance_cfg = OmegaConf.to_container(
        getattr(cfg.model, "segformer_guidance", {}), resolve=True)
    model = DistilledConch(
        cls_num_classes=cfg.dataset.cls_num_classes,
        num_prototypes_per_class=cfg.model.num_prototypes_per_class,
        prototype_feature_dim=cfg.model.prototype_feature_dim,
        clip_adapter=clip_adapter,
        enable_segformer_guidance=guidance_cfg.get("enable", True),
        segformer_backbone=guidance_cfg.get("backbone", "mit_b1"),
        segformer_checkpoint=guidance_cfg.get("checkpoint", None),
        guidance_layers=tuple(guidance_cfg.get("layers", (2,))),
        text_prompts=getattr(cfg.model, "text_prompts", None),
        n_ratio=cfg.model.n_ratio,
        pretrained=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for CAM generation; no GPU was detected.")

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    set_seed(42)
    cfg = OmegaConf.load(args.config)

    output_dir = args.out_dir or os.path.join(
        os.path.dirname(args.checkpoint), f"{args.split}_cams")
    os.makedirs(output_dir, exist_ok=True)
    cfg.work_dir.pred_dir = output_dir
    print(f"Saves CAM masks to: {output_dir}")

    print("Building model and loading weights...")
    model = build_model(cfg, args.checkpoint, device)

    print(f"Preparing {args.split} dataset...")
    _, cam_dataset = get_cls_dataset(
        cfg, split=args.split, enable_rotation=False, p=0.0)
    cam_loader = DataLoader(
        cam_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(args.num_workers, os.cpu_count() or 1),
        pin_memory=True,
    )
    print(f"Dataset size: {len(cam_dataset)} samples")

    print("\nGenerating CAMs...")
    generate_cam(
        model=model,
        data_loader=cam_loader,
        cfg=cfg,
    )
    print(f"\nDone. CAMs saved to {output_dir}")


if __name__ == "__main__":
    main()
