# Implementation adapted from PBIP by Qingchen Tang
# Source: https://github.com/QingchenTang/PBIP

import argparse
import datetime
import os

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from model.model import DistilledConch
from model.conch_adapter import ConchAdapter
from utils.hierarchical_utils import merge_to_parent_predictions
from utils.optimizer import PolyWarmupAdamW
from utils.pyutils import set_seed
from utils.trainutils import get_cls_dataset
from utils.validate import generate_cam, validate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--resume", type=str, default=None,
                        help="path to checkpoint")
    return parser.parse_args()


def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now().replace(microsecond=0)
    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta * scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


def get_device(gpu_id: int):
    return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


def build_clip_model(cfg, device):
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

    clip_model = ConchAdapter(
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
    clip_model.to(device)
    return clip_model


def build_dataloaders(cfg, num_workers):
    train_dataset, val_dataset = get_cls_dataset(cfg, split="valid")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.samples_per_gpu,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.samples_per_gpu * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def build_model(cfg, device, clip_adapter=None):
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
        pretrained=cfg.train.pretrained,
    )
    return model.to(device)


def build_optimizer(cfg, model):
    return PolyWarmupAdamW(
        params=model.parameters(),
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iter,
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power,
    )


def resume(args, model, optimizer, device):
    start_iter = 0
    best_metric = 0.0

    if args.resume is None:
        print("\nStarting training from scratch.")
        return start_iter, best_metric

    if not os.path.exists(args.resume):
        print(
            f"WARNING: Checkpoint file not found at {args.resume}. Starting from scratch.")
        return start_iter, best_metric

    print(f"\nResuming training from checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Optimizer state loaded successfully.")
    else:
        print("WARNING: Optimizer state not found in checkpoint. Starting with a fresh optimizer.")

    if "iter" in checkpoint:
        start_iter = checkpoint["iter"] + 1
        print(f"Resuming from iteration: {start_iter}")
    else:
        print(
            "WARNING: Iteration number not found in checkpoint. Starting from iteration 0.")

    if "best_mIoU" in checkpoint:
        best_metric = checkpoint["best_mIoU"]
        print(f"Loaded previous best mIoU: {best_metric:.4f}")

    return start_iter, best_metric


def save_best(model, optimizer, best_metric, cfg, n_iter, current_metric):
    if current_metric <= best_metric:
        return best_metric
    best_metric = current_metric
    save_path = os.path.join(cfg.work_dir.ckpt_dir, "best_cam.pth")
    torch.save(
        {
            "cfg": cfg,
            "iter": n_iter,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_mIoU": best_metric,
        },
        save_path,
        _use_new_zipfile_serialization=True,
    )
    print(f"\nSaved best model with mIOU: {best_metric:.4f}")
    return best_metric


def train(cfg, args):
    print("\nInitializing training...")
    torch.backends.cudnn.benchmark = True
    set_seed(42)

    device = get_device(args.gpu)
    print(f"Using device: {device}")
    num_workers = min(10, os.cpu_count())

    clip_model = build_clip_model(cfg, device)
    time0 = datetime.datetime.now().replace(microsecond=0)

    print("\nPreparing datasets...")
    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        cfg, num_workers)

    iters_per_epoch = len(train_loader)
    cfg.train.max_iters = cfg.train.epoch * iters_per_epoch
    cfg.train.eval_iters = iters_per_epoch
    cfg.scheduler.warmup_iter = cfg.scheduler.warmup_iter * iters_per_epoch

    model = build_model(cfg, device, clip_adapter=clip_model)
    optimizer = build_optimizer(cfg, model)
    start_iter, best_fuse234_dice = resume(args, model, optimizer, device)

    loss_function = nn.BCEWithLogitsLoss().to(device)
    lambda_struct = getattr(cfg.train, "l_struct", 0.0)

    scaler = torch.cuda.amp.GradScaler()
    model.train()

    print("\nStarting training...")
    train_loader_iter = iter(train_loader)

    for n_iter in range(start_iter, cfg.train.max_iters):
        try:
            _, inputs, cls_labels, _ = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            _, inputs, cls_labels, _ = next(train_loader_iter)

        inputs = inputs.to(device).float()
        cls_labels = cls_labels.to(device).float()

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            cls_logits = outputs[0]
            k_list = outputs[9] if len(outputs) > 9 else [
                cfg.model.num_prototypes_per_class] * cfg.dataset.cls_num_classes
            cls_merge = merge_to_parent_predictions(
                cls_logits, k_list, method=cfg.train.merge_train)
            cls_loss = loss_function(cls_merge, cls_labels)
            loss = cls_loss
            distill_loss = outputs[-1] if outputs else None
            if distill_loss is not None and lambda_struct > 0:
                loss = loss + lambda_struct * distill_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if (n_iter + 1) % 100 == 0:
            delta, eta = cal_eta(time0, n_iter + 1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]["lr"]
            torch.cuda.synchronize()
            cls_pred4 = (torch.sigmoid(cls_merge) > 0.5).float()
            all_cls_acc4 = (cls_pred4 == cls_labels).all(
                dim=1).float().mean() * 100
            avg_cls_acc4 = (
                (cls_pred4 == cls_labels).float().mean(dim=0)).mean() * 100
            print(
                f"Iter: {n_iter + 1}/{cfg.train.max_iters}; "
                f"Elapsed: {delta}; ETA: {eta}; "
                f"LR: {cur_lr:.3e}; Loss: {loss.item():.4f}; "
                f"Acc4: {all_cls_acc4:.2f}/{avg_cls_acc4:.2f}"
            )

        if (n_iter + 1) % cfg.train.eval_iters == 0 or (n_iter + 1) == cfg.train.max_iters:
            val_mIoU, val_mean_dice, val_fw_iu, val_iu_per_class, val_dice_per_class = validate(
                model=model,
                data_loader=val_loader,
                cfg=cfg,
                cls_loss_func=loss_function,
            )

            print("Validation results:")
            print(f"Val mIoU: {val_mIoU:.4f}")
            print(f"Val Mean Dice: {val_mean_dice:.4f}")
            print(f"Val FwIU: {val_fw_iu:.4f}")

            current_miou = val_mIoU
            print(f"mIOU (for saving): {current_miou:.4f}")

            best_fuse234_dice = save_best(
                model, optimizer, best_fuse234_dice, cfg, n_iter, current_miou)

    torch.cuda.empty_cache()
    end_time = datetime.datetime.now()
    total_training_time = end_time - time0
    print(f"Total training time: {total_training_time}")

    final_evaluation(cfg, device, num_workers, model, loss_function)


def final_evaluation(cfg, device, num_workers, model, loss_function):
    print("\n" + "=" * 80)
    print("POST-TRAINING EVALUATION")
    print("=" * 80)

    print("\nPreparing test dataset...")
    _, test_dataset = get_cls_dataset(
        cfg, split="test", enable_rotation=False, p=0.0)
    print(f"Test dataset loaded: {len(test_dataset)} samples")

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.samples_per_gpu * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    print("\nTesting on test dataset...")
    test_mIoU, test_mean_dice, test_fw_iu, test_iu_per_class, test_dice_per_class = validate(
        model=model, data_loader=test_loader, cfg=cfg, cls_loss_func=loss_function
    )

    print("Testing results:")
    print(f"Test mIoU: {test_mIoU:.4f}")
    print(f"Test Mean Dice: {test_mean_dice:.4f}")
    print(f"Test FwIU: {test_fw_iu:.4f}")

    print("\nPer-class IoU scores (FG classes + BG):")
    for i, score in enumerate(test_iu_per_class):
        label = f"Class {i}" if i < len(
            test_iu_per_class) - 1 else "Background"
        print(f"  {label}: {score*100:.4f}")

    print("\nPer-class Dice scores (FG classes + BG):")
    for i, score in enumerate(test_dice_per_class):
        label = f"Class {i}" if i < len(
            test_dice_per_class) - 1 else "Background"
        print(f"  {label}: {score*100:.4f}")
    print("=" * 80)


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.work_dir.dir = os.path.dirname(args.config)
    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(
        cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(
        cfg.work_dir.dir, cfg.work_dir.pred_dir)

    os.makedirs(cfg.work_dir.dir, exist_ok=True)
    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)

    print("\nArgs:", args)
    print("\nConfigs:", cfg)

    set_seed(0)
    train(cfg=cfg, args=args)


if __name__ == "__main__":
    main()
