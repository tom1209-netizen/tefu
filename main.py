# Implementation adapted from PBIP by Qingchen Tang
# Source: https://github.com/QingchenTang/PBIP

import argparse
import datetime
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from model.model import ClsNetwork
from model.conch_adapter import ConchAdapter
from utils.contrast_loss import InfoNCELossBG, InfoNCELossFG
from utils.diversity_loss import PrototypeDiversityRegularizer
from utils.fgbg_feature import FeatureExtractor, MaskAdapter_DynamicThreshold
from utils.hierarchical_utils import (
    expand_parent_to_subclass_labels,
    merge_subclass_cams_to_parent,
    merge_to_parent_predictions,
    pair_features,
)
from utils.optimizer import PolyWarmupAdamW
from utils.pyutils import AverageMeter, set_seed
from utils.trainutils import get_cls_dataset, get_mean_std
from utils.validate import generate_cam, validate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint")
    return parser.parse_args()


def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now().replace(microsecond=0)
    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta * scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


def monitor_diversity_loss(meter, loss_value):
    if loss_value is None:
        return None
    if isinstance(loss_value, torch.Tensor):
        loss_value = loss_value.detach().item()
    meter.add({"diversity_loss": loss_value})
    return meter.get("diversity_loss")


def get_device(gpu_id: int):
    return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


def build_clip_model(cfg, device):
    clip_cfg = getattr(cfg, "clip", None)
    clip_cfg = OmegaConf.to_container(clip_cfg, resolve=True) if clip_cfg is not None else {}

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
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def build_model(cfg, device, clip_adapter=None):
    model = ClsNetwork(
        backbone=cfg.model.backbone.config,
        stride=cfg.model.backbone.stride,
        cls_num_classes=cfg.dataset.cls_num_classes,
        num_prototypes_per_class=cfg.model.num_prototypes_per_class,
        prototype_feature_dim=cfg.model.prototype_feature_dim,
        clip_adapter=clip_adapter,
        n_ratio=cfg.model.n_ratio,
        pretrained=cfg.train.pretrained,
        enable_text_fusion=getattr(cfg.model, "enable_text_fusion", True),
        text_prompts=getattr(cfg.model, "text_prompts", None),
        fusion_dim=getattr(cfg.model, "fusion_dim", None),
        learnable_text_prompt=getattr(cfg.model, "learnable_text_prompt", False),
        prompt_init_scale=getattr(cfg.model, "prompt_init_scale", 0.02),
        prototype_init_mode=getattr(cfg.model, "prototype_init_mode", "text_learnable"),
        prototype_text_noise_std=getattr(cfg.model, "prototype_text_noise_std", 0.02),
        use_ctx_prompt=getattr(cfg.model, "use_ctx_prompt", False),
        ctx_prompt_len=getattr(cfg.model, "ctx_prompt_len", 8),
        ctx_class_specific=getattr(cfg.model, "ctx_class_specific", False),
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
        print(f"WARNING: Checkpoint file not found at {args.resume}. Starting from scratch.")
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
        print("WARNING: Iteration number not found in checkpoint. Starting from iteration 0.")

    if "best_mIoU" in checkpoint:
        best_metric = checkpoint["best_mIoU"]
        print(f"Loaded previous best mIoU: {best_metric:.4f}")

    return start_iter, best_metric


def compute_proto_text_contrast(feature_map, pseudo_mask, projected_p4, text_features, raw_prototypes, k_per_class, temp_img=0.07, temp_text=0.07):
    """
    Aligns image regions with prototypes and text to keep semantics anchored.
    Returns summed loss (image-proto + proto-text) or None if unavailable.
    """
    if projected_p4 is None or text_features is None or raw_prototypes is None:
        return None

    B, C, H, W = feature_map.shape
    device = feature_map.device

    feats = feature_map.permute(0, 2, 3, 1).reshape(B, -1, C)  # [B, HW, C]
    masks = pseudo_mask.view(B, -1)  # [B, HW]

    # Prototype means in feature space
    proto_feat = projected_p4.view(-1, k_per_class, projected_p4.shape[-1]).mean(dim=1)  # [classes, C]
    proto_feat = F.normalize(proto_feat, dim=-1)

    # Prototype means in proto/text space
    proto_raw = raw_prototypes.view(-1, k_per_class, raw_prototypes.shape[-1]).mean(dim=1)  # [classes, D]
    proto_raw = F.normalize(proto_raw, dim=-1)

    text_features = F.normalize(text_features, dim=-1)  # [classes, D]

    img_feats = []
    targets = []
    for b in range(B):
        present = masks[b].unique()
        for cls_idx in present:
            cls_idx = int(cls_idx.item())
            if cls_idx >= proto_feat.shape[0]:
                continue
            idxs = (masks[b] == cls_idx).nonzero(as_tuple=True)[0]
            if idxs.numel() == 0:
                continue
            cls_feat = feats[b, idxs].mean(dim=0)
            img_feats.append(cls_feat)
            targets.append(cls_idx)

    loss_parts = []
    if img_feats:
        img_feats = torch.stack(img_feats)  # [N, C]
        targets_t = torch.tensor(targets, device=device, dtype=torch.long)
        img_feats = F.normalize(img_feats, dim=-1)

        logits_img_proto = (img_feats @ proto_feat.t()) / max(temp_img, 1e-4)
        loss_parts.append(F.cross_entropy(logits_img_proto, targets_t))

    logits_proto_text = (proto_raw @ text_features.t()) / max(temp_text, 1e-4)
    targets_proto = torch.arange(proto_raw.shape[0], device=device)
    loss_parts.append(F.cross_entropy(logits_proto_text, targets_proto))

    if not loss_parts:
        return None
    return sum(loss_parts) / len(loss_parts)


def build_loss_components(cfg, device, clip_adapter):
    loss_function = nn.BCEWithLogitsLoss().to(device)
    mask_adapter = MaskAdapter_DynamicThreshold(alpha=cfg.train.mask_adapter_alpha)
    input_mean, input_std = get_mean_std(cfg.dataset.name)
    feature_extractor = FeatureExtractor(
        mask_adapter=mask_adapter,
        clip_adapter=clip_adapter,
        clip_size=getattr(cfg.model, "clip_size", None),
        input_mean=input_mean,
        input_std=input_std,
    )
    fg_loss_fn = InfoNCELossFG(temperature=0.07).to(device)
    bg_loss_fn = InfoNCELossBG(temperature=0.07).to(device)

    div_cfg = getattr(cfg.train, "diversity", None)
    diversity_loss_fn = PrototypeDiversityRegularizer(
        num_prototypes_per_class=cfg.model.num_prototypes_per_class,
        omega_window=(div_cfg.omega_window if div_cfg else 7),
        omega_min_mass=(div_cfg.omega_min_mass if div_cfg else 0.05),
        temperature=(div_cfg.temperature if div_cfg and hasattr(div_cfg, "temperature") else 0.07),
        sharpness_weight=(div_cfg.sharpness_weight if div_cfg and hasattr(div_cfg, "sharpness_weight") else 0.1),
        coverage_weight=(div_cfg.coverage_weight if div_cfg and hasattr(div_cfg, "coverage_weight") else 0.1),
        repulsion_weight=(div_cfg.repulsion_weight if div_cfg and hasattr(div_cfg, "repulsion_weight") else 0.5),
        repulsion_margin=(div_cfg.repulsion_margin if div_cfg and hasattr(div_cfg, "repulsion_margin") else 0.2),
        jeffreys_weight=(div_cfg.jeffreys_weight if div_cfg and hasattr(div_cfg, "jeffreys_weight") else 0.0),
        pool_size=(div_cfg.pool_size if div_cfg and hasattr(div_cfg, "pool_size") else None),
        debug=(div_cfg.debug if div_cfg and "debug" in div_cfg else False),
        debug_every=(div_cfg.debug_every if div_cfg and "debug_every" in div_cfg else 200),
    ).to(device)
    lambda_fuse = getattr(cfg.train, "l_fuse", 1.0)
    lambda_proto_text = getattr(cfg.train, "l_proto_text", 0.0)
    proto_text_temp = getattr(cfg.train, "l_proto_temp", 0.07)

    return {
        "cls": loss_function,
        "feature_extractor": feature_extractor,
        "fg": fg_loss_fn,
        "bg": bg_loss_fn,
        "diversity": diversity_loss_fn,
        "lambda_fuse": lambda_fuse,
        "lambda_proto_text": lambda_proto_text,
        "proto_text_temp": proto_text_temp,
    }


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
    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(cfg, num_workers)

    iters_per_epoch = len(train_loader)
    cfg.train.max_iters = cfg.train.epoch * iters_per_epoch
    cfg.train.eval_iters = iters_per_epoch
    cfg.scheduler.warmup_iter = cfg.scheduler.warmup_iter * iters_per_epoch

    model = build_model(cfg, device, clip_adapter=clip_model)
    optimizer = build_optimizer(cfg, model)
    start_iter, best_fuse234_dice = resume(args, model, optimizer, device)

    losses = build_loss_components(cfg, device, clip_model)
    loss_function = losses["cls"]
    feature_extractor = losses["feature_extractor"]
    fg_loss_fn = losses["fg"]
    bg_loss_fn = losses["bg"]
    diversity_loss_fn = losses["diversity"]
    lambda_fuse = losses["lambda_fuse"]
    lambda_proto_text = losses["lambda_proto_text"]
    proto_text_temp = losses["proto_text_temp"]

    scaler = torch.cuda.amp.GradScaler()
    model.train()

    print("\nStarting training...")
    train_loader_iter = iter(train_loader)
    diversity_meter = AverageMeter("diversity_loss")
    diversity_running_avg = None

    for n_iter in range(start_iter, cfg.train.max_iters):
        try:
            _, inputs, cls_labels, _ = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            _, inputs, cls_labels, _ = next(train_loader_iter)

        inputs = inputs.to(device).float()
        cls_labels = cls_labels.to(device).float()
        iter_diversity_loss = None

        with torch.cuda.amp.autocast():
            (cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, k_list, feature_map_for_diversity, cam_weights, projected_p4, text_features_out) = model(inputs)

            cls1_merge = merge_to_parent_predictions(cls1, k_list, method=cfg.train.merge_train)
            cls2_merge = merge_to_parent_predictions(cls2, k_list, method=cfg.train.merge_train)
            cls3_merge = merge_to_parent_predictions(cls3, k_list, method=cfg.train.merge_train)
            cls4_merge = merge_to_parent_predictions(cls4, k_list, method=cfg.train.merge_train)

            loss1 = loss_function(cls1_merge, cls_labels)
            loss2 = loss_function(cls2_merge, cls_labels)
            loss3 = loss_function(cls3_merge, cls_labels)
            loss4 = loss_function(cls4_merge, cls_labels)
            cls_loss = cfg.train.l1 * loss1 + cfg.train.l2 * loss2 + cfg.train.l3 * loss3 + cfg.train.l4 * loss4
            if cam_weights is not None:
                stacked_cls = torch.stack([cls2_merge, cls3_merge, cls4_merge], dim=-1)
                cam_weights = cam_weights / (cam_weights.sum(dim=-1, keepdim=True) + 1e-8)
                fused_cls = (stacked_cls * cam_weights).sum(dim=-1)
                loss_fused = loss_function(fused_cls, cls_labels)
                cls_loss = cls_loss + lambda_fuse * loss_fused

            if n_iter >= cfg.train.warmup_iters:
                subclass_labels = expand_parent_to_subclass_labels(cls_labels, k_list)
                cls4_expand = expand_parent_to_subclass_labels(cls4_merge, k_list)
                cls4_bir = (cls4 > cls4_expand).float() * subclass_labels
                batch_info = feature_extractor.process_batch(inputs, cam4, cls4_bir)

                contrastive_loss = None
                if batch_info is not None:
                    fg_features, bg_features = batch_info["fg_features"], batch_info["bg_features"]
                    set_info = pair_features(fg_features, bg_features, l_fea, cls4_bir)
                    fg_features, bg_features, fg_pro, bg_pro = (
                        set_info["fg_features"],
                        set_info["bg_features"],
                        set_info["fg_text"],
                        set_info["bg_text"],
                    )
                    fg_loss = fg_loss_fn(fg_features, fg_pro, bg_pro)
                    bg_loss = bg_loss_fn(bg_features, fg_pro, bg_pro)
                    contrastive_loss = fg_loss + bg_loss

                with torch.no_grad():
                    cam4_merged = merge_subclass_cams_to_parent(cam4, k_list, method=cfg.train.merge_train)
                    cam_max, _ = torch.max(cam4_merged, dim=1, keepdim=True)
                    background_score = torch.full_like(cam_max, 0.2)
                    full_cam = torch.cat([background_score, cam4_merged], dim=1)
                    pseudo_mask = torch.argmax(full_cam, dim=1)

                pseudo_mask_resized = F.interpolate(
                    pseudo_mask.unsqueeze(1).float(), size=feature_map_for_diversity.shape[2:], mode="nearest"
                ).squeeze(1).long()
                diversity_loss = diversity_loss_fn(feature_map_for_diversity, projected_p4, pseudo_mask_resized, global_step=n_iter)
                iter_diversity_loss = diversity_loss.detach().item()
                diversity_running_avg = monitor_diversity_loss(diversity_meter, diversity_loss)

                proto_text_loss = None
                if lambda_proto_text > 0:
                    proto_text_loss = compute_proto_text_contrast(
                        feature_map_for_diversity,
                        pseudo_mask_resized,
                        projected_p4,
                        text_features_out,
                        l_fea,
                        cfg.model.num_prototypes_per_class,
                        temp_img=proto_text_temp,
                        temp_text=proto_text_temp,
                    )

                lambda_sim = cfg.train.l5
                lambda_j = cfg.train.lambda_j
                loss = cls_loss + lambda_j * diversity_loss
                if proto_text_loss is not None:
                    loss = loss + lambda_proto_text * proto_text_loss
                if contrastive_loss is not None:
                    loss = loss + lambda_sim * (contrastive_loss + 0.0005 * torch.mean(cam4))
            else:
                loss = cls_loss

        if n_iter == cfg.train.warmup_iters:
            print(f"\n--- Iteration {n_iter}: Warm-up complete. Activating contrastive and diversity losses. ---\n")

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if (n_iter + 1) % 100 == 0:
            delta, eta = cal_eta(time0, n_iter + 1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]["lr"]
            torch.cuda.synchronize()
            cls_pred4 = (torch.sigmoid(cls4_merge) > 0.5).float()
            all_cls_acc4 = (cls_pred4 == cls_labels).all(dim=1).float().mean() * 100
            avg_cls_acc4 = ((cls_pred4 == cls_labels).float().mean(dim=0)).mean() * 100
            div_last_str = f"{iter_diversity_loss:.4f}" if iter_diversity_loss is not None else "N/A"
            div_avg_str = f"{diversity_running_avg:.4f}" if diversity_running_avg is not None else "N/A"
            print(
                f"Iter: {n_iter + 1}/{cfg.train.max_iters}; "
                f"Elapsed: {delta}; ETA: {eta}; "
                f"LR: {cur_lr:.3e}; Loss: {loss.item():.4f}; "
                f"DivLoss(last/avg): {div_last_str}/{div_avg_str}; "
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

            saving_grace_period = cfg.train.eval_iters
            if (n_iter + 1) > (cfg.train.warmup_iters + saving_grace_period):
                best_fuse234_dice = save_best(model, optimizer, best_fuse234_dice, cfg, n_iter, current_miou)
            else:
                print(f"--- In warm-up or grace period (current iter: {n_iter + 1}). Skipping best model check. ---")

    torch.cuda.empty_cache()
    end_time = datetime.datetime.now()
    total_training_time = end_time - time0
    print(f"Total training time: {total_training_time}")

    final_evaluation(cfg, device, num_workers, model, loss_function, train_dataset)


def final_evaluation(cfg, device, num_workers, model, loss_function, train_dataset):
    print("\n" + "=" * 80)
    print("POST-TRAINING EVALUATION AND CAM GENERATION")
    print("=" * 80)

    print("\nPreparing test dataset...")
    _, test_dataset = get_cls_dataset(cfg, split="test", enable_rotation=False, p=0.0)
    print(f"Test dataset loaded: {len(test_dataset)} samples")

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    print("\n1. Testing on test dataset...")
    print("-" * 50)

    test_mIoU, test_mean_dice, test_fw_iu, test_iu_per_class, test_dice_per_class = validate(
        model=model, data_loader=test_loader, cfg=cfg, cls_loss_func=loss_function
    )

    print("Testing results:")
    print(f"Test mIoU: {test_mIoU:.4f}")
    print(f"Test Mean Dice: {test_mean_dice:.4f}")
    print(f"Test FwIU: {test_fw_iu:.4f}")

    print("\nPer-class IoU scores (FG classes + BG):")
    for i, score in enumerate(test_iu_per_class):
        label = f"Class {i}" if i < len(test_iu_per_class) - 1 else "Background"
        print(f"  {label}: {score*100:.4f}")

    print("\nPer-class Dice scores (FG classes + BG):")
    for i, score in enumerate(test_dice_per_class):
        label = f"Class {i}" if i < len(test_dice_per_class) - 1 else "Background"
        print(f"  {label}: {score*100:.4f}")

    print("\n2. Generating CAMs for complete training dataset...")
    print("-" * 50)

    train_cam_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    print(f"Generating CAMs for all {len(train_dataset)} training samples...")
    print(f"Output directory: {cfg.work_dir.pred_dir}")

    best_model_path = os.path.join(cfg.work_dir.ckpt_dir, "best_cam.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        best_iter = checkpoint.get("iter", "unknown")
        print(f"Best model loaded successfully! (Saved at iteration: {best_iter})")
    else:
        print("Warning: Best model checkpoint not found, using current model state")
        print(f"Expected path: {best_model_path}")

    generate_cam(model=model, data_loader=train_cam_loader, cfg=cfg)

    print("\nFiles generated:")
    print(f"  Training CAM visualizations: {cfg.work_dir.pred_dir}/*.png")
    print(f"  Model checkpoint: {cfg.work_dir.ckpt_dir}/best_cam.pth")
    print("=" * 80)


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.work_dir.dir = os.path.dirname(args.config)
    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)

    os.makedirs(cfg.work_dir.dir, exist_ok=True)
    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)

    print("\nArgs:", args)
    print("\nConfigs:", cfg)

    set_seed(0)
    train(cfg=cfg, args=args)


if __name__ == "__main__":
    main()
