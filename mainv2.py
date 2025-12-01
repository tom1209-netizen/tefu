# Implementation adapted from PBIP by Qingchen Tang
# Source: https://github.com/QingchenTang/PBIP

import argparse
import datetime
import os
import numpy as np
import cv2 as cv
from omegaconf import OmegaConf
from tqdm import tqdm
import ttach as tta
from skimage import morphology

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# from utils_v2.diversity_loss import PrototypeDiversityRegularizer
from utils_v2.trainutils import get_cls_dataset
from utils_v2.optimizer import PolyWarmupAdamW
from utils_v2.pyutils import set_seed,AverageMeter
from utils_v2.evaluate import ConfusionMatrixAllClass
from utils_v2.hierarchical_utils import FeatureExtractor, MaskAdapter_DynamicThreshold, pair_features, merge_to_parent_predictions, merge_subclass_cams_to_parent, expand_parent_to_subclass_labels
from utils_v2.contrast_loss import InfoNCELossFG, InfoNCELossBG
from utils_v2.validate import generate_cam, validate
from model.ver2.model import ClsNetwork
# from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor

start_time = datetime.datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--gpu", type=int, default=0, help="gpu id")
parser.add_argument("--resume", type=str, default=None, help="path to the checkpoint to resume from")
args = parser.parse_args()

def get_seg_label(cams, inputs, label):
    """Generate segmentation labels from CAMs"""
    with torch.no_grad():
        b, c, h, w = inputs.shape
        label = label.view(b, -1, 1, 1).cpu().data.numpy()
        cams = cams.cpu().data.numpy()
        cams = np.maximum(cams, 0)
        
        # Normalize CAMs to [0,1]
        channel_max = np.max(cams, axis=(2, 3), keepdims=True)
        channel_min = np.min(cams, axis=(2, 3), keepdims=True)
        cams = (cams - channel_min) / (channel_max - channel_min + 1e-6)
        cams = cams * label
        cams = torch.from_numpy(cams).float()
        
        # Resize to input dimensions
        cams = F.interpolate(cams, size=(h, w), mode="bilinear", align_corners=True)
        cam_max = torch.max(cams, dim=1, keepdim=True)[0]
        bg_cam = (1 - cam_max) ** 10
        cam_all = torch.cat([cams, bg_cam], dim=1)

    return cams

def cal_eta(time0, cur_iter, total_iter):
    """Calculate elapsed time and estimated time to completion"""
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta * scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)

# def monitor_diversity_loss(meter, loss_value):
#     """
#     Update the running statistics for the diversity loss and return the current average.
#     meter: AverageMeter instance tracking a key named 'diversity_loss'
#     loss_value: torch.Tensor or float representing the diversity loss for the current batch
#     """
#     if loss_value is None:
#         return None
#     if isinstance(loss_value, torch.Tensor):
#         loss_value = loss_value.detach().item()
#     meter.add({'diversity_loss': loss_value})
#     return meter.get('diversity_loss')

def build_model(device):
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
    class_prompts = {
        0: ["invasive carcinoma", "malignant tumor cells", "invasive cancer"],
        1: ["benign tissue", "fibrous stroma", "non-malignant tissue"],
        2: ["immune infiltrate", "lymphocytes", "inflammatory cells"],
        3: ["necrosis", "dead tissue", "necrotic area"]
    }   
    
    CKPT_PATH = "/project/hnguyen2/mvu9/pretrained_checkpoints/conch_checkpoints/pytorch_model.bin"  
    
    model_conch, preprocess = create_model_from_pretrained(
        model_cfg="conch_ViT-B-16",
        checkpoint_path=CKPT_PATH,
        device="cuda",
        force_image_size=224,
        cache_dir="",
        hf_auth_token=None,
    )
 
    model = ClsNetwork(
        model_conch = model_conch, 
        cls_num_classes=4,
        num_prototypes_per_class=10,
        prototype_feature_dim=512,
        conch_img_feat_dim = 768, 
        stride=[4, 2, 2, 1],
        pretrained=True,
        n_ratio=0.5, 
        class_prompts = class_prompts, 
        device = device
                 
    )  
    return model.to(device), model_conch, class_prompts 

def train(cfg):
    
    print("\nInitializing training...")
    torch.backends.cudnn.benchmark = True  # Enable cudnn auto optimization
    
    num_workers = 16 # Optimize worker count based on CPU cores
    print(">>> num_workers", num_workers)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(42)
    
    # Preload CLIP model to GPU and set to eval mode
    # clip_model = clip_model.to(device)
    # clip_model.eval()
    
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    
    train_dataset, val_dataset = get_cls_dataset(cfg, split="valid")
    
    # Efficient data loading configuration
    train_loader = DataLoader(train_dataset,
                            batch_size=cfg.train.samples_per_gpu,
                            num_workers=num_workers,
                            pin_memory=True,
                            shuffle=True,
                            prefetch_factor=2,
                            persistent_workers=True)
    
    
    val_loader = DataLoader(val_dataset,
                          batch_size=128,
                          shuffle=False,
                          num_workers=num_workers,
                          pin_memory=True,
                          persistent_workers=True)

    iters_per_epoch = len(train_loader)
    cfg.train.max_iters = cfg.train.epoch * iters_per_epoch
    cfg.train.eval_iters = iters_per_epoch
    cfg.scheduler.warmup_iter = cfg.scheduler.warmup_iter * iters_per_epoch


    model, model_conch, class_prompts = build_model(device) 
    print("Freezing CONCH backbone (vision + text encoder)...")
    model_conch.eval()
    for param in model_conch.parameters():
        param.requires_grad = False

    # Also freeze the wrapped version inside CONCHFeatureExtractor
    if hasattr(model, 'image_encoder') and hasattr(model.image_encoder, 'model'):
        model.image_encoder.model.eval()
        for param in model.image_encoder.model.parameters():
            param.requires_grad = False

    # # Optional: double-check what is actually trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")
    
    # Optimizer configuration
    
    # optimizer = PolyWarmupAdamW(
    #     params=model.parameters(),
    #     lr=cfg.optimizer.learning_rate,
    #     weight_decay=cfg.optimizer.weight_decay,
    #     betas=cfg.optimizer.betas,
    #     warmup_iter=cfg.scheduler.warmup_iter,
    #     max_iter=cfg.train.max_iters,
    #     warmup_ratio=cfg.scheduler.warmup_ratio,
    #     power=cfg.scheduler.power
    # )
    optimizer = PolyWarmupAdamW(
            params=[p for p in model.parameters() if p.requires_grad],  # ← ONLY trainable ones!
            lr=cfg.optimizer.learning_rate,
            weight_decay=cfg.optimizer.weight_decay,
            betas=cfg.optimizer.betas,
            warmup_iter=cfg.scheduler.warmup_iter,
            max_iter=cfg.train.max_iters,
            warmup_ratio=cfg.scheduler.warmup_ratio,
            power=cfg.scheduler.power
        ) 
    # Resume training if a checkpoint path is provided
    start_iter = 0
    
    best_fuse234_dice = 0.0
    
    # Check if a resume path was provided in the command-line arguments
    if args.resume is not None:
        if os.path.exists(args.resume):
            print(f"\nResuming training from checkpoint: {args.resume}")
            # Load the checkpoint dictionary
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Load model state
            # Using strict=False is safer as it won't crash if the model architectures
            # have minor differences (e.g., a new layer was added).
            model.load_state_dict(checkpoint['model'], strict=False)
            
            # Load optimizer state
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("Optimizer state loaded successfully.")
            else:
                print("WARNING: Optimizer state not found in checkpoint. Starting with a fresh optimizer.")

            # Load the iteration number to continue from where we left off
            if 'iter' in checkpoint:
                start_iter = checkpoint['iter'] + 1 # Start from the next iteration
                print(f"Resuming from iteration: {start_iter}")
            else:
                print("WARNING: Iteration number not found in checkpoint. Starting from iteration 0.")
            
            if 'best_mIoU' in checkpoint:
                 best_fuse234_dice = checkpoint['best_mIoU']
                 print(f"Loaded previous best mIoU: {best_fuse234_dice:.4f}")

        else:
            print(f"WARNING: Checkpoint file not found at {args.resume}. Starting from scratch.")
    else:
        print("\nStarting training from scratch.")
    
    # Mixed precision training setup
    scaler = torch.cuda.amp.GradScaler()
    
    model.to(device)
    model.train()

    # Loss functions and feature extractor setup
    # Classification Loss
    loss_function = nn.BCEWithLogitsLoss().to(device)
    
    # Contrastive Loss components
    mask_adapter = MaskAdapter_DynamicThreshold(alpha=cfg.train.mask_adapter_alpha,)
    feature_extractor = FeatureExtractor(mask_adapter=mask_adapter)
    fg_loss_fn = InfoNCELossFG(temperature=0.07).to(device)
    bg_loss_fn = InfoNCELossBG(temperature=0.07).to(device)
    
    # Diversity Loss
    # Prototype diversity regularizer (encourages disjoint \Omega_c support)
    div_cfg = getattr(cfg.train, "diversity", None)
    from utils.diversity_loss import PrototypeDiversityRegularizer
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

    print("\nStarting training...")
    diversity_meter = AverageMeter('diversity_loss')
    diversity_running_avg = None

    # Convert start_iter back to epoch and batch indices
    start_epoch = start_iter // iters_per_epoch
    start_batch = start_iter % iters_per_epoch

    for epoch in range(start_epoch, cfg.train.epoch):
        print(f"\nEpoch {epoch + 1}/{cfg.train.epoch}")

        # Track training metrics for this epoch
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_batch_count = 0

        # Create progress bar for this epoch
        train_loader_iter = iter(train_loader)
        pbar = tqdm(total=iters_per_epoch, desc=f"Epoch {epoch + 1}", leave=False)

        # Skip batches if resuming from middle of epoch
        if epoch == start_epoch and start_batch > 0:
            for _ in range(start_batch):
                next(train_loader_iter)
            pbar.update(start_batch)

        batch_idx = start_batch if epoch == start_epoch else 0

        while batch_idx < iters_per_epoch:
            try:
                img_name, inputs, cls_labels, _ = next(train_loader_iter)
            except StopIteration:
                break

            # Calculate global iteration number
            n_iter = epoch * iters_per_epoch + batch_idx

            inputs = inputs.to(device).float()
            cls_labels = cls_labels.to(device).float()
            iter_diversity_loss = None

            with torch.cuda.amp.autocast():
                # Unpack the new return value from the model
                (cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, cls5, cam5,
                    prototypes, k_list, feature_map_for_diversity) = model(inputs)
                # --- Classification Loss (L_CLS) ---
                cls1_merge = merge_to_parent_predictions(cls1, k_list, method=cfg.train.merge_train)
                cls2_merge = merge_to_parent_predictions(cls2, k_list, method=cfg.train.merge_train)
                cls3_merge = merge_to_parent_predictions(cls3, k_list, method=cfg.train.merge_train)
                cls4_merge = merge_to_parent_predictions(cls4, k_list, method=cfg.train.merge_train)
                cls5_merge = merge_to_parent_predictions(cls5, k_list, method=cfg.train.merge_train)

                loss1 = loss_function(cls1_merge, cls_labels)
                loss2 = loss_function(cls2_merge, cls_labels)
                loss3 = loss_function(cls3_merge, cls_labels)
                loss4 = loss_function(cls4_merge, cls_labels)
                loss5 = loss_function(cls5_merge, cls_labels)

                cls_loss = cfg.train.l1 * loss1 + cfg.train.l2 * loss2 + cfg.train.l3 * loss3 + cfg.train.l4 * loss4 + cfg.train.l5 * loss5
                # --- Conditional application of other losses based on warm-up ---
                if n_iter >= cfg.train.warmup_iters:
                    subclass_labels = expand_parent_to_subclass_labels(cls_labels, k_list)
                    cls4_expand = expand_parent_to_subclass_labels(cls4_merge, k_list)
                    cls4_bir = (cls4 > cls4_expand).float() * subclass_labels
                    batch_info = feature_extractor.process_batch(inputs, cam4, cls4_bir, model_conch)

                    contrastive_loss = None
                    if batch_info is not None:
                    # === CORRECT: Use LEARNABLE text prototypes from PromptLearner ===
                    # This is what actually works and gives 40–50+ mIoU
                        with torch.no_grad():  # saves memory, not required for correctness
                            text_proto = model.prompt_learner.get_prototypes(class_prompts)
                            l_fea = text_proto['features']                    # [total_prototypes, 512]
                            l_fea = F.normalize(l_fea, dim=-1).to(inputs.device)  # crucial!

                        # Extract FG/BG visual features
                        fg_features = batch_info['fg_features']   # [N, 512]
                        bg_features = batch_info['bg_features']   # [N, 512]

                        # Pair them with corresponding text prototypes
                        set_info = pair_features(fg_features, bg_features, l_fea, cls4_bir)
                        fg_vis, bg_vis = set_info['fg_features'], set_info['bg_features']
                        fg_txt, bg_txt = set_info['fg_text'], set_info['bg_text']

                        # Compute contrastive losses
                        fg_loss = fg_loss_fn(fg_vis, fg_txt, bg_txt)
                        bg_loss = bg_loss_fn(bg_vis, fg_txt, bg_txt)
                        contrastive_loss = fg_loss + bg_loss 
                    # if batch_info is not None:
                    #     # Get text features from CONCH using encode_text with tokenized input
                    #     # Map prompts to prototypes (num_prototypes_per_class per class)
                    #     #=========================================================== 
                    #     with torch.no_grad():
                    #         from conch.open_clip_custom import get_tokenizer, tokenize
                    #         tokenizer = get_tokenizer()
                            
                    #         num_prototypes_per_class = cfg.model.num_prototypes_per_class
                    #         all_text_features = []
                            
                    #         # For each class, tokenize and encode its prompts
                    #         for cls_idx in sorted(class_prompts.keys()):
                    #             prompts = class_prompts[cls_idx]
                                
                    #             # Tokenize prompts using CONCH's tokenize function (same as text_encoder.py)
                    #             # tokenize() handles batching and padding automatically
                    #             token_ids_batch = tokenize(tokenizer, prompts).to(device)  # [num_prompts, 77]
                                
                    #             # Encode tokenized prompts using CONCH's encode_text
                    #             # model_conch.encode_text expects token_ids (tensors), not raw strings
                    #             cls_text_features = model_conch.encode_text(token_ids_batch, normalize=True)  # [num_prompts, 512]
                                
                    #             # Map encoded prompts to prototypes (repeat/average to match prototype count)
                    #             if len(prompts) == num_prototypes_per_class:
                    #                 # Perfect match: use directly
                    #                 all_text_features.append(cls_text_features)
                    #             elif len(prompts) < num_prototypes_per_class:
                    #                 # Fewer prompts than prototypes: repeat to match
                    #                 repeats = num_prototypes_per_class // len(prompts)
                    #                 remainder = num_prototypes_per_class % len(prompts)
                    #                 repeated = cls_text_features.repeat(repeats, 1)
                    #                 if remainder > 0:
                    #                     repeated = torch.cat([repeated, cls_text_features[:remainder]], dim=0)
                    #                 all_text_features.append(repeated)
                    #             else:
                    #                 # More prompts than prototypes: take first N or average
                    #                 all_text_features.append(cls_text_features[:num_prototypes_per_class])
                            
                    #         # Stack all class text features: [total_prototypes, 512]
                    #         l_fea = torch.cat(all_text_features, dim=0)  # [40, 512] for 4 classes * 10 prototypes
                    #     #===========================================================  
                    #     fg_features, bg_features = batch_info['fg_features'], batch_info['bg_features']
                    #     set_info = pair_features(fg_features, bg_features, l_fea, cls4_bir)
                    #     fg_features, bg_features, fg_pro, bg_pro = set_info['fg_features'], set_info['bg_features'], set_info['fg_text'], set_info['bg_text']
                    #     fg_loss = fg_loss_fn(fg_features, fg_pro, bg_pro)
                    #     bg_loss = bg_loss_fn(bg_features, fg_pro, bg_pro)
                    #     contrastive_loss = fg_loss + bg_loss

                    # Diversity loss computation
                    with torch.no_grad():
                        cam4_merged = merge_subclass_cams_to_parent(cam4, k_list, method=cfg.train.merge_train)
                        cam_max, _ = torch.max(cam4_merged, dim=1, keepdim=True)
                        background_score = torch.full_like(cam_max, 0.2)
                        full_cam = torch.cat([background_score, cam4_merged], dim=1)
                        pseudo_mask = torch.argmax(full_cam, dim=1)

                    pseudo_mask_resized = F.interpolate(
                        pseudo_mask.unsqueeze(1).float(),
                        size=feature_map_for_diversity.shape[2:],
                        mode='nearest').squeeze(1).long()
                    diversity_loss = diversity_loss_fn(feature_map_for_diversity, prototypes, pseudo_mask_resized, global_step=n_iter)
                    iter_diversity_loss = diversity_loss.detach().item()
                    diversity_running_avg = diversity_meter.get('diversity_loss') if 'diversity_loss' in diversity_meter._AverageMeter__data else None
                    diversity_meter.add({'diversity_loss': iter_diversity_loss})
                    diversity_running_avg = diversity_meter.get('diversity_loss')

                    # Total Loss = All components (matching main.py)
                    lambda_sim = cfg.train.l5  # Use config value (0.2) instead of hardcoded 0.1
                    lambda_j = cfg.train.lambda_j
                    loss = cls_loss + lambda_j * diversity_loss
                    if contrastive_loss is not None:
                        # Include CAM regularization term like main.py
                        loss = loss + lambda_sim * (contrastive_loss + 0.0005 * torch.mean(cam4))
                else:
                    # --- WARM-UP PHASE ---
                    loss = cls_loss

            # A one-time message to signal the end of the warm-up period
            if n_iter == cfg.train.warmup_iters:
                print(f"\n--- Iteration {n_iter}: Warm-up complete. Activating contrastive and diversity losses. ---\n")

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Accumulate training metrics
            train_loss_sum += loss.item()
            cls_pred4 = (torch.sigmoid(cls4_merge) > 0.5).float()
            all_cls_acc4 = (cls_pred4 == cls_labels).all(dim=1).float().mean() * 100
            train_acc_sum += all_cls_acc4.item()
            train_batch_count += 1

            # Update progress bar
            pbar.update(1)
            batch_idx += 1

            # Log progress every 20 batches or at end of epoch
            if batch_idx % 20 == 0 or batch_idx == iters_per_epoch:
                cur_lr = optimizer.param_groups[0]['lr']
                # div_last_str = f"{iter_diversity_loss:.4f}" if iter_diversity_loss is not None else "N/A"
                # div_avg_str = f"{diversity_running_avg:.4f}" if diversity_running_avg is not None else "N/A"

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{cur_lr:.2e}",
                    'acc': f"{all_cls_acc4:.1f}%",
                    # 'div': div_last_str
                })

        # Close progress bar for this epoch
        pbar.close()

        # Calculate average training metrics for this epoch
        avg_train_loss = train_loss_sum / train_batch_count if train_batch_count > 0 else 0.0
        avg_train_acc = train_acc_sum / train_batch_count if train_batch_count > 0 else 0.0

        # Reset start_batch for subsequent epochs
        start_batch = 0
        # Validation at end of each epoch
        val_mIoU, val_mean_dice, val_fw_iu, val_iu_per_class, val_dice_per_class, val_acc = validate(
            model=model,
            data_loader=val_loader,
            cfg=cfg,
            cls_loss_func=loss_function
        )

        # Print training results on one line
        print(f"\nTraining: Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.2f}%")
        
        # Print validation results on one line
        print(f"Validation: mIoU: {val_mIoU:.4f} | Mean Dice: {val_mean_dice:.4f} | FwIU: {val_fw_iu:.4f} | Acc: {val_acc:.2f}%")
        
        # Print per-class IoU on one line (iu_per_class is in 0-1 range, convert to percentage)
        iu_parts = []
        val_iu_list = val_iu_per_class.cpu().numpy() if hasattr(val_iu_per_class, 'cpu') else val_iu_per_class
        for i, score in enumerate(val_iu_list):
            label = f"Class {i}" if i < len(val_iu_list) - 1 else "Background"
            iu_parts.append(f"{label}: {score*100:.4f}")
        print(f"IoU by class: {' | '.join(iu_parts)}")

        # The variable current_miou is now just val_mIoU
        current_miou = val_mIoU
        print(f"mIOU (for saving): {current_miou:.4f}")

        # Save best model (warm-up disabled, so always save if better)
        # # Only consider saving if we are past the warm-up period
        # if n_iter >= cfg.train.warmup_iters:
        if current_miou > best_fuse234_dice:
            best_fuse234_dice = current_miou
            save_path = os.path.join(cfg.work_dir.ckpt_dir, "best_cam.pth")

            torch.save(
                {
                    "cfg": cfg,
                    "iter": n_iter,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_mIoU": best_fuse234_dice
                },
                save_path,
                _use_new_zipfile_serialization=True
            )
            print(f"\nSaved best model with mIOU: {best_fuse234_dice:.4f}")
        # else:
        #     print(f"--- In warm-up period (current iter: {n_iter + 1}). Skipping best model check. ---")

    torch.cuda.empty_cache()
    end_time = datetime.datetime.now()
    total_training_time = end_time - start_time
    print(f'Total training time: {total_training_time}')

    
    print("\n" + "="*80)
    print("POST-TRAINING EVALUATION AND CAM GENERATION")
    print("="*80)
 
    print("\nPreparing test dataset...")

    train_dataset, test_dataset = get_cls_dataset(cfg, split="test",enable_rotation=False,p=0.0)
    print(f"Test dataset loaded: {len(test_dataset)} samples")
    

    test_loader = DataLoader(test_dataset,
                    batch_size=128,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=True)

    print("\n1. Testing on test dataset...")
    print("-" * 50)
    
    test_mIoU, test_mean_dice, test_fw_iu, test_iu_per_class, test_dice_per_class, test_acc = validate(
        model=model,
        data_loader=test_loader,
        cfg=cfg,
        cls_loss_func=loss_function
    )   

    print("Testing results:")
    print(f"Test mIoU: {test_mIoU:.4f}")
    print(f"Test Mean Dice: {test_mean_dice:.4f}")
    print(f"Test FwIU: {test_fw_iu:.4f}")
    print(f"Test Acc: {test_acc:.2f}%")

    # Per-class IoU scores on single line
    iu_parts = []
    for i, score in enumerate(test_iu_per_class):
        label = f"Class {i}" if i < len(test_iu_per_class) - 1 else "Background"
        iu_parts.append(f"{label}: {score*100:.4f}")
    print(f"\nPer-class IoU scores (FG classes + BG):")
    print(f"  {' | '.join(iu_parts)}")

    # Per-class Dice scores on single line
    dice_parts = []
    for i, score in enumerate(test_dice_per_class):
        label = f"Class {i}" if i < len(test_dice_per_class) - 1 else "Background"
        dice_parts.append(f"{label}: {score*100:.4f}")
    print(f"\nPer-class Dice scores (FG classes + BG):")
    print(f"  {' | '.join(dice_parts)}")

    # print("\n2. Generating CAMs for complete training dataset...")
    # print("-" * 50)
    
    # train_cam_loader = DataLoader(train_dataset,
    #                         batch_size=1,
    #                         shuffle=False,
    #                         num_workers=num_workers,
    #                         pin_memory=True,
    #                         persistent_workers=True)

    # print(f"Generating CAMs for all {len(train_dataset)} training samples...")
    # print(f"Output directory: {cfg.work_dir.pred_dir}")

    # best_model_path = os.path.join(cfg.work_dir.ckpt_dir, "best_cam.pth")
    # if os.path.exists(best_model_path):
    #     print(f"Loading best model from: {best_model_path}")
    #     checkpoint = torch.load(best_model_path, map_location=device)
    #     model.load_state_dict(checkpoint["model"])
    #     best_iter = checkpoint.get("iter", "unknown")
    #     print(f"Best model loaded successfully! (Saved at iteration: {best_iter})")
    # else:
    #     print("Warning: Best model checkpoint not found, using current model state")
    #     print(f"Expected path: {best_model_path}")
    
    # generate_cam(model=model, data_loader=train_cam_loader, cfg=cfg)
    
    # print("\nFiles generated:")
    # print(f"  Training CAM visualizations: {cfg.work_dir.pred_dir}/*.png")
    # print(f"  Model checkpoint: {cfg.work_dir.ckpt_dir}/best_cam.pth")
    # print("="*80)
    
    
if __name__ == "__main__":
    cfg = OmegaConf.load(args.config)
    cfg.work_dir.dir = os.path.dirname(args.config)
    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)

    os.makedirs(cfg.work_dir.dir, exist_ok=True)
    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)

    print('\nArgs: %s' % args)
    print('\nConfigs: %s' % cfg)

    set_seed(0)
    train(cfg=cfg) 