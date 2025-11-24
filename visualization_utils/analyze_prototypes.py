import os
import argparse
import pickle as pkl
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

# Suppress matplotlib warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Import BOTH model architectures
from model.model import ClsNetwork as LearnableClsNetwork
from model.pbip_model import ClsNetwork as OriginalPbipNetwork

# Configuration & Palette
PALETTE = [
    [255, 0, 0],   # 0: TUM (Red)
    [0, 255, 0],   # 1: STR (Green)
    [0, 0, 255],   # 2: LYM (Blue)
    [153, 0, 255], # 3: NEC (Purple)
    [255, 255, 255],     # 4: BACK (White)
]
CLASS_NAMES = ['TUM', 'STR', 'LYM', 'NEC']

# Helper Functions
def get_validation_transform():
    MEAN = [0.66791496, 0.47791372, 0.70623304]
    STD = [0.1736589,  0.22564577, 0.19820057]
    return A.Compose([
        A.Normalize(MEAN, STD),
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

def generate_heatmap(image, activation_map):
    heatmap = cv2.applyColorMap(np.uint8(255 * activation_map), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    image_float = np.float32(image) / 255
    overlay = cv2.addWeighted(image_float, 0.5, heatmap, 0.5, 0)
    return np.uint8(overlay * 255)

def get_activation_map(feature_map, projected_prototypes, proto_idx, img_size):
    """
    Calculates the cosine similarity map for a single prototype.
    """
    B, C, H, W = feature_map.shape
    
    # Flatten and normalize pixel features
    feature_map_flat = feature_map.permute(0, 2, 3, 1).reshape(-1, C)
    feature_map_flat_norm = F.normalize(feature_map_flat, p=2, dim=1)
    
    # Get and normalize the single prototype vector
    proto_vec = projected_prototypes[proto_idx]
    proto_norm = F.normalize(proto_vec, p=2, dim=0)
    
    # Calculate cosine similarity
    cos_sim = torch.matmul(feature_map_flat_norm, proto_norm)
    
    # Reshape, ReLU, and Upsample
    act_map = F.relu(cos_sim.view(H, W))
    act_map_up = F.interpolate(act_map.unsqueeze(0).unsqueeze(0),
                               size=img_size,
                               mode='bilinear',
                               align_corners=False).squeeze()
    
    # Normalize to [0, 1] for heatmap
    map_min, map_max = act_map_up.min(), act_map_up.max()
    if map_max > map_min:
        act_map_norm = (act_map_up - map_min) / (map_max - map_min + 1e-8)
    else:
        act_map_norm = torch.zeros_like(act_map_up)
        
    return act_map_norm.cpu().numpy()

# Main Execution

def main(args):
    print("--- Fair Prototype Activation Visualization")
    
    # Load Config and Setup
    cfg = OmegaConf.load(args.config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    
    num_classes = cfg.dataset.cls_num_classes
    k_per_class = cfg.model.num_prototypes_per_class
    print(f"Config loaded. Using {k_per_class} prototypes per class.")

    # Load YOUR Learnable Model
    print(f"Loading Learnable checkpoint: {args.learnable_ckpt}")
    model_learnable = LearnableClsNetwork(
        backbone=cfg.model.backbone.config,
        cls_num_classes=num_classes,
        num_prototypes_per_class=k_per_class,
        prototype_feature_dim=cfg.model.prototype_feature_dim,
        n_ratio=cfg.model.n_ratio
    )
    ckpt_learnable = torch.load(args.learnable_ckpt, map_location=device)
    model_learnable.load_state_dict(ckpt_learnable['model'])
    model_learnable.to(device)
    model_learnable.eval()
    print("Learnable model loaded.")

    # Load Original PBIP Model
    print(f"Loading Original PBIP checkpoint: {args.pbip_ckpt}")
    model_pbip = OriginalPbipNetwork(
        backbone=cfg.model.backbone.config,
        cls_num_classes=num_classes,
        n_ratio=cfg.model.n_ratio,
        l_fea_path=cfg.model.label_feature_path
    )
    ckpt_pbip = torch.load(args.pbip_ckpt, map_location=device)
    model_pbip.load_state_dict(ckpt_pbip['model'])
    model_pbip.to(device)
    model_pbip.eval()
    print("Original PBIP model loaded.")

    # Load Transform
    transform = get_validation_transform()

    # Process Each Image
    print(f"\nProcessing {len(args.images)} target images...")
    for img_name in tqdm(args.images, desc="Images"):
        img_path = os.path.join(cfg.dataset.val_root, "test", "img", img_name)
        mask_path = os.path.join(cfg.dataset.val_root, "test", "mask", img_name)
        
        if not os.path.exists(img_path):
            print(f"  Warning: Image not found at {img_path}. Skipping.")
            continue
            
        image_orig_bgr = cv2.imread(img_path)
        if image_orig_bgr is None: continue
        
        image_orig_rgb = cv2.cvtColor(image_orig_bgr, cv2.COLOR_BGR2RGB)
        img_size = (image_orig_rgb.shape[0], image_orig_rgb.shape[1])
        mask_color = load_color_mask(mask_path, PALETTE)
        image_tensor = transform(image=image_orig_rgb)["image"].unsqueeze(0).to(device)
        # Get Features & Prototypes from BOTH models
        with torch.no_grad():
            # Get data from YOUR learnable model
            _x_all_learnable, _ = model_learnable.encoder(image_tensor)
            feature_map_learnable = _x_all_learnable[3].detach()
            prototypes_learnable = model_learnable.prototypes.detach()
            projected_learnable = model_learnable.l_fc4(prototypes_learnable).detach()
            
            # Get data from the ORIGINAL PBIP model
            _x_all_pbip, _ = model_pbip.encoder(image_tensor) 
            feature_map_pbip = _x_all_pbip[3].detach()
            _, _, _, _, _, _, _, _, prototypes_pbip, _ = model_pbip(image_tensor)
            prototypes_pbip = prototypes_pbip.detach().to(device) 
            
            # Now get the projected prototypes using the PBIP model's layer
            projected_pbip = model_pbip.l_fc4(prototypes_pbip).detach()

        # Create Visualization Grid (Horizontal Layout)
        num_rows = k_per_class + 1     # prototypes + header row
        num_cols = 1 + len(CLASS_NAMES) * 2  # each class has OURS + PBIP
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
        axes = np.atleast_2d(axes)

        for ax in axes.ravel():
            ax.axis("off")

        # Header (row 0)
        axes[0, 0].text(
            0.5, 0.5, "Prototype",
            ha='center', va='center', fontsize=25, fontweight='bold'
        )

        # Add class headers horizontally
        for i, class_name in enumerate(CLASS_NAMES):
            col_ours = 1 + i * 2
            col_pbip = col_ours + 1

            axes[0, col_ours].text(
                0.5, 0.5, f"{class_name}\n(OURS)",
                ha='center', va='center', fontsize=25, fontweight='bold'
            )
            axes[0, col_pbip].text(
                0.5, 0.5, f"{class_name}\n(PBIP)",
                ha='center', va='center', fontsize=25, fontweight='bold'
            )

        # Fill prototype rows
        for k_idx in range(k_per_class):
            # Label each prototype row
            axes[k_idx + 1, 0].text(
                0.5, 0.5, f"{k_idx + 1}",
                ha='center', va='center', fontsize=40, fontweight='bold'
            )

            for i, class_name in enumerate(CLASS_NAMES):
                proto_idx = i * k_per_class + k_idx

                # OURS
                col_ours = 1 + i * 2
                map_learnable = get_activation_map(feature_map_learnable, projected_learnable, proto_idx, img_size)
                heatmap_learnable = generate_heatmap(image_orig_bgr, map_learnable)
                axes[k_idx + 1, col_ours].imshow(cv2.cvtColor(heatmap_learnable, cv2.COLOR_BGR2RGB))

                # PBIP
                col_pbip = col_ours + 1
                map_original = get_activation_map(feature_map_pbip, projected_pbip, proto_idx, img_size)
                heatmap_original = generate_heatmap(image_orig_bgr, map_original)
                axes[k_idx + 1, col_pbip].imshow(cv2.cvtColor(heatmap_original, cv2.COLOR_BGR2RGB))

        # Layout & Save
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        save_path = os.path.join(args.out_dir, f"{os.path.splitext(img_name)[0]}_horizontal.png")
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=400)
        plt.close(fig)

        
    print(f"\nDone. Visualizations saved to {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and compare prototype activations.")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to the training config.yaml file for your learnable model.")
    parser.add_argument("--learnable_ckpt", type=str, required=True, 
                        help="Path to YOUR trained best_cam.pth checkpoint.")
    parser.add_argument("--pbip_ckpt", type=str, required=True, 
                        help="Path to the ORIGINAL PBIP trained checkpoint.")
    parser.add_argument("--images", type=str, nargs='+', required=True, 
                        help="List of image filenames from the test set (e.g., patient_01.png patient_02.png).")
    parser.add_argument("--out_dir", type=str, default="./prototype_visualizations", 
                        help="Directory to save the output comparison images.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use.")
    
    main(parser.parse_args())
