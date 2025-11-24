# Implementation adapted from PBIP by Qingchen Tang
# Source: https://github.com/QingchenTang/PBIP

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

# Suppress matplotlib warnings for cleaner output
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Import your model definition
from model.model import ClsNetwork

# Configuration & Palette

# Define the color palette for the masks
# Order: TUM, STR, LYM, NEC, BACK
PALETTE = [
    [255, 0, 0],   # 0: TUM (Red)
    [0, 255, 0],   # 1: STR (Green)
    [0, 0, 255],   # 2: LYM (Blue)
    [153, 0, 255], # 3: NEC (Purple)
    [255, 255, 255],     # 4: BACK (White)
]

# Define the class names in the correct order
CLASS_NAMES = ['TUM', 'STR', 'LYM', 'NEC']

# Helper Functions for Visualization
def get_validation_transform():
    """Gets the normalization transform for a validation image."""
    MEAN = [0.66791496, 0.47791372, 0.70623304]
    STD = [0.1736589,  0.22564577, 0.19820057]
    return A.Compose([
        A.Normalize(MEAN, STD),
        ToTensorV2(transpose_mask=True),
    ])

def load_color_mask(mask_path, palette):
    """Loads a ground truth mask and converts its indices to a color image."""
    try:
        mask_pil = Image.open(mask_path)
        mask_np = np.array(mask_pil)
        
        # Create an RGB image from the indexed mask
        color_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
        
        for class_idx, color in enumerate(palette):
            color_mask[mask_np == class_idx] = color
            
        return color_mask
    except FileNotFoundError:
        print(f"  Warning: No mask found at {mask_path}. Skipping.")
        return np.zeros((224, 224, 3), dtype=np.uint8) # Return a black image
    except Exception as e:
        print(f"  Error loading mask {mask_path}: {e}")
        return np.zeros((224, 224, 3), dtype=np.uint8)

def generate_heatmap(image, activation_map):
    """
    Overlays a grayscale activation map as a heatmap onto the original image.
    image: Original BGR image (H, W, 3).
    activation_map: Grayscale activation map (H, W), normalized 0-1.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * activation_map), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    image = np.float32(image) / 255
    overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
    return np.uint8(overlay * 255)

def get_activation_map(feature_map_flat, prototype_vec, feature_map_shape, img_size):
    """
    Calculates the cosine similarity map for a single prototype.
    """
    # Normalize the single prototype vector
    proto_norm = F.normalize(prototype_vec, p=2, dim=0)
    
    # Calculate cosine similarity
    # (Num_Pixels, 1)
    cos_sim = torch.matmul(feature_map_flat, proto_norm)
    
    # Reshape to 2D map
    act_map = cos_sim.view(feature_map_shape)
    
    # Visualization Normalization
    # Use ReLU to only show positive activations
    act_map = F.relu(act_map)
    
    # Upsample the map to the original image size
    act_map_up = F.interpolate(act_map.unsqueeze(0).unsqueeze(0),
                               size=img_size,
                               mode='bilinear',
                               align_corners=False).squeeze()
    
    # Normalize to [0, 1] for the heatmap
    map_min, map_max = act_map_up.min(), act_map_up.max()
    if map_max > map_min:
        act_map_norm = (act_map_up - map_min) / (map_max - map_min + 1e-8)
    else:
        act_map_norm = torch.zeros_like(act_map_up)
        
    return act_map_norm.cpu().numpy()

# Main Execution
def main(args):
    print("--- Prototype Activation Visualization")
    
    # Load Config and Setup
    cfg = OmegaConf.load(args.config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    
    num_classes = cfg.dataset.cls_num_classes
    k_per_class = cfg.model.num_prototypes_per_class
    print(f"Config loaded. Using {k_per_class} prototypes per class for {num_classes} classes.")

    # Load Learnable Model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = ClsNetwork(backbone=cfg.model.backbone.config,
                       stride=cfg.model.backbone.stride,
                       cls_num_classes=num_classes,
                       num_prototypes_per_class=k_per_class,
                       prototype_feature_dim=cfg.model.prototype_feature_dim,
                       n_ratio=cfg.model.n_ratio,
                       pretrained=False) # Not loading ImageNet weights
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    print("Learnable model loaded successfully.")

    # Load Original PBIP Prototypes
    original_proto_path = os.path.join(
        os.path.dirname(args.config), 
        cfg.model.label_feature_path + ".pkl"
    )
    print(f"Loading original prototypes from: {original_proto_path}")
    try:
        with open(original_proto_path, 'rb') as f:
            original_proto_data = pkl.load(f)
        original_protos = original_proto_data['features'].to(device)
        # Verify shape
        assert original_protos.shape[0] == num_classes * k_per_class
        print("Original PBIP prototypes loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load original prototypes from {original_proto_path}. {e}")
        return

    # Get Both Sets of Prototypes
    # Get the final learnable prototypes from the model
    learnable_protos = model.prototypes.detach()
    
    # Project both sets of prototypes using the model's trained projection layer
    # This ensures a fair comparison using the same feature space
    with torch.no_grad():
        projected_learnable_protos = model.l_fc4(learnable_protos).detach()
        projected_original_protos = model.l_fc4(original_protos).detach()

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
            
        # Load original image (for heatmaps) and mask
        image_orig_bgr = cv2.imread(img_path)
        if image_orig_bgr is None:
            print(f"  Warning: Could not read image {img_path}. Skipping.")
            continue
        
        image_orig_rgb = cv2.cvtColor(image_orig_bgr, cv2.COLOR_BGR2RGB)
        img_size = (image_orig_rgb.shape[0], image_orig_rgb.shape[1])
        
        mask_color = load_color_mask(mask_path, PALETTE)
        
        # Preprocess image for model
        image_tensor = transform(image=image_orig_rgb)["image"].unsqueeze(0).to(device)
        
        # Get Image Feature Map
        with torch.no_grad():
            # Get the multi-scale features from the encoder
            _x_all, _ = model.encoder(image_tensor)
            # We use the final, highest-res feature map
            feature_map = _x_all[3].detach() 
            
            B, C, H, W = feature_map.shape
            feature_map_shape = (H, W)
            
            # Flatten and normalize the pixel features
            feature_map_flat = feature_map.permute(0, 2, 3, 1).reshape(-1, C)
            feature_map_flat_norm = F.normalize(feature_map_flat, p=2, dim=1)

        # Create Visualization Grid
        # Rows: 1 (Header) + 4 classes * 2 methods
        # Cols: 1 (Label) + k prototypes
        num_rows = 1 + num_classes * 2
        num_cols = k_per_class + 1
        
        fig, axes = plt.subplots(num_rows, num_cols, 
                                 figsize=(num_cols * 4, num_rows * 4))
        
        # Flatten axes array for easier iteration
        axes = axes.ravel()
        for ax in axes:
            ax.axis('off') # Turn off all axes by default
            
        # Plot Header Row
        axes[0].imshow(image_orig_rgb)
        axes[0].set_title("Original Image", fontsize=16)
        axes[0].axis('on')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        
        axes[1].imshow(mask_color)
        axes[1].set_title("Ground Truth Mask", fontsize=16)
        axes[1].axis('on')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        
        # Plot Prototype Rows
        for c_idx, class_name in enumerate(CLASS_NAMES):
            row_offset_learnable = (1 + c_idx * 2) * num_cols
            row_offset_original = (2 + c_idx * 2) * num_cols
            
            # Set Row Labels
            axes[row_offset_learnable].text(0.5, 0.5, f"{class_name}\nLearnable", 
                                            ha='center', va='center', fontsize=14, fontweight='bold')
            axes[row_offset_original].text(0.5, 0.5, f"{class_name}\nOriginal (PBIP)", 
                                           ha='center', va='center', fontsize=14, fontweight='bold')
            
            for k_idx in range(k_per_class):
                proto_idx = c_idx * k_per_class + k_idx
                ax_idx_learnable = row_offset_learnable + k_idx + 1
                ax_idx_original = row_offset_original + k_idx + 1
                
                # Get Learnable Prototype Activation Map
                proto_learnable = projected_learnable_protos[proto_idx]
                map_learnable = get_activation_map(feature_map_flat_norm, proto_learnable, 
                                                   feature_map_shape, img_size)
                heatmap_learnable = generate_heatmap(image_orig_bgr, map_learnable)
                
                # Get Original PBIP Prototype Activation Map
                proto_original = projected_original_protos[proto_idx]
                map_original = get_activation_map(feature_map_flat_norm, proto_original, 
                                                  feature_map_shape, img_size)
                heatmap_original = generate_heatmap(image_orig_bgr, map_original)

                # Plot
                axes[ax_idx_learnable].imshow(cv2.cvtColor(heatmap_learnable, cv2.COLOR_BGR2RGB))
                axes[ax_idx_learnable].set_title(f"Prototype {k_idx + 1}")
                
                axes[ax_idx_original].imshow(cv2.cvtColor(heatmap_original, cv2.COLOR_BGR2RGB))
        
        # Save the Figure
        plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=0.5)
        save_path = os.path.join(args.out_dir, f"{os.path.splitext(img_name)[0]}_proto_comparison.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
    print(f"\nDone. Visualizations saved to {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and compare prototype activations.")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to the training config.yaml file.")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to the trained best_cam.pth model checkpoint.")
    parser.add_argument("--images", type=str, nargs='+', required=True, 
                        help="List of image filenames from the test set (e.g., patient_01.png patient_02.png).")
    parser.add_argument("--out_dir", type=str, default="./prototype_visualizations", 
                        help="Directory to save the output comparison images.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use.")
    
    main(parser.parse_args())