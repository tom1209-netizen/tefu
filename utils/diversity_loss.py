import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_jeffreys_similarity(distributions):
    """Calculates Jeffrey's Similarity for a list of distributions."""
    num_distributions = len(distributions)
    if num_distributions <= 1:
        # If there's only one prototype for a class, diversity is undefined (or perfect).
        return torch.tensor(0.0, device=distributions[0].device)

    total_similarity = 0.0
    num_pairs = 0

    # Ensure distributions are valid (sum to 1 and are non-negative)
    distributions = [F.softmax(d, dim=0) for d in distributions]

    for i in range(num_distributions):
        for j in range(i + 1, num_distributions):
            U = distributions[i]
            V = distributions[j]
            
            # Add a small epsilon to avoid log(0) issues in KL-divergence
            eps = 1e-10
            U_safe = U + eps
            V_safe = V + eps
            
            # Kullback-Leibler Divergence
            dkl_uv = F.kl_div(U_safe.log(), V_safe, reduction='sum')
            dkl_vu = F.kl_div(V_safe.log(), U_safe, reduction='sum')
            
            # Jeffrey's Divergence (symmetrized KL)
            jeffreys_divergence = dkl_uv + dkl_vu
            
            # Jeffrey's Similarity
            similarity = torch.exp(-jeffreys_divergence)
            total_similarity += similarity
            num_pairs += 1
            
    return total_similarity / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=distributions[0].device)


class PrototypeDiversityLoss(nn.Module):
    def __init__(self, num_prototypes_per_class):
        super().__init__()
        self.num_prototypes_per_class = num_prototypes_per_class

    def forward(self, feature_map, prototypes, gt_mask):
        """
        Calculates diversity loss using Cosine Similarity.
        
        Args:
            feature_map (torch.Tensor): Output from the encoder [B, D, H, W]
            prototypes (torch.Tensor): The learnable prototype vectors [Total_Prototypes, D_proto]
            gt_mask (torch.Tensor): Ground truth segmentation mask [B, H, W]
        """
        B, D, H, W = feature_map.shape
        device = feature_map.device
        total_loss = 0.0

        # Reshape for easier processing
        feature_map_flat = feature_map.view(B, D, H * W).permute(0, 2, 1) # -> [B, H*W, D]
        gt_mask_flat = gt_mask.view(B, H * W) # -> [B, H*W]

        num_classes = prototypes.shape[0] // self.num_prototypes_per_class

        for b in range(B): # Iterate over each image in the batch
            class_losses = []
            for c in range(num_classes): # Iterate over each class present in the image
                # Find pixels belonging to the current class 'c' in the ground truth
                class_pixel_indices = (gt_mask_flat[b] == c).nonzero(as_tuple=True)[0]
                
                if len(class_pixel_indices) == 0:
                    continue # Skip if this class is not in the ground truth for this image

                class_pixel_features = feature_map_flat[b, class_pixel_indices] # -> [Num_Pixels_c, D]

                # Get all prototypes for this class
                start_idx = c * self.num_prototypes_per_class
                end_idx = (c + 1) * self.num_prototypes_per_class
                class_prototypes = prototypes[start_idx:end_idx] # -> [Num_Proto_c, D_proto]
                
                # Normalize features and prototypes for cosine similarity calculation
                class_pixel_features_norm = F.normalize(class_pixel_features, p=2, dim=1)
                class_prototypes_norm = F.normalize(class_prototypes, p=2, dim=1)

                # Calculate v(Z, p) for each prototype of this class
                # v is the distribution of a prototype's activation over the class pixels
                distributions_v = []
                for p_k_norm in class_prototypes_norm:
                    # Calculate cosine similarity from this prototype to all class pixels
                    # Shape: [Num_Pixels_c]
                    similarities = torch.matmul(class_pixel_features_norm, p_k_norm)
                    
                    # The activation distribution is the softmax over these similarities.
                    v_distribution = F.softmax(similarities, dim=0)
                    distributions_v.append(v_distribution)

                # Calculate Jeffrey's Similarity for this class's prototypes 
                lj_c = calculate_jeffreys_similarity(distributions_v)
                class_losses.append(lj_c)

            if len(class_losses) > 0:
                # Average the similarity loss across all classes for this image
                total_loss += torch.mean(torch.stack(class_losses))

        return total_loss / B 