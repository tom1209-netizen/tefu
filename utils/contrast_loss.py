import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class InfoNCELossFG(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        print(f'Use InfoNCELossFG with temperature: {temperature}')

    def forward(self, fg_img_feature, fg_pro_feature, bg_pro_feature):
        # Normalize image features
        fg_img_feature_norm = F.normalize(fg_img_feature, p=2, dim=-1) # [N, D]
        fg_pro_feature_norm = F.normalize(fg_pro_feature, p=2, dim=-1) # [N, D]
        bg_pro_feature_norm = F.normalize(bg_pro_feature, p=2, dim=-1) # [N, L, D]
        
        # Calculate positive similarities
        # [N, D] * [N, D] -> sum(dim=1) -> [N]
        pos_sim = torch.sum(fg_img_feature_norm * fg_pro_feature_norm, dim=1) # [N]

        # Calculate negative similarities
        # [N, D] @ [N, D, L] -> [N, L] (using einsum for batched matmul)
        neg_sim = torch.einsum('nd,nld->nl', fg_img_feature_norm, bg_pro_feature_norm) # [N, L]

        # Temperature scaling
        pos_logits = pos_sim / self.temperature  # [N]
        neg_logits = neg_sim / self.temperature  # [N, L]

        # Standard InfoNCE: per-sample log-softmax
        logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)  # [N, 1 + L]
        log_denominator = torch.logsumexp(logits, dim=1)  # [N]
        loss = -(pos_logits - log_denominator).mean()

        return loss


class InfoNCELossBG(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        print(f'Use InfoNCELossBG with temperature: {temperature}')

    def forward(self, bg_img_feature, fg_pro_feature, bg_pro_feature):
        # Normalize image features
        bg_img_feature_norm = F.normalize(bg_img_feature, p=2, dim=-1) # [N, D]
        fg_pro_feature_norm = F.normalize(fg_pro_feature, p=2, dim=-1) # [N, D]
        bg_pro_feature_norm = F.normalize(bg_pro_feature, p=2, dim=-1) # [N, L, D]
        
        # Calculate positive similarities (BG img vs BG text)
        # [N, D] @ [N, D, L] -> [N, L] (using einsum)
        pos_sim = torch.einsum('nd,nld->nl', bg_img_feature_norm, bg_pro_feature_norm) # [N, L]
        
        # Calculate negative similarities (BG img vs FG text)
        # [N, D] * [N, D] -> sum(dim=1) -> [N]
        neg_sim = torch.sum(bg_img_feature_norm * fg_pro_feature_norm, dim=1) # [N]
        
        # Temperature scaling
        pos_logits = pos_sim / self.temperature  # [N, L]
        neg_logits = neg_sim / self.temperature  # [N]

        # Multi-positive InfoNCE: numerator is log-sum-exp over positives
        log_pos = torch.logsumexp(pos_logits, dim=1)  # [N]
        all_logits = torch.cat([pos_logits, neg_logits.unsqueeze(1)], dim=1)  # [N, L+1]
        log_denom = torch.logsumexp(all_logits, dim=1)  # [N]
        loss = -(log_pos - log_denom).mean()

        return loss

