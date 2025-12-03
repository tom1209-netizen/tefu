import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InfoNCELossFG(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        print(f'Use InfoNCELossFG with temperature: {temperature}')

    def forward(self, fg_img_feature, fg_pro_feature, bg_pro_feature, memory_queue=None):
        """
        fg_img_feature: [N, D] anchors
        fg_pro_feature: [N, D] positives
        bg_pro_feature: [N, L, D] in-batch negatives
        memory_queue:   [M, D] external negatives (optional)
        """
        fg_img_norm = F.normalize(fg_img_feature, p=2, dim=-1)
        fg_pro_norm = F.normalize(fg_pro_feature, p=2, dim=-1)
        bg_pro_norm = F.normalize(bg_pro_feature, p=2, dim=-1)

        pos_sim = torch.sum(fg_img_norm * fg_pro_norm,
                            dim=1).unsqueeze(1)  # [N,1]
        neg_sim_batch = torch.matmul(fg_img_norm.unsqueeze(1),
                                     bg_pro_norm.transpose(1, 2)).squeeze(1)  # [N,L]

        logits_list = [pos_sim / self.temperature,
                       neg_sim_batch / self.temperature]
        
        if memory_queue is not None:
            mem_norm = F.normalize(memory_queue, p=2, dim=-1)
            neg_sim_mem = torch.matmul(fg_img_norm, mem_norm.t())  # [N,M]
            logits_list.append(neg_sim_mem / self.temperature)
        logits = torch.cat(logits_list, dim=1)  # [N, 1+L(+M)]

        labels = torch.zeros(
            logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


class InfoNCELossBG(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        print(f'Use InfoNCELossBG with temperature: {temperature}')

    def forward(self, bg_img_feature, fg_pro_feature, bg_pro_feature):
        # Normalize image features
        bg_img_feature_norm = F.normalize(
            bg_img_feature, p=2, dim=-1)  # [N, D]
        fg_pro_feature_norm = F.normalize(
            fg_pro_feature, p=2, dim=-1)  # [N, D]
        bg_pro_feature_norm = F.normalize(
            bg_pro_feature, p=2, dim=-1)  # [N, L, D]

        # Calculate positive similarities (BG img vs BG text)
        # [N, D] @ [N, D, L] -> [N, L] (using einsum)
        pos_sim = torch.einsum(
            'nd,nld->nl', bg_img_feature_norm, bg_pro_feature_norm)  # [N, L]

        # Calculate negative similarities (BG img vs FG text)
        # [N, D] * [N, D] -> sum(dim=1) -> [N]
        neg_sim = torch.sum(bg_img_feature_norm *
                            fg_pro_feature_norm, dim=1)  # [N]

        # Temperature scaling
        pos_logits = pos_sim / self.temperature  # [N, L]
        neg_logits = neg_sim / self.temperature  # [N]

        # Multi-positive InfoNCE: numerator is log-sum-exp over positives
        log_pos = torch.logsumexp(pos_logits, dim=1)  # [N]
        all_logits = torch.cat(
            [pos_logits, neg_logits.unsqueeze(1)], dim=1)  # [N, L+1]
        log_denom = torch.logsumexp(all_logits, dim=1)  # [N]
        loss = -(log_pos - log_denom).mean()

        return loss
