from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeDiversityRegularizer(nn.Module):
    """
    Unified diversity loss:
      - Feature space: prototype repulsion (orthogonality) + coverage.
      - Spatial space: compactness (low variance) + separation (Jeffreys).
    """

    def __init__(
        self,
        num_prototypes_per_class: int = 3,
        repulsion_weight: float = 0.0,
        coverage_weight: float = 0.0,
        spatial_weight: float = 0.0,
        spatial_jeffreys_weight: float = 0.0,
        omega_min_mass: float = 0.05,
    ):
        super().__init__()
        self.k = num_prototypes_per_class
        self.repulsion_weight = repulsion_weight
        self.coverage_weight = coverage_weight
        self.spatial_weight = spatial_weight
        self.spatial_jeffreys_weight = spatial_jeffreys_weight
        self.min_mass = omega_min_mass

    def forward(
        self,
        cam_maps: Optional[torch.Tensor],
        prototypes: Optional[torch.Tensor] = None,
        proto_mask: Optional[torch.Tensor] = None,
        global_step=None,
    ):
        """
        Args:
            cam_maps: [B, P, H, W] prototype activation logits/maps.
            prototypes: [P, D] prototype vectors (for feature repulsion).
            proto_mask: [B, P] mask of active prototypes per image.
        """
        if cam_maps is None:
            if prototypes is not None:
                return prototypes.new_tensor(0.0)
            raise ValueError("cam_maps is required for diversity loss")

        B, P, H, W = cam_maps.shape
        device = cam_maps.device
        if proto_mask is None:
            proto_mask = torch.ones(B, P, device=device)
        else:
            proto_mask = proto_mask.float()

        loss = cam_maps.new_tensor(0.0)
        num_classes = max(1, P // max(self.k, 1))

        # Feature space constraints
        if prototypes is not None and (self.repulsion_weight > 0 or self.coverage_weight > 0):
            protos_norm = F.normalize(prototypes, dim=1)
            # Repulsion (orthogonality) per class
            if self.repulsion_weight > 0 and self.k > 1:
                rep_sum = cam_maps.new_tensor(0.0)
                for c in range(num_classes):
                    start = c * self.k
                    end = start + self.k
                    class_protos = protos_norm[start:end]  # [k, D]
                    sim = class_protos @ class_protos.t()
                    sim = sim - torch.diag(sim.diag())  # zero diagonal
                    rep_sum = rep_sum + sim.pow(2).mean()
                loss = loss + self.repulsion_weight * (rep_sum / num_classes)

        # Spatial constraints (also used for coverage)
        if self.spatial_weight > 0 or self.spatial_jeffreys_weight > 0 or self.coverage_weight > 0:
            # [B, P, HW] (spatial norm)
            probs = F.softmax(cam_maps.view(B, P, -1), dim=2)
            log_probs = probs.clamp_min(1e-8).log()
            # [B, P, H, W] (prototype competition)
            proto_probs = F.softmax(cam_maps, dim=1)

            # Coverage: balanced usage across active prototypes (use proto-wise probs)
            if self.coverage_weight > 0:
                coverage_losses = []
                for b in range(B):
                    active = (proto_mask[b] > 0).float()
                    active_count = active.sum()
                    if active_count < 1:
                        continue
                    weights = proto_probs[b] * \
                        active.view(-1, 1, 1)  # [P, H, W]
                    mass = weights.sum(dim=(1, 2))  # [P]
                    mass = mass / (mass.sum() + 1e-6)
                    active_mass = mass[active > 0]
                    target = torch.full_like(active_mass, 1.0 / active_count)
                    coverage_losses.append(F.mse_loss(active_mass, target))
                    coverage_losses.append(
                        F.relu(self.min_mass - active_mass).mean())
                if coverage_losses:
                    loss = loss + self.coverage_weight * \
                        (torch.stack(coverage_losses).mean())

            # Grid for spatial terms
            y = torch.arange(H, device=device,
                             dtype=cam_maps.dtype) / max(H - 1, 1)
            x = torch.arange(W, device=device,
                             dtype=cam_maps.dtype) / max(W - 1, 1)
            grid_y, grid_x = torch.meshgrid(y, x)
            grid_x = grid_x.reshape(-1)
            grid_y = grid_y.reshape(-1)

            # Compactness (variance)
            if self.spatial_weight > 0:
                mu_x = (probs * grid_x).sum(dim=2)  # [B, P]
                mu_y = (probs * grid_y).sum(dim=2)
                dist_sq = (
                    (grid_x.view(1, 1, -1) - mu_x.unsqueeze(2)).pow(2)
                    + (grid_y.view(1, 1, -1) - mu_y.unsqueeze(2)).pow(2)
                )
                var = (probs * dist_sq).sum(dim=2)  # [B, P]
                compact_mask_sum = proto_mask.sum() + 1e-6
                compactness = (var * proto_mask).sum() / compact_mask_sum
                loss = loss + self.spatial_weight * compactness

            # Separation (Jeffreys divergence between prototype maps)
            if self.spatial_jeffreys_weight > 0 and P > 1:
                sep_total = cam_maps.new_tensor(0.0)
                count = 0
                for b in range(B):
                    active_idx = (proto_mask[b] > 0).nonzero(as_tuple=True)[0]
                    n_active = active_idx.numel()
                    if n_active < 2:
                        continue
                    p_active = probs[b, active_idx]         # [A, HW]
                    log_p_active = log_probs[b, active_idx]  # [A, HW]
                    kl_matrix = (p_active.unsqueeze(
                        1) * (log_p_active.unsqueeze(1) - log_p_active.unsqueeze(0))).sum(dim=2)
                    jeff = kl_matrix + kl_matrix.t()
                    triu = torch.triu_indices(n_active, n_active, offset=1)
                    pair_div = jeff[triu[0], triu[1]]
                    sep_total = sep_total - pair_div.mean()
                    count += 1
                if count > 0:
                    loss = loss + self.spatial_jeffreys_weight * \
                        (sep_total / count)

        return loss
