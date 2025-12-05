import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from model.conch_adapter import ConchAdapter
from model.segform import mix_transformer


class StructuralConsistencyLoss(nn.Module):
    """
    Relational distillation: match pairwise token affinities between student
    (adapter-refined CONCH features) and teacher (SegFormer) maps.
    """

    def forward(self, student_map: torch.Tensor, teacher_map: torch.Tensor) -> torch.Tensor:
        if student_map is None or teacher_map is None:
            raise ValueError("student_map and teacher_map must not be None")

        b, c_s, h, w = student_map.shape
        teacher_map = F.interpolate(teacher_map, size=(h, w),
                                    mode="bilinear", align_corners=False)

        # Downsample very large maps for memory safety
        max_side = 64
        if h > max_side or w > max_side:
            student_map = F.adaptive_avg_pool2d(
                student_map, (max_side, max_side))
            teacher_map = F.adaptive_avg_pool2d(
                teacher_map, (max_side, max_side))

        student_tokens = student_map.view(
            b, c_s, -1).transpose(1, 2)  # [B, HW, Cs]
        teacher_tokens = teacher_map.view(
            b, teacher_map.shape[1], -1).transpose(1, 2)  # [B, HW, Ct]

        student_tokens = F.normalize(student_tokens, dim=-1)
        teacher_tokens = F.normalize(teacher_tokens, dim=-1)

        student_affinity = torch.bmm(
            student_tokens, student_tokens.transpose(1, 2))
        teacher_affinity = torch.bmm(
            teacher_tokens, teacher_tokens.transpose(1, 2))

        return F.mse_loss(student_affinity, teacher_affinity)


class FeatureRefinementHead(nn.Module):
    """
    Lightweight adapter to refine frozen CONCH features before distillation/CAMs.
    """

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class AdaptiveLayer(nn.Module):
    """
    Two-layer MLP to adapt prototype embeddings to a target feature dimension.
    """

    def __init__(self, in_dim, n_ratio, out_dim):
        super().__init__()
        hidden_dim = int(in_dim * n_ratio)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class DistilledConch(nn.Module):
    """
    Bare-minimum model:
      - Frozen CONCH backbone
      - Frozen SegFormer teacher for structural supervision
      - Trainable structure adapters on CONCH features
      - Text-initialized prototypes drive classification/CAMs
    """

    def __init__(
        self,
        cls_num_classes: int,
        num_prototypes_per_class: int = 1,
        prototype_feature_dim: int = 512,
        clip_adapter: ConchAdapter = None,
        enable_segformer_guidance: bool = True,
        segformer_backbone: str = "mit_b1",
        segformer_checkpoint: str = None,
        guidance_layers=(2,),
        text_prompts=None,
        n_ratio: float = 0.5,
        pretrained: bool = True,
    ):
        super().__init__()
        if clip_adapter is None:
            raise ValueError(
                "DistilledConch requires a CONCH adapter instance.")

        self.is_simplified = True
        self.cls_num_classes = cls_num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.total_prototypes = cls_num_classes * num_prototypes_per_class
        self.guidance_layers = guidance_layers

        # Frozen CONCH backbone
        self.clip_adapter = clip_adapter
        self.prototype_feature_dim = getattr(
            self.clip_adapter, "embed_dim", prototype_feature_dim)
        for p in self.clip_adapter.model.parameters():
            p.requires_grad_(False)

        # Frozen SegFormer teacher
        self.enable_segformer_guidance = enable_segformer_guidance
        self.segformer_teacher = None
        if self.enable_segformer_guidance:
            self.segformer_teacher = getattr(
                mix_transformer, segformer_backbone)(stride=[4, 2, 2, 1])
            if segformer_checkpoint:
                state_dict = torch.load(
                    segformer_checkpoint, map_location="cpu")
                state_dict = {
                    k: v for k, v in state_dict.items()
                    if k in self.segformer_teacher.state_dict()
                }
                self.segformer_teacher.load_state_dict(
                    state_dict, strict=False)
            self.segformer_teacher.eval()
            for p in self.segformer_teacher.parameters():
                p.requires_grad_(False)

        # Structure adapters (trainable)
        self.in_channels = [self.prototype_feature_dim] * 4
        self.structure_adapters = nn.ModuleList(
            [FeatureRefinementHead(ch) for ch in self.in_channels])
        self.structural_loss_fn = StructuralConsistencyLoss()

        # Text-initialized prototypes
        self.prototypes = nn.Parameter(torch.zeros(
            self.total_prototypes, self.prototype_feature_dim), requires_grad=True)
        self._init_prototypes(text_prompts)

        # Projectors for aligning prototypes to feature dims
        self.l_fc1 = AdaptiveLayer(
            self.prototype_feature_dim, n_ratio, self.in_channels[0])
        self.l_fc2 = AdaptiveLayer(
            self.prototype_feature_dim, n_ratio, self.in_channels[1])
        self.l_fc3 = AdaptiveLayer(
            self.prototype_feature_dim, n_ratio, self.in_channels[2])
        self.l_fc4 = AdaptiveLayer(
            self.prototype_feature_dim, n_ratio, self.in_channels[3])

        self.logit_scale = nn.Parameter(torch.ones([]) * (1 / 0.07))

        # Student (CONCH input) stats and teacher (SegFormer) stats
        self.register_buffer(
            "student_mean",
            torch.tensor([0.6679, 0.4779, 0.7062],
                         dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "student_std",
            torch.tensor([0.1737, 0.2256, 0.1982],
                         dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "teacher_mean",
            torch.tensor([0.485, 0.456, 0.406],
                         dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "teacher_std",
            torch.tensor([0.229, 0.224, 0.225],
                         dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def _init_prototypes(self, text_prompts):
        if text_prompts is None:
            trunc_normal_(self.prototypes, std=0.02)
            return
        print("Initializing prototypes from text...")
        with torch.no_grad():
            text_feats = self.clip_adapter.encode_text(
                text_prompts, normalize=True).to(self.prototypes.device)
            base = text_feats.unsqueeze(1).repeat(
                1, self.num_prototypes_per_class, 1)
            base = base.view(self.total_prototypes, -1)
            self.prototypes.copy_(base)

    def _get_teacher_feats(self, x: torch.Tensor):
        if not self.enable_segformer_guidance or self.segformer_teacher is None:
            return None
        # De-normalize from student stats, then re-normalize for teacher
        x_raw = x * self.student_std + self.student_mean
        x_teacher = (x_raw - self.teacher_mean) / self.teacher_std
        with torch.no_grad():
            feats, _ = self.segformer_teacher(x_teacher)
        return feats

    def _get_student_feats(self, x: torch.Tensor):
        with torch.no_grad():
            vis = self.clip_adapter.visual_intermediates(x, use_grad=False)
            while len(vis) < 4:
                vis.append(F.avg_pool2d(vis[-1], kernel_size=2, stride=2))
        # Gradients start at adapters
        return [adapter(f.detach()) for adapter, f in zip(self.structure_adapters, vis)]

    def forward(self, x):
        teacher_feats = self._get_teacher_feats(x)
        student_feats = self._get_student_feats(x)

        distill_loss = None
        if self.training and teacher_feats is not None:
            losses = []
            for idx in self.guidance_layers:
                if idx < len(student_feats) and idx < len(teacher_feats):
                    losses.append(self.structural_loss_fn(
                        student_feats[idx], teacher_feats[idx]))
            if losses:
                distill_loss = sum(losses) / len(losses)

        feat = student_feats[-1]
        B, C, H, W = feat.shape

        proto_proj = self.l_fc4(self.prototypes)
        proto_proj = F.normalize(proto_proj, dim=-1)

        feat_flat = feat.permute(0, 2, 3, 1).reshape(B, -1, C)
        feat_flat = F.normalize(feat_flat, dim=-1)

        logits = torch.matmul(feat_flat, proto_proj.t()) * self.logit_scale
        logits = logits.view(
            B, H, W, self.total_prototypes).permute(0, 3, 1, 2)

        logits_per_class = logits.view(
            B, self.cls_num_classes, self.num_prototypes_per_class, H, W)
        cam = logits_per_class.max(dim=2).values

        cls_logits = F.adaptive_avg_pool2d(cam, (1, 1)).view(B, -1)

        k_list = [self.num_prototypes_per_class] * self.cls_num_classes

        return (
            cls_logits,  # cls1
            cam,         # cam1
            cls_logits,  # cls2
            cam,         # cam2
            cls_logits,  # cls3
            cam,         # cam3
            cls_logits,  # cls4
            cam,         # cam4
            None,        # l_fea (unused)
            k_list,      # k_list
            None,        # feature_map_for_diversity
            None,        # cam_weights
            proto_proj,  # projected_p4
            None,        # text_features_out
            distill_loss,  # distill_loss
        )


__all__ = [
    "DistilledConch",
    "StructuralConsistencyLoss",
    "FeatureRefinementHead",
    "AdaptiveLayer",
]
