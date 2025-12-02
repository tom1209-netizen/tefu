import math
import os
import pickle as pkl
from functools import partial
from typing import Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from model.segform import mix_transformer
from model.conch_adapter import ConchAdapter


class StructuralConsistencyLoss(nn.Module):
    """
    Relational distillation loss: matches pairwise token affinities between
    student (ViT) and teacher (SegFormer) feature maps after spatial alignment.
    """

    def forward(self, student_map: torch.Tensor, teacher_map: torch.Tensor) -> torch.Tensor:
        if student_map is None or teacher_map is None:
            raise ValueError("student_map and teacher_map must not be None")

        b, c_s, h, w = student_map.shape
        teacher_map = F.interpolate(teacher_map, size=(
            h, w), mode="bilinear", align_corners=False)

        # Downsample very large maps to keep affinity matrices memory-friendly
        max_side = 64
        if h > max_side or w > max_side:
            student_map = F.adaptive_avg_pool2d(
                student_map, (max_side, max_side))
            teacher_map = F.adaptive_avg_pool2d(
                teacher_map, (max_side, max_side))
            b, c_s, h, w = student_map.shape

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
    Lightweight adapter to sharpen frozen CONCH features before distillation/CAMs.
    Channel-mixing + depthwise spatial filtering preserves efficiency.
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


class MetaNet(nn.Module):
    """
    Lightweight conditioner from CoCoOp: maps visual features to a context shift.
    """

    def __init__(self, vis_dim, ctx_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or max(1, vis_dim // 16)
        self.net = nn.Sequential(
            nn.Linear(vis_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, ctx_dim),
        )

    def forward(self, img_feat):
        return self.net(img_feat)


class CoCoOpLearner(nn.Module):
    """
    Instance-conditioned prompt learner (CoCoOp) that outputs per-image, per-class
    text embeddings given global visual features.
    """

    def __init__(self, clip_adapter: ConchAdapter, class_names: List[str], vis_dim: int, n_ctx: int = 4, ctx_init: Optional[str] = "a photo of a"):
        super().__init__()
        self.clip_adapter = clip_adapter
        self.class_names = [str(name).replace("_", " ")
                            for name in class_names]
        self.n_cls = len(self.class_names)
        self.n_ctx = n_ctx
        token_embedding = clip_adapter.get_token_embedding()
        if token_embedding is None:
            raise RuntimeError(
                "Unable to access CONCH token embedding for CoCoOp.")
        tok_weight = getattr(token_embedding, "weight", None)
        if tok_weight is None:
            raise RuntimeError("Token embedding missing weight parameter.")
        dtype = tok_weight.dtype
        dev = tok_weight.device
        embed_dim_tokens = tok_weight.shape[1]
        self.meta_net = MetaNet(vis_dim, embed_dim_tokens)

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            tokenized = clip_adapter.tokenize(ctx_init).to(dev)
            with torch.no_grad():
                embedding = token_embedding(tokenized).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :].clone()
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(self.n_ctx, embed_dim_tokens, dtype=dtype, device=dev)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * self.n_ctx)
        self.ctx = nn.Parameter(ctx_vectors)

        prompts = [prompt_prefix + " " + name +
                   "." for name in self.class_names]
        tokenized_prompts = clip_adapter.tokenize(prompts).to(dev)
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer(
            "token_prefix", embedding[:, :1, :], persistent=False)
        self.register_buffer(
            "token_suffix", embedding[:, 1 + self.ctx.shape[0]:, :], persistent=False)

    def forward(self, img_features: torch.Tensor) -> torch.Tensor:
        """
        img_features: [B, vis_dim] visual features (no grad to CLIP required).
        Returns text embeddings: [B, n_cls, embed_dim]
        """
        b = img_features.shape[0]
        bias = self.meta_net(img_features).unsqueeze(1)  # [B, 1, D]
        ctx = self.ctx.unsqueeze(0) + bias  # [B, n_ctx, D]

        prefix = self.token_prefix.unsqueeze(0).expand(b, -1, -1, -1)
        suffix = self.token_suffix.unsqueeze(0).expand(b, -1, -1, -1)
        ctx_expanded = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        prompts = torch.cat([prefix, ctx_expanded, suffix],
                            dim=2)  # [B, n_cls, L, D]
        prompts_flat = prompts.view(-1, prompts.shape[2], prompts.shape[3])
        text_features = self.clip_adapter.encode_text_with_embeddings(
            prompts_flat)
        text_features = text_features.view(b, self.n_cls, -1)
        return text_features


class AdaptiveLayer(nn.Module):
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
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TextPromptEncoder(nn.Module):
    """
    Lightweight wrapper around MedCLIP's text encoder to turn a list of
    class descriptions into normalized embeddings that can be learned via
    a small projection head.
    """

    def __init__(
        self,
        cls_num_classes: int,
        prompts: Optional[Union[List[str], str]],
        clip_adapter: ConchAdapter,
        projection_dim: Optional[int] = None,
        learnable_prompt: bool = False,
        prompt_init_scale: float = 0.02,
        use_ctx_prompt: bool = False,
        ctx_prompt_len: int = 8,
        ctx_class_specific: bool = False,
    ):
        super().__init__()
        self.clip_adapter = clip_adapter
        self.learnable_prompt = learnable_prompt
        # Context prompts are not wired for CONCH text yet.
        self.use_ctx_prompt = False
        self.n_ctx = ctx_prompt_len
        self.ctx_class_specific = ctx_class_specific

        prompts = prompts or self._build_default_prompts(cls_num_classes)
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts]
        prompts = [str(p) for p in prompts]
        self.prompts = prompts
        self.register_buffer("prompt_token_ids",
                             clip_adapter.tokenize(prompts), persistent=False)

        self.text_dim = clip_adapter.embed_dim
        proj_dim = projection_dim or self.text_dim
        if proj_dim != self.text_dim:
            self.text_proj = nn.Sequential(
                nn.Linear(self.text_dim, proj_dim),
                nn.ReLU(),
                nn.LayerNorm(proj_dim),
            )
        else:
            self.text_proj = nn.Identity()

        self.prompt_delta = None
        if learnable_prompt:
            delta = torch.zeros(cls_num_classes, proj_dim)
            nn.init.trunc_normal_(delta, std=prompt_init_scale)
            self.prompt_delta = nn.Parameter(delta)

    @staticmethod
    def _build_default_prompts(cls_num_classes):
        base_prompts = [
            "Histopathology patch showing invasive tumor epithelium.",
            "Histopathology patch rich in fibrous stroma and connective tissue.",
            "Histopathology patch dominated by lymphocytes and immune cells.",
            "Histopathology patch with necrotic tissue and cellular debris.",
        ]
        if cls_num_classes <= len(base_prompts):
            return base_prompts[:cls_num_classes]

        for idx in range(len(base_prompts), cls_num_classes):
            base_prompts.append(f"Histopathology patch for class {idx}.")
        return base_prompts

    def forward(self, device):
        token_ids = self.prompt_token_ids.to(device)
        base = self.clip_adapter.encode_text_tokens(token_ids, normalize=True)
        base = self.text_proj(base)
        if self.prompt_delta is not None:
            base = base + self.prompt_delta.to(device)
        return F.normalize(base, dim=-1)


class TextGuidedCamFusion(nn.Module):
    """
    Computes per-class attention over CAM scales using text prompts as queries
    and pooled image/prototype features as keys/values.
    """

    def __init__(
        self,
        cls_num_classes,
        level_dims,
        prototype_feature_dim,
        clip_adapter: ConchAdapter,
        fusion_dim=None,
        prompts=None,
        learnable_prompt=False,
        prompt_init_scale=0.02,
        use_ctx_prompt=False,
        ctx_prompt_len=8,
        ctx_class_specific=False,
    ):
        super().__init__()
        self.cls_num_classes = cls_num_classes
        self.fusion_dim = fusion_dim or prototype_feature_dim

        self.text_encoder = TextPromptEncoder(
            cls_num_classes=cls_num_classes,
            prompts=prompts,
            projection_dim=self.fusion_dim,
            clip_adapter=clip_adapter,
            learnable_prompt=learnable_prompt,
            prompt_init_scale=prompt_init_scale,
            use_ctx_prompt=use_ctx_prompt,
            ctx_prompt_len=ctx_prompt_len,
            ctx_class_specific=ctx_class_specific,
        )

        # Project concatenated image feature + prototype context to fusion_dim
        self.image_projectors = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim * 2),
                nn.Linear(dim * 2, fusion_dim),
                nn.ReLU()
            ) for dim in level_dims
        ])

        # Learnable temperature for the attention logits
        self.logit_scale = nn.Parameter(torch.ones(1))

    def forward(self, pooled_features, projected_prototypes, return_text=False):
        """
        pooled_features: list of [B, C] pooled image features per level
        projected_prototypes: list of [P, C] projected prototypes per level
        """
        if len(pooled_features) != len(projected_prototypes):
            raise ValueError(
                "pooled_features and projected_prototypes must have the same length")

        device = pooled_features[0].device
        text_features = self.text_encoder(device)  # [num_classes, fusion_dim]
        text_features = F.normalize(text_features, dim=-1)

        image_contexts = []
        for pooled, proto, projector in zip(pooled_features, projected_prototypes, self.image_projectors):
            proto_ctx = proto.mean(dim=0, keepdim=True).expand(
                pooled.shape[0], -1)
            fused = torch.cat([pooled, proto_ctx], dim=1)
            fused = projector(fused)
            image_contexts.append(F.normalize(fused, dim=-1))

        # [B, num_levels, fusion_dim]
        image_contexts = torch.stack(image_contexts, dim=1)
        logits = torch.einsum('kd,bld->bkl', text_features, image_contexts)
        logits = logits / math.sqrt(self.fusion_dim)
        logits = logits * torch.clamp(self.logit_scale, min=1e-4)
        weights = torch.softmax(logits, dim=-1)  # [B, num_classes, num_levels]
        if return_text:
            return weights, text_features
        return weights


class ClsNetwork(nn.Module):
    def __init__(self,
                 backbone='mit_b1',
                 cls_num_classes=4,
                 num_prototypes_per_class=10,
                 prototype_feature_dim=768,
                 clip_adapter: Optional[ConchAdapter] = None,
                 stride=[4, 2, 2, 1],
                 pretrained=True,
                 n_ratio=0.5,
                 enable_text_fusion=True,
                 text_prompts=None,
                 fusion_dim=None,
                 learnable_text_prompt=True,
                 prompt_init_scale=0.02,
                 # ["random", "text_fixed", "text_learnable", "text_prompt_tuned"]
                 prototype_init_mode="text_learnable",
                 prototype_text_noise_std=0.02,
                 use_ctx_prompt=False,
                 ctx_prompt_len=8,
                 ctx_class_specific=False,
                 enable_segformer_guidance=True,
                 segformer_backbone="mit_b1",
                 segformer_checkpoint: Optional[str] = None,
                 guidance_layers=(2,),
                 train_clip_visual: Optional[bool] = None,
                 input_mean: Optional[Union[list, tuple]] = None,
                 input_std: Optional[Union[list, tuple]] = None,
                 use_structure_adapter: bool = False,
                 enable_cocoop: bool = False,
                 cocoop_n_ctx: int = 4,
                 cocoop_ctx_init: Optional[str] = "a photo of a",
                 cocoop_class_names: Optional[List[str]] = None,
                 ):
        super().__init__()
        self.cls_num_classes = cls_num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.total_prototypes = cls_num_classes * num_prototypes_per_class
        self.stride = stride
        self.cam_fusion_levels = (1, 2, 3)  # use scales 2,3,4 for fusion
        self.clip_adapter = clip_adapter
        self.clip_visual_dim = None
        raw_guidance_layers = guidance_layers if guidance_layers is not None else (
            2,)
        self.guidance_layers = tuple(
            int(g) for g in raw_guidance_layers if isinstance(g, int) and g >= 0)

        self.prototype_feature_dim = prototype_feature_dim

        # Backbone Encoder
        self.use_clip_visual = backbone.startswith("conch")
        if self.use_clip_visual and self.clip_adapter is None:
            raise ValueError(
                "Backbone set to CONCH but clip_adapter is missing.")
        self.train_clip_visual = train_clip_visual
        if self.train_clip_visual is None:
            self.train_clip_visual = self.clip_adapter is not None and not getattr(
                self.clip_adapter, "freeze", True)
        self.enable_segformer_guidance = bool(
            enable_segformer_guidance and self.use_clip_visual)
        self.enable_cocoop = bool(
            enable_cocoop and self.use_clip_visual and self.clip_adapter is not None)
        self.segformer_teacher = None
        self.structural_loss_fn = StructuralConsistencyLoss(
        ) if self.enable_segformer_guidance else None
        if self.enable_segformer_guidance and not self.guidance_layers:
            self.guidance_layers = (2,)
        self.encoder = None
        if self.use_clip_visual:
            trunk = getattr(getattr(self.clip_adapter.model,
                            "visual", None), "trunk", None)
            visual_width = (
                getattr(trunk, "embed_dim", None)
                or getattr(trunk, "num_features", None)
                or self.prototype_feature_dim
            )
            self.clip_visual_dim = visual_width
            self.in_channels = [visual_width] * 4
            self.cocoop_learner = None
            if self.enable_cocoop:
                names = cocoop_class_names
                if names is None:
                    if text_prompts is not None and isinstance(text_prompts, (list, tuple)):
                        names = [str(p) for p in text_prompts]
                    else:
                        names = [f"Class {i}" for i in range(cls_num_classes)]
                self.cocoop_learner = CoCoOpLearner(
                    clip_adapter=self.clip_adapter,
                    class_names=names,
                    vis_dim=self.clip_adapter.embed_dim,
                    n_ctx=cocoop_n_ctx,
                    ctx_init=cocoop_ctx_init,
                )

            if self.enable_segformer_guidance:
                self.segformer_teacher = getattr(
                    mix_transformer, segformer_backbone)(stride=self.stride)
                teacher_ckpt = segformer_checkpoint or (
                    f"./pretrained/{segformer_backbone}.pth" if pretrained else None)
                if teacher_ckpt and os.path.exists(teacher_ckpt):
                    state_dict = torch.load(teacher_ckpt, map_location="cpu")
                    state_dict = {k: v for k, v in state_dict.items(
                    ) if k in self.segformer_teacher.state_dict()}
                    self.segformer_teacher.load_state_dict(
                        state_dict, strict=False)
                elif teacher_ckpt:
                    raise FileNotFoundError(
                        f"SegFormer checkpoint not found at {teacher_ckpt}. Aborting to avoid distilling from random weights.")
                for param in self.segformer_teacher.parameters():
                    param.requires_grad_(False)
                self.segformer_teacher.eval()
        else:
            self.encoder = getattr(
                mix_transformer, backbone)(stride=self.stride)
            self.in_channels = self.encoder.embed_dims

            # Loads pre-trained weights for the backbone (Same as original)
            if pretrained:
                state_dict = torch.load(
                    './pretrained/'+backbone+'.pth', map_location="cpu")
                state_dict.pop('head.weight', None)
                state_dict.pop('head.bias', None)
                state_dict = {k: v for k, v in state_dict.items(
                ) if k in self.encoder.state_dict().keys()}
                self.encoder.load_state_dict(state_dict, strict=False)

        # Learnable Prototypes
        # Instead of loading from a file, we create prototypes as a learnable parameter
        # The optimizer will update these vectors during training
        self.prototypes = nn.Parameter(torch.randn(
            self.total_prototypes, self.prototype_feature_dim), requires_grad=True)

        # Adaptive Layers to Project Prototypes
        # These now project the learnable prototypes to match each of the four feature scales
        self.l_fc1 = AdaptiveLayer(
            self.prototype_feature_dim, n_ratio, self.in_channels[0])
        self.l_fc2 = AdaptiveLayer(
            self.prototype_feature_dim, n_ratio, self.in_channels[1])
        self.l_fc3 = AdaptiveLayer(
            self.prototype_feature_dim, n_ratio, self.in_channels[2])
        self.l_fc4 = AdaptiveLayer(
            self.prototype_feature_dim, n_ratio, self.in_channels[3])

        # Other components from the original model are kept for compatibility
        self.pooling = F.adaptive_avg_pool2d

        # The k_list (number of prototypes per class) is now generated programmatically.
        # This is needed by the loss function in the training script.
        self.k_list = [self.num_prototypes_per_class] * self.cls_num_classes
        self.use_structure_adapter = bool(
            use_structure_adapter and self.use_clip_visual)
        self.structure_adapters = None
        if self.use_structure_adapter:
            self.structure_adapters = nn.ModuleList(
                [FeatureRefinementHead(ch) for ch in self.in_channels])

        # The learnable temperature parameters for cosine similarity are kept.
        self.logit_scale1 = nn.parameter.Parameter(
            torch.ones([1]) * (1 / 0.07))
        self.logit_scale2 = nn.parameter.Parameter(
            torch.ones([1]) * (1 / 0.07))
        self.logit_scale3 = nn.parameter.Parameter(
            torch.ones([1]) * (1 / 0.07))
        self.logit_scale4 = nn.parameter.Parameter(
            torch.ones([1]) * (1 / 0.07))

        self.text_fusion = None
        self.prototype_init_mode = prototype_init_mode
        self.prototype_text_noise_std = prototype_text_noise_std
        self.prototype_initialized = prototype_init_mode == "random"
        # Input normalization stats (used to re-normalize for SegFormer teacher)
        mean_list = input_mean if input_mean is not None else [0.0, 0.0, 0.0]
        std_list = input_std if input_std is not None else [1.0, 1.0, 1.0]
        self.register_buffer("input_mean", torch.tensor(
            mean_list, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("input_std", torch.tensor(
            std_list, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
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
        if enable_text_fusion:
            fusion_dim_val = fusion_dim or self.prototype_feature_dim
            level_dims = [self.in_channels[idx]
                          for idx in self.cam_fusion_levels]
            self.text_fusion = TextGuidedCamFusion(
                cls_num_classes=cls_num_classes,
                level_dims=level_dims,
                prototype_feature_dim=self.prototype_feature_dim,
                clip_adapter=self.clip_adapter,
                fusion_dim=fusion_dim_val,
                prompts=text_prompts,
                learnable_prompt=learnable_text_prompt,
                prompt_init_scale=prompt_init_scale,
                use_ctx_prompt=use_ctx_prompt,
                ctx_prompt_len=ctx_prompt_len,
                ctx_class_specific=ctx_class_specific,
            )

    def get_param_groups(self):
        regularized = []
        not_regularized = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

    def init_prototypes_from_text(self, device):
        if self.prototype_initialized or self.prototype_init_mode == "random":
            return None
        if self.text_fusion is None:
            return None
        with torch.no_grad():
            text_feats = self.text_fusion.text_encoder(device)  # [C, D]
            base = text_feats[:, None, :].repeat(
                1, self.num_prototypes_per_class, 1)
            base = base.view(self.total_prototypes, -1)
            if self.prototype_text_noise_std > 0:
                noise = torch.randn_like(base) * self.prototype_text_noise_std
                base = base + noise
            self.prototypes.data = base.to(device)
            if self.prototype_init_mode == "text_fixed":
                self.prototypes.requires_grad_(False)
            self.prototype_initialized = True
        return text_feats

    def forward(self, x):
        text_feats_init = self.init_prototypes_from_text(x.device)
        cocoop_text_feats = None
        distill_loss = None
        teacher_features = None
        if self.segformer_teacher is not None and self.training:
            self.segformer_teacher.eval()
            with torch.no_grad():
                x_raw = x * \
                    self.input_std.to(x.device) + self.input_mean.to(x.device)
                x_seg = (x_raw - self.teacher_mean.to(x.device)) / \
                    self.teacher_std.to(x.device)
                teacher_features, _ = self.segformer_teacher(x_seg)
        # Extract multi-scale feature maps
        if self.use_clip_visual:
            x_raw = x * self.input_std.to(x.device) + \
                self.input_mean.to(x.device)
            conch_mean = torch.tensor(
                [0.5, 0.5, 0.5], device=x.device).view(1, 3, 1, 1)
            conch_std = torch.tensor(
                [0.5, 0.5, 0.5], device=x.device).view(1, 3, 1, 1)
            x_conch = (x_raw - conch_mean) / conch_std
            cocoop_text_feats = None
            if self.enable_cocoop and self.cocoop_learner is not None:
                with torch.no_grad():
                    global_img_feat = self.clip_adapter.encode_image(
                        x_conch, normalize=True)
                cocoop_text_feats = self.cocoop_learner(global_img_feat)
            if self.use_structure_adapter:
                with torch.no_grad():
                    _x_all = self.clip_adapter.visual_intermediates(
                        x_conch, use_grad=False
                    )
                    if not _x_all:
                        raise RuntimeError(
                            "CONCH visual encoder did not return any feature maps.")
                    while len(_x_all) < 4:
                        _x_all.append(F.avg_pool2d(
                            _x_all[-1], kernel_size=2, stride=2))
                # Detach to ensure gradients flow only into adapters
                _x_all = [f.detach() for f in _x_all]
                _x_all = [adapter(fmap) for adapter, fmap in zip(
                    self.structure_adapters, _x_all)]
            else:
                _x_all = self.clip_adapter.visual_intermediates(
                    x_conch, use_grad=(self.train_clip_visual or (
                        self.enable_segformer_guidance and self.training))
                )
            if not _x_all:
                raise RuntimeError(
                    "CONCH visual encoder did not return any feature maps.")
            while len(_x_all) < 4:
                _x_all.append(F.avg_pool2d(
                    _x_all[-1], kernel_size=2, stride=2))
        else:
            _x_all, _ = self.encoder(x)

        if self.structural_loss_fn is not None and teacher_features is not None:
            guided_losses = []
            for idx in self.guidance_layers:
                if idx >= len(_x_all) or idx >= len(teacher_features):
                    continue
                guided_losses.append(self.structural_loss_fn(
                    _x_all[idx], teacher_features[idx]))
            if guided_losses:
                distill_loss = sum(guided_losses) / len(guided_losses)

        imshapes = [f.shape for f in _x_all]
        feats_flat = [
            f.permute(0, 2, 3, 1).reshape(f.shape[0], -1, f.shape[1]) for f in _x_all
        ]  # [B, HW, C] per level

        # Build prototypes: dynamic (CoCoOp) + static offsets, or static only
        protos_dynamic = None
        if self.enable_cocoop and cocoop_text_feats is not None:
            k = self.num_prototypes_per_class
            proto_offset = self.prototypes.view(1, self.cls_num_classes, k, -1)
            dynamic = cocoop_text_feats.unsqueeze(2).expand(-1, -1, k, -1)
            protos_dynamic = (dynamic + proto_offset).reshape(
                cocoop_text_feats.shape[0], -1, self.prototype_feature_dim
            )  # [B, total, D]
        protos_static = self.prototypes  # [total, D]

        # Project prototypes (static for compatibility, dynamic for CoCoOp logits)
        projected_static = (
            self.l_fc1(protos_static),
            self.l_fc2(protos_static),
            self.l_fc3(protos_static),
            self.l_fc4(protos_static),
        )
        projected_dynamic = None
        if protos_dynamic is not None:
            projected_dynamic = (
                self.l_fc1(protos_dynamic),
                self.l_fc2(protos_dynamic),
                self.l_fc3(protos_dynamic),
                self.l_fc4(protos_dynamic),
            )

        def compute_logits(feats, protos, scale):
            """
            feats: [B, HW, C_feat]
            protos: [total, C_proto] or [B, total, C_proto]
            returns: [B, total, HW]
            """
            feats_norm = feats / (feats.norm(dim=-1, keepdim=True) + 1e-6)
            if protos.dim() == 2:
                proto_norm = protos / \
                    (protos.norm(dim=-1, keepdim=True) + 1e-6)
                logits = torch.matmul(
                    feats_norm, proto_norm.t().float())  # [B, HW, P]
            else:
                proto_norm = protos / \
                    (protos.norm(dim=-1, keepdim=True) + 1e-6)
                logits = torch.bmm(
                    feats_norm, proto_norm.transpose(1, 2))  # [B, HW, P]
            logits = logits.permute(0, 2, 1)  # [B, P, HW]
            return scale * logits

        def logits_to_map(logits, h, w):
            return logits.view(logits.shape[0], logits.shape[1], h, w)

        # Select which prototype set to use for logits/CAMs
        p_set = projected_dynamic if projected_dynamic is not None else projected_static

        logits1 = compute_logits(feats_flat[0], p_set[0], self.logit_scale1)
        out1 = logits_to_map(logits1, imshapes[0][2], imshapes[0][3])
        cam1 = out1.clone().detach()
        cls1 = self.pooling(out1, (1, 1)).view(out1.shape[0], -1)

        logits2 = compute_logits(feats_flat[1], p_set[1], self.logit_scale2)
        out2 = logits_to_map(logits2, imshapes[1][2], imshapes[1][3])
        cam2 = out2.clone().detach()
        cls2 = self.pooling(out2, (1, 1)).view(out2.shape[0], -1)

        logits3 = compute_logits(feats_flat[2], p_set[2], self.logit_scale3)
        out3 = logits_to_map(logits3, imshapes[2][2], imshapes[2][3])
        cam3 = out3.clone().detach()
        cls3 = self.pooling(out3, (1, 1)).view(out3.shape[0], -1)

        logits4 = compute_logits(feats_flat[3], p_set[3], self.logit_scale4)
        out4 = logits_to_map(logits4, imshapes[3][2], imshapes[3][3])
        cam4 = out4.clone()
        cls4 = self.pooling(out4, (1, 1)).view(out4.shape[0], -1)

        feature_map_for_diversity = _x_all[3]

        cam_weights = None
        text_features_out = cocoop_text_feats if cocoop_text_feats is not None else text_feats_init
        if self.text_fusion is not None:
            pooled_feats = [
                self.pooling(_x_all[idx], (1, 1)).flatten(1)
                for idx in self.cam_fusion_levels
            ]
            proto_levels = [projected_static[1],
                            projected_static[2], projected_static[3]]
            fusion_out = self.text_fusion(
                pooled_feats, proto_levels, return_text=True)
            cam_weights, text_features_out = fusion_out

        return (cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4,
                self.prototypes, self.k_list, feature_map_for_diversity, cam_weights, projected_static[3], text_features_out, distill_loss)
