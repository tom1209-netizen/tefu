import math
import pickle as pkl
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor

from model.segform import mix_transformer

# A simple MLP to project prototype features to the correct dimension.
# This is still required to map the single set of learnable prototypes
# to the different feature dimensions of the four backbone scales.
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
        cls_num_classes,
        prompts=None,
        projection_dim=512,
        learnable_prompt=False,
        prompt_init_scale=0.02,
    ):
        super().__init__()
        self.processor = MedCLIPProcessor()
        self.text_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT).text_model
        # Ensure the underlying BERT returns hidden states for pooling
        if hasattr(self.text_model.model, "config"):
            self.text_model.model.config.output_hidden_states = True
        # We keep the language encoder frozen for stability
        self.text_model.requires_grad_(False)
        self.text_model.eval()
        self.learnable_prompt = learnable_prompt

        prompts = prompts or self._build_default_prompts(cls_num_classes)
        # OmegaConf ListConfig or other iterables need to be cast to a plain list of strings
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts]
        prompts = [str(p) for p in prompts]
        tokenized = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        self.register_buffer("prompt_input_ids", tokenized["input_ids"], persistent=False)
        self.register_buffer("prompt_attention_mask", tokenized["attention_mask"], persistent=False)

        text_width = self.text_model.projection_head.out_features
        self.text_proj = nn.Sequential(
            nn.Linear(text_width, projection_dim),
            nn.ReLU(),
            nn.LayerNorm(projection_dim)
        )
        self.prompt_delta = None
        if learnable_prompt:
            delta = torch.zeros(cls_num_classes, text_width)
            nn.init.trunc_normal_(delta, std=prompt_init_scale)
            self.prompt_delta = nn.Parameter(delta)

        with torch.no_grad():
            text_embeds = self.text_model(
                input_ids=self.prompt_input_ids,
                attention_mask=self.prompt_attention_mask
            )
            text_embeds = self.text_proj(text_embeds)
            text_embeds = F.normalize(text_embeds, dim=-1)
        self.register_buffer("text_embedding", text_embeds, persistent=False)

    @staticmethod
    def _build_default_prompts(cls_num_classes):
        base_prompts = [
            "Histopathology patch showing invasive tumor epithelium.",
            "Histopathology patch rich in fibrous stroma and connective tissue.",
            "Histopathology patch dominated by lymphocytes and immune cells.",
            "Histopathology patch with necrotic tissue and cellular debris."
        ]
        if cls_num_classes <= len(base_prompts):
            return base_prompts[:cls_num_classes]
        # Extend with generic prompts if more classes are present
        for idx in range(len(base_prompts), cls_num_classes):
            base_prompts.append(f"Histopathology patch for class {idx}.")
        return base_prompts

    def forward(self, device):
        if hasattr(self, "text_embedding") and self.text_embedding is not None:
            base = self.text_embedding.to(device)
        else:
            input_ids = self.prompt_input_ids.to(device)
            attention_mask = self.prompt_attention_mask.to(device)
            with torch.no_grad():
                base = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
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
        fusion_dim=None,
        prompts=None,
        learnable_prompt=False,
        prompt_init_scale=0.02,
    ):
        super().__init__()
        self.cls_num_classes = cls_num_classes
        self.fusion_dim = fusion_dim or prototype_feature_dim

        self.text_encoder = TextPromptEncoder(
            cls_num_classes=cls_num_classes,
            prompts=prompts,
            projection_dim=self.fusion_dim,
            learnable_prompt=learnable_prompt,
            prompt_init_scale=prompt_init_scale,
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
            raise ValueError("pooled_features and projected_prototypes must have the same length")

        device = pooled_features[0].device
        text_features = self.text_encoder(device)  # [num_classes, fusion_dim]
        text_features = F.normalize(text_features, dim=-1)

        image_contexts = []
        for pooled, proto, projector in zip(pooled_features, projected_prototypes, self.image_projectors):
            proto_ctx = proto.mean(dim=0, keepdim=True).expand(pooled.shape[0], -1)
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
        prototype_feature_dim=512,
        stride=[4, 2, 2, 1],
        pretrained=True,
        n_ratio=0.5,
        enable_text_fusion=True,
        text_prompts=None,
        fusion_dim=None,
        learnable_text_prompt=True,
        prompt_init_scale=0.02,
        prototype_init_mode="text_learnable",  # ["random", "text_fixed", "text_learnable", "text_prompt_tuned"]
        prototype_text_noise_std=0.02,
    ):
        super().__init__()
        self.cls_num_classes = cls_num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.total_prototypes = cls_num_classes * num_prototypes_per_class
        self.stride = stride
        self.cam_fusion_levels = (1, 2, 3)  # use scales 2,3,4 for fusion

        # Backbone Encoder (Same as original)
        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        # Loads pre-trained weights for the backbone (Same as original)
        if pretrained:
            state_dict = torch.load('./pretrained/'+backbone+'.pth', map_location="cpu")
            state_dict.pop('head.weight', None)
            state_dict.pop('head.bias', None)
            state_dict = {k: v for k, v in state_dict.items() if k in self.encoder.state_dict().keys()}
            self.encoder.load_state_dict(state_dict, strict=False)

        # Learnable Prototypes 
        # Instead of loading from a file, we create prototypes as a learnable parameter
        # The optimizer will update these vectors during training
        self.prototypes = nn.Parameter(torch.randn(self.total_prototypes, prototype_feature_dim), requires_grad=True)

        # Adaptive Layers to Project Prototypes 
        # These now project the learnable prototypes to match each of the four feature scales
        self.l_fc1 = AdaptiveLayer(prototype_feature_dim, n_ratio, self.in_channels[0])
        self.l_fc2 = AdaptiveLayer(prototype_feature_dim, n_ratio, self.in_channels[1])
        self.l_fc3 = AdaptiveLayer(prototype_feature_dim, n_ratio, self.in_channels[2])
        self.l_fc4 = AdaptiveLayer(prototype_feature_dim, n_ratio, self.in_channels[3])

        # Other components from the original model are kept for compatibility
        self.pooling = F.adaptive_avg_pool2d
        
        # The k_list (number of prototypes per class) is now generated programmatically.
        # This is needed by the loss function in the training script.
        self.k_list = [self.num_prototypes_per_class] * self.cls_num_classes
        
        # The learnable temperature parameters for cosine similarity are kept.
        self.logit_scale1 = nn.parameter.Parameter(torch.ones([1]) * (1 / 0.07))
        self.logit_scale2 = nn.parameter.Parameter(torch.ones([1]) * (1 / 0.07))
        self.logit_scale3 = nn.parameter.Parameter(torch.ones([1]) * (1 / 0.07))
        self.logit_scale4 = nn.parameter.Parameter(torch.ones([1]) * (1 / 0.07))

        self.text_fusion = None
        self.prototype_init_mode = prototype_init_mode
        self.prototype_text_noise_std = prototype_text_noise_std
        self.prototype_initialized = prototype_init_mode == "random"
        if enable_text_fusion:
            fusion_dim_val = fusion_dim or prototype_feature_dim
            level_dims = [self.in_channels[idx] for idx in self.cam_fusion_levels]
            self.text_fusion = TextGuidedCamFusion(
                cls_num_classes=cls_num_classes,
                level_dims=level_dims,
                prototype_feature_dim=prototype_feature_dim,
                fusion_dim=fusion_dim_val,
                prompts=text_prompts,
                learnable_prompt=learnable_text_prompt,
                prompt_init_scale=prompt_init_scale,
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
            base = text_feats[:, None, :].repeat(1, self.num_prototypes_per_class, 1)
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
        # Passes the input image through the SegFormer backbone to get multi-scale feature maps
        _x_all, _ = self.encoder(x) 

        imshapes = [f.shape for f in _x_all]
        
        # Flattens the feature maps from [B, C, H, W] to [B*H*W, C] for matching
        image_features = [f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]) for f in _x_all]
        _x1, _x2, _x3, _x4 = image_features
    
        # Projects the single set of learnable prototypes to match the feature dimensions at each scale
        projected_p1 = self.l_fc1(self.prototypes)
        projected_p2 = self.l_fc2(self.prototypes)
        projected_p3 = self.l_fc3(self.prototypes)
        projected_p4 = self.l_fc4(self.prototypes)
        
        # Scale 1 Calculation
        # Normalize pixel features for cosine similarity calculation
        _x1_norm = _x1 / _x1.norm(dim=-1, keepdim=True)
        # Normalize the projected prototypes as well for true cosine similarity
        p1_norm = projected_p1 / projected_p1.norm(dim=-1, keepdim=True)
        # Calculate cosine similarity between each pixel feature and all projected prototypes
        logits1 = self.logit_scale1 * _x1_norm @ p1_norm.t().float()
        # Reshape the output back into a map [B, C, H, W], where C is the total number of prototypes
        out1 = logits1.view(imshapes[0][0], imshapes[0][2], imshapes[0][3], -1).permute(0, 3, 1, 2) 
        cam1 = out1.clone().detach() # The Class Activation Map (CAM)
        # Apply Global Average Pooling to the CAM to get the image-level classification score
        cls1 = self.pooling(out1, (1, 1)).view(-1, self.total_prototypes)

        # Scale 2 Calculation
        _x2_norm = _x2 / _x2.norm(dim=-1, keepdim=True)
        p2_norm = projected_p2 / projected_p2.norm(dim=-1, keepdim=True)
        logits2 = self.logit_scale2 * _x2_norm @ p2_norm.t().float() 
        out2 = logits2.view(imshapes[1][0], imshapes[1][2], imshapes[1][3], -1).permute(0, 3, 1, 2) 
        cam2 = out2.clone().detach()
        cls2 = self.pooling(out2, (1, 1)).view(-1, self.total_prototypes)

        # Scale 3 Calculation
        _x3_norm = _x3 / _x3.norm(dim=-1, keepdim=True)
        p3_norm = projected_p3 / projected_p3.norm(dim=-1, keepdim=True)
        logits3 = self.logit_scale3 * _x3_norm @ p3_norm.t().float() 
        out3 = logits3.view(imshapes[2][0], imshapes[2][2], imshapes[2][3], -1).permute(0, 3, 1, 2) 
        cam3 = out3.clone().detach()
        cls3 = self.pooling(out3, (1, 1)).view(-1, self.total_prototypes)

        # Scale 4 Calculation 
        _x4_norm = _x4 / _x4.norm(dim=-1, keepdim=True)
        p4_norm = projected_p4 / projected_p4.norm(dim=-1, keepdim=True)
        logits4 = self.logit_scale4 * _x4_norm @ p4_norm.t().float() 
        out4 = logits4.view(imshapes[3][0], imshapes[3][2], imshapes[3][3], -1).permute(0, 3, 1, 2) 
        # For the final layer's CAM, we keep the gradient attached for the contrastive loss
        cam4 = out4.clone()
        cls4 = self.pooling(out4, (1, 1)).view(-1, self.total_prototypes)

        feature_map_for_diversity = _x_all[3]

        cam_weights = None
        text_features_out = text_feats_init
        if self.text_fusion is not None:
            pooled_feats = [
                self.pooling(_x_all[idx], (1, 1)).flatten(1)
                for idx in self.cam_fusion_levels
            ]
            proto_levels = [projected_p2, projected_p3, projected_p4]
            fusion_out = self.text_fusion(pooled_feats, proto_levels, return_text=True)
            cam_weights, text_features_out = fusion_out

        return (cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, 
                self.prototypes, self.k_list, feature_map_for_diversity, cam_weights, projected_p4, text_features_out)
