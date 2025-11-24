import pickle as pkl
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

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


class ClsNetwork(nn.Module):
    def __init__(self,
        backbone='mit_b1',
        cls_num_classes=4,
        num_prototypes_per_class=10,
        prototype_feature_dim=512,
        stride=[4, 2, 2, 1],
        pretrained=True,
        n_ratio=0.5
    ):
        super().__init__()
        self.cls_num_classes = cls_num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.total_prototypes = cls_num_classes * num_prototypes_per_class
        self.stride = stride

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

    def forward(self, x):
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

        return (cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, 
                self.prototypes, self.k_list, feature_map_for_diversity)