import logging
import os
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Primary CONCH imports (ViT backbone)
try:
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
except ImportError as exc:  # pragma: no cover - only hits if dependency missing
    raise ImportError(
        "CONCH is not installed. Install with `pip install git+https://github.com/mahmoodlab/CONCH.git`."
    ) from exc


class ConchAdapter(nn.Module):
    """
    Thin wrapper around the CONCH CLIP model that exposes unified text/image
    encoding plus optional multi-scale visual features (for hierarchical heads).
    """

    def __init__(
        self,
        model_name: str,
        checkpoint_path: Optional[str],
        device: torch.device,
        variant: str = "vit",
        force_image_size: Optional[int] = None,
        cache_dir: Optional[str] = None,
        hf_hub: Optional[str] = None,
        hf_token: Optional[str] = None,
        proj_contrast: bool = False,
        freeze: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.variant = variant.lower()
        self.force_image_size = force_image_size
        self.cache_dir = cache_dir
        self.hf_hub = hf_hub
        self.hf_token = hf_token
        self.proj_contrast = proj_contrast
        self.freeze = freeze

        self.tokenizer = get_tokenizer()
        self.model, self.preprocess = self._load_model()

        # Expose key metadata
        self.embed_dim = getattr(self.model, "embed_dim", None)
        if self.embed_dim is None and hasattr(self.model, "text_projection"):
            self.embed_dim = self.model.text_projection.shape[1]
        if self.embed_dim is None:
            raise RuntimeError(
                "Unable to infer CONCH embed_dim from loaded model.")

        visual = getattr(self.model, "visual", None)
        self.image_size = getattr(
            visual, "image_size", force_image_size or 448)
        self.image_mean = getattr(visual, "image_mean", (0.5, 0.5, 0.5))
        self.image_std = getattr(visual, "image_std", (0.5, 0.5, 0.5))

        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()
        else:
            self.model.train()

    def _load_model(self):
        """
        Loads a CONCH model. The public release exposes the ViT backbone; a
        ResNet variant can be plugged in by pointing to a compatible checkpoint
        and config via `model_name`/`checkpoint_path`.
        """
        checkpoint = self.hf_hub or self.checkpoint_path
        model_cfg = self.model_name
        if checkpoint is None:
            raise ValueError(
                "Please provide `checkpoint_path` or `hf_hub` in cfg.clip to load CONCH weights.")

        model, preprocess = create_model_from_pretrained(
            model_cfg,
            checkpoint_path=checkpoint,
            device=self.device,
            force_image_size=self.force_image_size,
            cache_dir=self.cache_dir,
            hf_auth_token=self.hf_token,
        )
        return model, preprocess

    # Text
    def tokenize(self, texts: Union[str, Iterable[str]]) -> torch.Tensor:
        """Tokenize raw strings using CONCH's tokenizer."""
        if isinstance(texts, str):
            texts = [texts]
        token_ids = tokenize(self.tokenizer, list(texts))
        return token_ids

    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
    ) -> torch.Tensor:
        tokens = self.tokenize(texts).to(self.device)
        return self.encode_text_tokens(tokens, normalize=normalize)

    def encode_text_tokens(self, token_ids: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        with torch.no_grad():
            text_latent = self.model.encode_text(
                token_ids.to(self.device), normalize=normalize)
        return text_latent

    # CoCoOp helpers
    def get_token_embedding(self):
        """
        Expose raw token embedding layer for building custom prompts.
        """
        if hasattr(self.model, "token_embedding"):
            return self.model.token_embedding
        if hasattr(getattr(self.model, "transformer", None), "token_embedding"):
            return self.model.transformer.token_embedding
        return None

    def encode_text_with_embeddings(self, embeddings: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Encode text when token embeddings are preassembled. Gradients are not
        propagated into the CONCH text encoder.
        """
        model = self.model
        device = embeddings.device
        with torch.no_grad():
            x = embeddings
            pos_emb = getattr(model, "positional_embedding", None)
            if pos_emb is not None:
                pe = pos_emb
                if pe.shape[0] < x.shape[1]:
                    pe = F.interpolate(
                        pe.unsqueeze(0).permute(0, 2, 1),
                        size=x.shape[1],
                        mode="linear",
                        align_corners=False,
                    ).permute(0, 2, 1).squeeze(0)
                x = x + pe[: x.shape[1], :].to(device)

            if hasattr(model, "transformer"):
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = model.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD

            if hasattr(model, "ln_final"):
                x = model.ln_final(x)

            x = x[:, -1, :]

            text_proj = getattr(model, "text_projection", None)
            if text_proj is not None:
                x = x @ text_proj

            if normalize:
                x = F.normalize(x, dim=-1)
        return x

    # Image
    def encode_image(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Encode images in the CONCH image-text space.
        By default, returns embeddings before the contrastive projection head so they
        stay aligned with text embeddings when proj_contrast=False.
        """
        proj = self.proj_contrast
        with torch.no_grad():
            feats = self.model.encode_image(
                images.to(self.device), normalize=normalize, proj_contrast=proj)
        return feats

    # Multi-scale features
    def visual_intermediates(self, images: torch.Tensor, use_grad: bool = False) -> List[torch.Tensor]:
        """
        Build a 4-level feature pyramid by tapping intermediate ViT blocks
        (SETR/UPerNet style):
            L3 -> upsample x4  (approx 1/4 scale)
            L6 -> upsample x2  (approx 1/8 scale)
            L9 -> keep         (approx 1/16 scale)
            L12 -> avgpool x2  (approx 1/32 scale)

        If intermediate blocks are unavailable, it falls back to pooling the
        final token map.

        Args:
            images: Input batch in the same normalization space expected by CONCH.
            use_grad: When True, keep the computation graph so ViT blocks can be
                fine-tuned (needed for structural distillation). Default keeps
                gradients detached for efficiency.
        """
        images = images.to(self.device)

        def _tokens_to_map(tok: torch.Tensor) -> Optional[torch.Tensor]:
            # tok: [B, L, C] with CLS at position 0
            b, l, c = tok.shape
            side = int((l - 1) ** 0.5)
            if side * side != (l - 1):
                return None
            patch = tok[:, 1:, :].permute(0, 2, 1).reshape(b, c, side, side)
            return patch

        def _collect_vit_blocks(target_idxs: List[int]) -> List[torch.Tensor]:
            trunk = getattr(getattr(self.model, "visual", None), "trunk", None)
            if trunk is None or not hasattr(trunk, "blocks"):
                return []
            outputs: Dict[int, torch.Tensor] = {}
            hooks = []
            for idx in target_idxs:
                if idx >= len(trunk.blocks):
                    continue
                hooks.append(
                    trunk.blocks[idx].register_forward_hook(
                        lambda _, __, out, idx=idx: outputs.__setitem__(idx, out))
                )
            try:
                _ = trunk(images)
            finally:
                for h in hooks:
                    h.remove()
            return [outputs[i] for i in sorted(outputs.keys()) if i in outputs]

        # 0-indexed block taps for a 12-layer ViT-B
        target_layers = [2, 5, 8, 11]
        ctx = torch.enable_grad() if use_grad else torch.no_grad()
        with ctx:
            block_tokens = _collect_vit_blocks(target_layers)

        # Fallback if no block captures succeeded
        if not block_tokens:
            with ctx:
                _, tokens = self.model._encode_image(images, normalize=False)
            fmap = _tokens_to_map(tokens) if tokens is not None else None
            return [fmap] if fmap is not None else []

        maps = [m for m in (_tokens_to_map(tok)
                            for tok in block_tokens) if m is not None]
        if not maps:
            return []

        # Pad/trim to 4 levels
        while len(maps) < 4:
            maps.append(maps[-1])
        maps = maps[:4]

        # Build hierarchical resolutions
        lvl1 = F.interpolate(maps[0], scale_factor=4,
                             mode="bilinear", align_corners=False)
        lvl2 = F.interpolate(maps[1], scale_factor=2,
                             mode="bilinear", align_corners=False)
        lvl3 = maps[2]
        lvl4 = F.avg_pool2d(maps[3], kernel_size=2, stride=2)

        return [lvl1, lvl2, lvl3, lvl4]

    # Utility
    def normalize_for_conch(self, images: torch.Tensor, input_mean: Tuple[float, ...], input_std: Tuple[float, ...]):
        """
        Renormalize tensors that were standardized with dataset statistics back
        into CONCH's expected mean/std space. Both mean/std should be tuples of length 3.
        """
        device = images.device
        in_mean = torch.tensor(input_mean, device=device).view(1, -1, 1, 1)
        in_std = torch.tensor(input_std, device=device).view(1, -1, 1, 1)
        tgt_mean = torch.tensor(
            self.image_mean, device=device).view(1, -1, 1, 1)
        tgt_std = torch.tensor(self.image_std, device=device).view(1, -1, 1, 1)

        images = images * in_std + in_mean
        images = (images - tgt_mean) / tgt_std
        return images
