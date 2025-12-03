import torch
import torch.nn.functional as F


class FeatureMemoryBank:
    """
    FIFO queue of features to provide a large set of negatives independent of
    the current batch size.
    """

    def __init__(self, feature_dim: int, size: int = 2048, device: str = "cuda"):
        self.size = size
        self.dim = feature_dim
        self.device = device
        self.queue = torch.randn(size, feature_dim, device=device)
        self.queue = F.normalize(self.queue, dim=1)
        self.ptr = 0

    def push(self, features: torch.Tensor):
        """
        Add new features to the queue (replaces oldest entries).
        Accepts [B, D] or [B, L, D]; flattens automatically.
        """
        if features is None:
            return
        if features.dim() > 2:
            features = features.view(-1, self.dim)
        feats = F.normalize(features.detach(), dim=1)
        bsz = feats.shape[0]

        if bsz >= self.size:
            self.queue = feats[-self.size:]
            self.ptr = 0
            return

        end = self.ptr + bsz
        if end <= self.size:
            self.queue[self.ptr:end] = feats
            self.ptr = end % self.size
        else:
            overflow = end - self.size
            self.queue[self.ptr:] = feats[:-overflow]
            self.queue[:overflow] = feats[-overflow:]
            self.ptr = overflow

    def get_negatives(self) -> torch.Tensor:
        return self.queue.detach()
