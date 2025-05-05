import math
import torch
from torch import nn
from typing import Optional

from ...utils.nested_tensor import NestedTensor

class PositionEmbeddingSine(nn.Module):
    """
    Standard version of the position embedding for 1D I/Q data.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        x_embed = not_mask.cumsum(1, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            x_embed = (x_embed - 0.5) / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = pos_x.permute(0, 2, 1)
        return pos

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned, adapted for 1D I/Q data.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.pos_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.pos_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        length = x.shape[1]
        i = torch.arange(length, device=x.device)
        pos_emb = self.pos_embed(i)
        pos = pos_emb.unsqueeze(0).repeat(x.shape[0], 1, 1).permute(0, 2, 1)
        return pos

def build_position_encoding(args):
    N_steps = args.hidden_dim #// 2
    if args.position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding


