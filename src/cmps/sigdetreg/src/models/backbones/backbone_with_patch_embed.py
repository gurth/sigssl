import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict
from ..utils.nested_tensor import NestedTensor


'''
Codes modified from SWIN Transformer
'''
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        seq_size (int): Sequence size.  Default: 131072.
        patch_size (int): Patch token size. Default: 32.
        in_chans (int): Number of input image channels. Default: 2.
        embed_dim (int): Number of linear projection output channels. Default: 80.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self,  patch_size=128, in_chans=2, embed_dim=256, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # TODO: Stride and overlapping
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class BackboneWithPatchEmbed(nn.Module):
    def __init__(self, backbone, train_backbone=True,
                    patch_size=32, in_chans=2, embed_dim=80,
                    load_backbone: str = ''
                 ):
        super(BackboneWithPatchEmbed, self).__init__()
        if load_backbone != '':
            backbone.load_state_dict(torch.load(load_backbone))
        if not train_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        self.backbone = backbone
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )

        self.num_channels = [256]

    def forward(self, tensor_list: NestedTensor):
        if isinstance(tensor_list, NestedTensor):
            xs = self.patch_embed(tensor_list.tensors.permute(0, 2, 1))
            input_lengths = xs.shape[-1]
            xs = xs.permute(0, 2, 1)

            xs, _ = self.backbone(xs, input_lengths)
            xs = xs.permute(0, 2, 1)

            out: Dict[str, NestedTensor] = {}

            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=xs.shape[-1:]).to(torch.bool)[0]
            out['0'] = NestedTensor(xs, mask)
        else:
            out = self.forward_non_nested(tensor_list)
        return out

    def forward_non_nested(self, tensors):
        xs = self.patch_embed(tensors)

        input_lengths = xs.shape[-1]
        xs = xs.permute(0, 2, 1)
        xs, _ = self.backbone(xs, input_lengths)

        out: Dict[str, NestedTensor] = {}

        out['0'] = xs.permute(0, 2, 1)
        return out