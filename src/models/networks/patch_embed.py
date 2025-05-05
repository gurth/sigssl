import torch
import torch.nn as nn

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

    def __init__(self,  patch_size=32, in_chans=2, embed_dim=80, norm_layer=None):
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

        
