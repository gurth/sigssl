import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from collections import namedtuple

from .conformer.encoder import ConformerEncoder
from .conformer.modules import Linear
from .patch_embed import PatchEmbed


class Conformer(nn.Module):
    """
    Conformer: Convolution-augmented Transformer for Speech Recognition
    The paper used a one-lstm Transducer decoder, currently still only implemented
    the conformer encoder shown in the paper.

    Args:
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of conformer encoder
        num_encoder_layers (int, optional): Number of conformer blocks
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths

    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by conformer.
        - **output_lengths** (batch): list of sequence output lengths
    """

    def __init__(
            self,
            input_dim: int = 80,
            encoder_dim: int = 512,
            num_encoder_layers: int = 17,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
            drop_path_rate=0.0
    ) -> None:
        super(Conformer, self).__init__()
        self.input_dim = input_dim
        self.encoder_dim = encoder_dim

        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            input_dropout_p=input_dropout_p,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
            drop_path_rate=drop_path_rate
        )

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return self.encoder.count_parameters()

    def update_dropout(self, dropout_p) -> None:
        """ Update dropout probability of model """
        self.encoder.update_dropout(dropout_p)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` and `targets` pair for training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically, for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)

        return encoder_outputs, encoder_output_lengths


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv1d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

# Modified from CenterNet (https://github.com/xingyizhou/CenterNet)
# Copyright (c) 2019 Xingyi Zhou. All rights reserved.

class Conforod(nn.Module):
    def __init__(
            self,
            settings,
            heads,
            head_conv,
            in_chans=2,
            final_kernel=1,
            wavelet_setting=None,
            drop_path_rate=0.0
    ):
        super(Conforod, self).__init__()
        self.heads = heads
        self.head_conv = head_conv
        self.wavelet_setting = wavelet_setting

        input_dim = settings['input_dim']
        output_dim = 144

        self.embed_dim = settings['encoder_dim']

        wt_scale = 1
        if self.wavelet_setting is not None:
            if not self.wavelet_setting["cuda"]:
                wt_scale = 2 ** self.wavelet_setting['level']

        # Patch embedding layer
        self.patch_embed = PatchEmbed(
            patch_size=settings["patch_size"] // wt_scale, in_chans=in_chans * wt_scale, embed_dim=input_dim
        )

        # Conformer encoder layer
        self.conformer = Conformer(
            input_dim=settings['input_dim'],
            encoder_dim=settings['encoder_dim'],
            num_encoder_layers=settings['num_encoder_layers'],
            num_attention_heads=settings['num_attention_heads'],
            conv_kernel_size=settings['conv_kernel_size'],
            drop_path_rate=drop_path_rate
        )

        self.backbone_prefixes =["patch_embed", "conformer"]

        # Predict hm, wt, reg from feature vector
        self.heads = heads

        if "ie" in self.heads:
            fc = nn.Linear(output_dim, 1, bias=False)
            self.__setattr__("ei", fc)
        else:
            for head in self.heads:
                classes = self.heads[head]

                if 'simclr' in head:
                    fc = nn.Sequential(
                        nn.AdaptiveAvgPool1d(1),
                        nn.Linear(output_dim, output_dim),
                        nn.ReLU(),
                        nn.Linear(output_dim, output_dim)
                    )

                if head_conv > 0:
                    fc = nn.Sequential(
                        nn.Conv1d(output_dim, head_conv,
                                  kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(head_conv, classes,
                                  kernel_size=final_kernel, stride=1,
                                  padding=final_kernel // 2, bias=True))
                    if 'hm' in head:
                        fc[-1].bias.data.fill_(-2.19)
                    else:
                        fill_fc_weights(fc)
                else:
                    fc = nn.Conv1d(output_dim, classes,
                                   kernel_size=final_kernel, stride=1,
                                   padding=final_kernel // 2, bias=True)
                    if 'hm' in head:
                        fc.bias.data.fill_(-2.19)
                    else:
                        fill_fc_weights(fc)

                self.__setattr__(head, fc)

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        inputs = self.patch_embed(inputs.transpose(1, 2)).transpose(1, 2)
        encoder_outputs, encoder_output_lengths = self.conformer(inputs, input_lengths)

        fields = self.heads
        if fields == {}:
            return encoder_outputs.transpose(1, 2)

        # 定义 namedtuple 类型
        HeadOutputs = namedtuple('HeadOutputs', fields)

        z = {}
        for head in self.heads:
            x = encoder_outputs.transpose(1, 2)
            # x = encoder_outputs.view(encoder_outputs.size(0), -1)
            # x = self.pre_fc(x)
            # x = x.view(-1, 10)
            # x = x.transpose(1, 2)

            z[head] = self.__getattr__(head)(x)

        z = HeadOutputs(**z)
        return z


def get_pose_conformer(heads, head_conv, model_scale='medium', wavelet_setting=None, drop_path_rate= 0.0):
    model_settings = {
        'mini': {
            "input_dim": 80,
            "encoder_dim": 144,
            "num_encoder_layers": 4,
            "num_attention_heads": 4,
            "conv_kernel_size": 31,
            "patch_size": 32,
        },
        'tiny': {
            "input_dim": 80,
            "encoder_dim": 144,
            "num_encoder_layers": 8,
            "num_attention_heads": 4,
            "conv_kernel_size": 31,
            "patch_size": 32,
        },
        'small': {
            "input_dim": 80,
            "encoder_dim": 144,
            "num_encoder_layers": 16,
            "num_attention_heads": 4,
            "conv_kernel_size": 31,
            "patch_size": 32,
        },
        'medium': {
            "input_dim": 80,
            "encoder_dim": 256,
            "num_encoder_layers": 16,
            "num_attention_heads": 4,
            "conv_kernel_size": 31,
            "patch_size": 32,
        },
        'large': {
            "input_dim": 80,
            "encoder_dim": 512,
            "num_encoder_layers": 17,
            "num_attention_heads": 8,
            "conv_kernel_size": 31,
            "patch_size": 32,
        }
        # Add more settings as needed
    }

    return Conforod(settings=model_settings[model_scale],
                    heads=heads,
                    head_conv=head_conv,
                    wavelet_setting=wavelet_setting,
                    drop_path_rate=drop_path_rate
                    )
