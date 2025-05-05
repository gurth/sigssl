import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from typing import Optional

from .patch_embed import PatchEmbed

from .conformer.convolution import (
    ConformerConvModule,
    Conv2dSubampling,
)
from .conformer.modules import Linear, ResidualConnectionModule
from .conformer.feed_forward import FeedForwardModule
from .conformer.attention import RelativeMultiHeadAttention
from .conformer.embedding import PositionalEncoding

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv1d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class MMSA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1):
        super(MMSA, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, pre_block: Tensor, mask: Optional[Tensor] = None):
        batch_size, seq_length, _ = inputs.size()
        pos_embedding = self.positional_encoding(seq_length)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, pre_block, pos_embedding=pos_embedding, mask=mask)

        return self.dropout(outputs)

class MRCM(nn.Module):
    """
    Residual Connection Module.
    outputs = (module(inputs) x module_factor + inputs x input_factor)
    """
    def __init__(self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0):
        super(MRCM, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: Tensor, pre_block: Tensor) -> Tensor:
        return (self.module(inputs, pre_block) * self.module_factor) + (inputs * self.input_factor)

class CNNBiLSTMSABlock(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 512,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
    ):
        super(CNNBiLSTMSABlock, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.layer1 = ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            )

        self.layer2 = MRCM(
                module=MMSA(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                ),
            )

        self.layer3 = ResidualConnectionModule(
                module=ConformerConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                ),
            )

        self.layer4 = ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            )

        self.layer_norm = nn.LayerNorm(encoder_dim)



    def forward(self, inputs: Tensor) -> Tensor:
        x = self.layer1(inputs)
        x = self.layer2(x, inputs)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer_norm(x)

        return x

class mConforod(nn.Module):
    def __init__(self,
                 settings,
                 heads,
                 head_conv,
                 in_chans=2,
                 final_kernel=1,
                 wavelet_setting=None):
        super(mConforod, self).__init__()
        self.heads = heads
        self.head_conv = head_conv
        self.wavelet_setting = wavelet_setting

        input_dim = settings['input_dim']
        output_dim = settings['encoder_dim']

        wt_scale = 1
        if self.wavelet_setting is not None:
            if not self.wavelet_setting["cuda"]:
                wt_scale = 2 ** self.wavelet_setting['level']

        # Patch embedding layer
        self.patch_embed = PatchEmbed(
            patch_size=settings["patch_size"] // wt_scale, in_chans=in_chans * wt_scale, embed_dim=input_dim
        )

        # Downsample
        self.conv_subsample = Conv2dSubampling(in_channels=1, out_channels=output_dim)
        self.input_projection = nn.Sequential(
            Linear(output_dim * (((input_dim - 1) // 2 - 1) // 2), output_dim),
            nn.Dropout(p=0.1),
        )

        self.layer_num = settings['num_encoder_layers']
        # self.layers = nn.ModuleList([CNNBiLSTMSA(
        #     input_dim=settings['lstm_dim'],
        #     lstm_dim=settings['lstm_dim'],
        #     num_attention_heads=settings['num_attention_heads'],
        #     conv_kernel_size=settings['conv_kernel_size'],
        # ) for _ in range(self.layer_num)])

        self.layers = nn.ModuleList([CNNBiLSTMSABlock(
            encoder_dim=settings['encoder_dim'],
            num_attention_heads=settings['num_attention_heads'],
            conv_kernel_size=settings['conv_kernel_size'],
        ) for _ in range(self.layer_num)])

        # Predict hm, wt, reg from feature vector
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
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

        outputs, output_lengths = self.conv_subsample(inputs, input_lengths)
        outputs = self.input_projection(outputs)

        # outputs = self.cnn_bilstm_sa(outputs.transpose(1, 2))
        for layer in self.layers:
            outputs = layer(outputs)

        z = {}
        for head in self.heads:
            x = outputs.transpose(1, 2)
            # x = encoder_outputs.view(encoder_outputs.size(0), -1)
            # x = self.pre_fc(x)
            # x = x.view(-1, 10)
            # x = x.transpose(1, 2)

            z[head] = self.__getattr__(head)(x)
        return z


def get_pose_mconformer(heads, head_conv, model_scale='medium', wavelet_setting=None):
    model_settings = {
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

    return mConforod(settings=model_settings[model_scale],
                    heads=heads,
                    head_conv=head_conv,
                    wavelet_setting=wavelet_setting
                    )
