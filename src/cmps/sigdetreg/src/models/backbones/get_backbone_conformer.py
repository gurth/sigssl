from .conformer.pose_conformer_backbone import pose_conformer_backbone
from .backbone_with_patch_embed import BackboneWithPatchEmbed

from ..networks.detr_utils.position_encoding import build_position_encoding
from .utils import Joiner

def get_backbone_conformer(
        arg,
        encoder_dim=256,
        num_attention_heads=8,
        feed_forward_expansion_factor=4,
        conv_expansion_factor=2,
        dropout_p=0.1,
        attn_dropout_p=0.1,
        conv_dropout_p=0.1,
        conv_kernel_size=31,
        half_step_residual=True,
):
    position_embedding = build_position_encoding(arg)

    backbone = BackboneWithPatchEmbed(
        pose_conformer_backbone(encoder_dim=encoder_dim,
                                num_attention_heads=num_attention_heads,
                                feed_forward_expansion_factor=feed_forward_expansion_factor,
                                conv_expansion_factor=conv_expansion_factor,
                                feed_forward_dropout_p=dropout_p,
                                attention_dropout_p=attn_dropout_p,
                                conv_dropout_p=conv_dropout_p,
                                conv_kernel_size=conv_kernel_size,
                                half_step_residual=half_step_residual)

    )

    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels

    return model