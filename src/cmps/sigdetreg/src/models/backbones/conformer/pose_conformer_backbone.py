from .encoder import ConformerEncoder


def pose_conformer_backbone(encoder_dim = 256,
                num_layers = 4,
                num_attention_heads = 8,
                feed_forward_expansion_factor = 4,
                conv_expansion_factor = 2,
                feed_forward_dropout_p = 0.1,
                attention_dropout_p = 0.1,
                conv_dropout_p = 0.1,
                conv_kernel_size = 31,
                half_step_residual = True):

    return ConformerEncoder(
            encoder_dim = encoder_dim,
            num_layers = num_layers,
            num_attention_heads = num_attention_heads,
            feed_forward_expansion_factor = feed_forward_expansion_factor,
            conv_expansion_factor = conv_expansion_factor,
            feed_forward_dropout_p = feed_forward_dropout_p,
            attention_dropout_p = attention_dropout_p,
            conv_dropout_p = conv_dropout_p,
            conv_kernel_size = conv_kernel_size,
            half_step_residual = half_step_residual,
    )
