import torch
import torch.nn.functional as F
from torch import nn, Tensor

class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        self.return_intermediate_dec = return_intermediate_dec
        self.normalize_before = normalize_before

        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                          dim_feedforward, dropout, activation)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, decoder_mask=None):
        # Flatten NxCxL to LxNxC
        bs, c, l = src.shape
        src = src.permute(2, 0, 1)
        pos_embed = pos_embed.permute(2, 0, 1)
        mask = mask.permute(1, 0)

        # Add positional embeddings to the source
        src = src + pos_embed

        tgt = torch.zeros_like(query_embed)
        memory = self.transformer.encoder(src, src_key_padding_mask=mask)

        # Add positional embeddings to the memory
        memory = memory + pos_embed

        hs = self._decode(tgt, memory, mask, pos_embed, query_embed, decoder_mask)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, l)

    def _decode(self, tgt, memory, mask, pos_embed, query_embed, decoder_mask):
        intermediate = []

        for i, layer in enumerate(self.transformer.decoder.layers):
            tgt2 = layer(tgt + query_embed, memory,
                         tgt_mask=decoder_mask,
                         memory_key_padding_mask=mask)
            if self.return_intermediate_dec:
                intermediate.append(tgt2)
            tgt = tgt2

        if self.return_intermediate_dec:
            return torch.stack(intermediate)

        return tgt.unsqueeze(0)

def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )

