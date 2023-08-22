import torch.nn as nn
from einops.layers.torch import Rearrange
from .gMHA_hadamard import HadamardEncoderLayer


class MLPMixer(nn.Module):
    def __init__(self,
                 nhid,
                 nlayer,
                 n_patches,
                 dropout=0,
                 with_final_norm=True
                 ):
        super().__init__()
        self.n_patches = n_patches
        self.with_final_norm = with_final_norm
        self.mixer_blocks = nn.ModuleList(
            [MixerBlock(nhid, self.n_patches, nhid*4, nhid//2, dropout=dropout) for _ in range(nlayer)])
        if self.with_final_norm:
            self.layer_norm = nn.LayerNorm(nhid)

    def forward(self, x, coarsen_adj, mask):
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        if self.with_final_norm:
            x = self.layer_norm(x)
        return x


class Hadamard(nn.Module):
    # Hadamard attention (default): (A ⊙ softmax(QK^T/sqrt(d)))V
    def __init__(self, nhid, dropout, nlayer, n_patches, nhead=8, batch_first=True):
        super().__init__()
        self.transformer_encoder = nn.ModuleList([HadamardEncoderLayer(
            d_model=nhid, dim_feedforward=nhid*2, nhead=nhead, batch_first=batch_first, dropout=dropout)for _ in range(nlayer)])

    def forward(self, x, coarsen_adj, mask):
        for layer in self.transformer_encoder:
            x = layer(x, A=coarsen_adj, src_key_padding_mask=mask)
        return x


class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b p d -> b d p'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d p -> b p d'),
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)