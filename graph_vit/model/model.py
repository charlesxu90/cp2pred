import torch.nn as nn
from torch_scatter import scatter
from einops.layers.torch import Rearrange

from ogb.graphproppred.mol_encoder import AtomEncoder
from ogb.graphproppred.mol_encoder import BondEncoder
from .gnn import GNN, MLP
from .mlp_mixer import MLPMixer
from .vit_mixer import Hadamard
from loguru import logger


class GraphMLPMixer(nn.Module):
    def __init__(self, nout=1, hidden_size=128, nlayer_gnn=4, nlayer_mixer=2, 
                 rw_dim=0, dropout=0, mixer_dropout=0, mixer_type='vit',
                 bn=True, residual=True, pooling='mean', n_patches=32, patch_rw_dim=8, 
                 n_enc_layer=1, n_out_layer=2):
        super().__init__()
        self.dropout = dropout
        self.pooling = pooling
        self.res = residual

        # Patch encoding
        self.input_encoder = AtomEncoder(hidden_size)
        self.edge_encoder = BondEncoder(hidden_size)

        self.rw_encoder = MLP(rw_dim, hidden_size, nlayer=n_enc_layer)
        self.patch_rw_encoder = MLP(patch_rw_dim, hidden_size, nlayer=n_enc_layer)

        self.gnns = nn.ModuleList([GNN(nin=hidden_size, nout=hidden_size, nlayer_gnn=1,
                                       bn=bn, dropout=dropout, res=residual) for _ in range(nlayer_gnn)])
        self.U = nn.ModuleList([MLP(hidden_size, hidden_size, nlayer=1, with_final_activation=True) for _ in range(nlayer_gnn-1)])
        self.reshape = Rearrange('(B p) d ->  B p d', p=n_patches)

        # Graph encoding
        self.transformer_encoder = Hadamard(nhid=hidden_size, dropout=mixer_dropout, nlayer=nlayer_mixer, n_patches=n_patches) \
            if mixer_type == 'vit' else MLPMixer(nhid=hidden_size, dropout=mixer_dropout, nlayer=nlayer_mixer, n_patches=n_patches)
        self.output_decoder = MLP(hidden_size, nout, nlayer=n_out_layer, with_final_activation=False)

    def forward(self, data):
        x = self.get_embd(data)
        x = self.output_decoder(x)  # Readout
        # logger.info(f'x shape: {x.shape}')
        return x

    def get_embd(self, data):
        # Initial encoding
        x = self.input_encoder(data.x.squeeze())
        x += self.rw_encoder(data.rw_pos_enc)  # Node PE

        if data.edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))
        edge_attr = self.edge_encoder(data.edge_attr)

        # Patch node encoding
        x = x[data.subgraphs_nodes_mapper]
        e = edge_attr[data.subgraphs_edges_mapper]
        edge_index = data.combined_subgraphs
        batch_x = data.subgraphs_batch
        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0, reduce=self.pooling)[batch_x]
                x = x + self.U[i-1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper, dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            x = gnn(x, edge_index, e)
        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)

        # MLP Mixer
        subgraph_x += self.patch_rw_encoder(data.patch_pe)  # Patch PE
        mixer_x = self.reshape(subgraph_x)
        mixer_x = self.transformer_encoder(mixer_x, data.coarsen_adj if hasattr(data, 'coarsen_adj') else None, ~data.mask)

        x = (mixer_x * data.mask.unsqueeze(-1)).sum(1) / data.mask.sum(1, keepdim=True)   # Global Average Pooling
        return x
