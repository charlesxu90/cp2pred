import torch
import torch.nn as nn
from torch_scatter import scatter
from einops.layers.torch import Rearrange
import torch.nn.functional as F

from .feature_encoder import FeatureEncoder
from .model_elements import Identity, MLP
from . import gMHA_wrapper
from . import gnn_wrapper
from loguru import logger

class GraphMLPMixer(nn.Module):

    def __init__(self,
                 nfeat_node, nfeat_edge,
                 nhid, nout,
                 nlayer_gnn,
                 nlayer_mlpmixer,
                 node_type, edge_type,
                 gnn_type,
                 gMHA_type='MLPMixer',
                 rw_dim=0,
                 lap_dim=0,
                 dropout=0,
                 mlpmixer_dropout=0,
                 bn=True,
                 res=True,
                 pooling='mean',
                 n_patches=32,
                 patch_rw_dim=0):

        super().__init__()
        self.dropout = dropout
        self.use_rw = rw_dim > 0
        self.use_lap = lap_dim > 0

        self.pooling = pooling
        self.res = res
        self.patch_rw_dim = patch_rw_dim

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(lap_dim, nhid, 1)
        if self.patch_rw_dim > 0:
            self.patch_rw_encoder = MLP(self.patch_rw_dim, nhid, 1)

        self.input_encoder = FeatureEncoder(node_type, nfeat_node, nhid)
        self.edge_encoder = FeatureEncoder(edge_type, nfeat_edge, nhid)

        self.gnns = nn.ModuleList([GNN(nin=nhid, nout=nhid, nlayer_gnn=1, gnn_type=gnn_type,
                                  bn=bn, dropout=dropout, res=res) for _ in range(nlayer_gnn)])
        self.U = nn.ModuleList(
            [MLP(nhid, nhid, nlayer=1, with_final_activation=True) for _ in range(nlayer_gnn-1)])

        self.reshape = Rearrange('(B p) d ->  B p d', p=n_patches)

        self.transformer_encoder = getattr(gMHA_wrapper, gMHA_type)(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)

        self.output_decoder = MLP(nhid, nout, nlayer=2, with_final_activation=False)

    def forward(self, data):
        x = self.input_encoder(data.x.squeeze())

        # Node PE
        if self.use_rw:
            x += self.rw_encoder(data.rw_pos_enc)
        if self.use_lap:
            x += self.lap_encoder(data.lap_pos_enc)
        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))
        edge_attr = self.edge_encoder(edge_attr)

        # Patch Encoder
        x = x[data.subgraphs_nodes_mapper]
        e = edge_attr[data.subgraphs_edges_mapper]
        edge_index = data.combined_subgraphs
        batch_x = data.subgraphs_batch
        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce=self.pooling)[batch_x]
                x = x + self.U[i-1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper,
                            dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            x = gnn(x, edge_index, e)
        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)

        # Patch PE
        if self.patch_rw_dim > 0:
            subgraph_x += self.patch_rw_encoder(data.patch_pe)
        mixer_x = self.reshape(subgraph_x)

        # MLPMixer
        mixer_x = self.transformer_encoder(mixer_x, data.coarsen_adj if hasattr(data, 'coarsen_adj') else None, ~data.mask)

        # Global Average Pooling
        x = (mixer_x * data.mask.unsqueeze(-1)).sum(1) / data.mask.sum(1, keepdim=True)
        # logger.info(f'x shape: {x.shape}')

        # Readout
        x = self.output_decoder(x)
        # logger.info(f'x shape: {x.shape}')
        return x



class GNN(nn.Module):
    def __init__(self,
                 nin,
                 nout,
                 nlayer_gnn,
                 gnn_type,
                 bn=True,
                 dropout=0.0,
                 res=True):
        super().__init__()
        self.dropout = dropout
        self.res = res

        self.convs = nn.ModuleList([getattr(gnn_wrapper, gnn_type)(
            nin, nin, bias=not bn) for _ in range(nlayer_gnn)])
        self.norms = nn.ModuleList(
            [nn.BatchNorm1d(nin) if bn else Identity() for _ in range(nlayer_gnn)])
        self.output_encoder = nn.Linear(nin, nout)

    def forward(self, x, edge_index, edge_attr):
        previous_x = x
        for layer, norm in zip(self.convs, self.norms):
            x = layer(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                x = x + previous_x
                previous_x = x

        x = self.output_encoder(x)
        return x
