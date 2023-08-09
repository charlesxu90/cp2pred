import logging
import torch
from torch import nn, optim
import torch.nn.functional as F
from .base_transformer import QuickGELU
from .vit import load_vit_model

logger = logging.getLogger(__name__)

class ConcatModel(nn.Module):
    def __init__(self, input_size, device='cuda', mlp_hid_size=None):
        super().__init__()
        self.mlp_hid_size = [256, 128, 64] if mlp_hid_size is None else [int(i) for i in mlp_hid_size.split(',')]

        mlp_layers = [[nn.Linear(input_size, self.mlp_hid_size[0]), nn.BatchNorm1d(self.mlp_hid_size[0]), QuickGELU()]] + \
            [[nn.Linear(self.mlp_hid_size[i], self.mlp_hid_size[i+1]), nn.BatchNorm1d(self.mlp_hid_size[i+1]), QuickGELU()]
             for i in range(len(self.mlp_hid_size) - 1)] + [[nn.Linear(self.mlp_hid_size[-1], 1)]]
        self.mlp = nn.Sequential(*[layer for layers in mlp_layers for layer in layers])
        self.device = device
        self.to(self.device)
        
    def forward(self, inputs):
        output = self.mlp(inputs)
        return output

    def configure_optimizers(self, learning_rate=1e-4):
        optimizer = optim.AdamW(params=self.parameters(), lr=learning_rate)
        return optimizer

class SmiModel(nn.Module):
    def __init__(self, input_size, device='cuda', mlp_hid_size=None):
        super().__init__()
        self.mlp_hid_size = [256, 128, 64] if mlp_hid_size is None else [int(i) for i in mlp_hid_size.split(',')]

        mlp_layers = [[nn.Linear(input_size, self.mlp_hid_size[0]), nn.BatchNorm1d(self.mlp_hid_size[0]), QuickGELU()]] + \
            [[nn.Linear(self.mlp_hid_size[i], self.mlp_hid_size[i+1]), nn.BatchNorm1d(self.mlp_hid_size[i+1]), QuickGELU()]
             for i in range(len(self.mlp_hid_size) - 1)] + [[nn.Linear(self.mlp_hid_size[-1], 1)]]
        self.mlp = nn.Sequential(*[layer for layers in mlp_layers for layer in layers])
        self.device = device
        self.to(self.device)
        
    def forward(self, inputs):
        x1, x2  = inputs
        x1 = self.mlp1(x1)
        x2 = self.mlp2(x2)
        x = torch.cat((x1, x2), dim=1)
        output = self.mlp(x)
        return output

    def configure_optimizers(self, learning_rate=1e-4):
        optimizer = optim.AdamW(params=self.parameters(), lr=learning_rate)
        return optimizer


class ViTModel(nn.Module):
    def __init__(self, load_ori_weights=True):
        super().__init__()
        self.vit, config = self._init_vit_model(load_ori_weights)
        self.head = nn.Linear(config.hidden_size, 1)

    def forward(self, inputs):
        embeddings, att_weights = self.vit.transformer(inputs)
        output = self.head(embeddings[:, 0])
        return output, att_weights

    def configure_optimizers(self, learning_rate=1e-4):
        optimizer = optim.AdamW(params=self.parameters(), lr=learning_rate)
        return optimizer
    
    def _init_vit_model(self, load_ori_weights=True):
        vit_model, config = load_vit_model(load_ori_weights)
        return vit_model, config
