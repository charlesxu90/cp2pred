import logging
import torch
from torch import nn, optim
import torch.nn.functional as F
from .base_transformer import QuickGELU

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
    def __init__(self, vit_model, hidden_size):
        super().__init__()
        self.vit = vit_model
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        embeddings, att_weights = self.vit(inputs)
        embedding_cls_token = embeddings[:, 0, :]
        output = self.head(embedding_cls_token)
        return output
