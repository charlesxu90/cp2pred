from torch import nn
from .base_transformer import QuickGELU
from .bert import BERT
from loguru import logger


class TaskPred(nn.Module):
    def __init__(self, bert_model: BERT, device='cuda', mlp_hid_size=None, output_size=2):
        super().__init__()
        self.bert = bert_model
        self.mlp_hid_size = [256, 128, 64] if mlp_hid_size is None else [int(i) for i in mlp_hid_size.split(',')]

        mlp_layers = [[nn.Linear(self.bert.transformer.width, self.mlp_hid_size[0]), nn.BatchNorm1d(self.mlp_hid_size[0]), QuickGELU()]] + \
            [[nn.Linear(self.mlp_hid_size[i], self.mlp_hid_size[i+1]), nn.BatchNorm1d(self.mlp_hid_size[i+1]), QuickGELU()] 
             for i in range(len(self.mlp_hid_size) - 1)] + [[nn.Linear(self.mlp_hid_size[-1], output_size)]]
        self.mlp = nn.Sequential(*[layer for layers in mlp_layers for layer in layers])
        self.device = device
        self.to(self.device)
        
    def get_bert_embd(self, tokens):
        outputs = self.bert.embed(tokens)
        reps = outputs[:, 0]  # CLS_token
        return reps

    def forward(self, inputs):
        tokens = self.bert.tokenize_inputs(inputs).to(self.device)
        embd = self.get_bert_embd(tokens)
        mlp_output = self.mlp(embd)
        return mlp_output.squeeze()

    def configure_optimizers(self, learning_rate=1e-4):
        return self.bert.configure_optimizers(learning_rate)
