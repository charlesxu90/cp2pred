from loguru import logger
import torch
from torch import nn, optim
import torch.nn.functional as F


torch.autograd.set_detect_anomaly(True)

class MolCL(nn.Module):
    def __init__(self, resnet, device, enc_width=2048, proj_dim=256, temp_scale=0.07, image_size=224):
        super().__init__()
        self.device = device
        self.temp = nn.Parameter(torch.tensor(1.0)) * temp_scale
        self.encoder = nn.Sequential(*list(resnet.children())[:-1]).to(self.device)
        self.proj = nn.Linear(enc_width, proj_dim)

    def forward(self, img, aug_img):
        img_embd = self.get_img_embd(img)
        aug_embd = self.get_img_embd(aug_img)

        img_feat = F.normalize(self.proj(img_embd), dim=-1)
        aug_feat = F.normalize(self.proj(aug_embd), dim=-1)

        #======= contrastive loss =======#
        sim_i2a = torch.mm(img_feat, aug_feat.T) / self.temp
        sim_a2i = torch.mm(aug_feat, img_feat.T) / self.temp

        targets = torch.zeros(sim_i2a.size()).to(self.device)
        targets.fill_diagonal_(1)

        loss_i2a = -torch.sum(F.log_softmax(sim_i2a, dim=-1) * targets, dim=-1).mean()
        loss_a2i = -torch.sum(F.log_softmax(sim_a2i, dim=-1) * targets, dim=-1).mean()

        loss_cl = (loss_i2a + loss_a2i) / 2
        return loss_cl
    
    def get_img_embd(self, img):
        embd = self.encoder(img).squeeze()
        return embd

    
    def configure_optimizers(self, learning_rate=1e-4):
        optimizer = optim.AdamW(params=self.parameters(), lr=learning_rate)
        return optimizer
