import os
from loguru import logger
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils.utils import save_model
from .model.cl_model import CLModel


class CLTrainer:

    def __init__(self, model: CLModel, output_dir, device='cuda', 
                 max_epochs=10, use_amp=False, grad_norm_clip=1.0, 
                 learning_rate=1e-4,lr_patience=20, lr_decay=0.5, min_lr=1e-5, weight_decay=0.0):
        self.model = model
        self.output_dir = output_dir
        self.grad_norm_clip = grad_norm_clip
        self.writer = SummaryWriter(self.output_dir)
        self.learning_rate = learning_rate
        self.device = device
        self.n_epochs = max_epochs
        self.use_amp = use_amp

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, raw_model.parameters()), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=lr_decay, patience=lr_patience, verbose=True)
        self.min_lr = min_lr
    
    def fit(self, train_loader, val_loader=None, test_loader=None, save_ckpt=True):
        model = self.model
        
        best_loss = np.float32('inf')
        for epoch in range(self.n_epochs):
            train_loss = self.train_epoch(epoch, model, train_loader)
            if val_loader is not None:
                val_loss = self.eval_epoch(epoch, model, val_loader, split='val')

            if test_loader is not None:
                test_loss = self.eval_epoch(epoch, model, test_loader, split='test')

            curr_loss = val_loss if 'val_loss' in locals() else train_loss
            
            if self.output_dir is not None and save_ckpt and curr_loss < best_loss:  # only save better loss
                best_loss = curr_loss
                self._save_model(self.output_dir, str(epoch+1), curr_loss)

            if self.optimizer.param_groups[0]['lr'] < float(self.min_lr):
                logger.info("Learning rate == min_lr, stop!")
                break
            self.scheduler.step(val_loss)

        if self.output_dir is not None and save_ckpt:  # save final model
            self._save_model(self.output_dir, 'final', curr_loss)

    def run_forward(self, model: CLModel, batch):
        # batch = tuple(t.to(self.device) for t in batch)
        x, x_aug = batch
        loss  = model.forward(x, x_aug)
        return loss
    
    def train_epoch(self, epoch, model, train_loader):
        model.train()
        losses = []
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for it, batch in pbar:
            if self.device == 'cuda':
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                    loss = self.run_forward(model, batch)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())
            else:
                loss = self.run_forward(model, batch)
            losses.append(loss.item())
            pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss:.5f}.")

            loss.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_norm_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        loss = float(np.mean(losses))
        logger.info(f'train epoch: {epoch + 1}/{self.n_epochs}, loss: {loss:.4f}')
        self.writer.add_scalar(f'train_loss', loss, epoch + 1)
        return loss
    
    @torch.no_grad()
    def eval_epoch(self, epoch, model, test_loader, split='test'):
        model.eval()
        losses = []

        pbar = enumerate(test_loader)
        for it, batch in pbar:
            if self.device == 'cuda':
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                    loss = self.run_forward(model, batch)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())
            else:
                loss = self.run_forward(model, batch)
            losses.append(loss.item())

        loss = float(np.mean(losses))
        logger.info(f'{split} epoch: {epoch + 1}/{self.n_epochs}, loss: {loss:.4f}')
        self.writer.add_scalar(f'{split}_loss', loss, epoch + 1)

        return loss

    def _save_model(self, base_dir, info, valid_loss):
        """ Save model with format: model_{info}_{valid_loss} """
        base_name = f'model_{info}_{valid_loss:.3f}'
        # logger.info(f'Save model {base_name}')
        save_model(self.model, base_dir, base_name)
