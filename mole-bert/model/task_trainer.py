import os
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from utils.utils import save_model, get_regresssion_metrics, get_metrics, LossAnomalyDetector
import torch.nn.functional as F


class TaskTrainer:

    def __init__(self, model, output_dir, grad_norm_clip=1.0, device='cuda', 
                 max_epochs=10, use_amp=False, task_type='regression',
                 learning_rate=1e-4,lr_patience=20, lr_decay=0.5, min_lr=1e-5, weight_decay=0.0):
        
        self.model = model
        self.output_dir = output_dir
        self.grad_norm_clip = grad_norm_clip
        self.writer = SummaryWriter(self.output_dir)
        self.learning_rate = learning_rate
        self.device = device
        self.n_epochs = max_epochs
        self.use_amp = use_amp
        self.task_type = task_type
        self.loss_anomaly_detector = LossAnomalyDetector(std_fold=20,)
        if task_type == 'regression':
            self.loss_fn = nn.MSELoss()
        elif task_type == 'classification':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise Exception(f'Unknown task type: {task_type}')

        self.optimizer = model.config_optimizer(lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=lr_decay, patience=lr_patience, verbose=True)
        self.min_lr = min_lr
        
    def fit(self, train_loader, val_loader=None, test_loader=None, save_ckpt=True):
        model = self.model.to(self.device)

        best_loss = np.float32('inf')
        for epoch in range(self.n_epochs):
            train_loss = self.train_epoch(epoch, model, train_loader)
            if val_loader is not None:
                val_loss, _ = self.eval_epoch(epoch, model, val_loader, split='val')

            if test_loader is not None:
                test_loss, _ = self.eval_epoch(epoch, model, test_loader, split='test')

            curr_loss = val_loss if 'val_loss' in locals() else train_loss
            
            if self.output_dir is not None and save_ckpt and curr_loss < best_loss:  # only save better loss
                best_loss = curr_loss
                self._save_model(self.output_dir, str(epoch+1), curr_loss)

            if self.optimizer.param_groups[0]['lr'] < float(self.min_lr):
                logger.info("Learning rate == min_lr, stop!")
                break
            self.scheduler.step(curr_loss)

        if self.output_dir is not None and save_ckpt:  # save final model
            self._save_model(self.output_dir, 'final', curr_loss)

    def run_forward(self, model, batch):
        if self.task_type == 'vqvae':
            batch = batch.to(self.device)
            loss = model(batch)
            return loss, None, None
        else:
            batch = batch.to(self.device)
            true = batch.y
            pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            # flatten the pred
            pred = pred.view(-1)
            is_labeled = batch.y == batch.y
            pred, true = pred[is_labeled], true[is_labeled]
            loss = self.loss_fn(pred.float(), true.float())
            return loss, pred, true
    
    def train_epoch(self, epoch, model, train_loader):
        model.train()
        losses = []

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        # pbar = enumerate(train_loader)
        for it, batch in pbar:
            if self.device == 'cuda':
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                    if self.task_type == 'pretrain':
                        loss, acc_node, acc_edge = self.run_forward(model, batch)
                    else:
                        loss, _, _ = self.run_forward(model, batch)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
            else:
                loss, _, _ = self.run_forward(model, batch)

            if self.loss_anomaly_detector(loss.item()):
                logger.info(f"Anomaly loss detected at epoch {epoch + 1} iter {it}: train loss {loss:.5f}.")
                del loss, batch
                continue
            else:
                losses.append(loss.item())
                pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss:.5f}.")
                # logger.debug(f"epoch {epoch + 1} iter {it}: train loss {loss:.5f}.")
                loss.backward()
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
        y_test = []
        y_test_hat = []

        pbar = enumerate(test_loader)
        for it, batch in pbar:
            if self.device == 'cuda':
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                    loss, y_hat, y = self.run_forward(model, batch)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())
            else:
                loss, y_hat, y = self.run_forward(model, batch)
            losses.append(loss.item())
            y_test_hat.append(y_hat.cpu().numpy())
            y_test.append(y.cpu().numpy())

        loss = float(np.mean(losses))
        logger.info(f'{split} epoch: {epoch + 1}, loss: {loss:.4f}')
        self.writer.add_scalar(f'{split}_loss', loss, epoch + 1)

        y_test = np.concatenate(y_test, axis=0).squeeze()
        y_test_hat = np.concatenate(y_test_hat, axis=0).squeeze()
        # logger.debug(f'y_test: {y_test}, y_test_hat: {y_test_hat}')
        if self.task_type == 'regression':
            metric = get_regresssion_metrics(y_test_hat, y_test, print_metrics=True)
            self.writer.add_scalar(f'{split}-mae', metric['mae'], epoch + 1)
        elif self.task_type == 'classification':
            metric = get_metrics(y_test_hat > 0.5, y_test, print_metrics=True)
            self.writer.add_scalar(f'{split}-acc', metric['acc'], epoch + 1)
        return loss, metric

    def _save_model(self, base_dir, info, valid_loss):
        """ Save model with format: model_{info}_{valid_loss} """
        base_name = f'model_{info}_{valid_loss:.3f}'
        # logger.info(f'Save model {base_name}')
        save_model(self.model, base_dir, base_name)
