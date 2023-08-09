import os
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils.utils import save_model, get_regresssion_metrics
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TaskTrainer:

    def __init__(self, model, output_dir, grad_norm_clip=1.0, device='cuda',
                 learning_rate=1e-4, max_epochs=10, use_amp=True, distributed=False, model_type='vit'):
        self.model = model
        self.output_dir = output_dir
        self.grad_norm_clip = grad_norm_clip
        self.writer = SummaryWriter(self.output_dir)
        self.learning_rate = learning_rate
        self.device = device
        self.n_epochs = max_epochs
        self.use_amp = use_amp
        self.distributed = distributed
        self.mse_loss = nn.MSELoss()
        self.model_type = model_type

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.optimizer = raw_model.configure_optimizers(self.learning_rate)
    
    def fit(self, train_loader, test_loader=None, save_ckpt=True):
        model = self.model
        
        if torch.cuda.is_available() and self.device == 'cuda' and self.distributed:  # for distributed parallel
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model.cuda())
            local_rank = int(os.environ['LOCAL_RANK'])
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])

        for epoch in range(self.n_epochs):
            train_loss = self.train_epoch(epoch, model, train_loader)
            if test_loader is not None:
                test_loss, _ = self.test_epoch(epoch, model, test_loader)

            curr_loss = test_loss if 'test_loss' in locals() else train_loss
            
            if self.output_dir is not None and save_ckpt: # save model in each epoch
                self._save_model(self.output_dir, str(epoch+1), curr_loss)

        if self.output_dir is not None and save_ckpt:  # save final model
            self._save_model(self.output_dir, 'final', curr_loss)

    def run_forward(self, model, batch):
        batch = tuple(t.to(self.device) for t in batch)
        feat, target = batch
        if self.model_type == 'vit':
            output, attn_weights = model.forward(feat.float())
            mse_loss = self.mse_loss(output.squeeze().float(), target.float())
        else:
            output  = model.forward(feat.float())
            mse_loss = self.mse_loss(output[:, 0], target.float())
        return mse_loss, output, target
    
    def train_epoch(self, epoch, model, train_loader):
        model.train()
        if self.distributed:
            train_loader.sampler.set_epoch(epoch)   # for distributed parallel
        losses = []
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for it, batch in pbar:
            if self.device == 'cuda':
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                    loss, _, _ = self.run_forward(model, batch)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())
            else:
                loss, _, _ = self.run_forward(model, batch)
            losses.append(loss.item())
            pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss:.5f}.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_norm_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        loss = float(np.mean(losses))
        logger.info(f'train epoch: {epoch + 1}/{self.n_epochs}, loss: {loss:.4f}')
        self.writer.add_scalar(f'train_loss', loss, epoch + 1)
        return loss
    
    def test_epoch(self, epoch, model, test_loader):
        model.eval()
        if self.distributed:
            test_loader.sampler.set_epoch(epoch)   # for distributed parallel
        losses = []
        y_test = []
        y_test_hat = []

        pbar = enumerate(test_loader)
        for it, batch in pbar:
            with torch.no_grad():
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
        logger.info(f'test epoch: {epoch + 1}/{self.n_epochs}, loss: {loss:.4f}')
        self.writer.add_scalar(f'test_loss', loss, epoch + 1)

        y_test = np.concatenate(y_test, axis=0)
        y_test_hat = np.concatenate(y_test_hat, axis=0)
        mae, mse, _, spearman, pearson = get_regresssion_metrics(y_test_hat, y_test, print_metrics=False)
        logger.info(f'eval, epoch: {epoch + 1}/{self.n_epochs}, spearman: {spearman.item():.3f}, pearson: {pearson.item():.3f}, mse: {mse.item():.3f}, mae: {mae.item():.3f}')
        self.writer.add_scalar('spearman', spearman, epoch + 1)

        return loss, spearman

    def _save_model(self, base_dir, info, valid_loss):
        """ Save model with format: model_{info}_{valid_loss} """
        base_name = f'model_{info}_{valid_loss:.3f}'
        # logger.info(f'Save model {base_name}')
        save_model(self.model, base_dir, base_name)
