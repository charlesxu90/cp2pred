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
                 learning_rate=1e-4, max_epochs=10, use_amp=True, distributed=False, high_threshold=-6):
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
        self.high_threshold = high_threshold
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def fit(self, train_loader, test_loader=None, val_loader=None, save_ckpt=True):
        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        optimizer = raw_model.configure_optimizers(self.learning_rate)
        
        if torch.cuda.is_available() and self.device == 'cuda' and self.distributed:  # for distributed parallel
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model.cuda())
            local_rank = int(os.environ['LOCAL_RANK'])
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
        
        def run_forward(batch):
            feat, target = batch
            feat, target = feat.to(self.device), target.to(self.device)
            output  = model.forward(feat.float())
            mse_loss = self.mse_loss(output[:, 0], target.float())
            target_label = target >= self.high_threshold
            output_label = output[:, 0] >= self.high_threshold
            bce_loss = self.bce_loss(output_label.float(), target_label.float())
            return mse_loss + bce_loss

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            loader = train_loader if is_train else test_loader
            if self.distributed:
                loader.sampler.set_epoch(epoch)   # for distributed parallel

            losses = []
            pbar = enumerate(loader)
            for it, batch in pbar:
                with torch.set_grad_enabled(is_train):
                    if self.device == 'cuda':
                        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                            loss = run_forward(batch)
                            loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                            losses.append(loss.item())
                    else:
                        loss = run_forward(batch)
                    losses.append(loss.item())

                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_norm_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            loss = float(np.mean(losses))
            logger.info(f'{split}, epoch: {epoch + 1}/{self.n_epochs}, loss: {loss:.4f}')
            self.writer.add_scalar(f'{split}_loss', loss, epoch + 1)
            return loss

        for epoch in range(self.n_epochs):
            train_loss = run_epoch('train')
            if test_loader is not None:
                test_loss = run_epoch('test')
            # if val_loader is not None and not self.distributed:
            #     self._eval_model(val_loader, epoch)

            curr_loss = test_loss if 'test_loss' in locals() else train_loss
            # save model in each epoch
            if self.output_dir is not None and save_ckpt:
                self._save_model(self.output_dir, str(epoch+1), curr_loss)

        if self.output_dir is not None and save_ckpt:  # save final model
            self._save_model(self.output_dir, 'final', curr_loss)

    def _save_model(self, base_dir, info, valid_loss):
        """
        Save a copy of the model with format: model_{info}_{valid_loss}
        """
        base_name = f'model_{info}_{valid_loss:.3f}'
        # logger.info(f'Save model {base_name}')
        save_model(self.model, base_dir, base_name)
    
    def predict(self, X_test):
        with torch.set_grad_enabled(False):
            y_hat = self.model.forward(X_test.float())
            return y_hat.squeeze()

    def _eval_model(self, val_loader, epoch):
        y_test = []
        y_test_hat = []

        for x, y in val_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.predict(x)
            y_test_hat.append(y_hat.cpu().numpy())
            y_test.append(y.cpu().numpy())

        y_test = np.concatenate(y_test, axis=0)
        y_test_hat = np.concatenate(y_test_hat, axis=0)
        mae, mse, _, spearman, pearson = get_regresssion_metrics(y_test_hat, y_test, print_metrics=False)
        logger.info(f'eval, epoch: {epoch + 1}/{self.n_epochs}, spearman: {spearman:.3f}, pearson: {pearson:.3f}, mse: {mse:.3f}, mae: {mae:.3f}')
        self.writer.add_scalar('spearman', spearman, epoch + 1)

