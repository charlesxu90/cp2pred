import os
import time
from tqdm import tqdm
from loguru import logger
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils.utils import save_model, LossAnomalyDetector
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dist import is_dist_avail_and_initialized, is_main_process


class BertTrainer:

    def __init__(self, model, output_dir, grad_norm_clip=1.0, device='cuda',
                 learning_rate=1e-4, max_epochs=10, use_amp=True):
        self.model = model
        self.output_dir = output_dir
        self.grad_norm_clip = grad_norm_clip
        self.writer = SummaryWriter(self.output_dir)
        self.learning_rate = learning_rate
        self.device = device
        self.n_epochs = max_epochs
        self.use_amp = use_amp
        self.loss_anomaly_detector = LossAnomalyDetector()
    
    def fit(self, train_loader, test_loader=None, save_ckpt=True):
        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        optimizer = raw_model.configure_optimizers(self.learning_rate)
        scheduler = CosineAnnealingWarmupRestarts(optimizer, max_lr=self.learning_rate, min_lr=0.001*self.learning_rate,
                                                  first_cycle_steps=len(train_loader)*self.n_epochs, warmup_steps=len(train_loader))
        
        if torch.cuda.is_available():  # for distributed parallel
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model.cuda())
            local_rank = int(os.environ['LOCAL_RANK'])
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            loader = train_loader if is_train else test_loader
            loader.sampler.set_epoch(epoch)  # for distributed parallel

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, x in pbar:
                with torch.set_grad_enabled(is_train):
                    x = model.module.tokenize_inputs(x).to(self.device) if hasattr(model, "module") else model.tokenize_inputs(x).to(self.device)
                    with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                        loss = model.forward(x)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus

                losses.append(loss.item())
                if is_train:
                    if self.loss_anomaly_detector(loss.item()):
                        logger.info(f"Anomaly loss detected at epoch {epoch + 1} iter {it}: train loss {loss:.5f}.")
                        del loss, x
                        continue  # Skip the current iteration if the loss is an anomaly
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_norm_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss.item()

                    pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss:.5f}.")
                    self.writer.add_scalar('step_loss', loss, epoch*len(loader) + it + 1)
                    self.writer.add_scalar('lr', scheduler.get_lr()[0], epoch*len(loader) + it + 1)

            loss = float(np.mean(losses))
            if is_main_process():
                logger.info(f'{split}, epoch: {epoch + 1}/{self.n_epochs}, loss: {loss:.4f}')
            self.writer.add_scalar(f'{split}_loss', loss, epoch + 1)

            return loss

        best_loss = np.float32('inf')
        for epoch in range(self.n_epochs):
            train_loss = run_epoch('train')
            if test_loader is not None:
                test_loss = run_epoch('test')

            curr_loss = test_loss if 'test_loss' in locals() else train_loss
            # save model in each epoch
            if self.output_dir is not None and save_ckpt and curr_loss < best_loss:  # only save better loss
                best_loss = curr_loss
                if is_main_process(): self._save_model(self.output_dir, str(epoch+1), curr_loss)

        if self.output_dir is not None and save_ckpt and is_main_process():  # save final model
            self._save_model(self.output_dir, 'final', curr_loss)

    def _save_model(self, base_dir, info, valid_loss):
        """
        Save a copy of the model with format: model_{info}_{valid_loss}
        """
        base_name = f'model_{info}_{valid_loss:.3f}'
        logger.info(f'Save model {base_name}')
        if not is_dist_avail_and_initialized() or is_main_process():  # for distributed parallel
            save_model(self.model, base_dir, base_name)

