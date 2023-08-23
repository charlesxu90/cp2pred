import argparse
import torch
import numpy as np
import random
import warnings
from loguru import logger
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record

from utils.utils import parse_config, log_GPU_info, load_model
from utils.dist import init_distributed, get_rank, is_main_process

from .dataset.dataset import create_dataset
from .model.model import GraphMLPMixer
from .task_trainer import TaskTrainer
# from .molcl import MolCL
warnings.filterwarnings("ignore", message="torch.distributed._all_gather_base is a private function")


def get_dataloaders(config):
    train_set, val_set, test_set = create_dataset(config)

    train_dataloader = DataLoader(train_set, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader


@record
def main(args, config):
    device = torch.device(args.device)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if is_main_process():
        log_GPU_info()
    
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config.data)
    nout = len(config.data.target_col.split(','))
    model = GraphMLPMixer(nout=nout, **config.model.graphvit, rw_dim=config.data.pos_enc.rw_dim, 
                          patch_rw_dim=config.data.pos_enc.patch_rw_dim, n_patches=config.data.metis.n_patches)
    
    # if args.ckpt_cl is not None:
    #     molcl = MolCL(model, device, **config.model.molcl)
    #     molcl = load_model(molcl, args.ckpt_cl, device)
    #     for r, c in zip(list(model.children())[:-1], list(molcl.encoder.children())):
    #         r = c
    model.to(device)
    
    logger.info(f"Start training")
    trainer = TaskTrainer(model, args.output_dir, **config.train)
    trainer.fit(train_dataloader, val_dataloader, test_dataloader)
    logger.info(f"Training finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_resnet.yaml')
    parser.add_argument('--output_dir', default='results/train_resnet/')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--ckpt_cl', default=None, type=str)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    
    config = parse_config(args.config)
    main(args, config)
