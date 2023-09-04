import argparse
import warnings
import logging
import torch
import numpy as np
import random
from pathlib import Path
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record

from utils.utils import parse_config, log_GPU_info, set_random_seed
from utils.dist import init_distributed, get_rank, is_main_process

from .dataset import load_image_data, ImageDataset
from .model.resnet import init_model, load_model_from_ckpt
from .model.molcl import MolCL
from .model.molcl_trainer import MolCLTrainer

warnings.filterwarnings("ignore", message="torch.distributed._all_gather_base is a private function")

def get_dataloaders(config):
    global_rank = get_rank()
    train_data, valid_data = load_image_data(config.input_path, feat_col=config.feat_col, 
                                             target_col=config.target_col, image_size=config.image_size)
    train_sampler = DistributedSampler(dataset=train_set, shuffle=True, rank=global_rank)
    train_dataloader = DataLoader(train_set, batch_size=config.batch_size, sampler=train_sampler, num_workers=config.num_workers, pin_memory=True)

    test_sampler = DistributedSampler(dataset=test_set, shuffle=False, rank=global_rank)
    test_dataloader = DataLoader(test_set, batch_size=config.batch_size, sampler=test_sampler, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    return train_dataloader, test_dataloader


@record
def main(args, config):
    init_distributed()
    global_rank = get_rank()
    device = torch.device(args.device)
    seed = args.seed + global_rank
    set_random_seed(seed)
    
    if is_main_process():
        log_GPU_info()
    
    train_dataloader, test_dataloader = get_dataloaders(config.data)
    
    resnet = init_model(**config.model.resnet)
    if args.ckpt is not None:
        resnet = load_model_from_ckpt(config.model.resnet.model_name, resnet, args.ckpt)
    resnet.to(device)

    model = MolCL(resnet, device, **config.model.molcl)
    
    logger.info(f"Start training")
    trainer = MolCLTrainer(model, args.output_dir, **config.train)
    trainer.fit(train_dataloader, test_dataloader)
    logger.info(f"Training finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='image_mol/pretrain_resnet.yaml')
    parser.add_argument('--output_dir', default='results/pretrain_resnet/')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--ckpt', default=None, type=str)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    
    config = parse_config(args.config)
    main(args, config)
