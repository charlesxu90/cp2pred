import argparse
import torch
import numpy as np
import random
import warnings
from loguru import logger
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record

from utils.utils import parse_config, log_GPU_info, load_model, set_random_seed
from utils.dist import init_distributed, get_rank, is_main_process

from .dataset import load_image_data_by_split
from .model.task_trainer import TaskTrainer
from .model.resnet import init_model, load_model_from_ckpt
from .model.molcl import MolCL
warnings.filterwarnings("ignore", message="torch.distributed._all_gather_base is a private function")

def get_dataloaders(config, val_split):
    global_rank = get_rank()
    batch_size, num_workers = config.batch_size, config.num_workers
    train_set, val_set, test_set = load_image_data_by_split(config.input_path, feat_col=config.feat_col, target_col=config.target_col, 
                                                            val_split=val_split, image_size=config.image_size)
    
    train_sampler = DistributedSampler(dataset=train_set, shuffle=True, rank=global_rank)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)

    val_sampler = DistributedSampler(dataset=val_set, shuffle=False, rank=global_rank)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler, shuffle=False, num_workers=num_workers, pin_memory=True)

    test_sampler = DistributedSampler(dataset=test_set, shuffle=False, rank=global_rank)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader

@record
def main(args, config):
    init_distributed()
    global_rank = get_rank()
    device = torch.device(args.device)
    seed = args.seed + global_rank
    set_random_seed(seed)

    if is_main_process():
        log_GPU_info()
    
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config.data, args.val_split)
    
    model = init_model(**config.model.resnet)
    if args.ckpt is not None:
        model = load_model_from_ckpt(config.model.resnet.model_name, model, args.ckpt)

    if args.ckpt_cl is not None:
        molcl = MolCL(model, device, **config.model.molcl)
        molcl = load_model(molcl, args.ckpt_cl, device)
        for r, c in zip(list(model.children())[:-1], list(molcl.encoder.children())):
            r = c
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
    parser.add_argument('--val_split', type=int, default=1, help='the split index of validation set, 1-5')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    
    config = parse_config(args.config)
    main(args, config)
