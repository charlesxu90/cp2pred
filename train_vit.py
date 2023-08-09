import argparse
import logging
import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import DataLoader

from utils.utils import parse_config, load_model, log_GPU_info
from dataset.dataset import load_image_data, TaskImageDataset
from model.task_model import ViTModel
from model.task_trainer import TaskTrainer
from torch.utils.data.distributed import DistributedSampler
from utils.dist import init_distributed, get_rank, is_main_process
from torch.distributed.elastic.multiprocessing.errors import record

from model.vit import VisionTransformer, get_b16_config

def get_dataloaders(config):
    global_rank = get_rank()
    train_data, valid_data = load_image_data(config.input_path, feat_names=config.feat_names, target_col=config.target_col,)
    train_set, test_set = TaskImageDataset(train_data), TaskImageDataset(valid_data)
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
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    logger = logging.getLogger(__name__)
    
    if is_main_process():
        log_GPU_info()
    
    train_dataloader, test_dataloader = get_dataloaders(config.data)
    
    if args.ckpt is None:
        model = ViTModel()
    else:
        model = ViTModel(load_ori_weights=False)
        model = load_model(model, args.ckpt, device)
    model.to(device)
    
    logger.info(f"Start training")
    trainer = TaskTrainer(model, args.output_dir, **config.train)
    trainer.fit(train_dataloader, test_dataloader)
    logger.info(f"Training finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_vit.yaml')
    parser.add_argument('--output_dir', default='results/train_vit/')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ckpt', default=None, type=str)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    
    config = parse_config(args.config)
    main(args, config)
