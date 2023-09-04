import argparse
import torch
import numpy as np
import warnings
from loguru import logger
from pathlib import Path
from torch_geometric.loader import DataLoader

from utils.utils import parse_config, set_random_seed, log_GPU_info, load_model

from .dataset.dataset import create_cl_dataset
from .model.model import GraphMLPMixer
from .model.cl_model import CLModel
from .cl_trainer import CLTrainer
warnings.filterwarnings("ignore", message="torch.distributed._all_gather_base is a private function")


def get_dataloaders(config):
    train_set, test_set = create_cl_dataset(config)

    train_dataloader = DataLoader(train_set, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    return train_dataloader, test_dataloader


def main(args, config):
    set_random_seed(args.seed)
    log_GPU_info()
    
    train_dataloader, test_dataloader = get_dataloaders(config.data)
    model = GraphMLPMixer(nout=1, **config.model.graphvit, rw_dim=config.data.pos_enc.rw_dim, 
                          patch_rw_dim=config.data.pos_enc.patch_rw_dim, n_patches=config.data.metis.n_patches)
    
    cl_model = CLModel(model, **config.model.cl_model)
    
    logger.info(f"Start training")
    trainer = CLTrainer(cl_model, args.output_dir, **config.train)
    trainer.fit(train_dataloader, test_dataloader)
    logger.info(f"Training finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_resnet.yaml')
    parser.add_argument('--output_dir', default='results/train_resnet/')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--ckpt_cl', default=None, type=str)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    
    config = parse_config(args.config)
    main(args, config)
