import argparse
import os
import logging
import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import DataLoader

from utils.utils import parse_config
from dataset.dataset import load_feat_data, TaskDataset

from model.task_model import ConcatModel
from model.task_trainer import TaskTrainer

def main(args, config):
    device = torch.device(config.train.device)
    seed = args.seed
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    logger = logging.getLogger(__name__)
    
    batch_size, num_workers = config.data.batch_size, config.data.num_workers
    train_data, valid_data = load_feat_data(config.data.input_path, feat_names=config.data.feat_names,)
    logger.info(f"train_data {len(train_data)}, valid_data {len(valid_data)}")
    feat_size = len(train_data[0][0])
    logger.info(f"Number of training features {feat_size}")

    train_set, test_set = TaskDataset(train_data), TaskDataset(valid_data)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    pred_model = ConcatModel(input_size=feat_size, device=device, **config.model)

    logger.info(f"Start training")
    trainer = TaskTrainer(pred_model, args.output_dir, **config.train)
    trainer.fit(train_dataloader, test_dataloader)
    logger.info(f"Training finished")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_conatmodel.yaml')
    parser.add_argument('--output_dir', default='results/train_conatmodel/')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    
    config = parse_config(args.config)
    main(args, config)
