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
import nni

def update_config_by_nni_params(config, params):
    config.train.learning_rate = params['lr']
    config.model.mlp_hid_size = ','.join([str(params[p]) for p in ['l1_dim', 'l2_dim', 'l3_dim', 'l4_dim']])
    return config

def main(params, args, config):
    device = torch.device(config.train.device)
    seed = 42
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    batch_size, num_workers = config.data.batch_size, config.data.num_workers
    train_data, valid_data = load_feat_data(config.data.input_path, feat_names=config.data.feat_names,)
    logger.info(f"train_data {len(train_data)}, valid_data {len(valid_data)}")
    feat_size = len(train_data[0][0])
    logger.info(f"Number of training features {feat_size}")

    train_set, test_set = TaskDataset(train_data), TaskDataset(valid_data)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    config = update_config_by_nni_params(config, params)
    
    model = ConcatModel(input_size=feat_size, device=device, **config.model)
    trainer = TaskTrainer(model, args.output_dir, **config.train)
    
    logger.info(f"Start training")
    for epoch in range(config.train.max_epochs):
        trainer.train_epoch(epoch, model, train_dataloader)
        _, spearman = trainer.test_epoch(epoch, model, test_dataloader)
        nni.report_intermediate_result(spearman)
    nni.report_final_result(spearman)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_conatmodel.yaml')
    parser.add_argument('--output_dir', default='results/train_conatmodel/')
    args = parser.parse_args()
    config = parse_config(args.config)

    params = {
        'l1_dim': 512,
        'l2_dim': 512,
        'l3_dim': 512,
        'l4_dim': 512,
        'lr': 0.001,
    }
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)

    main(params, args, config)
