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
from functools import partial

import ray
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

def update_config(config, params):
    config.train.learning_rate = params['lr']
    config.model.mlp_hid_size = ','.join([str(params[p]) for p in ['l1_dim', 'l2_dim', 'l3_dim', 'l4_dim']])
    return config

def main(config, args, base_config):

    config = update_config(base_config, config)

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

    
    model = ConcatModel(input_size=feat_size, device=device, **config.model)
    trainer = TaskTrainer(model, args.output_dir, **config.train)
    
    logger.info(f"Start training")
    for epoch in range(config.train.max_epochs):
        trainer.train_epoch(epoch, model, train_dataloader)
        _, spearman = trainer.test_epoch(epoch, model, test_dataloader)
        # raw_model = model.module if hasattr(model, "module") else model

        # checkpoint_data = {
        #     "epoch": epoch,
        #     "model_state_dict": raw_model.state_dict(),
        # }
        # checkpoint = Checkpoint.from_dict(checkpoint_data)
        session.report({"spearman": spearman}) #, checkpoint=checkpoint,)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_conatmodel.yaml')
    parser.add_argument('--output_dir', default='results/train_conatmodel/')
    args = parser.parse_args()
    base_config = parse_config(args.config)
    
    # ray.init()
    # ray.init(address='127.0.0.1', _node_ip_address='127.0.0.1')

    search_space_config = {
        "l1_dim": tune.choice([2**i for i in range(3, 10)]),
        "l2_dim": tune.choice([2**i for i in range(3, 10)]),
        "l3_dim": tune.choice([2**i for i in range(3, 10)]),
        "l4_dim": tune.choice([2**i for i in range(2, 8)]),
        "lr": tune.loguniform(1e-7, 1e-1),
    }

    scheduler = ASHAScheduler(
        max_t=base_config.train.max_epochs,
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(partial(main, args=args, base_config=base_config)),
            resources={"cpu": 2, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="spearman",
            mode="max",
            scheduler=scheduler,
            num_samples=100,
        ),
        param_space=search_space_config,
    )

    results = tuner.fit()

    best_result = results.get_best_result("spearman", "max", "last")
    print(f"Best trial config: {best_result.config}")
