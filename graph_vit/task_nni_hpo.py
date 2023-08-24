import os
import argparse
from loguru import logger
from pathlib import Path
from torch_geometric.loader import DataLoader

from utils.utils import parse_config, set_random_seed, log_GPU_info, load_model
from functools import partial
from .dataset.dataset import create_dataset
from .model.model import GraphMLPMixer
from .task_trainer import TaskTrainer
from .model.cl_model import CLModel

from .task_finetune import get_dataloaders
import nni


def update_config(config, tune_config):
    config.data.metis.n_patches = tune_config['n_patches']
    config.data.metis.drop_rate = tune_config['drop_rate']
    config.data.metis.num_hops = tune_config['num_hops']
    config.train.learning_rate = tune_config['lr']
    
    return config

def main(args, base_config, tune_config):
    set_random_seed(args.seed)
    log_GPU_info()

    config = update_config(base_config, tune_config)
    
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config.data)
    nout = len(config.data.target_col.split(','))
    model = GraphMLPMixer(nout=nout, **config.model.graphvit, rw_dim=config.data.pos_enc.rw_dim, 
                          patch_rw_dim=config.data.pos_enc.patch_rw_dim, n_patches=config.data.metis.n_patches)
    trainer = TaskTrainer(model, args.output_dir, **config.train)
    model = model.to(trainer.device)
    
    logger.info(f"Start training")
    for epoch in range(args.max_epochs):
        trainer.train_epoch(epoch, model, train_dataloader)
        _, spearman = trainer.eval_epoch(epoch, model, test_dataloader)
        nni.report_intermediate_result(spearman)
    nni.report_final_result(spearman)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='graph_vit/task_finetune.yaml')
    parser.add_argument('--output_dir', default='results/graph_vit/task_ray_tune_hpo')
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    base_config = parse_config(args.config)

    params = {
        "n_patches": 32,
        "drop_rate": 0.4,
        "num_hops": 2,
        "lr": 0.001,
    }

    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)  
    main(args, base_config, params)
