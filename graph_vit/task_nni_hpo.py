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
import ray
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

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
    for epoch in range(config.train.max_epochs):
        trainer.train_epoch(epoch, model, train_dataloader)
        _, spearman = trainer.eval_epoch(epoch, model, test_dataloader)
        # raw_model = model.module if hasattr(model, "module") else model

        # checkpoint_data = {
        #     "epoch": epoch,
        #     "model_state_dict": raw_model.state_dict(),
        # }
        # checkpoint = Checkpoint.from_dict(checkpoint_data)
        session.report({"spearman": spearman}) #, checkpoint=checkpoint,)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='graph_vit/task_finetune.yaml')
    parser.add_argument('--output_dir', default='results/graph_vit/task_ray_tune_hpo')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples/trials to run")
    parser.add_argument("--max_concur_trials", type=int, default=8, help="Maximum trials to start concurrently")

    args = parser.parse_args()
    base_config = parse_config(args.config)
    
    #Connect to Ray server
    ray.init(address=os.environ["ip_head"], _node_ip_address=os.environ["head_node_ip"],_redis_password=os.getenv('redis_password'))

    search_space_config = {
        "n_patches": tune.randint(24, 40),
        "drop_rate": tune.choice([0.1*x for x in range(0, 5, 1)]),
        "num_hops": tune.randint(1, 4),
        "lr": tune.loguniform(1e-6, 1e-1),
    }

    scheduler = ASHAScheduler(max_t=base_config.train.max_epochs, grace_period=1, reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(partial(main, args=args, base_config=base_config)),
            resources={"cpu": 4, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="spearman",
            mode="max",
            scheduler=scheduler,
            num_samples=args.num_samples,
            max_concurrent_trials=args.max_concur_trials,
        ),
        param_space=search_space_config,
    )

    results = tuner.fit()

    best_result = results.get_best_result("spearman", "max", "last")
    logger.info(f"Best trial config: {best_result.config}")
