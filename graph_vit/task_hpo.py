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

def update_config(config, params):
    config.train.learning_rate = params['lr']
    config.model.mlp_hid_size = ','.join([str(params[p]) for p in ['l1_dim', 'l2_dim', 'l3_dim', 'l4_dim']])
    return config

def main(config, args, base_config):
    set_random_seed(args.seed)
    log_GPU_info()

    config = update_config(base_config, config)
    
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config.data)
    nout = len(config.data.target_col.split(','))
    model = GraphMLPMixer(nout=nout, **config.model.graphvit, rw_dim=config.data.pos_enc.rw_dim, 
                          patch_rw_dim=config.data.pos_enc.patch_rw_dim, n_patches=config.data.metis.n_patches)
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
    parser.add_argument('--config', default='graph_vit/task_finetune.yaml')
    parser.add_argument('--output_dir', default='results/graph_vit/task_finetune')
    parser.add_argument('--seed', default=42, type=int)

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

    scheduler = ASHAScheduler(max_t=base_config.train.max_epochs, grace_period=1, reduction_factor=2)
    
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