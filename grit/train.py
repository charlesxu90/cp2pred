import argparse
from loguru import logger
from pathlib import Path
from torch_geometric.loader import DataLoader

from utils.utils import parse_config, set_random_seed, log_GPU_info, load_model
from .dataset.dataset import create_dataset
from .model.grit_model import GritTransformer
from .trainer import TaskTrainer


def get_dataloaders(config, val_split):
    train_set, val_set, test_set = create_dataset(config, val_split)

    train_dataloader = DataLoader(train_set, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader


def main(args, config):
    set_random_seed(args.seed)
    log_GPU_info()
    
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config.data, args.val_split)
    nout = len(config.data.target_col.split(','))
    model = GritTransformer(nout, **config.model.grit, ksteps=config.data.pos_enc_rrwp.ksteps)
    
    if args.ckpt is not None:
       model = load_model(model, args.ckpt)

    logger.info(f"Start training")
    trainer = TaskTrainer(model, args.output_dir, **config.train)
    trainer.fit(train_dataloader, val_dataloader, test_dataloader)
    logger.info(f"Training finished")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='graph_vit/task_finetune.yaml')
    parser.add_argument('--output_dir', default='results/graph_vit/task_finetune')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--ckpt_cl', default=None, type=str)
    parser.add_argument('--val_split', type=int, default=1, help='the split index of validation set, 1-5')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    
    config = parse_config(args.config)
    main(args, config)
