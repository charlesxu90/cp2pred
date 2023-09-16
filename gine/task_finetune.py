import argparse
import pickle
import os.path as osp
from loguru import logger
import numpy as np
from pathlib import Path

from torch_geometric.loader import DataLoader
from utils.utils import parse_config, set_random_seed, log_GPU_info, load_model

from .dataset.dataset import CycPepDataset
from .model.model import GNN_graphpred
from .model.task_trainer import TaskTrainer


def load_data(config, val_split=1):
    batch_size, num_workers = config.batch_size, config.num_workers
    dataset = CycPepDataset(config.root, smiles_col=config.smiles_col, target_col=config.target_col)

    split_file = osp.join(config.root, 'raw', "scaffold_k_fold_idxes.pkl")
    with open(split_file, 'rb') as f:
        split_idx = pickle.load(f)
    val_idx = split_idx[val_split]
    test_idx = split_idx[val_split+1] # test split is val_split-1
    train_splits = [split_idx[i] for i in range(len(split_idx))if i != val_split+1 and i != val_split]  # the rest are training data
    train_idx = np.concatenate(train_splits, axis=0)

    train_dataset, val_dataset, test_dataset = dataset[train_idx], dataset[val_idx], dataset[test_idx]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

def create_model(config, ckpt_pretrain=None):
    model = GNN_graphpred(config.num_layer, config.emb_dim, config.num_tasks, JK=config.JK, drop_ratio=config.drop_ratio, 
                          graph_pooling=config.graph_pooling, gnn_type=config.gnn_type)
    if ckpt_pretrain is not None:
        model_pretrain = GNN_graphpred(config.num_layer, config.emb_dim, 1310, JK=config.JK, drop_ratio=config.drop_ratio, 
                                       graph_pooling=config.graph_pooling, gnn_type=config.gnn_type)
        
        model_pretrain = load_model(model_pretrain, ckpt_pretrain)
        model.gnn = model_pretrain.gnn
    return model

def main(args, config):
    set_random_seed(args.seed)
    log_GPU_info()

    train_loader, val_loader, test_loader = load_data(config.data, val_split=args.val_split)
    # train_loader, val_loader, test_loader = get_loader(config.data)
    model = create_model(config.model, args.ckpt_pretrain)

    logger.info(f"Start training")
    trainer = TaskTrainer(model, args.output_dir, **config.train)
    trainer.fit(train_loader, val_loader, test_loader)
    logger.info(f"Training finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='gine/pretrain.yaml')
    parser.add_argument('--output_dir', default='results/gine/pretrain')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--ckpt_pretrain', default=None, type=str)
    parser.add_argument('--val_split', type=int, default=1, help='the split index of validation set, 1-5')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    
    config = parse_config(args.config)
    main(args, config)
