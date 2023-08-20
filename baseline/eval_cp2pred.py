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
from utils.utils import get_regresssion_metrics, load_model, get_metrics


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
    
def predict(model, X_test):
    with torch.set_grad_enabled(False):
        y_hat = model.forward(X_test.float())
    return y_hat.squeeze()

def eval_model(model, val_loader, device, classification=False, high_threshold=-6):
    y_test = []
    y_test_hat = []

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        y_hat = predict(model, x)
        y_test_hat.append(y_hat.cpu().numpy())
        y_test.append(y.cpu().numpy())

    y_test = np.concatenate(y_test, axis=0)
    y_test_hat = np.concatenate(y_test_hat, axis=0)
    if classification:
        y_test = y_test >= high_threshold
        y_test_hat = y_test_hat >= high_threshold
        score, sn, sp, mcc, auroc = get_metrics(y_test_hat, y_test, print_metrics=True)
    else:
        mae, score, _, spearman, pearson = get_regresssion_metrics(y_test_hat, y_test, print_metrics=True)
    return score

def compare_score(best_score, score, classification=False):
    if classification:
        return score > best_score
    else:
        return score < best_score

def main(args, config):
    device = torch.device(config.train.device)
    seed = args.seed
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    batch_size, num_workers = config.data.batch_size, config.data.num_workers
    _, valid_data = load_feat_data(config.data.input_path, feat_names=config.data.feat_names,)
    feat_size = len(valid_data[0][0])
    logger.info(f"Number of training features {feat_size}")

    test_set = TaskDataset(valid_data)
    val_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    pred_model = ConcatModel(input_size=feat_size, device=device, **config.model)

    logger.info(f"Start evaluating")
    ckpt_files = [f for f in os.listdir(args.ckpt_dir) if f.endswith('.pt')]
    best_score = float('inf') if not args.classification else -float('inf')
    for ckpt in ckpt_files:
        ckpt = os.path.join(args.ckpt_dir, ckpt)
        print(ckpt)

        model = load_model(pred_model, ckpt, device)
        score = eval_model(model, val_dataloader, device, classification=args.classification)
        
        if compare_score(best_score, score, classification=args.classification):
            best_score = score
            best_ckpt = ckpt
            logger.info(f'best_score: {best_score}, best_ckpt: {best_ckpt}')
    
    logger.info(f'best_score: {best_score}, best_ckpt: {best_ckpt}')
    model = load_model(pred_model, best_ckpt, device)
    eval_model(pred_model, val_dataloader, device, classification=args.classification)

    logger.info(f"Evaluating finished")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, help='path to checkpoint directory, containing .pt files')
    parser.add_argument('--config', default=None, type=str, help='path to config file')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--classification', action='store_true')

    args = parser.parse_args()

    config = parse_config(args.config)
    main(args, config)
