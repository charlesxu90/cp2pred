import os
import argparse
import logging
import torch
import numpy as np
import pandas as pd

from utils.utils import parse_config, load_model, get_regresssion_metrics
from model.bert import BERT
from model.task_model import TaskPred
from dataset.tokenizer import SmilesTokenizer, AATokenizer, HELMTokenizer, BPETokenizer

def load_task_model(ckpt, config, device='cuda', model_type='smi_bert'):
    if model_type == 'smi_bert':
        tokenizer = SmilesTokenizer(max_len=config.data.max_len)
    elif model_type == 'aa_bert':
        tokenizer = AATokenizer(max_len=config.data.max_len)
    elif config.data.type == 'bpe':
        tokenizer = BPETokenizer(bpe_path=config.data.bpe_path, max_len=config.data.max_len, cls=True)
    else:
        raise ValueError(f'Invalid model_type: {model_type}')

    model = BERT(tokenizer, **config.model.bert)
    pred_model = TaskPred(model, model_type=config.model.model_type, device=device, output_size=config.model.output_size)
    model = load_model(pred_model, ckpt, device)
    model.eval()
    return model

def eval_task_model(model, X_test, y_test):
    with torch.set_grad_enabled(False):
        output, _ = model.forward(X_test)
        y_hat = output.squeeze()

    y_test_hat = y_hat.cpu().numpy()
    mae, _, _, _, _  = get_regresssion_metrics(y_test_hat, y_test)
    return mae


def main(args, config):
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    logger = logging.getLogger(__name__)

    df_test = pd.read_csv('data/CycPeptMPDB/test.csv')
    y_test = df_test.score.values
    X_test = df_test.helm.values

    ckpt_files = [f for f in os.listdir(args.ckpt_dir) if f.endswith('.pt')]
    best_mae = float('inf')
    for ckpt in ckpt_files:
        ckpt = os.path.join(args.ckpt_dir, ckpt)
        print(ckpt)
        model = load_task_model(ckpt, config, device=args.device, model_type=args.model_type)
        mae = eval_task_model(model, X_test, y_test)

        if mae < best_mae:
            best_mae = mae
            best_ckpt = ckpt
            logger.info(f'best_mae: {best_mae}, best_ckpt: {best_ckpt}')
        
    logger.info(f'best_mae: {best_mae}, best_ckpt: {best_ckpt}')
    model = load_task_model(best_ckpt, config, device=args.device, model_type=args.model_type)
    eval_task_model(model, X_test, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='aa_bert', type=str, help='model type: aa_bert, smi_bert, molclip, pep_bart')
    parser.add_argument('--ckpt_dir', type=str, help='path to checkpoint directory')
    parser.add_argument('--config', default=None, type=str, help='path to config file')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()  
    config = parse_config(args.config)
    main(args, config)
