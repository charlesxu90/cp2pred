import argparse
import logging
import os
import torch
import numpy as np
import pandas as pd

from utils.utils import parse_config, load_model, get_metrics, get_regresssion_metrics
from model.bert import BERT
# from model.molclip import MolCLIP
# from model.pbart import PepBART
from dataset.tokenizer import SmilesTokenizer, AATokenizer, BPETokenizer, HELMTokenizer
from torch.distributed.elastic.multiprocessing.errors import record


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def load_bert_model(ckpt, config, device='cuda', model_type='smi_bert'):
    if model_type == 'smi_bert':
        tokenizer = SmilesTokenizer(max_len=config.data.max_len)
    elif model_type == 'helm_bert':
        tokenizer = HELMTokenizer(max_len=config.data.max_len)
    elif config.data.type == 'bpe':
        tokenizer = BPETokenizer(bpe_path=config.data.bpe_path, max_len=config.data.max_len)
    else:
        raise ValueError(f'Invalid model_type: {model_type}')

    model = BERT(tokenizer, **config.model)
    model = load_model(model, ckpt, device)
    model.eval()
    return model, device

def get_bert_embd(encoder, inputs, device='cuda',):
    with torch.no_grad():
        tokens = encoder.tokenize_inputs(inputs).to(device)
        batch_lens = (tokens != encoder.tokenizer.pad_token_id).sum(1)
        embd = encoder.embed(tokens)
        reps = []
        for i, tokens_len in enumerate(batch_lens):
            reps.append(embd[i, 1 : tokens_len - 1].mean(0))
    return torch.stack(reps)

def encode_with_bert(list, model, device='cuda', batch_size=128):
    reps = []
    for i in range(0, len(list), batch_size):
        reps.append(get_bert_embd(model, list[i : i + batch_size], device=device))
    reps = torch.cat(reps).cpu().numpy()
    # logger.info(f"list shape: {list.shape}, reps shape: {reps.shape}")
    return reps

def get_metric_by_clf(X_train, y_train, X_test, y_test, clf='xgb'):
    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    mae, _, _, _, _  = get_regresssion_metrics(y_hat, y_test)
    return mae

@record
def main(args, config):
    df_train = pd.read_csv('data/CycPeptMPDB/train.csv')
    df_test = pd.read_csv('data/CycPeptMPDB/test.csv')

    x_train = df_train[config.data.feat_col].values
    y_train = df_train[config.data.target_col].values
    x_test = df_test[config.data.feat_col].values
    y_test = df_test[config.data.target_col].values

    ckpt_files = [f for f in os.listdir(args.ckpt_dir) if f.endswith('.pt')]
    best_mae = float('inf')
    for ckpt in ckpt_files:
        ckpt = os.path.join(args.ckpt_dir, ckpt)
        print(ckpt)

        model, device = load_bert_model(ckpt=ckpt, config=config, device=args.device, model_type=args.model_type)
        X_train = encode_with_bert(x_train, model, device=device,)
        X_test = encode_with_bert(x_test, model, device=device, )

        logger.debug(f"data shape X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
        mae = get_metric_by_clf(X_train, y_train, X_test, y_test, clf=args.clf)
        if mae < best_mae:
            best_mae = mae
            best_ckpt = ckpt
            logger.info(f'best_mae: {best_mae}, best_ckpt: {best_ckpt}')
    logger.info(f'best_mae: {best_mae}, best_ckpt: {best_ckpt}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='smi_bert', type=str, help='model type: smi_bert, molclip, pep_bart')
    parser.add_argument('--ckpt_dir', type=str, help='path to checkpoint directory, containing .pt files')
    parser.add_argument('--config', default=None, type=str, help='path to config file')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--clf', default='rf', type=str, help='classifier: rf, svm, xgb')

    args = parser.parse_args()  
    config = parse_config(args.config)
    main(args, config)
