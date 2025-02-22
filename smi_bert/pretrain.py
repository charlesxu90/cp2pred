import argparse
import os
import torch
import numpy as np
import random
from pathlib import Path
from loguru import logger
from torch.utils.data import DataLoader

from utils.utils import parse_config, load_model, log_GPU_info
from .dataset.dataset import load_data, UniDataset
from .dataset.tokenizer import SmilesTokenizer, HELMTokenizer
from .model.bert import BERT
from .model.bert_trainer import BertTrainer
from torch.utils.data.distributed import DistributedSampler
from utils.dist import init_distributed, get_rank, is_main_process
from torch.distributed.elastic.multiprocessing.errors import record


@record
def main(args, config):
    init_distributed()
    global_rank = get_rank()

    device = torch.device(args.device)
    seed = args.seed + global_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if is_main_process():
        log_GPU_info()
    
    train_data, valid_data = load_data(config.data.input_path, feat_names=config.data.feat_col,)
    train_set, test_set = UniDataset(train_data), UniDataset(valid_data)
    train_sampler = DistributedSampler(dataset=train_set, shuffle=True, rank=global_rank)
    train_dataloader = DataLoader(train_set, batch_size=config.data.batch_size, sampler=train_sampler, num_workers=config.data.num_workers, pin_memory=True)

    test_sampler = DistributedSampler(dataset=test_set, shuffle=False, rank=global_rank)
    test_dataloader = DataLoader(test_set, batch_size=config.data.batch_size, sampler=test_sampler, shuffle=False, num_workers=config.data.num_workers, pin_memory=True)

    if config.data.type == 'smiles':
        tokenizer = SmilesTokenizer(max_len=config.data.max_len)
    else:
        raise Exception(f"Unknown data type: {config.data.type}")
    
    model = BERT(tokenizer=tokenizer, **config.model).to(device)
    if args.ckpt is not None:
        model = load_model(model, args.ckpt, device)
    
    logger.info(f"Start training")
    trainer = BertTrainer(model, args.output_dir, **config.train)
    trainer.fit(train_dataloader, test_dataloader)
    logger.info(f"Training finished")
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='seq_bert/pretrain_smi_bert.yaml')
    parser.add_argument('--output_dir', default='results/pretrain_smi_bert/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--ckpt', default=None, type=str, help='path to checkpoint to load')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    
    config = parse_config(args.config)
    main(args, config)
