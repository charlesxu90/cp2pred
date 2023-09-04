import argparse
import torch
import random
import warnings
import numpy as np
from pathlib import Path
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record

from utils.utils import parse_config, load_model, log_GPU_info, set_random_seed
from utils.dist import init_distributed, get_rank, is_main_process
from .dataset.dataset import load_task_dataset, TaskDataset
from .dataset.tokenizer import SmilesTokenizer, HELMTokenizer
from .model.bert import BERT
from .model.task_model import TaskPred
from .model.task_trainer import TaskTrainer
warnings.filterwarnings("ignore", message="torch.distributed._all_gather_base is a private function")


def get_dataloaders(config, distributed=True, val_split=1, global_rank=0):
    batch_size, num_workers = config.batch_size, config.num_workers
    
    train_set, val_set, test_set = load_task_dataset(config.input_path, smiles_col=config.smiles_col, 
                                                     target_col=config.target_col, val_split=val_split)
    
    train_sampler = DistributedSampler(dataset=train_set, shuffle=True, rank=global_rank) if distributed else None
    train_dataloader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)

    val_sampler = DistributedSampler(dataset=val_set, shuffle=False, rank=global_rank) if distributed else None
    val_dataloader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler, shuffle=False, num_workers=num_workers)

    test_sampler = DistributedSampler(dataset=test_set, shuffle=False, rank=global_rank) if distributed else None
    test_dataloader = DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=num_workers)
    
    return train_dataloader, val_dataloader, test_dataloader


@record
def main(args, config):
    device = torch.device(config.train.device)
    if config.train.distributed:
        init_distributed()
        global_rank = get_rank()
        seed = args.seed + global_rank
    else:
        seed = args.seed
    
    set_random_seed(seed)

    if is_main_process():
        log_GPU_info()
    
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config.data, distributed=config.train.distributed, 
                                                                        val_split=args.val_split, global_rank=global_rank)

    if config.data.type == 'smiles':
        tokenizer = SmilesTokenizer(max_len=config.data.max_len)
    else:
        raise Exception(f"Unknown data type: {config.data.type}")
    
    model = BERT(tokenizer=tokenizer, **config.model.bert).to(device)
    if args.ckpt is not None:
        bert_model = load_model(model, args.ckpt, device)

    pred_model = TaskPred(bert_model, device=device, output_size=config.model.output_size)

    logger.info(f"Start training")
    trainer = TaskTrainer(pred_model, args.output_dir, **config.train)
    trainer.fit(train_dataloader, test_dataloader, val_dataloader)
    logger.info(f"Training finished")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='seq_bert/smi_bert_task_finetune.yaml')
    parser.add_argument('--output_dir', default='results/smi_bert_task_finetune/')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--ckpt', default=None, type=str, help='path to checkpoint to load')
    parser.add_argument('--ckpt_model_type', default='bert', type=str, help='model type of checkpoint, bert, pep_bart, or molclip')
    parser.add_argument('--val_split', type=int, default=1, help='the split index of validation set, 1-5')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    
    config = parse_config(args.config)
    main(args, config)
