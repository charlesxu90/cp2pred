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

from utils.utils import parse_config, load_model, log_GPU_info
from utils.dist import init_distributed, get_rank, is_main_process
from .dataset import load_data, TaskDataset
from .tokenizer import SmilesTokenizer, HELMTokenizer
from .bert import BERT
from .task_model import TaskPred
from .task_trainer import TaskTrainer
warnings.filterwarnings("ignore", message="torch.distributed._all_gather_base is a private function")

@record
def main(args, config):
    device = torch.device(config.train.device)
    if config.train.distributed:
        init_distributed()
        global_rank = get_rank()
        seed = args.seed + global_rank
    else:
        seed = args.seed
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if is_main_process():
        log_GPU_info()
    
    batch_size, num_workers = config.data.batch_size, config.data.num_workers
    
    train_data, valid_data = load_data(config.data.input_path, feat_names=config.data.col_names,)
    train_set, test_set = TaskDataset(train_data), TaskDataset(valid_data)
    train_sampler = DistributedSampler(dataset=train_set, shuffle=True, rank=global_rank) if config.train.distributed else None
    train_dataloader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)

    test_sampler = DistributedSampler(dataset=test_set, shuffle=False, rank=global_rank) if config.train.distributed else None
    test_dataloader = DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=num_workers)
    # logger.debug(f"train_sampler: {len(train_sampler)}, test_sampler: {len(test_sampler)}")
    val_dataloader = DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=num_workers)
    
    if config.data.type == 'smiles':
        tokenizer = SmilesTokenizer(max_len=config.data.max_len)
    elif config.data.type == 'helm':
        tokenizer = HELMTokenizer(max_len=config.data.max_len)
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
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)    
    config = parse_config(args.config)
    main(args, config)
