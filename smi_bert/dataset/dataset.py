import pickle
import pandas as pd
import numpy as np
import os.path as osp
from loguru import logger
from torch.utils.data import Dataset
from utils.utils import get_path

np.seterr(divide='ignore', invalid='ignore')


def load_data(data_path, feat_names='SMILES'):
    col_names = feat_names.split(',')
    
    train_data = pd.read_csv(get_path(data_path, 'train.csv'))[col_names].values
    valid_data = pd.read_csv(get_path(data_path, 'test.csv'))[col_names].values
    return train_data, valid_data


class UniDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        self.len = len(self.dataset)
        return self.len

    def __getitem__(self, idx):
        item = self.dataset[idx][0]
        # logger.debug(f'UniDataset: {item}')
        return item
    

def load_task_dataset(data_path, smiles_col='smi', target_col='score', val_split=1):
    
    raw_file = osp.join(data_path, 'raw', "all.csv.gz")
    data = pd.read_csv(raw_file)[[smiles_col, target_col]].values

    split_file = osp.join(data_path, 'raw', "scaffold_k_fold_idxes.pkl")
    with open(split_file, 'rb') as f:
        splits = pickle.load(f)

    val_idx = splits[val_split]
    test_idx = splits[val_split+1] # test split is val_split-1
    train_splits = [splits[i] for i in range(len(splits))if i != val_split+1 and i != val_split]  # the rest are training data
    train_idx = np.concatenate(train_splits, axis=0)

    train_data, val_data, test_data = data[train_idx], data[val_idx], data[test_idx]

    train_dataset, val_dataset, test_dataset = TaskDataset(train_data), TaskDataset(val_data), TaskDataset(test_data)
    return train_dataset, val_dataset, test_dataset


class TaskDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        self.len = len(self.dataset)
        return self.len

    def __getitem__(self, idx):
        item = self.dataset[idx]
        [feat, prop] = item
        # logger.debug(f'TaskDataset: {feat}, {prop}')
        return feat, prop
