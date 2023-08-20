import pandas as pd
import numpy as np
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
