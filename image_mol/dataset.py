
import pickle
import pandas as pd
import numpy as np
import os.path as osp
from loguru import logger
from torch.utils.data import Dataset
from utils.utils import get_path
import logging
import torch
import cv2
from torchvision import transforms
from PIL import Image
from utils.img_utils import ImageAugmentation

np.seterr(divide='ignore', invalid='ignore')

logging.getLogger('PIL').setLevel(logging.WARNING)

def load_image_data(data_path, feat_col='smi_img', target_col='score', image_size=500):

    train_data = pd.read_csv(get_path(data_path, 'train.csv'))[[feat_col, target_col]].values
    test_data = pd.read_csv(get_path(data_path, 'test.csv'))[[feat_col, target_col]].values

    train_set, test_set = ImageDataset(train_data, image_size=image_size), ImageDataset(test_data, image_size=image_size)
    return train_set, test_set


def load_image_data_by_split(data_path, feat_col='smi_img', target_col='score', val_split=1, image_size=500):
    
    raw_file = osp.join(data_path, 'raw', "all.csv.gz")
    data = pd.read_csv(raw_file)[[feat_col, target_col]].values

    split_file = osp.join(data_path, 'raw', "scaffold_k_fold_idxes.pkl")
    with open(split_file, 'rb') as f:
        splits = pickle.load(f)

    val_idx = splits[val_split]
    test_idx = splits[val_split+1] # test split is val_split-1
    train_splits = [splits[i] for i in range(len(splits))if i != val_split+1 and i != val_split]  # the rest are training data
    train_idx = np.concatenate(train_splits, axis=0)

    train_data, val_data, test_data = data[train_idx], data[val_idx], data[test_idx]

    train_set = TaskImageDataset(train_data, image_size=image_size)
    val_set = TaskImageDataset(val_data, image_size=image_size)
    test_set = TaskImageDataset(test_data, image_size=image_size)

    return train_set, val_set, test_set
    

class ImageDataset(Dataset):
    def __init__(self, dataset, image_size=224):
        self.dataset = dataset
        self.img_augmentor = ImageAugmentation()
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((image_size, image_size)), transforms.ToTensor(), 
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])

    def __len__(self):
        self.len = len(self.dataset)
        return self.len

    def __getitem__(self, idx):
        item = self.dataset[idx]
        [filepath, prop] = item
        # logger.debug(f'ImageDataset: {filepath[0]}')
        img = cv2.imread(filepath[0])

        img_aug = self.img_augmentor(img)
        img = self.transform(img)
        img_aug = self.transform(img_aug)
        
        return img, img_aug, prop

class TaskImageDataset(Dataset):
    def __init__(self, dataset, image_size=224):
        self.dataset = dataset
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), 
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])

    def __len__(self):
        self.len = len(self.dataset)
        return self.len

    def __getitem__(self, idx):
        item = self.dataset[idx]
        [filepath, prop] = item
        # logger.debug(f'ImageDataset: {filepath}, {prop}')
        img = Image.open(filepath)
        img = self.transform(img)
        
        return img, prop

