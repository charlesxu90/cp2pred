import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from utils.utils import get_path
import logging
import torch
import cv2
from torchvision import transforms
from PIL import Image
from utils.img_utils import ImageAugmentation

np.seterr(divide='ignore', invalid='ignore')
logger = logging.getLogger(__name__)

logging.getLogger('PIL').setLevel(logging.WARNING)

def load_data(data_path, feat_names='SMILES'):
    col_names = feat_names.split(',')
    
    train_data = pd.read_csv(get_path(data_path, 'train.csv'))[col_names].values
    valid_data = pd.read_csv(get_path(data_path, 'test.csv'))[col_names].values
    return train_data, valid_data

def load_feat_data(data_path, feat_names='fps,monomer_dps', target_col='score'):

    df_train = pd.read_csv(get_path(data_path, 'train.csv'))
    df_test = pd.read_csv(get_path(data_path, 'test.csv'))
    y_train = df_train[target_col].values
    y_test = df_test[target_col].values

    features = feat_names.split(',')
    
    X_train_features = []
    X_test_features = []
    for feat in features:
        try:
            logger.debug(f"Loading feature: {feat}")
            X_train = np.load(get_path(data_path, f'X_train_{feat}.npy'))
            X_test = np.load(get_path(data_path, f'X_test_{feat}.npy'))
            if 'dps' in feat:
                # min-max normalize
                X_train = (X_train - np.min(X_train, axis=0)) / (np.max(X_train, axis=0) - np.min(X_train, axis=0))
                X_test = (X_test - np.min(X_test, axis=0)) / (np.max(X_test, axis=0) - np.min(X_test, axis=0))
                
                # drop nan columns
                col_idx = np.isnan(X_train).any(axis=0) + np.isnan(X_test).any(axis=0)
                X_train = X_train[:, ~col_idx]
                X_test = X_test[:, ~col_idx]
            X_train_features.append(X_train)
            X_test_features.append(X_test)
        except:
            raise ValueError(f'Feature {feat} not supported')

    X_train = np.concatenate(X_train_features, axis=1)
    X_test = np.concatenate(X_test_features, axis=1)
    logger.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    return list(zip(X_train, y_train)), list(zip(X_test, y_test))


def load_image_data(data_path, feat_names='smi_img', target_col='score'):

    df_train = pd.read_csv(get_path(data_path, 'img_train.csv'))
    df_test = pd.read_csv(get_path(data_path, 'img_test.csv'))
    # logger.debug(f"train data columns: {df_train.columns}")

    y_train = df_train[target_col].values
    y_test = df_test[target_col].values

    
    features = feat_names.split(',')
    X_train = df_train[features].values
    X_test = df_test[features].values

    return list(zip(X_train, y_train)), list(zip(X_test, y_test))


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


class CrossDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        self.len = len(self.dataset)
        return self.len

    def __getitem__(self, idx):
        item = self.dataset[idx]
        [item1, item2] = item
        # logger.debug(f'CrossDataset: {item1}, {item2}')
        return item1, item2
    

class TaskDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        self.len = len(self.dataset)
        return self.len

    def __getitem__(self, idx):
        item = self.dataset[idx]
        [feat, prop] = item
        # logger.debug(f'TaskDataset: {feat.shape}, {prop}')
        return feat, prop
    

def cl_collate(batch):
    """ collate function for siamese network """
    # return pair of sequences by split data into two halves (seq1, seq2, label1, label2, label)

    batch_size = len(batch)
    # logger.debug(f"batch_size: {batch_size}")
    # logger.debug(f"batch type: {type(batch)}")

    seq1_batch = [batch[i] for i in range(int(batch_size / 2))]
    seq2_batch = [batch[i + int(batch_size/2)] for i in range(int(batch_size / 2))]
    # logger.debug(f"seq1_batch: {seq1_batch}")
    seq1, label1 = zip(*seq1_batch)
    seq2, label2 = zip(*seq2_batch)
    # logger.debug(f"seq1: {seq1}, label1: {label1}")
    label = [label1[i] ^ label2[i] for i in range(int(batch_size / 2))]

    # covert label to tensor
    label1 = torch.tensor(label1)
    label2 = torch.tensor(label2)
    label = torch.tensor(label)
    
    return list(seq1), list(seq2), label1, label2, label


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
        # logger.debug(f'ImageDataset: {filepath[0]}, {prop}')
        img = Image.open(filepath[0])
        img_t = self.transform(img)
        
        return img_t, prop

