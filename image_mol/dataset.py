
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

logging.getLogger('PIL').setLevel(logging.WARNING)

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

