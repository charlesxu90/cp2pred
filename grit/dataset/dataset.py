import os.path as osp
import pickle
import numpy as np
import pandas as pd
import torch
from ogb.utils import smiles2graph
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from loguru import logger

from utils.graph_utils import log_loaded_dataset
from .transform import RRWPTransform


def create_dataset(config, val_split):
    pre_transform = RRWPTransform(**config.pos_enc_rrwp)
    if config.dataset_name == 'cycpeptmpdb':
        dataset = CycPepDataset(root=config.root, smiles_col=config.smiles_col, target_col=config.target_col, pre_transform=pre_transform)
    else:
        raise Exception(f'Unknown dataset: {config.dataset_name}')
    log_loaded_dataset(dataset)

    split_idx = dataset.get_idx_split()
    val_idx = split_idx[val_split]
    test_idx = split_idx[val_split+1] # test split is val_split-1
    train_splits = [split_idx[i] for i in range(len(split_idx))if i != val_split+1 and i != val_split]  # the rest are training data
    train_idx = np.concatenate(train_splits, axis=0)

    train_dataset, val_dataset, test_dataset = dataset[train_idx], dataset[val_idx], dataset[test_idx]
    
    torch.set_num_threads(config.num_workers)
    val_dataset = [x for x in val_dataset]  # Fixed for valid after enumeration
    test_dataset = [x for x in test_dataset]

    return train_dataset, val_dataset, test_dataset


class CycPepDataset(InMemoryDataset):
    def __init__(self, root='data/CycPeptMPDB', smiles_col='smi', target_col='score', smiles2graph=smiles2graph, transform=None, pre_transform=None):
        """
        The label represents the cyclic peptide permeability ['score'] or if it is cpp ['is_cpp'].

        Args:
            root (string): Root directory where the dataset should be saved.
            smiles2graph (callable): A callable function that converts a SMILES string into a graph object. We use the OGB featurization.
                * The default smiles2graph requires rdkit to be installed *
        """
        self.smiles2graph = smiles2graph
        self.smiles_col = smiles_col
        self.target_col = target_col
        super().__init__(root, transform, pre_transform)
        logger.info(f"Processed data path: {self.processed_paths[0]}")
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'cyc_peptide_dataset.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        pass

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, 'all.csv.gz'))
        smiles_list = data_df[self.smiles_col]

        logger.info('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            graph = self.smiles2graph(smiles)

            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            # logger.debug(f"y: {data_df[self.target_col].iloc[i]}")
            data.y = torch.Tensor([data_df[self.target_col].iloc[i]])  # label

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        logger.info('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        """ Get dataset splits.
        """
        split_file = osp.join(self.root, 'raw', "scaffold_k_fold_idxes.pkl")
        with open(split_file, 'rb') as f:
            splits = pickle.load(f)
        # split_dict = replace_numpy_with_torchtensor(splits)
        return splits
    
