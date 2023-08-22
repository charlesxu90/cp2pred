import os.path as osp
import pickle
import pandas as pd
import torch
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from loguru import logger
from .transform import GraphPartitionTransform, PositionalEncodingTransform


class CycPepDataset(InMemoryDataset):
    def __init__(self, root='data/CycPeptMPDB', target_col='score', smiles2graph=smiles2graph, transform=None, pre_transform=None):
        """

        The label represents the cyclic peptide permeability ['score'].

        Args:
            root (string): Root directory where the dataset should be saved.
            smiles2graph (callable): A callable function that converts a SMILES
                string into a graph object. We use the OGB featurization.
                * The default smiles2graph requires rdkit to be installed *
        """
        self.smiles2graph = smiles2graph
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
        smiles_list = data_df['smi']

        print('Converting SMILES strings into graphs...')
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
            data.y = data_df[self.target_col].iloc[i]  # label

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        """ Get dataset splits.

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        split_file = osp.join(self.root, "splits_random_stratified_peptide.pickle")
        with open(split_file, 'rb') as f:
            splits = pickle.load(f)
        split_dict = replace_numpy_with_torchtensor(splits)
        return split_dict
    

def create_dataset(config):
    pre_transform = PositionalEncodingTransform(rw_dim=config.pos_enc.rw_dim, lap_dim=config.pos_enc.lap_dim)
    transform_train = transform_eval = None

    if config.metis.n_patches > 0:
        _transform_train = GraphPartitionTransform(n_patches=config.metis.n_patches,
                                                   metis=config.metis.enable,
                                                   drop_rate=config.metis.drop_rate,
                                                   num_hops=config.metis.num_hops,
                                                   is_directed=False,
                                                   patch_rw_dim=config.pos_enc.patch_rw_dim,
                                                   patch_num_diff=config.pos_enc.patch_num_diff)

        _transform_eval = GraphPartitionTransform(n_patches=config.metis.n_patches,
                                                  metis=config.metis.enable,
                                                  drop_rate=0.0,
                                                  num_hops=config.metis.num_hops,
                                                  is_directed=False,
                                                  patch_rw_dim=config.pos_enc.patch_rw_dim,
                                                  patch_num_diff=config.pos_enc.patch_num_diff)

        transform_train = _transform_train
        transform_eval = _transform_eval

    dataset = CycPepDataset(root=config.root, target_col=config.target_col, pre_transform=pre_transform)
    split_idx = dataset.get_idx_split()
    train_dataset, val_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['val']], dataset[split_idx['test']]
    train_dataset.transform, val_dataset.transform, test_dataset.transform = transform_train, transform_eval, transform_eval

    torch.set_num_threads(config.num_workers)
    if not config.metis.online:
        train_dataset = [x for x in train_dataset]
    val_dataset = [x for x in val_dataset]
    test_dataset = [x for x in test_dataset]

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    dataset = CycPepDataset()
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[100])
    print(dataset[100].y)
    print(dataset.get_idx_split())
