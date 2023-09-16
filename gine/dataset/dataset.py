import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem
from itertools import repeat

import torch
from torch_geometric.data import Data, InMemoryDataset

from utils.graph_utils import log_loaded_dataset

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),  # Atom 1-119
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,  # chirality that hasn't been specified
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,  # tetrahedral: clockwise rotation
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,  # tetrahedral: counter-clockwise rotation
        Chem.rdchem.ChiralType.CHI_OTHER  # some unrecognized type of chirality
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, 
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, 
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, 
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [  # Bond type
        Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,  # for cis/trans
        Chem.rdchem.BondDir.ENDDOWNRIGHT  # for cis/trans
    ]
}

def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + [
                        allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + [
                            allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


class CycPepDataset(InMemoryDataset):
    def __init__(self,
                 root, 
                 smiles_col='smi', target_col='score', 
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,):
        self.root = root
        self.smiles_col = smiles_col
        self.target_col = target_col
        super(CycPepDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        logger.info(f"Processed data path: {self.processed_paths[0]}")
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get(self, idx):
        data = Data()
        for key in self._data.keys:
            item, slices = self._data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        return data

    @property
    def raw_file_names(self):
        return 'all.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        pass

    def process(self):
        data_smiles_list = []
        data_list = []

        input_df = pd.read_csv(self.raw_paths[0], sep=',')
        smiles_list = input_df[self.smiles_col]
        rdkit_mol_objs = [AllChem.MolFromSmiles(s) for s in smiles_list]
        labels = input_df[self.target_col]

        for i in tqdm(range(len(smiles_list))):
            rdkit_mol = rdkit_mol_objs[i]
            data = mol_to_graph_data_obj_simple(rdkit_mol)
            # manually add mol id
            data.id = torch.tensor([i])  # id here is the index of the mol in the dataset
            data.y = torch.tensor(labels[i])
            data_list.append(data)
            data_smiles_list.append(smiles_list[i])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir, 'smiles.csv'), index=False, header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
