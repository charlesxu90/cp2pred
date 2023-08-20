import argparse
import warnings
import joblib
import pickle
import numpy as np
import pandas as pd
import os.path as osp
from typing import List
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
from rdkit import Chem, rdBase, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, rdMolDescriptors
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


from utils.utils import get_regresssion_metrics, parse_config

rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def fingerprints_from_smiles(smiles: List, size=2048):
    """ Create ECFP fingerprints of smiles, with validity check """
    fps = []
    valid_mask = []
    for i, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile)
        valid_mask.append(int(mol is not None))
        fp = fingerprints_from_mol(mol, size=size) if mol else np.zeros((1, size))
        fps.append(fp)

    fps = np.concatenate(fps, axis=0)
    return fps, valid_mask


def fingerprints_from_mol(molecule, radius=3, size=2048, hashed=False):
    """ Create ECFP fingerprint of a molecule """
    if hashed:
        fp_bits = AllChem.GetHashedMorganFingerprint(molecule, radius, nBits=size)
    else:
        fp_bits = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=size)
    fp_np = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp_bits, fp_np)
    return fp_np.reshape(1, -1)


def getMolDescriptors(mol, missingVal=0):
    """ calculate the full list of descriptors for a molecule """

    values, names = [], []
    for nm, fn in Descriptors._descList:
        try:
            val = fn(mol)
        except:
            val = missingVal
        values.append(val)
        names.append(nm)

    custom_descriptors = {'hydrogen-bond donors': rdMolDescriptors.CalcNumLipinskiHBD,
                          'hydrogen-bond acceptors': rdMolDescriptors.CalcNumLipinskiHBA,
                          'rotatable bonds': rdMolDescriptors.CalcNumRotatableBonds,}
    
    for nm, fn in custom_descriptors.items():
        try:
            val = fn(mol)
        except:
            val = missingVal
        values.append(val)
        names.append(nm)
    return values,names


def get_pep_dps_from_smi(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        print(f"convert smi {smi} to molecule failed!")
        mol = None
    
    dps, _ = getMolDescriptors(mol)
    return np.array(dps)


def get_pep_dps(smi_list):
    return np.array([get_pep_dps_from_smi(smi) for smi in smi_list])


def plot_scatter_y(y, y_hat, x_label='y', y_label='y_hat', save_path=None, label='test'):
    plt.scatter(y, y_hat, alpha=0.1)
    mae, _, _, _, _ = get_regresssion_metrics(y, y_hat)
    mae = np.mean(np.abs(y - y_hat))
    logger.info(f'{label} MAE: {mae:.3f}, y mean: {np.mean(y):.3f}, y_hat mean: {np.mean(y_hat):.3f}')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_path is not None:
        plt.savefig(f'{save_path}_{label}_{mae:.3f}.pdf', dpi=300, bbox_inches='tight')


def load_data(config, ):
    logger.info(f"Start loading data")
    data_df = pd.read_csv(osp.join(config.root_path, 'raw', 'all.csv.gz'))
    split_file = osp.join(config.root_path, "splits_random_stratified_peptide.pickle")
    with open(split_file, 'rb') as f:
        split_idx = pickle.load(f)

    smiles_list = data_df['smi'].tolist()

    if not config.processed:
        Path(osp.join(config.root_path, 'processed')).mkdir(parents=True, exist_ok=True)  
        logger.info(f"Processing features for molecule fingerprints")
        X_fps = fingerprints_from_smiles(smiles_list)[0]
        np.save(osp.join(config.root_path, 'processed', 'X_fps.npy'), X_fps)
        logger.info(f"Processing features for molecule descriptors")
        X_dps = get_pep_dps(smiles_list)
        np.save(osp.join(config.root_path, 'processed', 'X_dps.npy'), X_dps)

    logger.info(f"Loading features {config.features}")
    features = config.features.split(',')
    X_features = []
    for feat in features:
        feat_path = osp.join(config.root_path, 'processed', f'X_{feat}.npy')
        try:
            X_feat = np.load(feat_path)
            X_features.append(X_feat)
        except:
            raise ValueError(f'Feature {feat} not found in dir {feat_path}!')

    X = np.concatenate(X_features, axis=1)
    y = data_df[config.label].values

    X_train, X_val, X_test = X[split_idx['train']], X[split_idx['val']], X[split_idx['test']]
    y_train, y_val, y_test = y[split_idx['train']], y[split_idx['val']], y[split_idx['test']]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def init_model(config):
    if config.model_type == 'xgboost':
        model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    elif config.model_type == 'rf':
        model = RandomForestRegressor()

    return model


def main(config):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(config.data)
    logger.info(f"Data splits: train {len(X_train)}, val {len(X_val)}, test {len(X_test)}")
    logger.info(f"Total features {X_train.shape[1]}")
    model = init_model(config.model)
    logger.info(f'Training {config.model.model_type} model on {config.model.task_type} task ...')
    model.fit(X_train, y_train)

    Path(config.model.save_dir).mkdir(parents=True, exist_ok=True)   
    save_path = osp.join(config.model.save_dir, f'{config.model.task_type}_{config.model.model_type}.pkl')
    joblib.dump(model, save_path)
    # model = joblib.load(save_path)


    logger.info(f'Evaluate on training dataset')
    y_hat = model.predict(X_train)
    plot_scatter_y(y_train, y_hat, x_label='y_val', y_label='y_hat_val', save_path=save_path, label='train')

    logger.info(f'Evaluate on validataion dataset')
    y_hat = model.predict(X_val)
    plot_scatter_y(y_val, y_hat, x_label='y_val', y_label='y_hat_val', save_path=save_path, label='val')

    logger.info(f'Evaluate on testing dataset')
    y_hat = model.predict(X_test)
    plot_scatter_y(y_test, y_hat, x_label='y_test', y_label='y_hat_test', save_path=save_path, label='test')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='baseline/config.yaml')

    args = parser.parse_args()
    config = parse_config(args.config)


    main(config)
