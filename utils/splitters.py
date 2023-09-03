from rdkit.Chem.Scaffolds import MurckoScaffold
from loguru import logger


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold

# # test generate_scaffold
# s = 'Cc1cc(Oc2nccc(CCC)c2)ccc1'
# scaffold = generate_scaffold(s)
# assert scaffold == 'c1ccc(Oc2ccccn2)cc1'

def k_fold_scaffold_split(smiles_list, k_fold=10):
    """
    Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    Split dataset by Bemis-Murcko scaffolds
    """
    logger.info(f"Totally {len(smiles_list)} smiles")
    # create dict of scaffold2idx {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [scaffold_set for (scaffold, scaffold_set) in 
                         sorted(all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)]
    
    logger.info(f"Totally {len(all_scaffold_sets)} scaffold sets")
    logger.info(f"In 75 quartile, each scaffold set has {len(all_scaffold_sets[len(all_scaffold_sets)//4])} smiles")

    # get indices
    n_cutoff = int(len(smiles_list) / k_fold)
    k_fold_idxes = []
    cur_fold = []
    for scaffold_set in all_scaffold_sets:
        if len(cur_fold) + len(scaffold_set) >= n_cutoff:
            cur_fold.extend(scaffold_set)
            k_fold_idxes.append(cur_fold)
            cur_fold = []
        else:
            cur_fold.extend(scaffold_set)
    k_fold_idxes.append(cur_fold)

    assert len(k_fold_idxes) == k_fold, f"Check the number of folds, it should be {k_fold}, but found {len(k_fold_idxes)}"
    for i in range(k_fold-1):
        for j in range(i+1, k_fold):
            assert len(set(k_fold_idxes[i]).intersection(set(k_fold_idxes[j]))) == 0

    return k_fold_idxes

def main(args):
    import pandas as pd
    import os
    import pickle

    # test scaffold_split
    smiles_list = pd.read_csv(args.raw_data_path)[args.smi_col].tolist()
    k_fold_idxes = k_fold_scaffold_split(smiles_list, k_fold=args.k_fold)
    unique_ids = set()
    for idxes in k_fold_idxes:
        unique_ids.update(idxes)

    assert len(unique_ids) == len(smiles_list)  # check that we did not have any missing or overlapping examples
    
    save_path = os.path.join(os.path.dirname(args.raw_data_path), f'scaffold_k_fold_idxes.pkl')
    pickle.dump(k_fold_idxes, open(save_path, 'wb'))
    logger.info(f"Save scaffold k-fold indices to {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', type=str, default='data/CycPeptMPDB/raw/all.csv.gz')
    parser.add_argument('--smi_col', type=str, default='smi')
    parser.add_argument('--k_fold', type=int, default=10)

    args = parser.parse_args()
    main(args)
