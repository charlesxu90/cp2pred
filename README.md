# Cyclic Cell Penetration Peptide prediction (CP2pred)

## Prepare environment

```shell
conda env create -f environment.yml -p ./env
conda activate ./env
```

Install pytorch-geometric
```shell
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
```

Install metis
```shell
conda install -c conda-forge metis
pip install metis

```