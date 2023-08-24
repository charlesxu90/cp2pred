# Cyclic Cell Penetration Peptide prediction (CP2pred)

## Prepare environment

Create conda environment
```shell
mamba env create -f environment.yml
conda activate cp2pred-env
```
Install pip requirements
```shell
pip install -r requirements.txt
```

Install pytorch and pytorch-geometric
```shell
# pip install torchmetrics
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
pip install torch_geometric

# Optional dependencies:
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

Install Ray Tune

```shell
pip install ray==2.2 'ray[tune]' 'ray[rllib]' 'ray[serve]'
```
