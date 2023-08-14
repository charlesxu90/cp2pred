#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=64G
#SBATCH --constraint="a100"
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --output=log-%x-slurm-%j.out
#SBATCH --error=log-%x-slurm-%j.out
#SBATCH --reservation=A100

module purge
module load gcc/11.1.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /ibex/user/xux/pep-pred/env


config_name='pretrain_resnet_ibex'
torchrun --nproc_per_node=2 pretrain_resnet.py --config configs/${config_name}.yaml --output_dir results/${config_name} --debug
