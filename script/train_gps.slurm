#!/bin/bash
#SBATCH --output=log-%x-%j.out
#SBATCH --error=log-%x-%j.out
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=4:00:00
#SBATCH --constraint="v100"

module purge
module load gcc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /ibex/user/xux/pep-pred/env

# python -m graph_vit.pretrain_graphvit_cl --config graph_vit/pretrain_graphvit_cl.yaml --output_dir results/graph_vit/pretrain_graphvit_cl

rm -rf data/kras_kd/processed/*.pt
i=1
python -m gps.train --config gps/config_kras_kd.yaml --output_dir results/gps/kras_kd_$i --val_split $i