#!/bin/bash
#SBATCH --output=log-%x-%j.out
#SBATCH --error=log-%x-%j.out
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=24:00:00
#SBATCH --constraint="v100"

module purge
module load gcc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/xux/work/cp2pred/env

# python -m graph_vit.pretrain_graphvit_cl --config graph_vit/pretrain_graphvit_cl.yaml --output_dir results/graph_vit/pretrain_graphvit_cl

rm -rf data/kras_kd/processed/*.pt
for i in {1..5};
do
nohup python -m graph_vit.train --config graph_vit/config_kras_kd.yaml --output_dir results/graph_vit/kras_kd_reg_$i --val_split $i 2>&1 >results/graph_vit/kras_kd_reg_$i.log &
done