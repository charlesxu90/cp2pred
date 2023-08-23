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
module load gcc/11.1.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /ibex/user/xux/pep-pred/env

# python -m graph_vit.pretrain_graphvit_cl --config graph_vit/pretrain_graphvit_cl.yaml --output_dir results/graph_vit/pretrain_graphvit_cl

python -m graph_vit.task_finetune --config graph_vit/task_finetune.yaml --output_dir results/graph_vit/task_finetune
# python -m graph_vit.task_finetune --config graph_vit/task_finetune.yaml --output_dir results/graph_vit/task_finetune --ckpt_cl results/graph_vit/pretrain_graphvit_cl/model_10_0.229.pt