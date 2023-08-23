#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=64G
#SBATCH --constraint="v100"
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --output=log-%x-%j.out
#SBATCH --error=log-%x-%j.out

module purge
module load gcc/11.1.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /ibex/user/xux/pep-pred/env

# python -m graph_vit.pretrain_graphvit_cl --config graph_vit/pretrain_graphvit_cl.yaml --output_dir results/graph_vit/pretrain_graphvit_cl

python -m graph_vit.task_finetune --config graph_vit/task_finetune.yaml --output_dir results/graph_vit/task_finetune
# python -m graph_vit.task_finetune --config graph_vit/task_finetune.yaml --output_dir results/graph_vit/task_finetune --ckpt_cl results/graph_vit/pretrain_graphvit_cl/model_10_0.229.pt