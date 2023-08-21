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

# Pretrain ResNet with contrastive learning, using ImageNet pretrained weights,
# torchrun --nproc_per_node=2 -m image_mol.pretrain_resnet --config image_mol/pretrain_resnet.yaml --output_dir results/resnet/pretrain_resnet

# Task finetune
torchrun --nproc_per_node=2 -m image_mol.task_finetune_resnet --config image_mol/task_finetune_resnet.yaml --output_dir results/resnet/task_finetune --ckpt_cl results/resnet/pretrain_resnet_ibex_imgnet_init/model_final_0.006.pt
torchrun --nproc_per_node=2 -m image_mol.task_finetune_resnet --config image_mol/task_cls_finetune_resnet.yaml --output_dir results/resnet/task_cls_finetune --ckpt_cl results/resnet/pretrain_resnet_ibex_imgnet_init/model_final_0.006.pt