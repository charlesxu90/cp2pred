#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=64G
#SBATCH --constraint="a100"
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --output=log-%x-%j.out
#SBATCH --error=log-%x-%j.out
#SBATCH --reservation=A100

module purge
module load gcc/11.1.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /ibex/user/xux/pep-pred/env

# Pretrain ResNet with contrastive learning, using ImageNet pretrained weights,
# torchrun --nproc_per_node=2 -m image_mol.pretrain_resnet --config image_mol/pretrain_resnet.yaml --output_dir results/resnet/pretrain_resnet

# Task finetune
# torchrun --nproc_per_node=2 -m image_mol.train --config image_mol/config_cls.yaml --output_dir results/resnet/cyc_cpp_cls --ckpt_cl results/resnet/pretrain_resnet_ibex_imgnet_init/model_100_0.027.pt --val_split 1

i=1

echo "split = $i in image_mol"
torchrun --nproc_per_node=2 -m image_mol.train --config image_mol/config_kras.yaml --output_dir results/resnet/kras_reg_$i --ckpt_cl results/resnet/pretrain_resnet_ibex_imgnet_init/model_100_0.027.pt --val_split $i