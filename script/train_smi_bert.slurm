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

module purge
module load gcc/11.1.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /ibex/user/xux/pep-pred/env


# Pretrain
# torchrun --nproc_per_node=2 -m seq_bert.pretrain_bert --config seq_bert/pretrain_smi_bert.yaml --output_dir results/smi_bert/pretrain_smi_bert_ibex

# Finetune
# torchrun --nproc_per_node=2 -m seq_bert.pretrain_bert --config seq_bert/smi_bert_finetune.yaml --output_dir results/smi_bert/smi_bert_finetune --ckpt results/smi_bert/pretrain_smi_bert_ibex/model_final_0.384.pt

# Task finetune
torchrun --nproc_per_node=2 -m seq_bert.task_finetune --config seq_bert/smi_bert_task_finetune.yaml --output_dir results/smi_bert/smi_bert_task_finetune --ckpt results/smi_bert/smi_bert_finetune/model_19_0.223.pt
torchrun --nproc_per_node=2 -m seq_bert.task_finetune --config seq_bert/smi_bert_cls_task_finetune.yaml --output_dir results/smi_bert/smi_bert_cls_task_finetune --ckpt results/smi_bert/smi_bert_finetune/model_19_0.223.pt
