#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=64G
#SBATCH --constraint="v100"
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --output=log-%x-slurm-%j.out
#SBATCH --error=log-%x-slurm-%j.err
## SBATCH --reservation=v100

module load gcc/11.1.0

nnictl create --config tune_config.yaml -p 8081