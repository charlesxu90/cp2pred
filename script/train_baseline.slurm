#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=100
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=batch
#SBATCH --output=log-%x-%j.out
#SBATCH --error=log-%x-%j.out

module purge
module load gcc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /ibex/user/xux/pep-pred/env

i=1
python -m baseline.train --config baseline/config_kras_kd.yaml --val_split $i --output_dir results/baseline/kras_kd_$i