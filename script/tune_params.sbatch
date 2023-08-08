#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --output=log-%x-slurm-%j.out
#SBATCH --error=log-%x-slurm-%j.err
#SBATCH --reservation=A100
#SBATCH --constraint="a100"

module purge
module load gcc/11.1.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /ibex/user/xux/pep-pred/env

# Get ssh-tunnel info
export XDG_RUNTIME_DIR=/tmp node=$(hostname -s) 
user=$(whoami) 
submit_host=${SLURM_SUBMIT_HOST} 
port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo ${node} pinned to port ${port} on ${submit_host} 

echo -e "
Create ssh tunnel to view the webpage of the job: 

ssh -L ${port}:${node}.ibex.kaust.edu.sa:${port} ${user}@glogin.ibex.kaust.edu.sa 

Then copy the link provided by your web-server and replace the hostname with localhost before pasting it in your local browser.
" >&2

# launch jupyter server
jupyter-lab --no-browser --port=${port} --port-retries=0  --ip=${node}.ibex.kaust.edu.sa

# launch nni server
# nnictl create --config tune_config.yaml -p ${port}

