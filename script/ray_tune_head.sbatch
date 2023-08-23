#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=3
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --output=log-%x-slurm-%j.out
#SBATCH --error=log-%x-slurm-%j.out
#SBATCH --reservation=A100
#SBATCH --constraint="a100"

module purge
module load gcc/11.1.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /ibex/user/xux/pep-pred/env

NUM_WORKERS=8

# Create head node
export head_node_ip=$(hostname -I | cut -d " " -f 2)
export server_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export redis_password=${SLURM_JOBID}
export dashboard_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export ip_head=${head_node_ip}:${server_port}
echo "${ip_head} ${redis_password} ${dashboard_port}" > head_node_info

ray start --node-ip-address ${head_node_ip} --port ${server_port} --redis-password=${redis_password} --head  \
	--dashboard-port ${dashboard_port} --dashboard-host=$HOSTNAME --num-cpus ${SLURM_CPUS_PER_TASK} --block &

# Get ssh-tunnel to view the dashboard
user=$(whoami)

echo -e "
Create ssh tunnel to view the webpage of the job: 

ssh -L ${dashboard_port}:${HOSTNAME}.ibex.kaust.edu.sa:${dashboard_port} ${user}@glogin.ibex.kaust.edu.sa 

Then copy the link provided by your web-server and replace the hostname with localhost before pasting it in your local browser.
" >&2

# Launch worker nodes
sleep 20
job_ids=()
for (( i=1; i<=${NUM_WORKERS}; i++ ))
 do
   job_ids[$i]=$(sbatch -x $SLURM_NODELIST script/ray_worker_node.slurm | cut -d " " -f 4)
 done 

while [ ! -z $(squeue -n ray_worker -t PD -h -o %A) ]
do
	echo "Waiting for worker(s) to start"
    sleep 30
done
sleep 20
ray status --address ${ip_head} --redis_password ${redis_password}

# ==== Code to run ray tune ==== #
# The following is the command to run ray tune, the most critical part to set for a task
python -u hello_tune.py --num-samples=100 --max-concurrent-trials=20

# Shutdown worker nodes
touch $PWD/shutdown.lock
sleep 20

# Shutdown head node
ray stop
rm $PWD/shutdown.lock
rm head_node_info
echo " Stopped ray on Head node: $(/bin/hostname)"
