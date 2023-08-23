#!/bin/bash
#SBATCH --job-name=ray_worker
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=6
#SBATCH --time=24:00:00
#SBATCH --constraint="v100"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /ibex/user/xux/pep-pred/env

export ip_head=$(cat ./ray.head_node_info | cut -d " " -f 1)
export head_node_ip=$(echo ${ip_head} | cut -d ":" -f 1)
export redis_password=$(cat ./ray.head_node_info | cut -d " " -f 2)
export dashboard_port=$(cat ./ray.head_node_info | cut -d " " -f 3)

ray start --address ${ip_head}  --redis-password ${redis_password} --num-cpus 6 --block &

sleep 20
ray status --address ${ip_head} --redis_password ${redis_password}
#sleep 10

# worker shutdown strategy
if [ -f "ray.shutdown.lock" ] ; then
  echo " Stopping ray on Node: $(/bin/hostname)"
  ray stop
else
  while [ ! -f "ray.shutdown.lock" ]; 
   do
     sleep 20
   done   
fi
