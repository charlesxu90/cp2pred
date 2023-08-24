#!/bin/bash

num_workers=2
# Start the head node
ray start --head --port=6379 
sleep 20

# Start the worker nodes
for i in $(seq 1 $num_workers):
    do 
    ray start --address='10.67.24.210:6379'
done
sleep 20

# Run the task
ray status

python -m graph_vit.task_ray_tune_hpo  --config graph_vit/task_finetune.yaml --output_dir results/graph_vit/task_finetune --num_samples=100 --max_concur_trials=8 --local

# Stop the nodes
# ray stop