#==== 5 fold CV ====#

# rm -rf data/CycPeptMPDB/processed/*.pt
# python -m gcn.train --config gcn/config_cls.yaml --output_dir results/gcn/cyc_cpp_cls --val_split 1
# python -m gcn.train --config gcn/config_reg.yaml --output_dir results/gcn/cyc_cpp_reg --val_split 1

# i=1
# nohup python -m gcn.train --config gcn/config_reg.yaml --output_dir results/gcn/cyc_cpp_reg_$i --val_split $i 2>&1 >results/gcn/gcn_reg_$i.log &

for i in {2..5};
do
nohup python -m gcn.train --config gcn/config_reg.yaml --output_dir results/gcn/cyc_cpp_reg_$i --val_split $i 2>&1 >results/gcn/gcn_reg_$i.log &
done


# python -m gine.task_finetune --config gine/config_cls.yaml --output_dir results/gine/config_cls --ckpt_pretrain  results/gine/pretrain_graphpred/model_final_0.692.pt --val_split 1
# python -m gine.task_finetune --config gine/config_reg.yaml --output_dir results/gine/config_reg --ckpt_pretrain  results/gine/pretrain_graphpred/model_final_0.692.pt --val_split 1

# i=1
# nohup python -m gine.task_finetune --config gine/config_reg.yaml --output_dir results/gine/config_reg_$i --ckpt_pretrain  results/gine/pretrain_graphpred/model_final_0.692.pt --val_split $i 2>&1 >results/gine/gine_reg_$i.log &

for i in {2..5};
do
nohup python -m gine.task_finetune --config gine/config_reg.yaml --output_dir results/gine/config_reg_$i --ckpt_pretrain  results/gine/pretrain_graphpred/model_final_0.692.pt --val_split $i 2>&1 >results/gine/gine_reg_$i.log &
done
