#==== 5 fold CV ====#
rm -rf data/CycPeptMPDB/processed/*.pt
# python -m mole-bert.task_finetune --config mole-bert/config_reg.yaml --output_dir results/mole-bert/cyc_cpp_reg  --ckpt_cl results/mole-bert/pretrain/model_1.pt --val_split 1

i=1
nohup python -m mole-bert.task_finetune --config mole-bert/config_reg.yaml --output_dir results/mole-bert/cyc_cpp_reg_$i  --ckpt_cl results/mole-bert/pretrain/model_1.pt --val_split $i 2>&1 >results/mole-bert/mole-bert_reg_$i.log &

# for i in {2..5};
# do
#     nohup python -m mole-bert.task_finetune --config mole-bert/config_reg.yaml --output_dir results/mole-bert/cyc_cpp_reg_$i  --ckpt_cl results/mole-bert/pretrain/model_1.pt --val_split $i 2>&1 >results/mole-bert/mole-bert_reg_$i.log &
# done
