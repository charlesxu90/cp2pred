#==== 5 fold CV ====#
# torchrun --nproc_per_node=2 -m smi_bert.task_finetune --config smi_bert/task_config_cls.yaml --output_dir results/smi_bert/cyc_cpp_cls --ckpt results/smi_bert/pretrain_ft/model_19_0.221.pt --val_split 1

# torchrun --nproc_per_node=2 -m smi_bert.task_finetune --config smi_bert/task_config_reg.yaml --output_dir results/smi_bert/cyc_cpp_reg --ckpt results/smi_bert/pretrain_ft/model_19_0.221.pt --val_split 1

# i=1
# nohup torchrun --nproc_per_node=2 -m smi_bert.task_finetune --config smi_bert/task_config_reg.yaml --output_dir results/smi_bert/cyc_cpp_reg_$i --ckpt results/smi_bert/pretrain_ft/model_19_0.221.pt --val_split $i 2>&1 >results/smi_bert/smi_bert_reg_$i.log &

# i=5
# nohup torchrun --nproc_per_node=2 -m smi_bert.task_finetune --config smi_bert/task_config_reg.yaml --output_dir results/smi_bert/cyc_cpp_reg_$i --ckpt results/smi_bert/pretrain_ft/model_19_0.221.pt --val_split $i 2>&1 >results/smi_bert/smi_bert_reg_$i.log &

# i=2 # 1
# nohup torchrun --nproc_per_node=2 -m smi_bert.task_finetune --config smi_bert/task_config_cls.yaml --output_dir results/smi_bert/cyc_cpp_cls_$i --ckpt results/smi_bert/pretrain_ft/model_19_0.221.pt --val_split $i 2>&1 >results/smi_bert/smi_bert_cls_$i.log &

# sleep 60m
# i=3
# nohup torchrun --nproc_per_node=2 -m smi_bert.task_finetune --config smi_bert/task_config_cls.yaml --output_dir results/smi_bert/cyc_cpp_cls_$i --ckpt results/smi_bert/pretrain_ft/model_19_0.221.pt --val_split $i 2>&1 >results/smi_bert/smi_bert_cls_$i.log &

# sleep 60m
# i=4
# nohup torchrun --nproc_per_node=2 -m smi_bert.task_finetune --config smi_bert/task_config_cls.yaml --output_dir results/smi_bert/cyc_cpp_cls_$i --ckpt results/smi_bert/pretrain_ft/model_19_0.221.pt --val_split $i 2>&1 >results/smi_bert/smi_bert_cls_$i.log &

# sleep 60m
# i=5
# nohup torchrun --nproc_per_node=2 -m smi_bert.task_finetune --config smi_bert/task_config_cls.yaml --output_dir results/smi_bert/cyc_cpp_cls_$i --ckpt results/smi_bert/pretrain_ft/model_19_0.221.pt --val_split $i 2>&1 >results/smi_bert/smi_bert_cls_$i.log &


