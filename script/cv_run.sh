#==== 5 fold CV ====#

# rm -rf data/CycPeptMPDB/processed/*.pt
# python -m gcn.train --config gcn/config_cls.yaml --output_dir results/gcn/cyc_cpp_cls --val_split 1
# python -m gcn.train --config gcn/config_reg.yaml --output_dir results/gcn/cyc_cpp_reg --val_split 1

# i=1
# nohup python -m gcn.train --config gcn/config_reg.yaml --output_dir results/gcn/cyc_cpp_reg_$i --val_split $i 2>&1 >results/gcn/gcn_reg_$i.log &

# for i in {2..5};
# do
# nohup python -m gcn.train --config gcn/config_reg.yaml --output_dir results/gcn/cyc_cpp_reg_$i --val_split $i 2>&1 >results/gcn/gcn_reg_$i.log &
# done


# python -m gine.task_finetune --config gine/config_cls.yaml --output_dir results/gine/config_cls --ckpt_pretrain  results/gine/pretrain_graphpred/model_final_0.692.pt --val_split 1
# python -m gine.task_finetune --config gine/config_reg.yaml --output_dir results/gine/config_reg --ckpt_pretrain  results/gine/pretrain_graphpred/model_final_0.692.pt --val_split 1

# i=1
# nohup python -m gine.task_finetune --config gine/config_reg.yaml --output_dir results/gine/config_reg_$i --ckpt_pretrain  results/gine/pretrain_graphpred/model_final_0.692.pt --val_split $i 2>&1 >results/gine/gine_reg_$i.log &

# for i in {2..5};
# do
# nohup python -m gine.task_finetune --config gine/config_reg.yaml --output_dir results/gine/config_reg_$i --ckpt_pretrain  results/gine/pretrain_graphpred/model_final_0.692.pt --val_split $i 2>&1 >results/gine/gine_reg_$i.log &
# done

#==== KRAS ====#

# rm -rf data/kras/processed/*.pt
## GCN
# python -m gcn.train --config gcn/config_kras.yaml --output_dir results/gcn/kras_reg --val_split 1

# for i in {2..5};
# do
# nohup python -m gcn.train --config gcn/config_kras.yaml --output_dir results/gcn/kras_reg_$i --val_split $i 2>&1 >results/gcn/kras_reg_$i.log &
# done

## GINE
# i=1
# nohup python -m gine.task_finetune --config gine/config_kras.yaml --output_dir results/gine/kras_$i --ckpt_pretrain  results/gine/pretrain_graphpred/model_final_0.692.pt --val_split $i 2>&1 >results/gine/kras_reg_$i.log &

# for i in {2..5};
# do
# nohup python -m gine.task_finetune --config gine/config_kras.yaml --output_dir results/gine/kras_$i --ckpt_pretrain  results/gine/pretrain_graphpred/model_final_0.692.pt --val_split $i 2>&1 >results/gine/kras_reg_$i.log &
# done

## GPS
# i=1
# nohup python -m gps.train --config gps/config_kras.yaml --output_dir results/gps/kras_reg_$i --val_split $i 2>&1 >results/gps/kras_reg_$i.log &

# for i in {2..5};
# do
# nohup python -m gps.train --config gps/config_kras.yaml --output_dir results/gps/kras_reg_$i --val_split $i 2>&1 >results/gps/kras_reg_$i.log &
# done

## Graph ViT
i=2
nohup python -m graph_vit.train --config graph_vit/config_kras.yaml --output_dir results/graph_vit/kras_reg_$i --val_split $i 2>&1 >results/graph_vit/kras_reg_$i.log &

# for i in {2..5};
# do
# nohup python -m graph_vit.train --config graph_vit/config_kras.yaml --output_dir results/graph_vit/kras_reg_$i --val_split $i 2>&1 >results/graph_vit/kras_reg_$i.log &
# done

## Grit
# i=1
# nohup python -m grit.train --config grit/config_kras.yaml --output_dir results/grit/kras_reg_$i --val_split $i 2>&1 >results/grit/kras_reg_$i.log &

# for i in {2..5};
# do
# nohup python -m grit.train --config grit/config_kras.yaml --output_dir results/grit/kras_reg_$i --val_split $i 2>&1 >results/grit/kras_reg_$i.log &
# done

## MGT
# i=1
# nohup python -m mgt.train --config mgt/config_kras.yaml --output_dir results/mgt/kras_reg_$i --val_split $i 2>&1 >results/mgt/kras_reg_$i.log &

# for i in {2..5};
# do
# nohup python -m mgt.train --config mgt/config_kras.yaml --output_dir results/mgt/kras_reg_$i --val_split $i 2>&1 >results/mgt/kras_reg_$i.log &
# done

## Mole-BERT
# i=1
# nohup python -m mole-bert.task_finetune --config mole-bert/config_kras.yaml --output_dir results/mole-bert/kras_reg_$i  --ckpt_cl results/mole-bert/pretrain/model_1.pt --val_split $i 2>&1 >results/mole-bert/kras_reg_$i.log &

# for i in {2..5};
# do
# nohup python -m mole-bert.task_finetune --config mole-bert/config_kras.yaml --output_dir results/mole-bert/kras_reg_$i  --ckpt_cl results/mole-bert/pretrain/model_1.pt --val_split $i 2>&1 >results/mole-bert/kras_reg_$i.log &
# done
