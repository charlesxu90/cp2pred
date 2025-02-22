#==== Split data into 10 folds ====#
# python utils/splitters.py --raw_data_path data/CycPeptMPDB/raw/all.csv.gz --smi_col smi
# python utils/splitters.py --raw_data_path data/kras/raw/all.csv.gz --smi_col smi
#  python utils/splitters.py --raw_data_path data/kras_kd/raw/all.csv.gz --k_fold 9

#==== Baseline ====#
# python -m baseline.train --config baseline/config_cls.yaml --val_split 5
# python -m baseline.train --config baseline/config_reg.yaml --val_split 6
# i=1
# python -m baseline.train --config baseline/config_kras.yaml --val_split $i --transform true --output_dir results/baseline/kras_$i

#==== SMILES BERT ====#
# Pretrain
# torchrun --nproc_per_node=2 -m smi_bert.pretrain --config smi_bert/pretrain_config.yaml --output_dir results/smi_bert/pretrain

# Finetune
# torchrun --nproc_per_node=2 -m smi_bert.pretrain --config smi_bert/pretrain_ft_config.yaml --output_dir results/smi_bert/pretrain_ft --ckpt results/smi_bert/pretrain/model_final_0.384.pt

# Task finetune
# torchrun --nproc_per_node=2 -m smi_bert.task_finetune --config smi_bert/task_config_cls.yaml --output_dir results/smi_bert/cyc_cpp_cls --ckpt results/smi_bert/pretrain_ft/model_19_0.221.pt --val_split 1

# torchrun --nproc_per_node=2 -m smi_bert.task_finetune --config smi_bert/task_config_reg.yaml --output_dir results/smi_bert/cyc_cpp_reg --ckpt results/smi_bert/pretrain_ft/model_19_0.221.pt --val_split 1

#==== ImageMol ResNet ====#
# Pretrain ResNet with contrastive learning, using ImageNet pretrained weights, with image size 500x500
# torchrun --nproc_per_node=2 -m image_mol.pretrain --config image_mol/pretrain_config.yaml --output_dir results/resnet/pretrain2

# Task finetune
# torchrun --nproc_per_node=2 -m image_mol.train --config image_mol/config_cls.yaml --output_dir results/resnet/cyc_cpp_cls --ckpt_cl results/resnet/pretrain/model_100_0.027.pt --val_split 1
# torchrun --nproc_per_node=2 -m image_mol.train --config image_mol/config_reg.yaml --output_dir results/resnet/cyc_cpp_reg --ckpt_cl results/resnet/pretrain/model_100_0.027.pt --val_split 1

#==== GPS ====#
# rm -rf data/CycPeptMPDB/processed/*.pt
# python -m gps.train --config gps/config_cls.yaml --output_dir results/gps/cyc_cpp_cls --val_split 1
# python -m gps.train --config gps/config_reg.yaml --output_dir results/gps/cyc_cpp_reg --val_split 1

#==== Graph ViT ====#
# rm -rf data/CycPeptMPDB/processed/*.pt
# python -m graph_vit.train --config graph_vit/config_cls.yaml --output_dir results/graph_vit/cyc_cpp_cls  --val_split 1
# python -m graph_vit.train --config graph_vit/config_reg.yaml --output_dir results/graph_vit/cyc_cpp_reg --val_split 1

#==== Grit ====#
# rm -rf data/CycPeptMPDB/processed/*.pt
# python -m grit.train --config grit/config_cls.yaml --output_dir results/grit/cyc_cpp_cls --val_split 1
# python -m grit.train --config grit/config_reg.yaml --output_dir results/grit/cyc_cpp_reg  --val_split 1

#==== MGT ====#
# rm -rf data/CycPeptMPDB/processed/*.pt
# python -m mgt.train --config mgt/config_cls.yaml --output_dir results/mgt/cyc_cpp_cls --val_split 1
# python -m mgt.train --config mgt/config_reg.yaml --output_dir results/mgt/cyc_cpp_reg --val_split 1

#==== Mole-BERT ====#
# python -m mole-bert.train_vae --config mole-bert/train_vae.yaml --output_dir results/mole-bert/train_vae
# python -m mole-bert.pretrain --config mole-bert/pretrain.yaml --output_dir results/mole-bert/pretrain --vae_ckpt results/mole-bert/train_vae/model_38_0.121.pt
# rm -rf data/CycPeptMPDB/processed/*.pt
# python -m mole-bert.task_finetune --config mole-bert/config_cls.yaml --output_dir results/mole-bert/cyc_cpp_cls  --ckpt_cl results/mole-bert/pretrain/model_1.pt --val_split 1
# python -m mole-bert.task_finetune --config mole-bert/config_reg.yaml --output_dir results/mole-bert/cyc_cpp_reg  --ckpt_cl results/mole-bert/pretrain/model_1.pt --val_split 1

#==== GCN ====#
# rm -rf data/CycPeptMPDB/processed/*.pt
# python -m gcn.train --config gcn/config_cls.yaml --output_dir results/gcn/cyc_cpp_cls --val_split 1
# python -m gcn.train --config gcn/config_reg.yaml --output_dir results/gcn/cyc_cpp_reg --val_split 1

#==== GINE ====#
# rm -rf data/CycPeptMPDB/processed/*.pt
# python -m gine.pretrain_contextpred --config gine/pretrain_contextpred.yaml --output_dir results/gine/pretrain_contextpred
# python -m gine.pretrain_graphpred --config gine/pretrain_graphpred.yaml --output_dir results/gine/pretrain_graphpred --ckpt_pretrain results/gine/pretrain/pretrain_contextpred2.pth
# python -m gine.task_finetune --config gine/config_cls.yaml --output_dir results/gine/config_cls --ckpt_pretrain  results/gine/pretrain_graphpred/model_final_0.692.pt --val_split 1
# python -m gine.task_finetune --config gine/config_reg.yaml --output_dir results/gine/config_reg --ckpt_pretrain  results/gine/pretrain_graphpred/model_final_0.692.pt --val_split 1
