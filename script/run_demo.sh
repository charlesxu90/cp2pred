#==== Split data into 10 folds ====#
# python utils/splitters.py --raw_data_path data/CycPeptMPDB/raw/all.csv.gz --smi_col smi

#==== Baseline ====#
# python -m baseline.train --config baseline/config_cls.yaml --val_split 5
# python -m baseline.train --config baseline/config_reg.yaml --val_split 1

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
# python -m gps.train --config gps/config_cls.yaml --output_dir results/gps/cyc_cpp_cls --val_split 1 2>&1 >gps_cls_1.log
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

#==== GCN ====#

#==== GINE ====#
