#==== Split data into 10 folds ====#
# python utils/splitters.py --raw_data_path data/CycPeptMPDB/raw/all.csv.gz --smi_col smi

#==== Baseline ====#
# python -m baseline.train --config baseline/config_cls.yaml --val_split 5
python -m baseline.train --config baseline/config_reg.yaml --val_split 1

#==== SMILES BERT ====#
# Pretrain
# torchrun --nproc_per_node=2 -m seq_bert.pretrain_bert --config seq_bert/pretrain_smi_bert.yaml --output_dir results/pretrain_smi_bert

# Finetune
# torchrun --nproc_per_node=2 -m seq_bert.pretrain_bert --config seq_bert/smi_bert_finetune.yaml --output_dir results/smi_bert_finetune --ckpt results/pretrain_smi_bert_ibex/model_final_0.384.pt

# Task finetune
# torchrun --nproc_per_node=2 -m seq_bert.task_finetune --config seq_bert/smi_bert_task_finetune.yaml --output_dir results/smi_bert_task_finetune --ckpt results/smi_bert_finetune/model_4_0.348.pt
# torchrun --nproc_per_node=2 -m seq_bert.task_finetune --config seq_bert/smi_bert_cls_task_finetune.yaml --output_dir results/smi_bert_cls_task_finetune --ckpt results/smi_bert_finetune/model_4_0.348.pt

#==== ImageMol ResNet ====#
# Pretrain ResNet with contrastive learning, using ImageNet or ImageMol pretrained weights, 
# ImageNet is better with size 500x500
# torchrun --nproc_per_node=2 -m image_mol.pretrain_resnet --config image_mol/pretrain_resnet.yaml --output_dir results/resnet/pretrain_resnet

# Task finetune
# torchrun --nproc_per_node=2 -m image_mol.task_finetune_resnet --config image_mol/task_finetune_resnet.yaml --output_dir results/resnet/task_finetune --ckpt_cl results/resnet/pretrain_resnet_ibex_imgnet_init/model_100_0.027.pt
# torchrun --nproc_per_node=2 -m image_mol.task_finetune_resnet --config image_mol/task_cls_finetune_resnet.yaml --output_dir results/resnet/task_cls_finetune --ckpt_cl results/resnet/pretrain_resnet_ibex_imgnet_init/model_100_0.027.pt

#==== Graph ViT ====#
# python -m graph_vit.pretrain_graphvit_cl --config graph_vit/pretrain_graphvit_cl.yaml --output_dir results/graph_vit/pretrain_graphvit_cl
# python -m graph_vit.task_finetune --config graph_vit/task_finetune.yaml --output_dir results/graph_vit/task_finetune 
# python -m graph_vit.task_finetune --config graph_vit/task_finetune.yaml --output_dir results/graph_vit/task_finetune --ckpt_cl results/graph_vit/pretrain_graphvit_cl/model_10_0.229.pt

# NNI HPO
# python -m graph_vit.task_nni_hpo  --config graph_vit/task_finetune.yaml
# nnictl create --config ./graphvit_nni_hpo.config --port 8080

#==== GPS ====#
# rm -rf data/CycPeptMPDB/processed/*.pt
# python -m gps.task_finetune --config gps/task_finetune.yaml --output_dir results/gps/task_finetune3

#==== Grit ====#
# rm -rf data/CycPeptMPDB/processed/*.pt
# python -m grit.task_finetune --config grit/task_finetune.yaml --output_dir results/grit/task_finetune3 

#==== MGT ====#
# python -m mgt.task_finetune --config mgt/task_finetune.yaml --output_dir results/mgt/task_finetune_reg2
