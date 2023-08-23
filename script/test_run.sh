#==== Baseline ====#
# python -m baseline.train --config baseline/config.yaml
# python -m baseline.train --config baseline/config_cls.yaml

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
python -m graph_vit.task_finetune --config graph_vit/task_finetune.yaml --output_dir results/graph_vit/task_finetune --ckpt results/graph_vit/task_finetune_vit2/model_22_0.886.pt
# python -m graph_vit.task_finetune --config graph_vit/task_finetune.yaml --output_dir results/graph_vit/task_finetune --ckpt_cl results/graph_vit/pretrain_graphvit_cl/model_10_0.229.pt
