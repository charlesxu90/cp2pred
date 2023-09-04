#==== Split data into 10 folds ====#
# python utils/splitters.py --raw_data_path data/CycPeptMPDB/raw/all.csv.gz --smi_col smi

#==== Baseline ====#
# python -m baseline.train --config baseline/config_cls.yaml --val_split 5
# python -m baseline.train --config baseline/config_reg.yaml --val_split 1

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
# rm -rf data/CycPeptMPDB/processed/*.pt
# python -m graph_vit.cl_pretrain --config graph_vit/cl_pretrain_config.yaml --output_dir results/graph_vit/cl_pretrain
# python -m graph_vit.train --config graph_vit/config_cls.yaml --output_dir results/graph_vit/cyc_cpp_cls  --val_split 1
# python -m graph_vit.train --config graph_vit/config_reg.yaml --output_dir results/graph_vit/cyc_cpp_reg --val_split 1

# NNI HPO
# python -m graph_vit.nni_hpo  --config graph_vit/config_reg.yaml  # test
# nnictl create --config ./graphvit_nni_hpo.config --port 8080

#==== GPS ====#
# rm -rf data/CycPeptMPDB/processed/*.pt
# python -m gps.train --config gps/config_cls.yaml --output_dir results/gps/cyc_cpp_cls --val_split 1
# python -m gps.train --config gps/config_reg.yaml --output_dir results/gps/cyc_cpp_reg --val_split 1

#==== Grit ====#
rm -rf data/CycPeptMPDB/processed/*.pt
python -m grit.train --config grit/config_cls.yaml --output_dir results/grit/cyc_cpp_cls --val_split 1
# python -m grit.train --config grit/config_reg.yaml --output_dir results/grit/cyc_cpp_reg  --val_split 1

#==== MGT ====#
# python -m mgt.task_finetune --config mgt/task_finetune.yaml --output_dir results/mgt/task_finetune_reg2
