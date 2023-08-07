#==== Pretrain ====#
# HELM BPE Tokenizer
# python -m dataset.bpe --config configs/helm_bpe.yaml --output_dir results/helm_bpe
# config_name='train_helms_bpe_bert_test'

# SMILES BPE Tokenizer
# python -m dataset.bpe --config configs/smi_bpe.yaml --output_dir results/smi_bpe
# config_name='train_smi_bpe_bert_test'

# HELM BERT
# config_name='train_helms_bert_test'

# SMILES BERT
# config_name='train_smi_bert_test'
# torchrun --nproc_per_node=2 train_bert.py --config configs/${config_name}.yaml --output_dir results/${config_name} --debug

#==== Finetune BERT ====#
# SMILES BERT
# config_name='train_smi_bert_tune'
# torchrun --nproc_per_node=2 train_bert.py --config configs/${config_name}.yaml --output_dir results/${config_name} --debug  --ckpt results/train_smi_bert/model_2_1.966.pt

#==== evaluate pretrain ====#
# HELM BERT
# python eval_pretrain.py --config configs/train_helms_bpe_bert_test.yaml  --ckpt_dir results/train_helms_bpe_bert/ --clf xgb --model_type bpe
# python eval_pretrain.py --config configs/train_smi_bpe_bert_test.yaml  --ckpt_dir results/train_smi_bpe_bert/ --clf xgb --model_type bpe
# python eval_pretrain.py --config configs/train_helms_bert_test.yaml  --ckpt_dir results/train_helms_bert/ --clf xgb --model_type helm_bert
# python eval_pretrain.py --config configs/train_smi_bert_test.yaml  --ckpt_dir results/train_smi_bert/ --clf xgb --model_type smi_bert
# python eval_pretrain.py --config configs/train_smi_bert_test.yaml  --ckpt_dir results/train_smi_bert_tune/ --clf xgb --model_type smi_bert

#=== Task train ===#
# config_name='train_concatmodel_fps'
# python train_cp2pred.py --config configs/${config_name}.yaml --output_dir results/${config_name} --debug
# python eval_cp2pred.py --config configs/${config_name}.yaml --ckpt_dir results/${config_name} --debug

config_name='train_concatmodel_dps'
# python train_cp2pred.py --config configs/${config_name}.yaml --output_dir results/${config_name} --debug
# python eval_cp2pred.py --config configs/${config_name}.yaml --ckpt_dir results/${config_name} --debug

# config_name='train_concatmodel_smi'
# python train_cp2pred.py --config configs/${config_name}.yaml --output_dir results/${config_name} --debug
# python eval_cp2pred.py --config configs/${config_name}.yaml --ckpt_dir results/${config_name} --debug

# config_name='train_concatmodel_fps_dps'
# python train_cp2pred.py --config configs/${config_name}.yaml --output_dir results/${config_name} --debug
# python eval_cp2pred.py --config configs/${config_name}.yaml --ckpt_dir results/${config_name} --debug --classification

config_name='train_concatmodel_fps_smi'
# python train_cp2pred.py --config configs/${config_name}.yaml --output_dir results/${config_name} --debug
# python eval_cp2pred.py --config configs/bak/${config_name}.yaml --ckpt_dir results/${config_name}/ckpt/ --debug --classification

config_name='train_concatmodel_dps_smi'
# python train_cp2pred.py --config configs/${config_name}.yaml --output_dir results/${config_name} --debug
# python eval_cp2pred.py --config configs/${config_name}.yaml --ckpt_dir results/${config_name} --debug

config_name='train_concatmodel_fps_dps_smi'
# python train_cp2pred.py --config configs/${config_name}.yaml --output_dir results/${config_name} --debug
# python eval_cp2pred.py --config configs/${config_name}.yaml --ckpt_dir results/${config_name} --debug
