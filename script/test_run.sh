#==== Pretrain ====#
# HELM BPE Tokenizer
# python -m dataset.bpe --config configs/helm_bpe.yaml --output_dir results/helm_bpe

# HELM BERT
torchrun --nproc_per_node=2 train_bert.py --config configs/train_helms_bert_test.yaml --output_dir results/train_helms_bert_test --debug

#==== task finetune ====#
# HELM BERT
# torchrun --nproc_per_node=2 task_finetune.py --config configs/CPP924_aa_bert.yaml --output_dir results/CPP924_aa_bert --debug --ckpt results/train_aa_bert_L40/model_12_2.523.pt

#==== evaluate ====#
# HELM BERT
# torchrun --nproc_per_node=2 evaluate.py --config configs/CPP924_aa_bert.yaml --output_dir results/CPP924_aa_bert --debug --ckpt results/train_aa_bert_L40/model_12_2.523.pt