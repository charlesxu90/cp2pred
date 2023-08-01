#==== Pretrain ====#
# HELM BPE Tokenizer
# python -m dataset.bpe --config configs/helm_bpe.yaml --output_dir results/helm_bpe

# HELM BERT
torchrun --nproc_per_node=2 train_bert.py --config configs/train_helms_bert_test.yaml --output_dir results/train_helms_bert_test --debug

#==== evaluate pretrain ====#
# HELM BERT
# python eval_pretrain.py --config configs/train_helms_bert_test.yaml  --ckpt_dir results/train_helms_bert_test/ --clf xgb --model_type bpe

# python eval_pretrain.py --config configs/train_helms_bert_test.yaml  --ckpt_dir results/train_helms_bert/ --clf xgb --model_type bpe

# python eval_pretrain.py --config configs/train_helms_bert_test.yaml  --ckpt_dir results/train_helms_bert2/ --clf xgb --model_type bpe
