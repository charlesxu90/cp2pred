#==== Pretrain ====#

# AA_BERT
torchrun --nproc_per_node=2 train_bert.py --config configs/train_helms_bert_test.yaml --output_dir results/train_helms_bert_test --debug

#==== task specific finetune ====#
# AA BERT
# torchrun --nproc_per_node=2 task_finetune.py --config configs/CPP924_aa_bert.yaml --output_dir results/CPP924_aa_bert --debug --ckpt results/train_aa_bert_L40/model_12_2.523.pt