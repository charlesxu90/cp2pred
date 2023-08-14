#==== Pretrain ====#
config_name='pretrain_resnet'
torchrun --nproc_per_node=2 pretrain_resnet.py --config configs/${config_name}.yaml --output_dir results/${config_name} --debug

#=== Task train ===#
# config_name='train_concatmodel_fps_dps'
# python train_cp2pred.py --config configs/${config_name}.yaml --output_dir results/${config_name} --debug
# python eval_cp2pred.py --config configs/${config_name}.yaml --ckpt_dir results/${config_name} --debug # --classification

# config_name='train_resnet'
# torchrun --nproc_per_node=2 train_resnet.py --config configs/${config_name}.yaml --output_dir results/${config_name} --debug
# torchrun --nproc_per_node=2 train_resnet.py --config configs/${config_name}.yaml --output_dir results/${config_name}_imagemol --debug --ckpt data/resnet_ckpt/ImageMol.pth.tar
