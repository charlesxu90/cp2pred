#==== Pretrain ====#

#=== Task train ===#
config_name='train_concatmodel_fps_dps'
# python train_cp2pred.py --config configs/${config_name}.yaml --output_dir results/${config_name} --debug
python eval_cp2pred.py --config configs/${config_name}.yaml --ckpt_dir results/${config_name} --debug # --classification