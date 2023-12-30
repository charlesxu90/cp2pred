#==== KRAS Kd ====#

## Baseline
# for i in {1..5};
# do
# nohup python -m baseline.train --config baseline/config_kras_kd.yaml --val_split $i --transform true --output_dir results/baseline/kras_kd_$i  2>&1 >results/baseline/kras_kd_$i.log &
# done

## ImageMol ResNet
i=1
torchrun --nproc_per_node=2 -m image_mol.train --config image_mol/config_kras_kd.yaml --output_dir results/resnet/kras_kd_$i --ckpt_cl results/resnet/pretrain/model_100_0.027.pt --val_split $i
# for i in {1..5};
# do
# nohup torchrun --nproc_per_node=2 -m image_mol.train --config image_mol/config_kras_kd.yaml --output_dir results/resnet/kras_kd_$i --ckpt_cl results/resnet/pretrain/model_100_0.027.pt --val_split $i  2>&1 >results/resnet/kras_kd_$i.log &
# done