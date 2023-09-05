#==== 5 fold CV ====#
# rm -rf data/CycPeptMPDB/processed/*.pt

# python -m graph_vit.train --config graph_vit/config_cls.yaml --output_dir results/graph_vit/cyc_cpp_cls  --val_split 1
# python -m graph_vit.train --config graph_vit/config_reg.yaml --output_dir results/graph_vit/cyc_cpp_reg --val_split 1

# i=3
# nohup python -m graph_vit.train --config graph_vit/config_cls.yaml --output_dir results/graph_vit/cyc_cpp_cls_$i --val_split $i 2>&1 >results/graph_vit/graph_vit_cls_$i.log &


# python -m mgt.train --config mgt/config_cls.yaml --output_dir results/mgt/cyc_cpp_cls --val_split 1
# python -m mgt.train --config mgt/config_reg.yaml --output_dir results/mgt/cyc_cpp_reg --val_split 1
# i=1
# nohup python -m mgt.train --config mgt/config_reg.yaml --output_dir results/mgt/cyc_cpp_reg_$i --val_split $i 2>&1 >results/mgt/mgt_reg_$i.log &

for i in {2..5};
do
    nohup python -m mgt.train --config mgt/config_reg.yaml --output_dir results/mgt/cyc_cpp_reg_$i --val_split $i 2>&1 >results/mgt/mgt_reg_$i.log &
done
