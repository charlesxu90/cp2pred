#==== 5 fold CV ====#
# rm -rf data/CycPeptMPDB/processed/*.pt
# python -m gps.train --config gps/config_cls.yaml --output_dir results/gps/cyc_cpp_cls --val_split 1 2>&1 >gps_cls_1.log
# python -m gps.train --config gps/config_reg.yaml --output_dir results/gps/cyc_cpp_reg --val_split 1

# rm -rf data/CycPeptMPDB/processed/*.pt
# i=1
# nohup python -m gps.train --config gps/config_cls.yaml --output_dir results/gps/cyc_cpp_cls_$i --val_split $i 2>&1 >gps_cls_$i.log &
# for i in {2..5};
# do
#     nohup python -m gps.train --config gps/config_cls.yaml --output_dir results/gps/cyc_cpp_cls_$i --val_split $i 2>&1 >gps_cls_$i.log &
# done

# rm -rf data/CycPeptMPDB/processed/*.pt
# i=1
# nohup python -m gps.train --config gps/config_reg.yaml --output_dir results/gps/cyc_cpp_reg_$i --val_split $i 2>&1 >gps_reg_$i.log &
# for i in {2..5};
# do
#     nohup python -m gps.train --config gps/config_reg.yaml --output_dir results/gps/cyc_cpp_reg_$i --val_split $i 2>&1 >gps_reg_$i.log &
# done
