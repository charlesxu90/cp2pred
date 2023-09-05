#==== Graph ViT HPO ====#
# rm -rf data/CycPeptMPDB/processed/*.pt
# python -m graph_vit.cl_pretrain --config graph_vit/cl_pretrain_config.yaml --output_dir results/graph_vit/cl_pretrain
# python -m graph_vit.train --config graph_vit/config_reg.yaml --output_dir results/graph_vit/cyc_cpp_reg --val_split 1

# NNI HPO
# python -m graph_vit.nni_hpo  --config graph_vit/config_reg.yaml  # test
# nnictl create --config ./nni_graphvit.config --port 8080
