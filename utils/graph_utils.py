from loguru import logger
import torch
import numpy as np

def log_loaded_dataset(dataset):
    """ Log dataset information. """
    logger.info(f"Info of the loaded dataset:")
    data = dataset._data
    logger.info(f"  {data}")
    logger.info(f"  undirected: {dataset[0].is_undirected()}")
    logger.info(f"  num graphs: {len(dataset)}")

    total_num_nodes = 0
    if hasattr(data, 'num_nodes'):
        total_num_nodes = data.num_nodes
    elif hasattr(data, 'x'):
        total_num_nodes = data.x.size(0)

    logger.info(f"  avg num_nodes/graph: {total_num_nodes // len(dataset)}")
    logger.info(f"  num node features: {dataset.num_node_features}")
    logger.info(f"  num edge features: {dataset.num_edge_features}")
    
