import os
from functools import partial
from multiprocessing import Pool
from multiprocessing import cpu_count
from tqdm import tqdm
from .utils import coloring_sample
from itertools import repeat
from datasets.utils_h5 import H5DatasetBuilder
import numpy as np

def create_coloring_dataset(config):
    
    # Load Config
    id = config['id']
    train_size = config['train_size']
    valid_size = config['valid_size']
    test_size = config['test_size']
    n_min_nodes = config['n_min_nodes']
    n_max_nodes = config['n_max_nodes']
    n_pixels = config['n_pixels']

    sampling_function = partial(coloring_sample,
                                n_max_nodes=n_max_nodes,
                                n_pixels=n_pixels,
                                sigma_filter=0.4,
                                sigma_noise=0.02,
                                node_weight_treshold=None,
                                edge_weight_treshold=None,
                                precompute_shortest_paths=True)

    node_labels = ["BLUE", "GREEN", "RED", "YELLOW"]
    edge_labels = ["NOEDGE", "EDGE"]
    
    # Generate samples
    total_size = train_size + valid_size + test_size
    with Pool(processes=cpu_count()//2) as pool:
        samples = list(tqdm(pool.imap(sampling_function, [n_min_nodes]*total_size), total=total_size))
    train_samples = samples[:train_size]
    valid_samples = samples[train_size:train_size+valid_size]
    test_samples = samples[train_size+valid_size:]
    
    # Init h5py file
     # Create HDF5 file
    path_h5 = "data/h5/COLORING_" + config['id'] + ".h5"
    dataset = H5DatasetBuilder.create(path_h5, 
                                      n_max_nodes=n_max_nodes, 
                                      node_labels=node_labels,
                                      edge_labels=edge_labels, # Including "no edge" label
                                      overwrite=True)
    
    # Can cause memory issues if dataset is too large
    for split, split_samples in zip(['train','valid','test'], [train_samples, valid_samples, test_samples]):
        
        #imgs = np.zeros((len(split_samples), n_pixels, n_pixels, 3), dtype=np.uint8)
        node_labels = np.zeros((len(split_samples), n_max_nodes), dtype=np.uint8)
        edge_labels = np.zeros((len(split_samples), n_max_nodes, n_max_nodes), dtype=np.uint8)
        SP_matrix = np.zeros((len(split_samples), n_max_nodes, n_max_nodes), dtype=np.uint8)
        node_mask = np.ones((len(split_samples), n_max_nodes), dtype=bool)

        for i, sample in enumerate(tqdm(split_samples, desc=f"Processing {split} samples")):
            img, graph = sample
            n_nodes = graph['node_colors'].shape[0]
            
            # Get image
            #imgs[i] = img

            # Get node labels (one-hot encoding)
            node_labels[i, :n_nodes] = graph['node_colors']
        
            # Get edge labels (one-hot encoding)
            edge_labels[i, :n_nodes, :n_nodes] = graph['adjacency_matrix']

            # Get SP matrix
            SP_matrix[i, :n_nodes, :n_nodes] = graph['SP_matrix']
            
            # Get node mask
            node_mask[i, :n_nodes] = 0

        # Add to dataset
        #dataset.append_data(f'{split}/images', imgs)
        dataset.append_data(f'{split}/node_labels', node_labels)
        dataset.append_data(f'{split}/edge_labels', edge_labels)
        dataset.append_data(f'{split}/node_mask', node_mask)
        dataset.append_data(f'{split}/SP_matrix', SP_matrix)
