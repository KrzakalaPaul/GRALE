import os
import requests
import gzip
import shutil
import time 
import pandas as pd 
from functools import partial
from multiprocessing import Pool
from multiprocessing import cpu_count
from tqdm import tqdm
import subprocess
from .utils import smiles2graph
from datasets.utils_h5 import DatasetHDF5
import numpy as np
from sklearn.model_selection import train_test_split

def download_PUBCHEM():

    url = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"
    csv_path = "data/raw/PUBCHEM.csv"
    gz_path = csv_path + ".gz"

    if not os.path.isfile(csv_path):
        print("Downloading raw PUBCHEM files, this can take a few minutes...")
        start_time = time.time()
        # 1. Download full .gz file in memory and save once
        response = requests.get(url)
        with open(gz_path, "wb") as f:
            f.write(response.content)
        # 2. Decompress .gz into .csv
        with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=1024*1024)  # 1MB buffer
        # 3. Remove temporary .gz
        os.remove(gz_path)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'PUBCHEM dataset downloaded successfully. Took {elapsed_time:.2f} seconds.')
        
        print("Create small version for development")
        df = pd.read_csv(csv_path, sep="\t", nrows=1000000, index_col=0, header=None)
        df.to_csv("data/raw/PUBCHEM_dev.csv", header=False, index=True, sep='\t')
        print("Small version created successfully")
        
    else:
        print("PUBCHEM File already exists, skipping download")

def process_PUBCHEM(config, dev=False):

    # Load Config
    valid_size = config['valid_size']
    n_max_nodes = config['n_max_nodes']
    chunk_size = config['chunk_size_preprocessing']
    valid_atomic_nums = config['valid_atomic_nums']
    valid_bond_types = config['valid_bond_types']
    
    # Create HDF5 file
    path_h5 = "data/h5/PUBCHEM_" + config['id'] + ".h5"
    dataset = DatasetHDF5.create(path_h5, n_max_nodes=n_max_nodes, overwrite=True)

    # Get chunks of smiles
    csv_path = "data/raw/PUBCHEM.csv" if not dev else "data/raw/PUBCHEM_dev.csv"
    n_smiles_max = int(subprocess.run(['wc', '-l', csv_path], stdout=subprocess.PIPE).stdout.decode().split()[0])
    n_chunks = n_smiles_max // chunk_size + 1
    chunk_iter = pd.read_csv(csv_path, chunksize=chunk_size, sep='\t', index_col=0, header=None)

    n_threads = cpu_count() 
    smiles2graph_with_config = partial(smiles2graph, n_max_nodes=n_max_nodes, valid_atomic_nums=valid_atomic_nums, valid_bond_types=valid_bond_types)
    n_valid_mols = 0
    n_mols_valid = 0
    for chunk in tqdm(chunk_iter, total=n_chunks, unit="chunk", desc="Processing smiles by chunks of size {}".format(chunk_size)):
        smiles_list = chunk.iloc[:,0].values
        # Convert SMILES to graphs using smiles2graph + multiprocessing
        with Pool(processes=n_threads) as pool:
            graphs = list(pool.imap(smiles2graph_with_config, smiles_list))
        # Filter out None graphs
        graphs = [g for g in graphs if g is not None]
        # Split into train and valid sets
        if n_mols_valid < valid_size:
            graphs_train, graphs_valid = train_test_split(graphs, test_size=0.05, random_state=42)
            n_mols_valid += len(graphs_valid)
            # Add valid data to dataset
            dataset.append_data('valid/node_labels', np.stack([g['node_labels'] for g in graphs_valid], axis=0))
            dataset.append_data('valid/edge_labels', np.stack([g['edge_labels'] for g in graphs_valid], axis=0))
            dataset.append_data('valid/node_mask', np.stack([g['node_mask'] for g in graphs_valid], axis=0))
            dataset.append_data('valid/SP_matrix', np.stack([g['SP_matrix'] for g in graphs_valid], axis=0))
        else:
            graphs_train = graphs
        # Build arrays 
        node_labels = np.stack([g['node_labels'] for g in graphs_train], axis=0)
        edge_labels = np.stack([g['edge_labels'] for g in graphs_train], axis=0)
        node_mask = np.stack([g['node_mask'] for g in graphs_train], axis=0)
        SP_matrix = np.stack([g['SP_matrix'] for g in graphs_train], axis=0)
        # Add to dataset
        dataset.append_data('train/node_labels', node_labels)
        dataset.append_data('train/edge_labels', edge_labels)
        dataset.append_data('train/node_mask', node_mask)
        dataset.append_data('train/SP_matrix', SP_matrix)

        # Update valid_mols count
        n_valid_mols += len(graphs)

    print(f"Total valid molecules: {n_valid_mols}/{n_smiles_max}")
    print(f"Save {n_mols_valid} for valid set")
