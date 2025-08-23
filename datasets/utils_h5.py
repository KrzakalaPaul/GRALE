import h5py
import os
import numpy as np

class H5DatasetBuilder:
    """
    Helps to create and manage an HDF5 file for storing graph datasets.
    The file is organized as follows:
    - attributes:
        - n_max_nodes: int - maximum number of nodes in any graph
        - n_node_labels: int - number of node labels
        - n_edge_labels: int - number of edge labels
        - padding_value: str - value used for padding (e.g. "nan")
        - chunk_size_storage: int - chunk size for HDF5 storage
    - train: 
        - node_labels: (n_mols_train, n_max_nodes) - labels of nodes in the graph
        - edge_labels: (n_mols_train, n_max_nodes, n_max_nodes) - labels of edges in the graph
        - node_mask: (n_mols_train, n_max_nodes) - mask for nodes in the graph
        - SP_matrix: (n_mols_train, n_max_nodes, n_max_nodes) - shortest path matrix
    - valid:
        # Same as train, but for validation set
    - test: 
        # Same as train, but for test set
    """
    def __init__(self, path_h5):
        """
        Load an existing HDF5 file.
        """
        self.path_h5 = path_h5
    
        # Check if file exists
        if not os.path.exists(path_h5):
            raise FileNotFoundError(f"HDF5 file {path_h5} does not exist. Please create it first.")
        
        # Load attributes
        with h5py.File(path_h5, 'r') as f:
            self.n_max_nodes = f.attrs['n_max_nodes']
            self.padding_value = f.attrs['padding_value']
            self.chunk_size_storage = f.attrs['chunk_size_storage']
            self.n_node_labels = len(f.attrs['node_labels'])
            self.n_edge_labels = len(f.attrs['edge_labels'])
        
    @classmethod
    def create(cls, path_h5: str, 
               n_max_nodes: int, 
               node_labels: int, 
               edge_labels: int, # Including "no edge" label
               padding_value: str = "nan", 
               chunk_size_storage: int = 100000, 
               overwrite: bool = False):
        """
        Create a new HDF5 file to store the dataset.
        """
        # Check if file exists
        if os.path.exists(path_h5):
            if overwrite:
                os.remove(path_h5)
            else:
                raise FileExistsError(f"HDF5 file {path_h5} already exists. Set overwrite=True to overwrite it.")
            
        # Add config to the file
        with h5py.File(path_h5, 'w') as f:
            f.attrs['n_max_nodes'] = n_max_nodes
            f.attrs['padding_value'] = padding_value
            f.attrs['chunk_size_storage'] = chunk_size_storage
            f.attrs['node_labels'] = np.array(node_labels, dtype='S')
            f.attrs['edge_labels'] = np.array(edge_labels, dtype='S')

        file = cls(path_h5)
        with h5py.File(path_h5, 'a') as f:
            file.init_split(f, 'train')
            file.init_split(f, 'valid')
            file.init_split(f, 'test')
            
        return file
    
    def init_split(self, f, split):
        '''
        Initialize a split in the HDF5 file.
        '''
        group = f.create_group(split)
        n_max_nodes = self.n_max_nodes
        chunk_size_storage = self.chunk_size_storage
        group.create_dataset('node_labels', shape = (0, n_max_nodes), maxshape=(None, n_max_nodes), dtype=np.uint8, chunks=(chunk_size_storage, n_max_nodes))
        group.create_dataset('edge_labels', shape = (0, n_max_nodes, n_max_nodes), maxshape=(None, n_max_nodes, n_max_nodes), dtype=np.uint8, chunks=(chunk_size_storage, n_max_nodes, n_max_nodes))
        group.create_dataset('node_mask', shape = (0, n_max_nodes), maxshape=(None, n_max_nodes), dtype=bool, chunks=(chunk_size_storage, n_max_nodes))
        group.create_dataset('SP_matrix', shape = (0, n_max_nodes, n_max_nodes), maxshape=(None, n_max_nodes, n_max_nodes), dtype=np.uint8, chunks=(chunk_size_storage, n_max_nodes, n_max_nodes))
        
    def append_data(self, dataset_path, new_data):
        '''
        Append data to the given dataset, e.g. 'train/mols/node_labels'.
        '''
        with h5py.File(self.path_h5, 'a') as f:
            dataset = f[dataset_path]
            current_rows = dataset.shape[0]
            new_rows = current_rows + new_data.shape[0]
            dataset.resize((new_rows, *dataset.shape[1:]))
            dataset[current_rows:new_rows] = new_data



def print_h5_structure(path):
    '''
    Print the structure of an HDF5 file. For arrays print shape and dtype.
    '''
    with h5py.File(path, 'r') as f:
        def print_group(name, obj):
            indent = '  ' * (name.count('/') - 1)
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}{name}: shape={obj.shape}, dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{indent}{name}/")
        f.visititems(print_group)