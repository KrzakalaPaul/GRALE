import numpy as np
import h5py
from torch.utils.data import Dataset   
import torch
from .dense_data import DenseData, BatchedDenseData
import lightning.pytorch as pl
from torch.utils.data import DataLoader

def concat_data(data1, data2):
    """
    Concatenates two dictionaries of tensors along the first dimension.
    """
    for key in data1.keys():
        if key in data2:
            data1[key] = np.concatenate((data1[key], data2[key]), axis=0)
        else:
            raise KeyError(f"Key '{key}' not found in both datasets.")
    return data1

def load_in_memory(h5_dataset, start_idx=0, end_idx=None):
    data_in_memory = {}
    for key in h5_dataset.keys():
        data_in_memory[key] = h5_dataset[key][start_idx:end_idx]
    return data_in_memory

class LazyDataset():
    '''
    Lazy loading of data from h5 file.
    Loads data in memory only when needed.
    This is can be very uneficient if data is not accessed sequentially (in particular in DataLoader with shuffling and multiple workers).
    
    We use it only for the "get_epoch_data" method, which loads all data for one "epoch" in memory.
    '''
    def __init__(self, path_h5, n_data_epoch, split = 'train'):
        self.lazy_data = h5py.File(path_h5, 'r')[split]
        self.n_data = self.lazy_data[f'node_mask'].shape[0]
        self.n_data_epoch = n_data_epoch
        n_iters = self.n_data // n_data_epoch 
        print(f"The full {split} dataset is of size {self.n_data}.")
        print(f"Every epoch a chunk of size {self.n_data_epoch} is loaded.")
        print(f"The entire dataset will be iterated every {n_iters:.1f} epochs.")
        if n_iters < 2:
            print("Consider using fully loading the data for better efficiency.")
        if n_iters > 100:
            print("Consider increasing n_data_epoch for better efficiency.")
        self.pointer = 0

    def get_data_epoch_starting_from(self, pointer):
        '''
        Returns a subset of the dataset starting from pointer
        '''
        start = pointer 
        end = start + self.n_data_epoch
        if end <= self.n_data:
            subset = load_in_memory(self.lazy_data, start_idx=start, end_idx=end)
            pointer = end
        else:
            # wrap around
            subset1 = load_in_memory(self.lazy_data, start_idx=start, end_idx=None)
            subset2 = load_in_memory(self.lazy_data, start_idx=0, end_idx=end % self.n_data)
            subset = concat_data(subset1, subset2)
            pointer = end % self.n_data
        return subset, pointer
        
    def get_data_epoch(self, fixed_pointer=False):
        '''
        Returns data for one epoch 
        Set fixed_pointer=True to always start from the beginning of the dataset.
        '''
        pointer = 0 if fixed_pointer else self.pointer
        subset, self.pointer = self.get_data_epoch_starting_from(pointer)
        return subset
    
    def __len__(self):
        return self.n_data

class InMemoryDataset(Dataset):
    '''
    Dataset for one epoch. All data is in memory.
    '''
    def __init__(self, data, metadata):
        '''
        Data: Dict of numpy arrays
            - node_mask: shape (n_nodes,)
            - node_labels: shape (n_nodes, n_node_labels)
            - edge_labels: shape (n_nodes, n_nodes, n_edge_labels)
            - SP_matrix: shape (n_nodes, n_nodes)
        Metadata: Dict with metadata
            - n_max_nodes: int
            - n_node_labels: int
            - n_edge_labels: int
        '''
        self.data = data
        self.metadata = metadata
        size = self.check()
        self.size = size

    def check(self):
        '''
        Check that all tensors have the same first dimension size.
        '''
        size = None
        for key, value in self.data.items():
            if size is None:
                size = value.shape[0]
            else:
                assert size == value.shape[0], f"All tensors must have the same first dimension size, key '{key}': has shape {value.shape}"
        
        # check that "node_mask" is present
        assert 'node_mask' in self.data, "'node_mask' key not found in data"
        assert 'node_labels' in self.data, "'node_labels' key not found in data"
        assert 'edge_labels' in self.data, "'edge_labels' key not found in data"
        assert 'SP_matrix' in self.data, "'SP_matrix' key not found in data"
        
        assert 'n_max_nodes' in self.metadata, "'n_max_nodes' key not found in metadata"
        assert 'n_node_labels' in self.metadata, "'n_node_labels' key not found in metadata"
        assert 'n_edge_labels' in self.metadata, "'n_edge_labels' key not found in metadata"
        
        return size
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        '''
        Returns: 
            input: DenseData object
        '''
        node_mask = self.data['node_mask'][idx]          # shape (n_nodes,)
        node_labels = self.data['node_labels'][idx]      # shape (n_nodes, n_node_labels)
        edge_labels = self.data['edge_labels'][idx]      # shape (n_nodes, n_nodes, n_edge_labels)
        SP_matrix = self.data['SP_matrix'][idx]          # shape (n_nodes, n_nodes)
        
        h = ~torch.tensor(node_mask, dtype=torch.bool)
        node_labels = torch.nn.functional.one_hot(torch.tensor(node_labels, dtype=torch.long), num_classes=self.metadata['n_node_labels']).to(torch.float32)
        edge_labels = torch.tensor(edge_labels, dtype=torch.long)
        adjacency_matrix = (edge_labels > 0).to(torch.float32)
        edge_labels = torch.nn.functional.one_hot(edge_labels, num_classes=self.metadata['n_edge_labels']).to(torch.float32)
        SP_matrix = torch.tensor(SP_matrix, dtype=torch.float32)
        
        input = DenseData(size=h.shape[0], h=h, nodes={'labels': node_labels}, edges={'adjacency': adjacency_matrix, 'labels': edge_labels, 'SP': SP_matrix})

        return input

def custom_collate_fn(inputs_list):
    return BatchedDenseData.from_list(inputs_list)
    

class DataModule(pl.LightningDataModule):
    def __init__(self, path_h5: str, batch_size: int = 32, n_data_epoch: int = None, n_data_valid: int = None, n_workers: int = 0):
        '''
        If n_data_epoch = None, use the entire dataset each epoch and load it fully in memory.
        Else, load only n_data_epoch samples each epoch.
        '''
        super().__init__()
        
        # Load metadata
        with h5py.File(path_h5, 'r') as f:
            n_max_nodes = f.attrs['n_max_nodes']
            n_edge_labels = len(f.attrs['edge_labels'])
            n_node_labels = len(f.attrs['node_labels'])

        self.metadata = {
            "n_max_nodes": n_max_nodes,
            "n_edge_labels": n_edge_labels,
            "n_node_labels": n_node_labels
        }

        # Load data (either fully in memory or lazily)
        print('---')
        if n_data_epoch == None:
            print(f'Loading the full train dataset of in memory. If this causes out of memory issues, consider setting n_data_epoch.')
            self.dataset_train = InMemoryDataset(load_in_memory(h5py.File(path_h5, 'r')['train']), self.metadata)
            print(f'Train dataset size: {len(self.dataset_train)}')
            self.lazy = False
        else:
            self.dataset_train = LazyDataset(path_h5, n_data_epoch=n_data_epoch, split='train')
            self.lazy = True
        print('---')
        if n_data_valid == None:
            print(f'Loading the full valid dataset in memory. If this causes out of memory issues, consider setting n_data_valid.')
            self.dataset_valid = InMemoryDataset(load_in_memory(h5py.File(path_h5, 'r')['valid']), self.metadata)
            print(f'Valid dataset size: {len(self.dataset_valid)}')
        else:
            print(f'Loading only {n_data_valid} samples of the valid dataset in memory.')
            self.dataset_valid = InMemoryDataset(load_in_memory(h5py.File(path_h5, 'r')['valid'], end_idx=n_data_valid), self.metadata)
        print('---')
        self.batch_size = batch_size
        self.n_data_epoch = n_data_epoch
        self.n_workers = n_workers

    def train_dataloader(self):
        if self.lazy:
            data_epoch = self.dataset_train.get_data_epoch()
            dataset_epoch = InMemoryDataset(data_epoch, self.metadata)
            return DataLoader(dataset_epoch, batch_size=self.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=self.n_workers,
                              pin_memory=True, non_blocking=True, prefetch_factor=2)
        else:
            return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=self.n_workers,
                              pin_memory=True, non_blocking=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=self.n_workers,
                          pin_memory=True, non_blocking=True, prefetch_factor=2)