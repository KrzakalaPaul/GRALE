import torch.nn as nn
import torch
from GRALE.model.utils import BatchedDenseData
from torch.utils.data import DataLoader
from .utils import update_running_average
import pickle

class AbstractTargetBuilder(nn.Module):
    '''
    Abstract class for the input builder module.
    This part of the model should be non-trainable.
    '''
    def forward(self, data: BatchedDenseData, target_size: int):    
        '''
        Input: data (BatchedDenseData) with attributes:
            data.nodes.labels = one-hot encoding of node labels
            data.edges.SP = shortest path matrix
            data.edges.adjacency = adjacency matrix
            data.edges.labels = one-hot encoding of edge labels
        Outputs: 
            targets (BatchedDenseData) with attributes:.
            targets.nodes.labels shape (batch_size, n_nodes_max, node_labels_dim) (one-hot encoding of node labels)
            targets.nodes.features shape (batch_size, n_nodes_max, node_features_dim) (node features)
            targets.edges.labels (batch_size, n_nodes_max, n_nodes_max, edge_labels_dim) (one-hot encoding of edge labels)
            targets.edges.features (batch_size, n_nodes_max, n_nodes_max, edge_features_dim) (edge features)
        '''  
        h = data.h.clone()
        node_labels = data.nodes.labels
        edge_labels = data.edges.labels
        node_features = ...
        edge_features = ...
        A = data.edges.adjacency
        targets = BatchedDenseData(h, 
                                   nodes={'labels': node_labels, 'features': node_features}, 
                                   edges={'labels': edge_labels, 'features': edge_features, 'adjacency': A})
        targets.pad_(target_size)
        return targets
        
class DiffusionTargetBuilder(AbstractTargetBuilder):
    '''
    A basic target builder with:
    node_features = 1/n AF where F = node labels and A = adjacency matrix (Feature diffusion)
    edge_features = 1/n SP where SP = shortest path matrix                             
    '''
    def get_node_features(self, data: BatchedDenseData):
        A = data.edges.adjacency
        N = A.sum(dim=-1, keepdim=True) # Number of neighbors
        F = data.nodes.labels
        node_features = torch.einsum('bij,bjd->bid', A, F) / (N.unsqueeze(-1) + 1e-6) 
        return node_features
    
    def get_edge_features(self, data: BatchedDenseData):
        edge_features = data.edges.SP.to(torch.float).unsqueeze(-1)
        n_nodes = edge_features.size(1)
        edge_features = edge_features / n_nodes  # Normalize by max possible
        return edge_features        

    def forward(self, data: BatchedDenseData, target_size = None):
        '''
        returns inputs, targets
        '''
        h = data.h.clone()
        node_labels = data.nodes.labels
        edge_labels = data.edges.labels
        node_features = self.get_node_features(data)
        edge_features = self.get_edge_features(data)
        A = data.edges.adjacency
        targets = BatchedDenseData(h, 
                                   nodes={'labels': node_labels, 'features': node_features}, 
                                   edges={'labels': edge_labels, 'features': edge_features, 'adjacency': A})
        if target_size is not None:
            targets.pad_(target_size)
        return targets

def get_target_builder(config):
    return ...