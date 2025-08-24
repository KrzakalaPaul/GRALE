import torch.nn as nn
import torch
from utils import BatchedDenseData
from torch.utils.data import DataLoader
from .utils import update_running_average
import os
import pickle

class AbstractTargetBuilder(nn.Module):
    '''
    Featurizer class to build target from data
    # IN :
    data (BatchedDenseData) has following attributes:
    data.nodes.labels = one-hot encoding of node labels
    data.edges.SP = shortest path matrix
    data.edges.adjacency = adjacency matrix
    data.edges.labels = one-hot encoding of edge labels
    "
    # Out:
    targets (BatchedDenseData) has following attributes:
    targets.nodes.labels shape (batch_size, n_nodes, node_labels_dim) (one-hot encoding of node labels)
    targets.nodes.features shape (batch_size, n_nodes, node_features_dim) (node features)
    targets.edges.labels (batch_size, n_nodes, n_nodes, edge_labels_dim) (one-hot encoding of edge labels)
    targets.edges.features (batch_size, n_nodes, n_nodes, edge_features_dim) (edge features)
    '''
    
    def __init__(self):
        super(AbstractTargetBuilder, self).__init__()
        
    
    def forward(self, data: BatchedDenseData, target_size: int):      
        '''
        returns inputs, targets
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

    def fit(self, dataloader: DataLoader, max_samples = 100000, device = 'cuda'):
        '''
        Use dataloader to compute statistics required for normalization
        Must also use to define node_features_dim, edge_features_dim, X_dim, E_dim
        '''
        self.node_features_dim = ...
        self.edge_features_dim = ...    
    
        
class TargetBuilder(AbstractTargetBuilder):
    '''
    A basic target builder with:
    
    node_features = normalize([AF, A^2F, ..., A^kF]) where F = node labels and A = adjacency matrix (Feature diffusion)
    edge_features = normalize([SP]) where SP = shortest path matrix                             
    '''
    
    def __init__(self, k_hops = 1):
        super(TargetBuilder, self).__init__()
        self.k_hops = k_hops
        self.node_features_dim = None
        self.node_features_mean = None
        self.node_features_std = None
        self.edge_features_dim = None
        self.edge_features_mean = None
        self.edge_features_std = None

        
    def save(self, path):
        dic = {'k_hops': self.k_hops,
               'node_features_mean': self.node_features_mean.data,
               'node_features_std': self.node_features_std.data,
               'edge_features_mean': self.edge_features_mean.data,
               'edge_features_std': self.edge_features_std.data,
               'node_features_dim': self.node_features_dim,
               'edge_features_dim': self.edge_features_dim
              }
        with open(path, 'wb') as f:
            pickle.dump(dic, f)
        
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            dic = pickle.load(f)
        obj = cls(k_hops=dic['k_hops'])
        obj.node_features_mean = nn.Parameter(dic['node_features_mean'])
        obj.node_features_std = nn.Parameter(dic['node_features_std'])
        obj.edge_features_mean = nn.Parameter(dic['edge_features_mean'])
        obj.edge_features_std = nn.Parameter(dic['edge_features_std'])
        obj.node_features_dim = dic['node_features_dim']
        obj.edge_features_dim = dic['edge_features_dim']
        return obj
        
    def get_node_features(self, data: BatchedDenseData, normalize = True):
        node_labels = data.nodes.labels
        A = data.edges.adjacency
        # Compute node features [A^1 F, A^2 F, ..., A^k F] where F = node_labels
        node_features = [torch.bmm(A,node_labels)]
        for _ in range(self.k_hops-1):
            F = node_features[-1]
            node_features.append(torch.bmm(A,F))
        node_features = torch.cat(node_features, dim=-1)
        node_features = node_features.to(torch.float)
        if normalize:
            node_features = (node_features - self.node_features_mean) / (self.node_features_std+1e-9)
            node_features = torch.clamp(node_features, -5, 5)
        return node_features

    def get_edge_features(self, data: BatchedDenseData, normalize = True):
        SP = data.edges.SP
        # Compute edge features 
        edge_features = SP.to(torch.float).unsqueeze(-1)
        if normalize:
            edge_features = (edge_features - self.edge_features_mean) / (self.edge_features_std+1e-9)
            edge_features = torch.clamp(edge_features, -5, 5)
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



    def fit(self, dataloader: DataLoader, max_samples = 100000, device = 'cuda'):
        '''
        Compute average/standard deviation of node and edge features    
        '''
        
        ex_data = next(iter(dataloader))
        
        node_labels_dim = ex_data.nodes.labels.shape[-1]
        edge_labels_dim = ex_data.edges.labels.shape[-1]
        
        self.node_features_dim = node_labels_dim * self.k_hops
        self.edge_features_dim = 1
        
        node_features_running_avg = torch.zeros(self.node_features_dim)
        node_features_squared_running_avg = torch.zeros(self.node_features_dim)
        n_nodes_seen = 0
        
        edge_features_running_avg = torch.zeros(self.edge_features_dim)
        edge_features_squared_running_avg = torch.zeros(self.edge_features_dim)
        n_edges_seen = 0
        
        n_samples = 0
        for i, data in enumerate(dataloader):
            
            data = data.to(device)
            
            h = data.h.clone() # (batch_size, n_nodes)
            h_nodes = h.flatten() # (batch_size * n_nodes)
            h_edges = h[:,:,None] * h[:,None,:] # (batch_size, n_nodes, n_nodes)
            h_edges = h_edges.flatten() # (batch_size * n_nodes * n_nodes)
            
            node_features = self.get_node_features(data, normalize=False)
            node_features = node_features.flatten(start_dim=0, end_dim=1) # (batch_size * n_nodes, node_features_dim)
            node_features_running_avg, _ = update_running_average(node_features_running_avg, n_nodes_seen, node_features, h_nodes)
            node_features_squared_running_avg, n_nodes_seen = update_running_average(node_features_squared_running_avg, n_nodes_seen, node_features**2, h_nodes)

            edge_features = self.get_edge_features(data, normalize=False)
            edge_features = edge_features.flatten(start_dim=0, end_dim=2) # (batch_size * n_nodes * n_nodes, edge_features_dim)
            edge_features_running_avg, _ = update_running_average(edge_features_running_avg, n_edges_seen, edge_features, h_edges)
            edge_features_squared_running_avg, n_edges_seen = update_running_average(edge_features_squared_running_avg, n_edges_seen, edge_features**2, h_edges)
            
            n_samples += data.batchsize
            if n_samples > max_samples:
                break
    
        self.node_features_mean = nn.Parameter(node_features_running_avg, requires_grad=False)
        self.node_features_std = nn.Parameter(torch.sqrt(node_features_squared_running_avg - node_features_running_avg**2), requires_grad=False)
        
        self.edge_features_mean = nn.Parameter(edge_features_running_avg, requires_grad=False)
        self.edge_features_std = nn.Parameter(torch.sqrt(edge_features_squared_running_avg - edge_features_running_avg**2), requires_grad=False)
        
        return self.node_features_dim, self.edge_features_dim
