import torch.nn as nn
import torch
from GRALE.data import BatchedDenseData
from abc import abstractmethod

class AbstractInputBuilder(nn.Module):
    '''
    Abstract class for the input builder module.
    This part of the model should be non-trainable.
    '''
    
    def __init__(self, n_nodes_max):
        super(AbstractInputBuilder, self).__init__()
        self.n_nodes_max = n_nodes_max
        
    
    @abstractmethod
    def forward(self, data: BatchedDenseData):
        '''
        Input: data (BatchedDenseData) with attributes:
            data.nodes.labels = one-hot encoding of node labels
            data.edges.SP = shortest path matrix
            data.edges.adjacency = adjacency matrix
            data.edges.labels = one-hot encoding of edge labels
        Outputs: 
            data (BatchedDenseData) with same attributes. 
            The data must be padded to self.n_nodes_max nodes.
        '''
        inputs = ...
        return inputs
    
class BasicInputBuilder(AbstractInputBuilder):
    '''
    Basic input builder that pads the input graph to n_nodes_max nodes.
    '''
    
    def __init__(self, n_nodes_max):
        super(BasicInputBuilder, self).__init__(n_nodes_max)
        
    def forward(self, data: BatchedDenseData):
        size = data.size
        if size != self.n_nodes_max:
            data.pad_(self.n_nodes_max)
        return data
    
class MaskingInputBuilder(AbstractInputBuilder):
    '''
    Input builder that randomly masks nodes and edges, and pads the input graph to n_nodes_max nodes.
    '''
    
    def __init__(self, n_nodes_max, node_masking_ratio=0.1):
        super(MaskingInputBuilder, self).__init__(n_nodes_max)
        self.node_masking_ratio = node_masking_ratio
        
    def forward(self, data: BatchedDenseData):
        
        # Clone data to avoid modifying the original data
        data = data.clone()
        
        # Pad to n_nodes_max
        size = data.size
        batchsize = data.batchsize
        if size != self.n_nodes_max:
            data.pad_(self.n_nodes_max)
        
        # Randomly mask nodes, edge (i,j) is masked if either node i or j is masked
        not_masked_node = torch.rand(batchsize, size, device=data.h.device) > self.node_masking_ratio
        not_masked_edge = not_masked_node.unsqueeze(1) * not_masked_node.unsqueeze(2) # 
        
        # If a node is masked its labels are set to a new "masked" label (add a new dimension to the one-hot encoding and set it to 1)
        labels = data.nodes.labels
        new_labels = torch.ones(batchsize, size, labels.size(-1) + 1, device=labels.device) * (~not_masked_node).unsqueeze(-1)
        new_labels[:, :, :-1] = labels * not_masked_node.unsqueeze(-1)
        data.nodes.labels = new_labels

        # If a edge is masked, the corresponding entry in the adjacency matrix is set to 0
        data.edges.adjacency = data.edges.adjacency * not_masked_edge

        # If a edge is masked, the corresponding entry in the SP matrix is set to -1
        data.edges.SP = data.edges.SP * not_masked_edge + (~not_masked_edge) * -1

        # If a edge is masked its labels are set to a new "masked" label (add a new dimension to the one-hot encoding and set it to 1)
        labels_edges = data.edges.labels
        new_labels_edges = torch.ones(batchsize, size, size, labels_edges.size(-1) + 1, device=labels_edges.device) * (~not_masked_edge).unsqueeze(-1)
        new_labels_edges[:, :, :, :-1] = labels_edges * not_masked_edge.unsqueeze(-1)
        data.edges.labels = new_labels_edges
        
        return data
    
def get_input_builder(config):
    if "mask_rate" in config:
        if config['mask_rate'] > 0:
            return MaskingInputBuilder(config['n_nodes_max'], node_masking_ratio=config['mask_rate'])
        print("Warning: mask_rate <= 0, using BasicInputBuilder")
    return BasicInputBuilder(config['n_nodes_max'])