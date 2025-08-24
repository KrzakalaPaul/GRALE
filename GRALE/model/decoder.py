import torch.nn as nn
from torch import Tensor
from GRALE.data import BatchedDenseData
from GRALE.attention import EvoformerDecoder, TransformerDecoder
from .utils import MLP_head, build_E_from_X

class AbstractGraphDecoder(nn.Module):
    '''
    Abstract class for Decoder part of the Graph Autoencoder.
    '''

    def forward(self, data: BatchedDenseData):
        '''
        Input: graph_embedding of shape (batch_size, graph_embedding_size, graph_embedding_dim)
        Outputs: 
            node_embeddings of shape (batch_size, n_nodes_max, node_model_dim)
            output (BatchedDenseData) with attributes:
                outputs.h shape (batch_size, n_nodes_max) (node activations)
                outputs.nodes.labels shape (batch_size, n_nodes_max, node_labels_dim) (one-hot encoding of node labels)
                outputs.nodes.features shape (batch_size, n_nodes_max, n_node_features) (node features)
                outputs.edges.A (batch_size, n_nodes_max, n_nodes_max, edge_labels_dim) (one-hot encoding of edge labels)
                outputs.edges.labels (batch_size, n_nodes_max, n_nodes_max, edge_labels_dim) (one-hot encoding of edge labels)
                outputs.edges.features (batch_size, n_nodes_max, n_nodes_max, n_edge_features) (edge features)
        All outputs are logits.
        '''
        
        h = ...
        node_labels = ...
        node_features = ...
        node_embeddings = ...
        A = ...
        edge_labels = ...
        edge_features = ...
        
        
        outputs = BatchedDenseData(h = h,
                                   nodes={'labels': node_labels, 'features': node_features}, 
                                   edges={'labels': edge_labels, 'features': edge_features, 'adjacency': A})
        
        return node_embeddings, outputs
    
class EvoformerGraphDecoder(AbstractGraphDecoder):
    '''
    Initialize X and E with learnable queries
    X, E = EvoformerDecoder(X, E, graph_embedding)
    node_embeddings, node_labels,node_features = MLP(X)
    A, edge_labels, edge_features = MLP(E)
    
    Cost: O(M³+KM²) M = n_nodes_max, K = graph_embedding_size
    '''
    
    def __init__(self, 
                 node_labels_dim,
                 node_features_dim,
                 node_model_dim,
                 node_hidden_dim,
                 edge_labels_dim,
                 edge_features_dim,
                 edge_model_dim,
                 edge_hidden_dim,
                 graph_embedding_dim,
                 n_nodes_max,
                 n_heads,
                 n_layers,
                 dropout_attn,
                 dropout_skip,
                 dropout_mlp,
                 norm_post_or_pre='pre',
                 triangular_multiplication_mode='symmetric',
                 triangular_attention_mode='none'
                 ):
        
        super().__init__()
        
        self.EvoformerDecoder = EvoformerDecoder(n_layers=n_layers,
                                                 src_dim=graph_embedding_dim, 
                                                 trgt_node_dim=node_model_dim,
                                                 trgt_edge_dim=edge_model_dim,
                                                 trgt_node_hidden_dim=node_hidden_dim,
                                                 trgt_edge_hidden_dim=edge_hidden_dim,
                                                 trgt_size=n_nodes_max,
                                                 n_heads=n_heads, 
                                                 dropout_attn=dropout_attn,
                                                 dropout_skip=dropout_skip, 
                                                 dropout_mlp=dropout_mlp,
                                                 norm_post_or_pre=norm_post_or_pre,
                                                 triangular_multiplication_mode=triangular_multiplication_mode,
                                                 triangular_attention_mode=triangular_attention_mode)
        
        self.head_h = MLP_head(in_dim = node_model_dim, hidden_dim = None, out_dim = 1, dropout = 0.)
        self.head_node_embeddings = MLP_head(in_dim = node_model_dim, hidden_dim = node_hidden_dim, out_dim = node_model_dim, dropout = 0.)
        self.head_node_labels = MLP_head(in_dim = node_model_dim, hidden_dim = None, out_dim = node_labels_dim, dropout = 0.)
        self.head_node_features = MLP_head(in_dim = node_model_dim, hidden_dim = node_hidden_dim, out_dim = node_features_dim, dropout = 0.)
        self.head_A = MLP_head(in_dim = edge_model_dim, hidden_dim = None, out_dim = 1, dropout = 0.)
        self.head_edge_labels = MLP_head(in_dim = edge_model_dim, hidden_dim = None, out_dim = edge_labels_dim, dropout = 0.)
        self.head_edge_features = MLP_head(in_dim = edge_model_dim, hidden_dim = edge_hidden_dim, out_dim = edge_features_dim, dropout = 0.)
    
    
    def forward(self, graph_embedding: Tensor):
        
        X, E = self.EvoformerDecoder(graph_embedding)
        
        h = self.head_h(X).squeeze(-1)
        node_embeddings = self.head_node_embeddings(X)
        node_labels = self.head_node_labels(X)
        node_features = self.head_node_features(X)
        A = self.head_A(E).squeeze(-1)
        edge_labels = self.head_edge_labels(E)
        edge_features = self.head_edge_features(E)
        
        outputs = BatchedDenseData(h = h,
                                   nodes={'labels': node_labels, 'features': node_features}, 
                                   edges={'labels': edge_labels, 'features': edge_features, 'adjacency': A})
        
        return node_embeddings, outputs
    
def get_decoder(config):
    return ...