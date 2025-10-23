from networkx import nodes
import torch.nn as nn
import torch
from GRALE.data import BatchedDenseData
from .utils import MLP, flatten_upper_diagonal, SineEmbedding, build_E_from_X, pad_node_embeddings, build_laplacian_node_pos
from GRALE.attention import EvoformerEncoder, TransformerDecoder, TransformerEncoder, mask_nodes_to_edges
from functools import partial
from abc import abstractmethod

class AbstractGraphEncoder(nn.Module):
    '''
    Abstract class for encoder part of the Graph Autoencoder.
    '''
    
    @abstractmethod
    def forward(self, inputs: BatchedDenseData):
        '''
        Input: data (BatchedDenseData) with attributes:
            data.nodes.labels = one-hot encoding of node labels of shape (batch_size, n_nodes_max, n_node_labels)
            data.edges.SP = shortest path matrix of shape (batch_size, n_nodes_max, n_nodes_max)
            data.edges.adjacency = adjacency matrix of shape (batch_size, n_nodes_max, n_nodes_max)
            data.edges.labels = one-hot encoding of edge labels of shape (batch_size, n_nodes_max, n_nodes_max, n_edge_labels)
        Outputs: 
            node_embeddings of shape (batch_size, n_nodes_max, node_embedding_dim)
            graph_embedding of shape (batch_size, graph_embedding_size, graph_embedding_dim)
        '''
        node_embeddings = ...
        graph_embedding = ...
        return node_embeddings, graph_embedding
    
          
class EvoformerGraphEncoder(AbstractGraphEncoder):
    '''
    Initializes nodes features as:
        X = [node_labels, node_pos] where node_pos is laplacian positional encoding
    Initializes edge features as:
        E_ij = [node_labels_i, node_labels_j, edge_labels_ij, Fourier(SP_ij)]
    node_embeddings, Z = EvoformerEncoder(X, E)
    graph_embedding = cross_attention(graph_embedding_queries, flatten(Z))
    
    Cost: O(M³+KM²+K²) M = n_nodes, K = graph_embedding_size
    '''
    def __init__(self, 
                 node_labels_dim,
                 node_pos_dim,
                 node_model_dim,
                 node_hidden_dim,
                 edge_labels_dim, 
                 edge_model_dim,
                 edge_hidden_dim,
                 graph_embedding_dim,
                 graph_embedding_size,
                 n_nodes_max,
                 n_heads,
                 n_layers,
                 dropout_attn,
                 dropout_skip,
                 dropout_mlp,
                 edge_pos_dim = 64,
                 norm_post_or_pre='pre',
                 triangular_multiplication_mode='symmetric',
                 triangular_attention_mode='none',
                 ):
        
        super().__init__()
        
        self.n_nodes_max = n_nodes_max
        node_pos_dim = min(node_pos_dim, n_nodes_max-1)  # max n_nodes_max-1 non-trivial eigenvectors
        self.positionnal_nodes = partial(build_laplacian_node_pos, n_eigvecs=node_pos_dim)
        self.positionnal_edges = SineEmbedding(edge_pos_dim)
        
        X_dim = node_labels_dim + node_pos_dim
        E_dim = edge_labels_dim + 2*X_dim + edge_pos_dim

        self.mlp_nodes = MLP(X_dim, node_hidden_dim, node_model_dim, dropout_mlp)
        self.mlp_edges = MLP(E_dim, edge_hidden_dim, edge_model_dim, dropout_mlp)
        
        self.EvoformerEncoder = EvoformerEncoder(n_layers=n_layers,
                                                node_dim=node_model_dim, 
                                                edge_dim=edge_model_dim,
                                                node_hidden_dim=node_hidden_dim,
                                                edge_hidden_dim=edge_hidden_dim,
                                                n_heads=n_heads, 
                                                dropout_attn=dropout_attn,
                                                dropout_skip=dropout_skip, 
                                                dropout_mlp=dropout_mlp,
                                                norm_post_or_pre=norm_post_or_pre,
                                                triangular_multiplication_mode=triangular_multiplication_mode,
                                                triangular_attention_mode=triangular_attention_mode)
        
        self.TransformerDecoder = TransformerDecoder(n_layers=n_layers,
                                                    src_dim=edge_model_dim,
                                                    trgt_dim=graph_embedding_dim, 
                                                    trgt_hidden_dim=graph_embedding_dim,
                                                    trgt_size=graph_embedding_size,
                                                    n_heads=n_heads, 
                                                    dropout_attn=dropout_attn,
                                                    dropout_skip=dropout_skip, 
                                                    dropout_mlp=dropout_mlp,
                                                    norm_post_or_pre=norm_post_or_pre,
                                                    )
        
    def nodes_and_edges_embedding(self, data):
        node_labels = data.nodes.labels
        node_pos = self.positionnal_nodes(data.edges.adjacency).to(node_labels.dtype)
        X = torch.concat([node_labels, node_pos], dim=-1)
        E1 = build_E_from_X(X)
        E2 = data.edges.labels
        E3 = self.positionnal_edges(data.edges.SP)
        E = torch.cat([E1, E2, E3], dim=-1)
        return X, E                       
                                       
    def forward(self, data: BatchedDenseData):
        h = data.h
        mask_nodes = ~h
        mask_edges = mask_nodes_to_edges(mask_nodes)
        X, E = self.nodes_and_edges_embedding(data)
        # Preprocess node and edge features
        X = self.mlp_nodes(X)
        E = self.mlp_edges(E)
        # Pass through EvoformerEncoder
        node_embeddings, Z = self.EvoformerEncoder(X = X, E = E, mask_nodes = mask_nodes, mask_edges = mask_edges)
        # free memory
        del E, X 
        # Compute graph embedding with cross-attention
        Z = Z + Z.transpose(-3, -2)
        Z = flatten_upper_diagonal(Z)
        mask_Z = flatten_upper_diagonal(mask_edges)
        graph_embedding = self.TransformerDecoder(X_src=Z, mask_src=mask_Z)
        return node_embeddings, graph_embedding

class EvoformerGraphEncoderV2(AbstractGraphEncoder):
    '''
    Same as V1 but graph embedding comes from special nodes added to the graph instead of cross-attention.
    
    Initializes nodes features as:
        X = [node_labels, node_pos] where node_pos is laplacian positional encoding
    Initializes edge features as:
        E_ij = [node_labels_i, node_labels_j, edge_labels_ij, Fourier(SP_ij)]
    
    Then adds graph_embedding_size special nodes with learnable embeddings to the graph.
        X' = [X, learnable queries]
        E' = [[E, 0],
              [0, learnable queries]]
    X', E' = EvoformerEncoder(X', E')

    Cost: O((K+M)³) M = n_nodes, K = graph_embedding_size
    '''
    def __init__(self, 
                 node_labels_dim,
                 node_pos_dim,
                 node_model_dim,
                 node_hidden_dim,
                 edge_labels_dim, 
                 edge_model_dim,
                 edge_hidden_dim,
                 graph_embedding_dim,
                 graph_embedding_size,
                 n_nodes_max,
                 n_heads,
                 n_layers,
                 dropout_attn,
                 dropout_skip,
                 dropout_mlp,
                 edge_pos_dim = 64,
                 norm_post_or_pre='pre',
                 triangular_multiplication_mode='symmetric',
                 triangular_attention_mode='none',
                 ):
        
        super().__init__()
        
        self.n_nodes_max = n_nodes_max
        node_pos_dim = min(node_pos_dim, n_nodes_max-1)  # max n_nodes_max-1 non-trivial eigenvectors
        self.positionnal_nodes = partial(build_laplacian_node_pos, n_eigvecs=node_pos_dim)
        self.positionnal_edges = SineEmbedding(edge_pos_dim)
        
        X_dim = node_labels_dim + node_pos_dim
        E_dim = edge_labels_dim + 2*X_dim + edge_pos_dim

        self.mlp_nodes = MLP(X_dim, node_hidden_dim, node_model_dim, dropout_mlp)
        self.mlp_edges = MLP(E_dim, edge_hidden_dim, edge_model_dim, dropout_mlp)
        
        self.queries_nodes = nn.Parameter(torch.rand(1,graph_embedding_size, node_model_dim))
        self.queries_edges = nn.Parameter(torch.rand(1,graph_embedding_size,graph_embedding_size, edge_model_dim))
        
        self.EvoformerEncoder = EvoformerEncoder(n_layers=n_layers,
                                                node_dim=node_model_dim, 
                                                edge_dim=edge_model_dim,
                                                node_hidden_dim=node_hidden_dim,
                                                edge_hidden_dim=edge_hidden_dim,
                                                n_heads=n_heads, 
                                                dropout_attn=dropout_attn,
                                                dropout_skip=dropout_skip, 
                                                dropout_mlp=dropout_mlp,
                                                norm_post_or_pre=norm_post_or_pre,
                                                triangular_multiplication_mode=triangular_multiplication_mode,
                                                triangular_attention_mode=triangular_attention_mode)
        
        self.mlp_graph = MLP(node_model_dim, node_hidden_dim, graph_embedding_dim, dropout_mlp)
        
    def nodes_and_edges_embedding(self, data):
        node_labels = data.nodes.labels
        node_pos = self.positionnal_nodes(data.edges.adjacency).to(node_labels.dtype)
        X = torch.concat([node_labels, node_pos], dim=-1)
        E1 = build_E_from_X(X)
        E2 = data.edges.labels
        E3 = self.positionnal_edges(data.edges.SP)
        E = torch.cat([E1, E2, E3], dim=-1)
        return X, E
    
    def add_special_nodes(self, mask_nodes, X, E):
        '''
        Inputs:
            X of shape (batch_size, n_nodes, node_model_dim)
            E of shape (batch_size, n_nodes, n_nodes, edge_model_dim)
        Outputs:
            X of shape (batch_size, n_nodes + graph_embedding_size, node_model_dim)
            E of shape (batch_size, n_nodes + graph_embedding_size, n_nodes + graph_embedding_size, edge_model_dim)
        '''
        batch_size, n_nodes, node_model_dim = X.shape
        _,_,_, edge_model_dim = E.shape
        queries_nodes = self.queries_nodes.repeat(batch_size,1,1)
        queries_edges = self.queries_edges.repeat(batch_size,1,1,1)
        n_queries = queries_nodes.shape[1]
        
        # For mask just add ones for the special nodes
        mask_nodes = torch.cat([mask_nodes, torch.ones(batch_size, n_queries, device=mask_nodes.device, dtype=mask_nodes.dtype)], dim=1)
        # For nodes X' = [X, queries_nodes]
        X = torch.cat([X, queries_nodes], dim=1)
        # For edges: E'=[[E,0],[0,queries_edges]]
        E_top = torch.cat([E, torch.zeros(batch_size, n_nodes, n_queries, edge_model_dim, device=E.device, dtype=E.dtype)], dim=2)
        E_bottom = torch.cat([torch.zeros(batch_size, n_queries, n_nodes, edge_model_dim, device=E.device, dtype=E.dtype), queries_edges], dim=2)
        E = torch.cat([E_top, E_bottom], dim=1)
        return mask_nodes, X, E

    def separat_embeddings(self, X):
        '''
        Inputs:
            X of shape (batch_size, n_nodes + graph_embedding_size, node_model_dim)
        Outputs:
            node_embeddings of shape (batch_size, n_nodes, node_model_dim)
            graph_embedding of shape (batch_size, graph_embedding_size, graph_embedding_dim)
        '''
        batch_size, n_nodes_plus_queries, node_model_dim = X.shape
        n_queries = self.queries_nodes.shape[1]
        n_nodes = n_nodes_plus_queries - n_queries
        node_embeddings, graph_embedding = X[:,:n_nodes,:], X[:,n_nodes:,:]
        graph_embedding = self.mlp_graph(graph_embedding)
        return node_embeddings, graph_embedding

    def forward(self, data: BatchedDenseData):
        # Get node and edge embeddings initialization as in V1
        h = data.h
        mask_nodes = ~h
        X, E = self.nodes_and_edges_embedding(data)
        X = self.mlp_nodes(X)
        E = self.mlp_edges(E)
        # Add specials nodes
        mask_nodes, X, E, = self.add_special_nodes(mask_nodes, X, E)
        mask_edges = mask_nodes_to_edges(mask_nodes)
        # Pass through EvoformerEncoder
        X, _ = self.EvoformerEncoder(X = X, E = E, mask_nodes = mask_nodes, mask_edges = mask_edges)
        # free memory
        del E
        # Extract graph embedding from the special nodes
        node_embeddings, graph_embedding = self.separat_embeddings(X)
        return node_embeddings, graph_embedding
    
import inspect
def get_encoder(config):
    encoder_model = config.get('encoder_model')
    if encoder_model == 'evoformer':
        EncoderClass = EvoformerGraphEncoder
    elif encoder_model == 'evoformer2':
        EncoderClass = EvoformerGraphEncoderV2
    else:
        print("Warning: encoder_model not recognized, using 'evoformer' by default.")
        EncoderClass = EvoformerGraphEncoder
    params = inspect.signature(EncoderClass).parameters
    valid_args = {k: v for k, v in config.items() if k in params}
    if "mask_rate" in config:
        if config['mask_rate'] > 0:
            valid_args['edge_labels_dim'] += 1  # add one dimension for the mask label
            valid_args['node_labels_dim'] += 1  # add one dimension for the mask label
    return EncoderClass(**valid_args)
