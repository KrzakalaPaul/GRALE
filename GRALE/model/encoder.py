import torch.nn as nn
import torch
from utils import BatchedDenseData
from .utils import MLP, GIN, flatten_upper_diagonal, SineEmbedding, build_E_from_X, pad_node_embeddings, build_random_node_pos
from .attention import EvoformerEncoder, TransformerDecoder, TransformerEncoder, mask_nodes_to_edges

class AbstractGraphEncoder(nn.Module):
    '''
    Abstract class for encoder part of the Graph Autoencoder.
    '''

    def forward(self, data: BatchedDenseData):
        '''
        Input: data (BatchedDenseData) with attributes:
            data.nodes.labels = one-hot encoding of node labels
            data.edges.SP = shortest path matrix
            data.edges.adjacency = adjacency matrix
            data.edges.labels = one-hot encoding of edge labels
        Outputs: 
            node_embeddings of shape (batch_size, n_nodes_max, node_model_dim)
            node_embeddings_mask of shape (batch_size, n_nodes_max)
            graph_embedding of shape (batch_size, graph_embedding_size, graph_embedding_dim)
        '''
        node_embeddings = ...
        graph_embedding = ...
        node_embeddings_mask = ...
        return node_embeddings, node_embeddings_mask, graph_embedding
    
          
class EvoformerGraphEncoder(AbstractGraphEncoder):
    '''
    X = [node_labels]
    E = [node_labels_i, node_labels_j, edge_labels_ij, SP_ij]
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
                 triangular_attention_mode='none'):
        
        super().__init__()
        
        self.n_nodes_max = n_nodes_max
        self.positionnal_nodes = nn.Linear(n_nodes_max, node_pos_dim)
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
        batchsize, n_nodes_batch, _ = data.nodes.labels.shape
        node_labels = data.nodes.labels
        node_pos = build_random_node_pos(batchsize, n_nodes_batch, self.n_nodes_max).to(node_labels)
        node_pos = self.positionnal_nodes(node_pos)
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
        node_embeddings, node_masks = pad_node_embeddings(node_embeddings, mask_nodes, self.n_nodes_max)
        # free memory
        del E, X 
        # Compute graph embedding with cross-attention
        Z = Z + Z.transpose(-3, -2)
        Z = flatten_upper_diagonal(Z)
        mask_Z = flatten_upper_diagonal(mask_edges)
        graph_embedding = self.TransformerDecoder(X_src=Z, mask_src=mask_Z)
        return node_embeddings, node_masks, graph_embedding

class GNNGraphEncoder(AbstractGraphEncoder):
        '''
        X_i = Linear([node_labels_i, sum_j edge_labels_ij])
        X = GNN(X,A) 
        X = TransformerEncoder(X)
        node_embeddings = MLP(X)
        graph_embedding = cross_attention(graph_embedding_queries, X)
        Cost: O(M²+KM+K²) M = n_nodes, K = graph_embedding_size
        '''
        def __init__(self, 
                    node_labels_dim,
                    node_pos_dim,
                    node_model_dim,
                    node_hidden_dim,
                    edge_labels_dim,
                    graph_embedding_dim,
                    graph_embedding_size,
                    n_nodes_max,
                    n_heads,
                    n_layers,
                    n_layers_gnn,
                    dropout_attn,
                    dropout_skip,
                    dropout_mlp,
                    norm_post_or_pre='pre',
                    ):
            
            super().__init__()

            self.n_nodes_max = n_nodes_max
            self.linear_pos = nn.Linear(n_nodes_max, node_pos_dim)
            self.mlp_nodes_in = MLP(node_labels_dim+node_pos_dim+edge_labels_dim, node_hidden_dim, node_model_dim, dropout_mlp)
            self.mlp_nodes_out_1 = MLP(node_model_dim, node_hidden_dim, node_model_dim, dropout_mlp)
            self.mlp_nodes_out_2 = MLP(node_model_dim, node_hidden_dim, node_model_dim, dropout_mlp)
            
            self.GIN = GIN(model_dim=node_model_dim, hidden_dim=node_hidden_dim, n_layers=n_layers_gnn, dropout=dropout_mlp)
            
            self.TransformerEncoder = TransformerEncoder(n_layers=n_layers,
                                                        node_dim=node_model_dim,
                                                        node_hidden_dim=node_hidden_dim, 
                                                        n_heads=n_heads, 
                                                        dropout_attn=dropout_attn,
                                                        dropout_skip=dropout_skip, 
                                                        dropout_mlp=dropout_mlp,
                                                        norm_post_or_pre='pre'
                                                        )
     
            
            self.TransformerDecoder = TransformerDecoder(n_layers=n_layers,
                                                        src_dim=node_model_dim,
                                                        trgt_dim=graph_embedding_dim, 
                                                        trgt_hidden_dim=graph_embedding_dim,
                                                        trgt_size=graph_embedding_size,
                                                        n_heads=n_heads, 
                                                        dropout_attn=dropout_attn,
                                                        dropout_skip=dropout_skip, 
                                                        dropout_mlp=dropout_mlp,
                                                        norm_post_or_pre=norm_post_or_pre,
                                                        )
        
        
        def forward(self, data: BatchedDenseData):
            batchsize, n_nodes_batch, _ = data.nodes.labels.shape
            h = data.h
            mask_nodes = ~h
            node_labels = data.nodes.labels
            node_pos = build_random_node_pos(batchsize, n_nodes_batch, self.n_nodes_max).to(node_labels)
            node_pos = self.linear_pos(node_pos)
            edge_labels = data.edges.labels # (batch_size, n_nodes, n_nodes, edge_labels_dim)
            A = data.edges.adjacency # (batch_size, n_nodes, n_nodes)
            
            X_edges = (A.unsqueeze(-1) * edge_labels).sum(dim=-2) # (batch_size, n_nodes, edge_labels_dim)
            X = torch.cat([node_labels, node_pos, X_edges], dim=-1)
            X = self.mlp_nodes_in(X)
            
            X = self.GIN(X, A)
            X = self.TransformerEncoder(X, mask=mask_nodes)
            
            node_embeddings = self.mlp_nodes_out_1(X)
            node_embeddings, node_masks = pad_node_embeddings(node_embeddings, mask_nodes, self.n_nodes_max)
            
            Z = self.mlp_nodes_out_2(X)
            graph_embedding = self.TransformerDecoder(X_src=Z, mask_src=mask_nodes)
           
            return node_embeddings, node_masks, graph_embedding