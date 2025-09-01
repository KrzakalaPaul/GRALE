import torch.nn as nn
import torch
from torch import Tensor
from GRALE.attention import EvoformerSetMatching
from GRALE.loss.utils import pairwise_L1_norm
from GRALE.loss.matching import batched_sinkhorn_projection, batched_hungarian_projection, batched_log_sinkhorn_projection

# Note: inputs = unsorted, outputs = sorted (by the model).
# The matcher part of the model is responsible for finding the permutation that will reorder the outputs to match the inputs.

class AbstractMatcher(nn.Module):
    '''
    Abstract class for Matcher part of the Graph Autoencoder.
    '''
    def forward(self, node_embeddings_inputs: Tensor, node_embeddings_outputs: Tensor, hard = False):
        '''
        Input: 
            node_masks_inputs of shape (batch_size, n_nodes_max)
            node_embeddings_inputs of shape (batch_size, n_nodes_max, node_model_dim)   
            node_embeddings_outputs of shape (batch_size, n_nodes_max, node_model_dim)
        Outputs:
            permutations to apply to the outputs to match the orderering of the inputs i.e.
                permutation_list: list of numpy array (n_nodes_max,) if hard 
                permutation_matrices: Tensor (batch_size, n_nodes_max, n_nodes_max) else
            log_solver: dict of additional information to log
        '''
        if hard:
            permutation_list = ...
            log_solver = ...
            return permutation_list, log_solver
        else:
            permutation_matrices = ...
            log_solver = ...
            return permutation_matrices, log_solver
            
class SinkhornMatcher(nn.Module):
    '''
    a_i = Linear(node_embeddings_outputs_i)
    b_j = Linear(node_embeddings_inputs_j)
    aff_ij = <a_i, b_j>
    K = 0.5 softmax(aff_ij, dim = 0) + 0.5 softmax(aff_ij, dim = 1) (approximate marginals)
    P = Sinkhorn(K) (K iterations of Sinkhorn projections, marginal might not be fully respected if it didn't converge)
    Cost is quadratic.
    '''
    
    def __init__(self, 
                 node_model_dim,
                 matcher_dim,
                 n_nodes_max,
                 max_iter_sinkhorn = 100,
                 tol_sinkhorn = 1e-3,
                 fixed_n_iters_sinkhorn = False,
                 normalize_cost_matrix = True,
                 epsilon = 1e-4
                 ):
        
        super().__init__()
        
        self.node_padding_features = nn.Parameter(torch.randn(1,1,node_model_dim))
        nn.init.xavier_uniform_(self.node_padding_features)
        self.positionnal_encoding_outputs = nn.Parameter(torch.randn(1,n_nodes_max,node_model_dim))
        nn.init.xavier_uniform_(self.positionnal_encoding_outputs)
        self.linear_inputs = nn.Linear(node_model_dim,matcher_dim)
        self.linear_outputs = nn.Linear(node_model_dim,matcher_dim)
    
        self.max_iter = max_iter_sinkhorn
        self.tol = tol_sinkhorn
        self.fixed_n_iters = fixed_n_iters_sinkhorn
        self.normalize_cost_matrix = normalize_cost_matrix
        self.epsilon = epsilon
        
    def forward(self, node_embeddings_inputs: Tensor, node_masks_inputs: Tensor, node_embeddings_outputs: Tensor, hard = False):
        
        batchsize, n_nodes_max, _ = node_embeddings_inputs.shape
        
        # Add padding features to node_embeddings_inputs
        padding = self.node_padding_features.expand(batchsize,n_nodes_max,-1)
        mask = node_masks_inputs.unsqueeze(-1)
        node_embeddings_inputs = (~mask)*node_embeddings_inputs + mask*padding
        # Add positionnal encoding to outputs
        node_embeddings_outputs = node_embeddings_outputs + self.positionnal_encoding_outputs
        # Compute affinity
        a = self.linear_outputs(node_embeddings_outputs)
        b = self.linear_inputs(node_embeddings_inputs)
        C = pairwise_L1_norm(a,b)
        if self.normalize_cost_matrix:
            C = C/C.sum(dim=(1,2),keepdim=True)
        C = C/self.epsilon
        log_K = -C
        if hard:
            return batched_hungarian_projection(log_K, metric='F')
        else:
            P, log_P, log_solver = batched_log_sinkhorn_projection(log_K,max_iter=self.max_iter,tol=self.tol, fixed_n_iters = self.fixed_n_iters)
            return P, log_solver

import inspect
def get_matcher(config):
    params = inspect.signature(SinkhornMatcher).parameters
    valid_args = {k: v for k, v in config.items() if k in params}
    return SinkhornMatcher(**valid_args)