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
            
class SinkhornMatcher(AbstractMatcher):
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

class SoftsortMatcher(AbstractMatcher):
    '''
    score_j = <node_embeddings_inputs_j, U > where U is a learnable parameter
    sorted_score_i = sort(score_i)
    C_ij = |sorted_score_i - score_j|
    P = softmax(-C/epsilon) (row-wise by default)
    Note: node_embeddings_outputs is not used in this model.
    Cost is mostly linear.
    '''

    def __init__(self, 
                 node_model_dim,
                 epsilon = 1e-4,
                 normalize_cost_matrix = True,
                 row_wise = True
                 ):

        super().__init__()

        self.node_padding_features = nn.Parameter(torch.randn(1,1,node_model_dim))
        nn.init.xavier_uniform_(self.node_padding_features)
        self.scoring = nn.Linear(node_model_dim,1)

        self.epsilon = epsilon
        self.normalize_cost_matrix = normalize_cost_matrix
        self.row_wise = row_wise

    def forward(self, node_embeddings_inputs: Tensor, node_masks_inputs: Tensor, node_embeddings_outputs: Tensor, hard = False):

        batchsize, input_size, _ = node_embeddings_inputs.shape

        node_embeddings_inputs = node_embeddings_inputs + 0*node_embeddings_outputs.sum() # To avoid unused variable warning

        # Add padding features to node_embeddings_inputs
        padding = self.node_padding_features.expand(batchsize,input_size,-1)
        mask = node_masks_inputs.unsqueeze(-1)
        node_embeddings_inputs = (~mask)*node_embeddings_inputs + mask*padding
        # Compute scores
        scores = self.scoring(node_embeddings_inputs).squeeze(-1)    

        if hard:
            permutations_list = [torch.argsort(torch.argsort(scores_i)).detach().cpu().numpy() for scores_i in scores] # argsort twice to get inverse permutation
            log_solver = {}
            return permutations_list, log_solver
        else:
            scores_sorted = scores.sort(dim=1)[0]
            C = pairwise_L1_norm(scores_sorted.unsqueeze(-1),scores.unsqueeze(-1))
            if self.normalize_cost_matrix:
                C = C/C.sum(dim=(1,2),keepdim=True)
            C = C/self.epsilon
            if self.row_wise:
                permutation_matrices = torch.softmax(-C,dim=1)
            else:
                permutation_matrices = torch.softmax(-C,dim=2)
            summand_f = permutation_matrices.sum(dim=2)
            summand_g = permutation_matrices.sum(dim=1)
            max_err_a = torch.amax(torch.abs(1 - summand_f), dim=-1).mean()
            max_err_b = torch.amax(torch.abs(1 - summand_g), dim=-1).mean()
            log_solver = {'sinkhorn marginal error (rows)': max_err_a.item(), 'sinkhorn marginal error (cols)': max_err_b.item()} 
            return permutation_matrices, log_solver
        
import inspect
def get_matcher(config):
    matcher_model = config.pop('matcher_model')
    if matcher_model == 'softsort':
        MatcherClass = SoftsortMatcher
    elif matcher_model == 'sinkhorn':
        MatcherClass = SinkhornMatcher
    params = inspect.signature(MatcherClass).parameters
    valid_args = {k: v for k, v in config.items() if k in params}
    return MatcherClass(**valid_args)