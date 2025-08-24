import torch.nn as nn
import torch
from torch import Tensor
from .attention import EvoformerSetMatching
from .loss.utils import pairwise_L1_norm
from .loss.matching import batched_sinkhorn_projection, batched_hungarian_projection, batched_log_sinkhorn_projection

# Note: inputs = unsorted, outputs = sorted (by the model).
# The Permuter part of the model is responsible for finding the permutation that will reorder the outputs to match the inputs.

class AbstractPermuter(nn.Module):
    '''
    Abstract class for Permuter part of the Graph Autoencoder.
    '''
    def forward(self, node_embeddings_inputs: Tensor, node_masks_inputs: Tensor, node_embeddings_outputs: Tensor, hard = False):
        '''
        Input: 
            node_masks_inputs of shape (batch_size, n_nodes)
            node_embeddings_inputs of shape (batch_size, n_nodes, node_model_dim)   
            node_embeddings_outputs of shape (batch_size, n_nodes, node_model_dim)
        Outputs:
            permutations to apply to the outputs to match the orderering of the inputs i.e.
                permutation_list: list of numpy array (n_nodes,) if hard 
                permutation_matrices: Tensor (batch_size, n_nodes, n_nodes) else
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
        
        
class SoftsortPermuter(nn.Module):
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
                 epsilon,
                 row_wise = True
                 ):
        
        super().__init__()
        
        self.node_padding_features = nn.Parameter(torch.randn(1,1,node_model_dim))
        nn.init.xavier_uniform_(self.node_padding_features)
        self.scoring = nn.Linear(node_model_dim,1)
        
        self.epsilon = epsilon
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
            
class SinkhornPermuter(nn.Module):
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
                 permuter_dim,
                 n_nodes,
                 add_positionnal_outputs = True,
                 max_iter_sinkhorn = 100,
                 tol_sinkhorn = 1e-5,
                 last_iter_grad_sinkhorn = False,
                 fixed_n_iters_sinkhorn = False,
                 normalize_cost_matrix = True,
                 epsilon = 1e-4
                 ):
        
        super().__init__()
        
        self.node_padding_features = nn.Parameter(torch.randn(1,1,node_model_dim))
        nn.init.xavier_uniform_(self.node_padding_features)
        self.positionnal_encoding_outputs = nn.Parameter(torch.randn(1,n_nodes,node_model_dim))
        nn.init.xavier_uniform_(self.positionnal_encoding_outputs)
        self.linear_inputs = nn.Linear(node_model_dim,permuter_dim)
        self.linear_outputs = nn.Linear(node_model_dim,permuter_dim)
        
        self.add_positionnal_outputs = add_positionnal_outputs
        self.max_iter = max_iter_sinkhorn
        self.tol = tol_sinkhorn
        self.last_iter_grad = last_iter_grad_sinkhorn
        self.fixed_n_iters = fixed_n_iters_sinkhorn
        self.normalize_cost_matrix = normalize_cost_matrix
        self.epsilon = epsilon
        
    def forward(self, node_embeddings_inputs: Tensor, node_masks_inputs: Tensor, node_embeddings_outputs: Tensor, hard = False):
        
        batchsize, n_nodes, _ = node_embeddings_inputs.shape
        
        # Add padding features to node_embeddings_inputs
        padding = self.node_padding_features.expand(batchsize,n_nodes,-1)
        mask = node_masks_inputs.unsqueeze(-1)
        node_embeddings_inputs = (~mask)*node_embeddings_inputs + mask*padding
        # Add positionnal encoding to outputs
        if self.add_positionnal_outputs:
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
            P, log_P, log_solver = batched_log_sinkhorn_projection(log_K,max_iter=self.max_iter,tol=self.tol, last_iter_grad = self.last_iter_grad, fixed_n_iters = self.fixed_n_iters)
            return P, log_solver
        '''
        # Aff log
        aff = torch.einsum('bik,bjk->bij',a,b)
        log_K = torch.log_softmax(aff,dim=1)
        if hard:
            return batched_hungarian_projection(log_K, metric='F')
        else:
            P, log_P, log_solver = batched_log_sinkhorn_projection(log_K,max_iter=self.max_iter,tol=self.tol, last_iter_grad = self.last_iter_grad, fixed_n_iters = self.fixed_n_iters)
            return P, log_solver
        # Aff 
        aff = torch.einsum('bik,bjk->bij',a,b)
        aff1 = torch.softmax(aff,dim=1)
        aff2 = torch.softmax(aff,dim=2)
        K = 0.5 * aff1 + 0.5 * aff2
        if hard:
            return batched_hungarian_projection(K, metric='KL')
        else:
            return batched_sinkhorn_projection(K,max_iter=self.max_iter,tol=self.tol, last_iter_grad = self.last_iter_grad, fixed_n_iters = self.fixed_n_iters)
        # Dists
        C = pairwise_L1_norm(a,b)
        if self.normalize_cost_matrix:
            C = C/C.sum(dim=(1,2),keepdim=True)
        C = C/self.epsilon
        log_K = -C
        '''
class EvoformerPermuter(nn.Module):
    '''
    aff_ij = EvorformerSetMatching(node_embeddings_outputs, node_embeddings_inputs)
    K = 0.5 softmax(aff_ij, dim = 0) + 0.5 softmax(aff_ij, dim = 1) (approximate marginals)
    P = Sinkhorn(K) (K iterations of Sinkhorn projections, marginal might not be fully respected if it didn't converge)
    Cost is Cubic.
    '''
    
    def __init__(self, 
                 node_model_dim,
                 node_hidden_dim,
                 edge_model_dim,
                 edge_hidden_dim,
                 n_nodes,
                 n_heads,
                 n_layers,
                 dropout_attn,
                 dropout_skip,
                 dropout_mlp,
                 norm_post_or_pre='pre',
                 triangular_multiplication_mode='symmetric',
                 triangular_attention_mode='none',
                 symmetric=False,
                 max_iter_sinkhorn = 0,
                 tol_sinkhorn = 1e-5,
                 last_iter_grad_sinkhorn = False,
                 fixed_n_iters_sinkhorn = False,
                 add_positionnal_outputs = True
                 ):
        
        super().__init__()
        
        self.node_padding_features = nn.Parameter(torch.randn(1,1,node_model_dim))
        nn.init.xavier_uniform_(self.node_padding_features)
        self.positionnal_encoding_outputs = nn.Parameter(torch.randn(1,n_nodes,node_model_dim))
        nn.init.xavier_uniform_(self.positionnal_encoding_outputs)
        
        self.evoformer = EvoformerSetMatching(n_layers=n_layers,
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
                                            triangular_attention_mode=triangular_attention_mode,
                                            symmetric=symmetric)
        self.linear_aff = nn.Linear(edge_model_dim,1)

        self.add_positionnal_outputs = add_positionnal_outputs
        self.max_iter = max_iter_sinkhorn
        self.tol = tol_sinkhorn
        self.last_iter_grad = last_iter_grad_sinkhorn
        self.fixed_n_iters = fixed_n_iters_sinkhorn
        
    def forward(self, node_embeddings_inputs: Tensor, node_masks_inputs: Tensor, node_embeddings_outputs: Tensor, hard = False):
        
        batchsize, n_nodes, _ = node_embeddings_inputs.shape
        
        # Add padding features to node_embeddings_inputs
        padding = self.node_padding_features.expand(batchsize,n_nodes,-1)
        mask = node_masks_inputs.unsqueeze(-1)
        node_embeddings_inputs = (~mask)*node_embeddings_inputs + mask*padding
        # Add positionnal encoding to outputs
        if self.add_positionnal_outputs:
            node_embeddings_outputs = node_embeddings_outputs + self.positionnal_encoding_outputs
        # Compute affinity
        aff = self.evoformer(node_embeddings_outputs, node_embeddings_inputs)
        aff = self.linear_aff(aff).squeeze(-1)
        aff1 = torch.softmax(aff,dim=1)
        aff2 = torch.softmax(aff,dim=2)
        K = 0.5 * aff1 + 0.5 * aff2
        if hard:
            return batched_hungarian_projection(K, metric='KL')
        else:
            return batched_sinkhorn_projection(K,max_iter=self.max_iter,tol=self.tol, last_iter_grad = self.last_iter_grad, fixed_n_iters = self.fixed_n_iters)