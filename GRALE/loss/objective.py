from .linear_losses import *
from .quadratic_losses import *
from .regularization import *

class GraphAutoencoderObjective():
    '''
    Differentiable objective for training graph autoencoders
    '''
    
    def __init__(self, alpha, mask_self_loops=False):

        self.alpha = alpha
        self.h_objective = LinearBCE()
        self.node_features_objective = LinearL2()
        self.node_labels_objective = LinearCE()
        self.adjacency_objective = QuadraticBCE(mask_self_loops=mask_self_loops)
        self.edge_features_objective = QuadraticL2(mask_self_loops=mask_self_loops)
        self.edge_labels_objective = QuadraticCE(mask_self_loops=mask_self_loops)
        self.marginal_objective = MarginalKL()
        
    def __call__(self, outputs, targets, permutation_matrices):
        '''
        Returns:
            - loss: tensor of shape (batch_size,)
            - log: dictionary of scalar values (averaged over the batch)
        '''
        
        alpha_h = self.alpha['h']
        alpha_node_features = self.alpha['node_features']
        alpha_node_labels = self.alpha['node_labels']
        alpha_edge_features = self.alpha['edge_features']
        alpha_edge_labels = self.alpha['edge_labels']
        alpha_adjacency = self.alpha['adjacency']
        alpha_marginals = self.alpha['marginals']
        
        h_preds = outputs.h
        node_features_preds = outputs.nodes.features
        node_labels_preds = outputs.nodes.labels
        edge_features_preds = outputs.edges.features
        edge_labels_preds = outputs.edges.labels
        A_preds = outputs.edges.adjacency
        
        h_targets = targets.h.to(h_preds.dtype)
        node_features_targets = targets.nodes.features
        node_labels_targets = targets.nodes.labels
        edge_features_targets = targets.edges.features
        edge_labels_targets = targets.edges.labels
        A_targets = targets.edges.adjacency
        
        size_targets = h_targets.sum(dim=-1)
        size_max = h_targets.size(-1)
        weight = h_targets 
        batch_size = h_targets.size(0)
        
        if permutation_matrices is None:
            loss_h = self.h_objective.forward_aligned(F1 = h_preds, F2 = h_targets)/size_max
            loss_node_features = self.node_features_objective.forward_aligned(F1 = node_features_preds, F2 = node_features_targets, weight = weight)/size_targets
            loss_node_labels = self.node_labels_objective.forward_aligned(F1 = node_labels_preds, F2 = node_labels_targets, weight = weight)/size_targets
            loss_edge_features = self.edge_features_objective.forward_aligned(C1 = edge_features_preds, C2 = edge_features_targets, weight = weight)/size_targets**2
            loss_edge_labels = self.edge_labels_objective.forward_aligned(C1 = edge_labels_preds, C2 = edge_labels_targets, weight = weight)/size_targets**2
            loss_adjacency = self.adjacency_objective.forward_aligned(C1 = A_preds, C2 = A_targets, weight = weight)/size_targets**2
            loss_marginals = torch.zeros_like(loss_h)
        else:
            loss_h = self.h_objective.forward(T = permutation_matrices, F1 = h_preds, F2 = h_targets)/size_max
            loss_node_features = self.node_features_objective.forward(T = permutation_matrices, F1 = node_features_preds, F2 = node_features_targets, weight_2 = weight)/size_targets
            loss_node_labels = self.node_labels_objective.forward(T = permutation_matrices, F1 = node_labels_preds, F2 = node_labels_targets, weight_2 = weight)/size_targets
            loss_edge_features = self.edge_features_objective.forward(T = permutation_matrices, C1 = edge_features_preds, C2 = edge_features_targets, weight_2 = weight)/size_targets**2
            loss_edge_labels = self.edge_labels_objective.forward(T = permutation_matrices, C1 = edge_labels_preds, C2 = edge_labels_targets, weight_2 = weight)/size_targets**2
            loss_adjacency = self.adjacency_objective.forward(T = permutation_matrices, C1 = A_preds, C2 = A_targets, weight_2 = weight)/size_targets**2
            loss_marginals = self.marginal_objective.forward(permutation_matrices)
            
        loss = alpha_h*loss_h + alpha_node_features*loss_node_features + alpha_node_labels*loss_node_labels + alpha_edge_features*loss_edge_features + alpha_edge_labels*loss_edge_labels + alpha_adjacency*loss_adjacency + alpha_marginals*loss_marginals 
        
        loss_h_batch = loss_h.sum() / batch_size
        loss_node_features_batch = loss_node_features.sum() / batch_size
        loss_node_labels_batch = loss_node_labels.sum() / batch_size
        loss_edge_features_batch = loss_edge_features.sum() / batch_size
        loss_edge_labels_batch = loss_edge_labels.sum() / batch_size
        loss_adjacency_batch = loss_adjacency.sum() / batch_size
        loss_marginals_batch = loss_marginals.sum() / batch_size
        loss_batch = loss.sum() / batch_size

        log = {'loss': loss_batch.item(),
               'loss_h': loss_h_batch.item(),
               'loss_node_features': loss_node_features_batch.item(),
               'loss_node_labels': loss_node_labels_batch.item(),
               'loss_edge_features': loss_edge_features_batch.item(),
               'loss_edge_labels': loss_edge_labels_batch.item(),
               'loss_adjacency': loss_adjacency_batch.item(),
               'loss_marginals': loss_marginals_batch.item()
               }
        
        return loss, log

class GraphAutoencoderMetric():
    '''
    Non - Differentiable metric for evaluating graph autoencoders
    '''
    
    def __init__(self, mask_self_loops=False):
        
        self.h_objective = LinearBinaryAccuracy()
        self.node_labels_objective = LinearAccuracy()
        self.adjacency_objective = QuadraticBinaryAccuracy(mask_self_loops=mask_self_loops)
        self.edge_labels_objective = QuadraticAccuracy(mask_self_loops=mask_self_loops)
        
    def __call__(self, outputs, targets, permutation_matrices):
        '''
        Returns:
            - loss: tensor of shape (batch_size,)
            - log: dictionary of scalar values (averaged over the batch)
        '''
        
        h_preds = outputs.h
        node_labels_preds = outputs.nodes.labels
        edge_labels_preds = outputs.edges.labels
        A_preds = outputs.edges.adjacency
        
        h_targets = targets.h.to(h_preds.dtype)
        node_labels_targets = targets.nodes.labels
        edge_labels_targets = targets.edges.labels
        A_targets = targets.edges.adjacency
        
        size_targets = h_targets.sum(dim=-1)
        size_max = h_targets.size(-1)
        weight = h_targets
        batchsize = h_targets.size(0)
        
        if permutation_matrices is None:
            loss_h = self.h_objective.forward_aligned(F1 = h_preds, F2 = h_targets)/size_max
            loss_node_labels = self.node_labels_objective.forward_aligned(F1 = node_labels_preds, F2 = node_labels_targets, weight = weight)/size_targets
            loss_edge_labels = self.edge_labels_objective.forward_aligned(C1 = edge_labels_preds, C2 = edge_labels_targets, weight = weight)/size_targets**2
            loss_adjacency = self.adjacency_objective.forward_aligned(C1 = A_preds, C2 = A_targets, weight = weight)/size_targets**2
        else:
            loss_h = self.h_objective.forward(T = permutation_matrices, F1 = h_preds, F2 = h_targets)/size_max
            loss_node_labels = self.node_labels_objective.forward(T = permutation_matrices, F1 = node_labels_preds, F2 = node_labels_targets, weight_2 = weight)/size_targets
            loss_edge_labels = self.edge_labels_objective.forward(T = permutation_matrices, C1 = edge_labels_preds, C2 = edge_labels_targets, weight_2 = weight)/size_targets**2
            loss_adjacency = self.adjacency_objective.forward(T = permutation_matrices, C1 = A_preds, C2 = A_targets, weight_2 = weight)/size_targets**2
        
        edit_graph = (size_targets**2)*loss_edge_labels/2 + size_targets*loss_node_labels
        acc_graph = torch.where(edit_graph<1e-5, torch.tensor(1.), torch.tensor(0.))
        
        acc_h_batch = 1 - loss_h.sum() / batchsize
        acc_node_labels_batch = 1 - loss_node_labels.sum() / batchsize
        acc_edge_labels_batch = 1 - loss_edge_labels.sum() / batchsize
        acc_adjacency_batch = 1 - loss_adjacency.sum() / batchsize
        edit_graph_batch = edit_graph.sum() / batchsize
        acc_graph_batch = acc_graph.sum() / batchsize

        log = {'acc_h': acc_h_batch.item(),
               'acc_node_labels': acc_node_labels_batch.item(),
               'acc_adjacency': acc_adjacency_batch.item(),
               'acc_edge_labels': acc_edge_labels_batch.item(),
               'acc_graph': acc_graph_batch.item(),
               'edit_graph': edit_graph_batch.item()
               }
        
        return edit_graph, log
    
    

    
    