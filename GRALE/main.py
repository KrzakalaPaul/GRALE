import torch.nn as nn
import torch
import os
import yaml
from .loss.matching import batched_hungarian_projection, permutations_list_to_matrices
from .loss.objective import GraphAutoencoderObjective, GraphAutoencoderMetric
from .model.encoder import *
from .model.decoder import *
from .model.permuter import *
from .model.target_builder import *
from .model.utils import *

class GraphAutoencoder(nn.Module):
    
    def __init__(self, encoder: AbstractGraphEncoder, decoder: AbstractGraphDecoder, permuter: AbstractPermuter, config: dict):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.permuter = permuter
        self.n_nodes_max = config['n_nodes_max']
        self.config = config
        
    def encode(self, inputs):
        node_embeddings, node_masks, graph_embedding = self.encoder(inputs)
        return graph_embedding
    
    def decode(self, graph_embedding, logits = False):
        node_embeddings, outputs = self.decoder(graph_embedding)
        if not logits:
            outputs = self.format_logits(outputs)
        return outputs
    
    def predict_permutation(self, inputs, hard_permuter = False):
        node_embeddings_inputs, node_masks_inputs, graph_embedding = self.encoder(inputs)
        node_embeddings_outputs, outputs = self.decoder(graph_embedding)
        if hard_permuter:
            permutation_list, _ = self.permuter(node_embeddings_inputs, node_masks_inputs, node_embeddings_outputs, hard = True)
            return permutation_list
        else:
            permutation_matrices, _ = self.permuter(node_embeddings_inputs, node_masks_inputs, node_embeddings_outputs, hard = False)
            return permutation_matrices
    
    def format_logits(self, outputs):
        outputs.h = torch.sigmoid(outputs.h)
        outputs.nodes.labels = torch.softmax(outputs.nodes.labels, dim = -1)
        outputs.edges.adjacency= torch.sigmoid(outputs.edges.adjacency)
        outputs.edges.labels = torch.softmax(outputs.edges.labels, dim = -1)
        return outputs
    
    def forward(self, inputs, hard_permuter=False, logits=False, return_permutations=False):
        node_embeddings_inputs, node_masks_inputs, graph_embedding = self.encoder(inputs)
        node_embeddings_outputs, outputs = self.decoder(graph_embedding)
        if hard_permuter:
            permutation_list, _ = self.permuter(node_embeddings_inputs, node_masks_inputs, node_embeddings_outputs, hard = True)
            outputs.align_(permutation_list)
        else:
            permutation_matrices, _ = self.permuter(node_embeddings_inputs, node_masks_inputs, node_embeddings_outputs, hard = False)
            outputs.permute_(permutation_matrices)
        if not logits:
            outputs = self.format_logits(outputs)
        if return_permutations:
            if hard_permuter:
                return outputs, permutation_list
            else:
                return outputs, permutation_matrices
        return outputs
    
    def loss(self, inputs, targets, alpha, hard_permuter=False, mask_self_loops=False):
        node_embeddings_inputs, node_masks_inputs, graph_embedding = self.encoder(inputs)
        node_embeddings_outputs, outputs = self.decoder(graph_embedding)
        permutation_matrices, log_permuter = self.permuter(node_embeddings_inputs, node_masks_inputs, node_embeddings_outputs, hard = False)
        if hard_permuter:
            permutation_list, _ = batched_hungarian_projection(permutation_matrices, metric = 'F')    
            permutation_matrices_hard = permutations_list_to_matrices(permutation_list)
            permutation_matrices = (permutation_matrices_hard - permutation_matrices).detach() + permutation_matrices
        objective = GraphAutoencoderObjective(alpha,mask_self_loops=mask_self_loops)
        loss, log_loss = objective(outputs, targets, permutation_matrices) 
        loss = loss.mean()
        log = log_loss | log_permuter
        return loss, log
    
    def metric(self, inputs, targets, alpha, mask_self_loops=False):
        '''
        Return all metrics:
            - Training Objective (Soft Permuter)
            - Training Objective (Hard Permuter)
            - Evaluation Metric  (Hard Permuter)
        '''
        objective = GraphAutoencoderObjective(alpha,mask_self_loops=mask_self_loops)
        metric = GraphAutoencoderMetric(mask_self_loops=mask_self_loops)
        
        node_embeddings_inputs, node_masks_inputs, graph_embedding = self.encoder(inputs)
        node_embeddings_outputs, outputs = self.decoder(graph_embedding)
        
        # Compute Soft Permuter
        permutation_matrices, log_permuter = self.permuter(node_embeddings_inputs, node_masks_inputs, node_embeddings_outputs, hard = False)
        
        _, log_objective_soft = objective(outputs, targets, permutation_matrices) 
        
        # Apply Hard Permuter
        permutation_list, log_permuter = self.permuter(node_embeddings_inputs, node_masks_inputs, node_embeddings_outputs, hard = True)
        aligned_outputs = outputs.clone()
        aligned_outputs.align_(permutation_list)
    
        _, log_objective_hard = objective(aligned_outputs, targets, None) 
        log_metric_hard = metric(self.format_logits(aligned_outputs), targets, None)
        
        return log_objective_soft, log_objective_hard, log_metric_hard
        
def build_model(config):
    
    # Dimensions nodes
    node_labels_dim = config['node_labels_dim']
    node_features_dim = config['node_features_dim']
    node_model_dim = config['node_model_dim']
    node_pos_dim = config['node_pos_dim']
    node_hidden_dim = config['node_hidden_dim']
    # Dimensions edges
    edge_labels_dim = config['edge_labels_dim']
    edge_features_dim = config['edge_features_dim']
    edge_model_dim = config['edge_model_dim']
    edge_hidden_dim = config['edge_hidden_dim']
    edge_pos_dim = config['edge_pos_dim']
    # Dimensions graph
    graph_embedding_dim = config['graph_embedding_dim']
    graph_embedding_size = config['graph_embedding_size']
    # Attention parameters
    n_heads = config['n_heads']
    n_layers = config['n_layers']
    dropout_attn = config['dropout_attn']
    dropout_skip = config['dropout_skip']
    dropout_mlp = config['dropout_mlp']
    norm_post_or_pre = config['norm_post_or_pre']
    # Permuter parameters
    add_positionnal_outputs = config['add_positionnal_outputs']
    max_iter_sinkhorn = config['max_iter_sinkhorn']
    tol_sinkhorn = config['tol_sinkhorn']
    last_iter_grad_sinkhorn  = config['last_iter_grad_sinkhorn']
    fixed_n_iters_sinkhorn = config['fixed_n_iters_sinkhorn']
    # Misc
    n_nodes_max = config['n_nodes_max']
    
    # Build encoder
    encoder_model = config['encoder_model']
    triangular_attention_mode_encoder = config['triangular_attention_mode_encoder']
    triangular_multiplication_mode_encoder = config['triangular_multiplication_mode_encoder']
    if encoder_model == 'gnn':
        n_layers_gnn = config['n_layers_gnn']
        encoder = GNNGraphEncoder(
            node_labels_dim=node_labels_dim,
            node_model_dim=node_model_dim,
            node_pos_dim=node_pos_dim,
            node_hidden_dim=node_hidden_dim,
            edge_labels_dim=edge_labels_dim,
            graph_embedding_dim=graph_embedding_dim,
            graph_embedding_size=graph_embedding_size,
            n_nodes_max=n_nodes_max,
            n_heads=n_heads,
            n_layers=n_layers,
            n_layers_gnn=n_layers_gnn,
            dropout_attn=dropout_attn,
            dropout_skip=dropout_skip,
            dropout_mlp=dropout_mlp,
            norm_post_or_pre=norm_post_or_pre
        )
    elif encoder_model == 'evoformer':
        encoder = EvoformerGraphEncoder(
                node_labels_dim=node_labels_dim,
                node_pos_dim=node_pos_dim,
                node_model_dim=node_model_dim,
                node_hidden_dim=node_hidden_dim,
                edge_labels_dim=edge_labels_dim, 
                edge_model_dim=edge_model_dim,
                edge_hidden_dim=edge_hidden_dim,
                graph_embedding_dim=graph_embedding_dim,
                graph_embedding_size=graph_embedding_size,
                n_nodes_max=n_nodes_max,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout_attn=dropout_attn,
                dropout_skip=dropout_skip,
                dropout_mlp=dropout_mlp,
                edge_pos_dim = edge_pos_dim,
                norm_post_or_pre=norm_post_or_pre,
                triangular_multiplication_mode=triangular_multiplication_mode_encoder,
                triangular_attention_mode=triangular_attention_mode_encoder
            )
        
    # Build decoder
    decoder_model = config['decoder_model']
    triangular_attention_mode_decoder = config['triangular_attention_mode_decoder']
    triangular_multiplication_mode_decoder = config['triangular_multiplication_mode_decoder']
    if decoder_model == 'transformer':
        decoder = TransformerGraphDecoder(
            node_labels_dim=node_labels_dim,
            node_features_dim=node_features_dim,
            node_model_dim=node_model_dim,
            node_hidden_dim=node_hidden_dim,
            edge_labels_dim=edge_labels_dim,
            edge_features_dim=edge_features_dim,
            edge_model_dim=edge_model_dim,
            edge_hidden_dim=edge_hidden_dim,
            graph_embedding_dim=graph_embedding_dim,
            n_nodes_max=n_nodes_max,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout_attn=dropout_attn,
            dropout_skip=dropout_skip,
            dropout_mlp=dropout_mlp,
            norm_post_or_pre=norm_post_or_pre
        )
    elif decoder_model == 'evoformer':
        decoder = EvoformerGraphDecoder(
                node_labels_dim=node_labels_dim,
                node_features_dim=node_features_dim,
                node_model_dim=node_model_dim,
                node_hidden_dim=node_hidden_dim,
                edge_labels_dim=edge_labels_dim,
                edge_features_dim=edge_features_dim,
                edge_model_dim=edge_model_dim,
                edge_hidden_dim=edge_hidden_dim,
                graph_embedding_dim=graph_embedding_dim,
                n_nodes_max=n_nodes_max,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout_attn=dropout_attn,
                dropout_skip=dropout_skip,
                dropout_mlp=dropout_mlp,
                norm_post_or_pre=norm_post_or_pre,
                triangular_multiplication_mode=triangular_multiplication_mode_decoder,
                triangular_attention_mode=triangular_attention_mode_decoder
            )        
        
    # Build permuter        
    permuter_model = config['permuter_model']
    epsilon = config['epsilon']
    normalize_cost_matrix = config['normalize_cost_matrix']
    if permuter_model == 'softsort':
        permuter = SoftsortPermuter(
                node_model_dim = node_model_dim,
                epsilon = epsilon,
                row_wise = True
            )
    elif permuter_model == 'sinkhorn':
        permuter_dim = config['permuter_dim']
        permuter = SinkhornPermuter( 
                        node_model_dim=node_model_dim,
                        permuter_dim=permuter_dim,
                        n_nodes=n_nodes_max,
                        add_positionnal_outputs = add_positionnal_outputs,
                        max_iter_sinkhorn = max_iter_sinkhorn,
                        tol_sinkhorn = tol_sinkhorn,
                        last_iter_grad_sinkhorn = last_iter_grad_sinkhorn,
                        fixed_n_iters_sinkhorn = fixed_n_iters_sinkhorn,
                        epsilon = epsilon,
                        normalize_cost_matrix = normalize_cost_matrix
                    )
    elif permuter_model == 'evoformer':
        triangular_multiplication_mode_permuter = config['triangular_multiplication_mode_permuter']
        triangular_attention_mode_permuter = config['triangular_attention_mode_permuter']
        permuter = EvoformerPermuter(
                node_model_dim=node_model_dim,
                node_hidden_dim=node_hidden_dim,
                edge_model_dim=edge_model_dim,
                edge_hidden_dim=edge_hidden_dim,
                n_nodes=n_nodes_max,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout_attn=dropout_attn,
                dropout_skip=dropout_skip,
                dropout_mlp=dropout_mlp,
                norm_post_or_pre=norm_post_or_pre,
                triangular_multiplication_mode=triangular_multiplication_mode_permuter,
                triangular_attention_mode=triangular_attention_mode_permuter,
                symmetric=False,
                max_iter_sinkhorn = max_iter_sinkhorn,
                tol_sinkhorn = tol_sinkhorn,
                add_positionnal_outputs = add_positionnal_outputs,
                last_iter_grad_sinkhorn = last_iter_grad_sinkhorn,
                fixed_n_iters_sinkhorn = fixed_n_iters_sinkhorn
                )
        
    model = GraphAutoencoder(encoder, decoder, permuter, config)
    
    return model

def save_model(model: GraphAutoencoder,path,module=False):
    config = model.config
    os.makedirs(path, exist_ok=True)
    with open(path+'/config.yaml', 'w') as f:
        yaml.dump(config, f)
    if module:
        torch.save(model.module.state_dict(), path+'/state_dict.pth')
    else:
        torch.save(model.state_dict(), path+'/state_dict.pth')

def load_model(path, config = None):  
    if config is None:
        with open(path+'/config.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    model = build_model(config)
    model.load_state_dict(torch.load(path+'/state_dict.pth'))
    return model
    

