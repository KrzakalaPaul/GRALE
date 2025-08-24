import torch
import os
import yaml
from .loss.objective import GraphAutoencoderObjective, GraphAutoencoderMetric
from .model import get_encoder, get_decoder, get_matcher, get_target_builder, get_input_builder
from GRALE.data import BatchedDenseData
import pytorch_lightning as pl
    
class GRALE(pl.LightningModule):
    
    def __init__(self, config: dict):
        super().__init__()
        self.encoder = get_encoder(config)
        self.decoder = get_decoder(config)
        self.matcher = get_matcher(config)
        self.target_builder = get_target_builder(config)
        self.input_builder = get_input_builder(config)
        self.n_nodes_max = config['n_nodes_max']
        self.config = config
        self.training_objective = GraphAutoencoderObjective(get_alpha(config))
        self.validation_metric = GraphAutoencoderMetric()
        
    def training_step(self, batch: BatchedDenseData):
        
        # Prepare inputs and targets
        inputs = self.input_builder(batch)
        targets = self.target_builder(batch)
        node_masks_inputs = ~inputs.h
        
        # Pass encoder and decoder
        node_embeddings_inputs, graph_embedding = self.encoder(inputs)
        node_embeddings_outputs, outputs = self.decoder(graph_embedding)
        
        # Pass node embeddings to the matcher
        permutation_matrices, log_matcher = self.matcher(node_embeddings_inputs, node_masks_inputs, node_embeddings_outputs, hard = False)

        # Compute loss
        loss, log_loss = self.training_objective(outputs, targets, permutation_matrices)

        # Log metrics
        pass
    
    def validation_step(self, batch: BatchedDenseData):
        
        # Prepare inputs and targets
        inputs = self.input_builder(batch)
        targets = self.target_builder(batch)
        node_masks_inputs = ~inputs.h
        
        # Pass encoder and decoder
        node_embeddings_inputs, graph_embedding = self.encoder(inputs)
        node_embeddings_outputs, outputs = self.decoder(graph_embedding)
        
        # Pass node embeddings to the matcher
        permutation_list, log_matcher = self.matcher(node_embeddings_inputs, node_masks_inputs, node_embeddings_outputs, hard = True)
        
        # Align outputs to inputs
        aligned_outputs = outputs.clone()
        aligned_outputs.align_(permutation_list)
        
        # Compute loss
        metric, log_metric = self.validation_metric(self.format_logits(aligned_outputs), targets, permutation_matrices=None)

        # Log metrics
        pass
    
    def encode(self, data: BatchedDenseData):
        '''
        Returns only the graph embedding from the encoder
        '''
        inputs = self.input_builder(data)
        node_embeddings, graph_embedding = self.encoder(inputs)
        return graph_embedding

    def decode(self, graph_embedding: torch.Tensor, logits = False):
        '''
        Returns the output from the decoder
        '''
        node_embeddings, outputs = self.decoder(graph_embedding)
        if not logits:
            outputs = self.format_logits(outputs)
        return outputs

    def canonical_permutation(self, data: BatchedDenseData, hard_matcher = False):
        '''
        Predicts the permutation that canonizes the input graph.
        # Note: this is actually the permutation that aligns the output graph to the input graph. 
        # To "canonize" the input graph, one needs to apply the inverse permutation.
        '''
        inputs = self.input_builder(data)
        node_masks_inputs = ~inputs.h
        node_embeddings_inputs, graph_embedding = self.encoder(inputs)
        node_embeddings_outputs, outputs = self.decoder(graph_embedding)
        if hard_matcher:
            permutation_list, _ = self.matcher(node_embeddings_inputs, node_masks_inputs, node_embeddings_outputs, hard = True)
            return permutation_list
        else:
            permutation_matrices, _ = self.matcher(node_embeddings_inputs, node_masks_inputs, node_embeddings_outputs, hard = False)
            return permutation_matrices
    
    def format_logits(self, outputs):
        '''
        Apply activation functions to the outputs of the decoder
        '''
        outputs.h = torch.sigmoid(outputs.h)
        outputs.nodes.labels = torch.softmax(outputs.nodes.labels, dim = -1)
        outputs.edges.adjacency = torch.sigmoid(outputs.edges.adjacency)
        outputs.edges.labels = torch.softmax(outputs.edges.labels, dim = -1)
        return outputs

    def forward(self, data: BatchedDenseData, hard_matcher=False, logits=False, return_permutations=False):
        '''
        Forward propagation of the full model: Encoder + Decoder + Matcher
        The matcher is used to align the output graph to the input graph.
        '''
        inputs = self.input_builder(data)
        node_masks_inputs = ~inputs.h
        node_embeddings_inputs, graph_embedding = self.encoder(inputs)
        node_embeddings_outputs, outputs = self.decoder(graph_embedding)
        if hard_matcher:
            permutation_list, _ = self.matcher(node_embeddings_inputs, node_masks_inputs, node_embeddings_outputs, hard = True)
            outputs.align_(permutation_list)
        else:
            permutation_matrices, _ = self.matcher(node_embeddings_inputs, node_masks_inputs, node_embeddings_outputs, hard = False)
            outputs.permute_(permutation_matrices)
        if not logits:
            outputs = self.format_logits(outputs)
        if return_permutations:
            if hard_matcher:
                return outputs, permutation_list
            else:
                return outputs, permutation_matrices
        return outputs

def get_alpha(config):
    return {'h': config['alpha_h'], 
            'node_features': config['alpha_node_features'],
            'node_labels': config['alpha_node_labels'], 
            'edge_features': config['alpha_edge_features'], 
            'edge_labels': config['alpha_edge_labels'], 
            'adjacency': config['alpha_adjacency'], 
            'marginals': config['alpha_marginals']}