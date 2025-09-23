import torch
import os
import yaml
from .loss.objective import GraphAutoencoderObjective, GraphAutoencoderMetric
from .model import get_encoder, get_decoder, get_matcher, get_target_builder, get_input_builder
from GRALE.data import BatchedDenseData
import lightning.pytorch as pl
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

class GRALE_model(pl.LightningModule):
    
    def __init__(self, config: dict):
        super().__init__()
        self.encoder = get_encoder(config)
        self.decoder = get_decoder(config)
        self.matcher = get_matcher(config)
        self.target_builder = get_target_builder(config)
        self.input_builder = get_input_builder(config)
        self.n_nodes_max = config['n_nodes_max']
        self.training_objective = GraphAutoencoderObjective(get_alpha(config))
        self.validation_metric = GraphAutoencoderMetric()
        self.save_hyperparameters(config)
        
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
        # self.log_dict(log_loss, on_epoch=True, batch_size=inputs.batchsize)
        # lr = self.trainer.optimizers[0].param_groups[0]['lr']
        # self.log("lr", lr, prog_bar=False, on_step=True, on_epoch=False)

        return loss.mean()
        
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
        self.log_dict(log_metric, on_epoch=True, batch_size=inputs.batchsize, sync_dist=True)
        
        # Also get the training loss for logging
        permutation_matrices, log_matcher = self.matcher(node_embeddings_inputs, node_masks_inputs, node_embeddings_outputs, hard = False)
        loss, log_loss = self.training_objective(outputs, targets, permutation_matrices)
        self.log_dict(log_loss, on_epoch=True, batch_size=inputs.batchsize, sync_dist=True)
        
        return metric.mean()

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
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["base_lr"])

        warmup_steps = self.hparams.n_warmup_steps
        total_steps = self.hparams.n_grad_steps

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            # cosine annealing after warmup
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda),
            "interval": "step",     # update per step
            "frequency": 1
        }

        return [optimizer], [scheduler]

def get_alpha(config):
    return {'h': config['alpha_h'], 
            'node_features': config['alpha_node_features'],
            'node_labels': config['alpha_node_labels'], 
            'edge_features': config['alpha_edge_features'], 
            'edge_labels': config['alpha_edge_labels'], 
            'adjacency': config['alpha_adjacency'], 
            'marginals': config['alpha_marginals']}
    
    
