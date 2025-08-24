import torch.nn as nn
import torch
from torch_geometric.nn.dense import DenseGINConv

class MLP(nn.Module):
    '''
    Always include a non-linearity.
    In particular: setting hidden_dim to None will result in a linear layer + non-linearity (usefull for lightweight FF block).
    '''
    def __init__(self, in_dim, hidden_dim, out_dim, dropout = 0):
        super().__init__()
        if hidden_dim is None:
            self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim),
                                    nn.ReLU(),
                                    nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
                                    )
        else:
            self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                                    nn.Linear(hidden_dim, out_dim))
    def forward(self, x):
        return self.mlp(x)
    
class MLP_head(nn.Module):
    '''
    Does not include a non-linearity if hidden_dim is None (usefull for the last layer of the model).
    '''
    def __init__(self, in_dim, hidden_dim, out_dim, dropout = 0):
        super().__init__()
        if hidden_dim is None:
            self.mlp = nn.Linear(in_dim, out_dim)
        else:
            self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                                    nn.Linear(hidden_dim, out_dim))
    def forward(self, x):
        return self.mlp(x)
    
class GIN(nn.Module):
    def __init__(self, model_dim, hidden_dim, n_layers, dropout = 0):
        super().__init__()
        self.layers = nn.ModuleList([DenseGINConv(MLP(model_dim, hidden_dim, model_dim, dropout)) for _ in range(n_layers)])
    def forward(self, x, A):
        for layer in self.layers:
            x = layer(x, A)
        return x
        
class SineEmbedding(nn.Module):
    '''
    Use to encode ints into a fixed-size vector 
    '''
    def __init__(self,pos_dim):
        super().__init__()
        assert pos_dim % 2 == 0
        self.freqs = 1.0 / (10000 ** (torch.arange(0, pos_dim, 2).float() / pos_dim))
    def forward(self, x):
        freqs = self.freqs.to(x.device)
        x = x[..., None] * freqs[None, :]
        sin = torch.sin(x)
        cos = torch.cos(x)
        return torch.cat([sin, cos], dim=-1)
    
def update_running_average(avg,n_avg,x,h):
    '''
    avg a tensor of shape (d,) computed on n_avg samples
    x a tensor of shape (n,d) (n new samples)
    h a tensor of shape (n,) (1 if the sample is valid, 0 otherwise)
    '''
    n = h.sum() # number of new samples                                   
    new_avg = (x*h.unsqueeze(-1)).sum(dim=0)/n # average of new samples
    new_n_avg = n_avg + n
    new_avg = (n_avg * avg + n * new_avg) / new_n_avg
    return new_avg, new_n_avg
    
def flatten_upper_diagonal(E):
    # E of shape (batch_size, n_nodes, n_nodes, E_dim)
    # returns E of shape (batch_size, n_nodes * (n_nodes + 1) // 2, E_dim)
    n_nodes = E.shape[1]
    indices = torch.triu_indices(n_nodes, n_nodes, offset=0)
    E_flat = E[:, indices[0], indices[1]]
    return E_flat

def build_E_from_X(X, undirected_edges = False):
    '''
    E_ij = [X_i, X_j] if undirected_edges else X_i + X_j
    '''
    batchsize, size, dim = X.shape
    if undirected_edges:
        return X.unsqueeze(1).expand(-1,size,-1,-1) + X.unsqueeze(2).expand(-1,-1,size,-1)
    return torch.concatenate((X.unsqueeze(1).expand(-1,size,-1,-1),X.unsqueeze(2).expand(-1,-1,size,-1)),dim=-1)

def freeze(module:nn.Module):
    for param in module.parameters():
        param.requires_grad = False
        
def unfreeze(module:nn.Module):
    for param in module.parameters():
        param.requires_grad = True
        
def pad_along_dim(x, padding_size, dim):
    # put dimension to pad at the end
    x = x.transpose(dim, -1)
    # pad the last dimension
    x = torch.nn.functional.pad(x, (0, padding_size))
    # transpose back
    x = x.transpose(dim, -1)
    return x

def pad_node_embeddings(node_embeddings, node_masks, n_nodes_max):
        '''
        Inputs: 
            node_embeddings: torch.Tensor of shape (batch_size, n_nodes_batch, node_embedding_dim)  
            node_masks: torch.Tensor of shape (batch_size, n_nodes_batch)
        Return:
            node_embeddings: torch.Tensor of shape (batch_size, n_nodes_max, node_embedding_dim)  
            node_masks: torch.Tensor of shape (batch_size, n_nodes_max)  
        '''
        n_nodes_batch = node_embeddings.shape[1]
        padding_size = n_nodes_max - n_nodes_batch
        node_embeddings = pad_along_dim(node_embeddings, padding_size, 1)
        node_masks = torch.nn.functional.pad(node_masks, (0, padding_size), value = True)
        return node_embeddings, node_masks
    
def build_random_node_pos(batchsize, n_nodes_batch, n_nodes_max):
    x = torch.stack([torch.randperm(n_nodes_max) for _ in range(batchsize)]) 
    x = x[:, :n_nodes_batch]
    x = torch.nn.functional.one_hot(x, n_nodes_max)
    return x


def build_laplacian_node_pos(A, n_eigvecs):
    '''
    A: torch.Tensor of shape (batchsize, n_nodes_batch, n_nodes_batch)
    '''
    D = A.sum(dim=-1)
    L = torch.diag_embed(D) - A
    eigvals, eigvecs = torch.linalg.eigh(L.to(float))
    return eigvecs[:, :, 1:n_eigvecs+1] 