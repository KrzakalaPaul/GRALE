import torch.nn as nn
from .blocks import *
from copy import deepcopy 

def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])

def with_pos_embed(tensor, pos: Optional[Tensor]):
    return tensor if pos is None else tensor + pos

def reset_parameters(self):
    for p in self.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
class TransformerEncoderLayer(nn.Module):
    '''
    Set Encoder Layer, use to update X of shape (shape: n_nodes, node_dim) 
    '''
    def __init__(self, 
                 node_dim, 
                 node_hidden_dim,
                 n_heads, 
                 dropout_attn = 0.0,
                 dropout_skip=0.0, 
                 dropout_mlp=0.0,
                 norm_post_or_pre='pre'):
        
        super().__init__()
        
        self.SAB = SAB(model_dim=node_dim, n_heads=n_heads, dropout_attn=dropout_attn, dropout_skip=dropout_skip, norm_post_or_pre=norm_post_or_pre)
        self.FFB = FFB(in_dim=node_dim,out_dim=node_dim,hidden_dim=node_hidden_dim,dropout_mlp=dropout_mlp,dropout_skip=dropout_skip,norm_post_or_pre=norm_post_or_pre)
        
    def forward(self, X, mask=None):
        X = self.SAB(X, mask=mask, attn_bias=None)
        X = self.FFB(X)
        return X
    
class TransformerEncoder(nn.Module):
    '''
    Stack of Set Encoder Layers
    '''
    def __init__(self, n_layers, **kwargs):
        super().__init__()
        layer = TransformerEncoderLayer(**kwargs)
        self.last_layer_norm = nn.LayerNorm(kwargs['node_dim']) if (kwargs.get('norm_post_or_pre', 'pre') == 'pre') else nn.Identity()
        self.layers = _get_clones(layer, n_layers)
        reset_parameters(self)

    def forward(self, X, X_pos=None,  mask=None):
        for layer in self.layers:
            X = with_pos_embed(X,X_pos)
            X = layer(X, mask=mask)
        X = self.last_layer_norm(X)
        return X
    
class EvoformerEncoderLayer(nn.Module):
    '''
    Graph Encoder Layer, use to update X (shape: n_nodes, node_dim) and E (shape: n_nodes, n_nodes, edge_dim)
    '''
    def __init__(self,
                 node_dim,
                 edge_dim, 
                 node_hidden_dim,
                 edge_hidden_dim,
                 n_heads, 
                 dropout_attn = 0.0,
                 dropout_skip=0.0, 
                 dropout_mlp=0.0,
                 norm_post_or_pre='pre',
                 triangular_multiplication_mode='symmetric',
                 triangular_attention_mode='none'): 
        
        super().__init__()
        
        assert triangular_multiplication_mode in ['symmetric', 'col', 'row', 'none']
        assert triangular_attention_mode in ['symmetric', 'col', 'row', 'none']
        self.use_triangular_multiplication = triangular_multiplication_mode != 'none'
        self.use_triangular_attention = triangular_attention_mode != 'none'
        self.linear_pair_bias = nn.Linear(edge_dim, n_heads)
        self.SAB = SAB(model_dim=node_dim, n_heads=n_heads, dropout_attn=dropout_attn, dropout_skip=dropout_skip, norm_post_or_pre=norm_post_or_pre)
        self.FFB_nodes = FFB(in_dim=node_dim,out_dim=node_dim,hidden_dim=node_hidden_dim,dropout_mlp=dropout_mlp,dropout_skip=dropout_skip,norm_post_or_pre=norm_post_or_pre)
        self.OPB = OPB(node_dim, edge_dim, hidden_dim=32, dropout_skip=dropout_skip,norm_post_or_pre=norm_post_or_pre)
        if self.use_triangular_multiplication:
            self.TMB = TMB(model_dim=edge_dim,hidden_dim=edge_dim//2,dropout_skip=dropout_skip,norm_post_or_pre=norm_post_or_pre, mode=triangular_multiplication_mode)
        if self.use_triangular_attention:
            self.TAB = TAB(model_dim=edge_dim,n_heads=n_heads,dropout_attn=dropout_attn,dropout_skip=dropout_skip,norm_post_or_pre=norm_post_or_pre, mode=triangular_attention_mode)
        self.FFB_edges = FFB(in_dim=edge_dim,out_dim=edge_dim,hidden_dim=edge_hidden_dim,dropout_mlp=dropout_mlp,dropout_skip=dropout_skip,norm_post_or_pre=norm_post_or_pre)
            
    def forward(self, X, E, mask_nodes=None, mask_edges=None):
        attn_bias = self.linear_pair_bias(E).permute(0,3,1,2) 
        X = self.SAB(X, mask=mask_nodes, attn_bias=attn_bias)
        X = self.FFB_nodes(X)
        E = self.OPB(X, X, E)
        if self.use_triangular_multiplication:
            E = self.TMB(E, mask_edges=mask_edges)
        if self.use_triangular_attention:
            E = self.TAB(E, mask_edges=mask_edges)
        E = self.FFB_edges(E)
        return X, E

class EvoformerEncoder(nn.Module):
    '''
    Stack of Graph Encoder Layers
    '''
    def __init__(self, n_layers, **kwargs):
        
        super().__init__()
        
        layer = EvoformerEncoderLayer(**kwargs)
        self.last_layer_norm_X = nn.LayerNorm(kwargs['node_dim']) if (kwargs.get('norm_post_or_pre', 'pre') == 'pre') else nn.Identity()
        self.last_layer_norm_E = nn.LayerNorm(kwargs['edge_dim']) if (kwargs.get('norm_post_or_pre', 'pre') == 'pre') else nn.Identity()
        self.layers = _get_clones(layer, n_layers)
        reset_parameters(self)

    def forward(self, X, E, X_pos=None, E_pos=None, mask_nodes=None, mask_edges=None):
        for layer in self.layers:
            X = with_pos_embed(X,X_pos)
            E = with_pos_embed(E,E_pos)
            X,E = layer(X, E, mask_nodes=mask_nodes, mask_edges=mask_edges)
        X = self.last_layer_norm_X(X)
        E = self.last_layer_norm_E(E)   
        return X, E
        
class TransformerDecoderLayer(nn.Module):
    '''
    Set Decoder Layer, use to decode X_trgt (shape: n_nodes_trgt, src_dim) from a X_src (shape: n_nodes_src, trgt_dim)
    '''
    def __init__(self,
                 src_dim,
                 trgt_dim, 
                 trgt_hidden_dim,
                 trgt_size,
                 n_heads, 
                 dropout_attn = 0.0,
                 dropout_skip=0.0, 
                 dropout_mlp=0.0,
                 norm_post_or_pre='pre'):
        
        super().__init__()
        
        self.queries = nn.Parameter(torch.rand(1,trgt_size, trgt_dim))
        self.SAB = SAB(model_dim=trgt_dim, n_heads=n_heads, dropout_attn=dropout_attn, dropout_skip=dropout_skip, norm_post_or_pre=norm_post_or_pre)
        self.CAB = CAB(src_dim=src_dim, trgt_dim=trgt_dim, n_heads=n_heads, dropout_attn=dropout_attn, dropout_skip=dropout_skip, norm_post_or_pre=norm_post_or_pre)
        self.FFB = FFB(in_dim=trgt_dim,out_dim=trgt_dim,hidden_dim=trgt_hidden_dim,dropout_mlp=dropout_mlp,dropout_skip=dropout_skip,norm_post_or_pre=norm_post_or_pre)
        
    def forward(self, X_src, mask_src, X_trgt, is_first_layer = False):
        X_trgt = with_pos_embed(X_trgt, self.queries)                      # Usally we add trgt queries only in the first layer. But this is most expressive.
        if not is_first_layer:
            X_trgt = self.SAB(X_trgt, mask=None, attn_bias=None)
        else:
            X_trgt = X_trgt + 0*sum(p.sum() for p in self.SAB.parameters()) # Bypass this layer but still get gradients (zeros). This is a trick for compatibility with DDP.
        X_trgt = self.CAB(X_src, X_trgt, mask_src=mask_src)
        X_trgt = self.FFB(X_trgt)
        return  X_trgt
        
class TransformerDecoder(nn.Module):
    '''
    Stack of Set Decoder Layers
    '''
    
    def __init__(self, n_layers, **kwargs):
        super().__init__()
        layer = TransformerDecoderLayer(**kwargs)
        self.last_layer_norm = nn.LayerNorm(kwargs['trgt_dim']) if (kwargs.get('norm_post_or_pre', 'pre') == 'pre') else nn.Identity()
        self.layers = _get_clones(layer, n_layers)
        self.trgt_size = kwargs['trgt_size']
        self.trgt_dim = kwargs['trgt_dim']
        reset_parameters(self)
        
    def forward(self, X_src, mask_src = None):
        X_trgt = torch.zeros(X_src.shape[0], self.trgt_size, self.trgt_dim, device=X_src.device, dtype=X_src.dtype)
        first_layer = True
        for layer in self.layers:
            X_trgt = layer(X_src, mask_src, X_trgt, is_first_layer=first_layer)
            first_layer = False
        X_trgt = self.last_layer_norm(X_trgt)
        return X_trgt
        
class EvoformerDecoderLayer(nn.Module):
    '''
    Graph Decoder Layer, use to decode X_trgt (shape: n_nodes_trgt, src_dim) E_trgt and (shape: n_nodes_trgt, n_nodes_trgt, trgt_node_dim)  from a X_src (shape: n_nodes_src, trgt_edge_dim)
    '''
    
    def __init__(self,
                 src_dim, 
                 trgt_node_dim,
                 trgt_edge_dim,
                 trgt_node_hidden_dim,
                 trgt_edge_hidden_dim,
                 trgt_size,
                 n_heads, 
                 dropout_attn = 0.0,
                 dropout_skip=0.0, 
                 dropout_mlp=0.0,
                 norm_post_or_pre='pre',
                 triangular_multiplication_mode='symmetric',
                 triangular_attention_mode='none'): 
        
        super().__init__()
        
        assert triangular_multiplication_mode in ['symmetric', 'col', 'row', 'none']
        assert triangular_attention_mode in ['symmetric', 'col', 'row', 'none']
        self.use_triangular_multiplication = triangular_multiplication_mode != 'none'
        self.use_triangular_attention = triangular_attention_mode != 'none'
        self.queries_nodes = nn.Parameter(torch.rand(1,trgt_size, trgt_node_dim))
        self.queries_edges = nn.Parameter(torch.rand(1,trgt_size,trgt_size, trgt_edge_dim))
        self.linear_pair_bias = nn.Linear(trgt_edge_dim, n_heads)
        self.SAB = SAB(model_dim=trgt_node_dim, n_heads=n_heads, dropout_attn=dropout_attn, dropout_skip=dropout_skip, norm_post_or_pre=norm_post_or_pre)
        self.CAB_nodes = CAB(src_dim=src_dim, trgt_dim=trgt_node_dim, n_heads=n_heads, dropout_attn=dropout_attn, dropout_skip=dropout_skip, norm_post_or_pre=norm_post_or_pre)
        self.FFB_nodes = FFB(in_dim=trgt_node_dim,out_dim=trgt_node_dim,hidden_dim=trgt_node_hidden_dim,dropout_mlp=dropout_mlp,dropout_skip=dropout_skip,norm_post_or_pre=norm_post_or_pre)
        self.OPB = OPB(node_dim=trgt_node_dim,edge_dim=trgt_edge_dim,hidden_dim=32,dropout_skip=dropout_skip,norm_post_or_pre=norm_post_or_pre)
        if self.use_triangular_multiplication:
            self.TMB = TMB(model_dim=trgt_edge_dim,hidden_dim=trgt_edge_dim//2,dropout_skip=dropout_skip,norm_post_or_pre=norm_post_or_pre, mode=triangular_multiplication_mode)
        if self.use_triangular_attention:
            self.TAB = TAB(model_dim=trgt_edge_dim,n_heads=n_heads,dropout_attn=dropout_attn,dropout_skip=dropout_skip,norm_post_or_pre=norm_post_or_pre, mode=triangular_attention_mode)
        self.CAB_edges = CAB(src_dim=src_dim, trgt_dim=trgt_edge_dim, n_heads=n_heads, dropout_attn=dropout_attn, dropout_skip=dropout_skip, norm_post_or_pre=norm_post_or_pre)
        self.FFB_edges = FFB(in_dim=trgt_edge_dim,out_dim=trgt_edge_dim,hidden_dim=trgt_edge_hidden_dim,dropout_mlp=dropout_mlp,dropout_skip=dropout_skip,norm_post_or_pre=norm_post_or_pre)

    def forward(self, X_src, mask_src, X_trgt, E_trgt, is_first_layer = False):
        X_trgt = with_pos_embed(X_trgt, self.queries_nodes)   
        E_trgt = with_pos_embed(E_trgt, self.queries_edges)
        if not is_first_layer:
            attn_bias = self.linear_pair_bias(E_trgt).permute(0,3,1,2)
            X_trgt = self.SAB(X_trgt, mask=None, attn_bias=attn_bias)
        else:
            # Bypass this layer but still get gradients (zeros). This is a trick for compatibility with DDP.
            X_trgt = X_trgt + 0*sum(p.sum() for p in self.linear_pair_bias.parameters())
            X_trgt = X_trgt + 0*sum(p.sum() for p in self.SAB.parameters())
        X_trgt = self.CAB_nodes(X_src, X_trgt, mask_src=mask_src)
        X_trgt = self.FFB_nodes(X_trgt)
        E_trgt = self.OPB(X_trgt, X_trgt, E_trgt)
        if self.use_triangular_multiplication:
            E_trgt = self.TMB(E_trgt, mask_edges=None)
        if self.use_triangular_attention:
            E_trgt = self.TAB(E_trgt, mask_edges=None)
        E_trgt = E_trgt.flatten(1,2)
        E_trgt = self.CAB_edges(X_src, E_trgt, mask_src=mask_src)
        E_trgt = E_trgt.view(E_trgt.shape[0], X_trgt.shape[1], X_trgt.shape[1], -1)
        E_trgt = self.FFB_edges(E_trgt)
        return X_trgt, E_trgt
 
class EvoformerDecoder(nn.Module):
    '''
    Stack of EvoformerDecoderLayers
    '''
    def __init__(self, n_layers, **kwargs):
        super().__init__()
        
        layer = EvoformerDecoderLayer(**kwargs)
        self.last_layer_norm_X = nn.LayerNorm(kwargs['trgt_node_dim']) if (kwargs.get('norm_post_or_pre', 'pre') == 'pre') else nn.Identity()
        self.last_layer_norm_E = nn.LayerNorm(kwargs['trgt_edge_dim']) if (kwargs.get('norm_post_or_pre', 'pre') == 'pre') else nn.Identity()
        self.layers = _get_clones(layer, n_layers)
        self.trgt_size = kwargs['trgt_size']
        self.trgt_node_dim = kwargs['trgt_node_dim']
        self.trgt_edge_dim = kwargs['trgt_edge_dim']
        reset_parameters(self)
    
    def forward(self, X_src, mask_src=None):
        X_trgt = torch.zeros(X_src.shape[0], self.trgt_size, self.trgt_node_dim, device=X_src.device, dtype=X_src.dtype)
        E_trgt = torch.zeros(X_src.shape[0], self.trgt_size, self.trgt_size, self.trgt_edge_dim, device=X_src.device, dtype=X_src.dtype)
        first_layer = True
        for layer in self.layers:
            X_trgt, E_trgt = layer(X_src, mask_src, X_trgt, E_trgt, is_first_layer=first_layer)
            first_layer = False
        X_trgt = self.last_layer_norm_X(X_trgt)
        E_trgt = self.last_layer_norm_E(E_trgt)
        return X_trgt, E_trgt
    
class EvoformerSetMatchingLayer(nn.Module):
    '''
    Evoformer Set Matching Layer, use to predict pairs embeddings (shape: n_nodes_1, n_nodes_2, edge_dim) between X_1 (shape: n_nodes_1, node_dim) and X_2 (shape: n_nodes_2, node_dim)
    '''
    
    def __init__(self,
                 node_dim, 
                 edge_dim,
                 node_hidden_dim,
                 edge_hidden_dim,
                 n_heads, 
                 dropout_attn = 0.0,
                 dropout_skip=0.0, 
                 dropout_mlp=0.0,
                 norm_post_or_pre='pre',
                 triangular_multiplication_mode='symmetric',
                 triangular_attention_mode='none',
                 symmetric=False): 
        
        super().__init__()
        
        assert triangular_multiplication_mode in ['symmetric', 'col', 'row', 'none']
        assert triangular_attention_mode in ['symmetric', 'col', 'row', 'none']
        if symmetric:
            assert triangular_multiplication_mode in ['symmetric','none']
            assert triangular_attention_mode in ['symmetric', 'none']

        self.use_triangular_multiplication = triangular_multiplication_mode != 'none'
        self.use_triangular_attention = triangular_attention_mode != 'none'
        
        self.linear_pair_bias_1 = nn.Linear(edge_dim, n_heads)
        self.SAB_1 = SAB(model_dim=node_dim, n_heads=n_heads, dropout_attn=dropout_attn, dropout_skip=dropout_skip, norm_post_or_pre=norm_post_or_pre)
        self.CAB_1 = CAB(src_dim=node_dim, trgt_dim=node_dim, n_heads=n_heads, dropout_attn=dropout_attn, dropout_skip=dropout_skip, norm_post_or_pre=norm_post_or_pre)
        self.FFB_nodes_1 = FFB(in_dim=node_dim,out_dim=node_dim,hidden_dim=node_hidden_dim,dropout_mlp=dropout_mlp,dropout_skip=dropout_skip,norm_post_or_pre=norm_post_or_pre)
        
        self.linear_pair_bias_2 = deepcopy(self.linear_pair_bias_1) if not symmetric else self.linear_pair_bias_1
        self.CAB_2 = deepcopy(self.CAB_1) if not symmetric else self.CAB_1
        self.SAB_2 = deepcopy(self.SAB_1) if not symmetric else self.SAB_1
        self.FFB_nodes_2 = deepcopy(self.FFB_nodes_1) if not symmetric else self.FFB_nodes_1
        
        self.OPB = OPB(node_dim, edge_dim, hidden_dim=32, dropout_skip=dropout_skip,norm_post_or_pre=norm_post_or_pre,symmetric=symmetric)
        if self.use_triangular_multiplication:
            self.TMB = TMB(model_dim=edge_dim,hidden_dim=edge_dim//2,dropout_skip=dropout_skip,norm_post_or_pre=norm_post_or_pre, mode=triangular_multiplication_mode)
        if self.use_triangular_attention:
            self.TAB = TAB(model_dim=edge_dim,n_heads=n_heads,dropout_attn=dropout_attn,dropout_skip=dropout_skip,norm_post_or_pre=norm_post_or_pre, mode=triangular_attention_mode)
        self.FFB_edges = FFB(in_dim=edge_dim,out_dim=edge_dim,hidden_dim=edge_hidden_dim,dropout_mlp=dropout_mlp,dropout_skip=dropout_skip,norm_post_or_pre=norm_post_or_pre)

    def forward(self, X_1, X_2, C_12 = None, mask_1 = None, mask_2 = None, mask_12 = None):
        
        # Self Attention on X_1 and X_2
        X_1 = self.SAB_1(X_1, mask=mask_1)
        X_2 = self.SAB_2(X_2, mask=mask_2)
        
        # Cross Attention between X_1 and X_2
        attn_bias_1 = self.linear_pair_bias_1(C_12).permute(0,3,1,2) 
        X_1_old = X_1
        X_1 = self.CAB_1(src = X_2, trgt = X_1, mask_src=mask_2, attn_bias=attn_bias_1)
        
        attn_bias_2 = self.linear_pair_bias_2(C_12).permute(0,3,2,1)
        X_2 = self.CAB_2(src = X_1_old, trgt = X_2, mask_src=mask_1, attn_bias=attn_bias_2)
        
        # FFN on X_1 and X_2
        X_1 = self.FFB_nodes_1(X_1)
        X_2 = self.FFB_nodes_2(X_2)
        
        # Update C_12
        C_12 = self.OPB(X_1, X_2, C_12)
        if self.use_triangular_multiplication:
            C_12 = self.TMB(C_12, mask_12)
        if self.use_triangular_attention:
            C_12 = self.TAB(C_12, mask_12)
        C_12 = self.FFB_edges(C_12)
        return X_1, X_2, C_12
        
class EvoformerSetMatching(nn.Module):
    '''
    Stack of EvoformerSetMatchingLayer
    '''
    def __init__(self, n_layers, **kwargs):
        super().__init__()
        
        layer = EvoformerSetMatchingLayer(**kwargs)
        self.last_layer_norm_C12 = nn.LayerNorm(kwargs['edge_dim']) if (kwargs.get('norm_post_or_pre', 'pre') == 'pre') else nn.Identity()
        self.layers = _get_clones(layer, n_layers)
        self.edge_dim = kwargs['edge_dim']
        reset_parameters(self)
    
    def forward(self, X_1, X_2, mask_1 = None, mask_2 = None, mask_12 = None):
        C_12 = torch.zeros(X_1.shape[0],X_1.shape[1], X_2.shape[1], self.edge_dim, dtype=X_1.dtype, device=X_1.device)
        for layer in self.layers:
            X_1, X_2, C_12 = layer(X_1, X_2, C_12, mask_1, mask_2, mask_12)
        C_12 = self.last_layer_norm_C12(C_12)
        return C_12
          
if __name__ == '__main__':
    
    src_dim = 64
    trgt_node_dim = 64
    trgt_edge_dim = 64
    trgt_node_hidden_dim = 128
    trgt_edge_hidden_dim = 128
    trgt_size = 20
    n_heads = 8
    batchsize = 72
    input_size = 10
    
    model = TransformerDecoder(5,
                        src_dim=src_dim,
                        trgt_dim=trgt_node_dim, 
                        trgt_hidden_dim=trgt_node_hidden_dim,
                        trgt_size=trgt_size,
                        n_heads=n_heads, 
                        dropout_attn = 0.0,
                        dropout_skip=0.0, 
                        dropout_mlp=0.0,
                        norm_post_or_pre='pre'
                        )
    
    layer = TransformerDecoderLayer(src_dim,
                            trgt_node_dim, 
                            trgt_node_hidden_dim,
                            trgt_size,
                            n_heads, 
                            dropout_attn = 0.0,
                            dropout_skip=0.0, 
                            dropout_mlp=0.0,
                            norm_post_or_pre='pre'
                            )
    i = 4
    
    X_src = torch.rand(batchsize, input_size, src_dim, requires_grad=True)
    mask_src = torch.zeros(batchsize, input_size, dtype=torch.bool)
    mask_src[:,i] = 1
    X_trgt = torch.zeros(batchsize, trgt_size, trgt_node_dim)
    
    '''
    out = layer(X_src, mask_src, X_trgt)
    out[0,0,:].sum().backward()
    information = (X_src.grad[0,:,0] != 0)
    print(information)
    '''
    model.eval()
    out = model(X_src, mask_src)
    out[0,0,:].sum().backward()
    print(out[0,0,:].sum())
    information = (X_src.grad[0,:,0] != 0)
    print(information)