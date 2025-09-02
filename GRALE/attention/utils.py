from torch import Tensor
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional
from torch.utils.checkpoint import checkpoint
import torch.jit as jit

class MLP(nn.Module):
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
    
@torch.jit.script
def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, attn_bias: Optional[Tensor] = None, dropout: float = 0.0, scale: Optional[float] = None):
    '''
    query, key, value of shape (B, H, M, d)
    attn_bias of shape (B, H, M, M)
    '''
    if scale is None:
        scale = 1.0 / query.shape[-1] ** 0.5
    query = query * scale
    attn = torch.einsum('bhid,bhjd->bhij', query, key)
    if attn_bias is not None:
        attn = attn + attn_bias
    attn = attn.softmax(-1)
    attn = F.dropout(attn, dropout) if dropout > 0 else attn
    out = torch.einsum('bhij,bhjd->bhid', attn, value)
    return out

class MultiHeadAttention(nn.Module):
    def __init__(self,dim_Q, dim_K, dim_V, n_heads, dropout=0.0):
        super().__init__()

        self.dropout = dropout
        self.n_heads = n_heads
        self.dim_Q = dim_Q
        self.head_dim = dim_Q // self.n_heads
        assert dim_Q % self.n_heads == 0, "Embedding dim is not divisible by nheads"
        
        self.fc_Q = nn.Linear(dim_Q, dim_Q)
        self.fc_K = nn.Linear(dim_K, dim_Q)
        self.fc_V = nn.Linear(dim_V, dim_Q)
        self.fc_O = nn.Linear(dim_Q, dim_Q)

        self.dropout = dropout
        
    def forward(self, Q, K, V, attn_bias=None, mask=None):
        '''
        Q of shape (B, N, d)
        K, V of shape (B, M, d)
        attn_bias of shape (B, H, N, M) or (B, N, M)
        mask of shape (B, M)
        '''
        
        batchsize, N, dim_Q = Q.shape
        _, M, _ = K.shape
        
        # Add head dim to mask and attn_bias
        if mask is not None:
            mask = mask.unsqueeze(1)
        if attn_bias is not None:
            attn_bias = attn_bias.unsqueeze(1) if attn_bias.dim() == 3 else attn_bias
        
        # Put mask into attn_bias
        if mask is not None:
            attn_bias = torch.zeros(batchsize,1,N,M,device=Q.device,dtype=Q.dtype) if attn_bias is None else attn_bias
            attn_bias = attn_bias.masked_fill(mask.unsqueeze(-2), -float('inf')) if mask is not None else attn_bias
        
        Q = self.fc_Q(Q)
        K = self.fc_K(K)
        V = self.fc_V(V)

        Q = Q.view(batchsize, N, self.n_heads, self.head_dim).permute(0,2,1,3)    
        K = K.view(batchsize, M, self.n_heads, self.head_dim).permute(0,2,1,3)   
        V = V.view(batchsize, M, self.n_heads, self.head_dim).permute(0,2,1,3) 
    
        out = scaled_dot_product_attention(Q, K, V, attn_bias, dropout = self.dropout if self.training else 0.0)
        out = out.permute(0,2,1,3).reshape(batchsize, N, self.dim_Q)
        
        out = self.fc_O(out)
        
        return out
    
def mask_nodes_to_edges(mask_nodes: Tensor):
    # mask_nodes: b x m
    # mask_edges: b x m x m
    mask_edges = mask_nodes.unsqueeze(-1) | mask_nodes.unsqueeze(-2)
    return mask_edges

@torch.jit.script
def triangular_self_attention_row(query: Tensor, key: Tensor, value: Tensor, bias_col: Optional[Tensor] = None, mask_edges: Optional[Tensor] = None, dropout: float = 0.0, scale: Optional[float] = None):
        # query, key, value:  b x h x n x m x d
        # ij can attend to all ik
        # trgl_biais: b x h x m x m is the column bias i.e. attn_ijk = < Q_ij, K_ik > is biased by biais_jk
        # mask_edges: b x n x m is the mask for edges
        
        if scale is None:
            scale = 1.0 / query.shape[-1] ** 0.5
        query = query * scale
        attn = torch.einsum('bhijd,bhikd->bhijk', query, key) # attn_ijk = attention from (i,j) to (i,k)
        if bias_col is not None:
            attn = attn + bias_col.unsqueeze(-3) # biais for ijk = trgl_biais_jk
        if mask_edges is not None:
            attn = attn.masked_fill(mask_edges.unsqueeze(-2), -float('inf')) # set attn_ijk to -inf if mask_ik = 0
        attn = attn.softmax(-1)
        attn = F.dropout(attn, dropout) if dropout > 0 else attn
        output = torch.einsum('bhijk,bhikd->bhijd', attn, value) 
        return output

@torch.jit.script
def triangular_self_attention_col(query: Tensor, key: Tensor, value: Tensor, bias_row: Optional[Tensor] = None, mask_edges: Optional[Tensor] = None, dropout: float = 0.0, scale: Optional[float] = None):
        # query, key, value:  b x h x n x m x d
        # ij can attend to all kj
        # bias_row: b x h x n x n is the row bias i.e. attn ijk = < Q_ij, K_kj > is biased by biais_ki
        # mask_edges: b x n x m is the mask for edges
        
        if scale is None:
            scale = 1.0 / query.shape[-1] ** 0.5
        query = query * scale
        attn = torch.einsum('bhijd,bhkjd->bhijk', query, key) # attn_ijk = attention from (i,j) to (k,j)
        if bias_row is not None:
            triag_biais = bias_row.transpose(-2,-1)
            attn = attn + triag_biais.unsqueeze(-2) # biais for ijk = trgl_biais_ki
        if mask_edges is not None:
            mask_edges = mask_edges.transpose(-2,-1)
            attn = attn.masked_fill(mask_edges.unsqueeze(-3), -float('inf')) # set attn_ijk to -inf if mask_kj = 0
        attn = attn.softmax(-1)
        attn = F.dropout(attn, dropout) if dropout > 0 else attn
        output = torch.einsum('bhijk,bhkjd->bhijd', attn, value) 
        return output
    
class MultiHeadSelfAttentionEdges(nn.Module):
    def __init__(self,model_dim, n_heads, dropout=0.0, mode = 'symmetric'):
        super().__init__()

        self.dropout = dropout
        self.n_heads = n_heads
        self.dim_Q = model_dim
        self.head_dim = model_dim // n_heads  
        
        assert model_dim % self.n_heads == 0, "Embedding dim is not divisible by nheads"
        
        self.fc_Q = nn.Linear(model_dim, model_dim, bias=False)
        self.fc_K = nn.Linear(model_dim, model_dim, bias=False)
        self.fc_V = nn.Linear(model_dim, model_dim, bias=False)
        self.fc_O = nn.Linear(model_dim, model_dim)
        self.fc_G = nn.Linear(model_dim, model_dim)
        self.fc_B = nn.Linear(model_dim, n_heads, bias=False)

        self.dropout = dropout
        
        assert mode in ['symmetric', 'row', 'col'], "mode should be either 'symmetric', 'row' or 'col'"
        self.mode = mode
        
    def forward(self, E, mask_edges = None):
        '''
        Q, K, V of shape (B, M, N, d)
        mask_edges of shape (B, M, N)
        '''
        batchsize, M, N, model_dim = E.size()
        
        Q = self.fc_Q(E)
        K = self.fc_K(E)
        V = self.fc_V(E)
        G = self.fc_G(E)
        B = self.fc_B(E) if M == N else None # Biais for the triangular attention is not well defined if M != N
        
        Q = Q.view(batchsize, M, N, self.n_heads, self.head_dim).permute(0,3,1,2,4)
        K = K.view(batchsize, M, N, self.n_heads, self.head_dim).permute(0,3,1,2,4)
        V = V.view(batchsize, M, N, self.n_heads, self.head_dim).permute(0,3,1,2,4)
        B = B.view(batchsize, M, N, self.n_heads).permute(0,3,1,2) if B is not None else None
        
        if mask_edges is not None:
            mask_edges = mask_edges.unsqueeze(1).repeat(1,self.n_heads,1,1)

        if self.mode == 'row':
            out = triangular_self_attention_row(Q, K, V, bias_col=B, mask_edges=mask_edges, dropout = self.dropout if self.training else 0.0)
        elif self.mode == 'col':
            out = triangular_self_attention_col(Q, K, V, bias_row=B, mask_edges=mask_edges, dropout = self.dropout if self.training else 0.0)
        else:
            out = triangular_self_attention_row(Q, K, V, bias_col=B, mask_edges=mask_edges, dropout = self.dropout if self.training else 0.0) + triangular_self_attention_col(Q, K, V, bias_row=B, mask_edges=mask_edges, dropout = self.dropout if self.training else 0.0)
        out = out.permute(0,2,3,1,4).reshape(batchsize, M, N, model_dim)
        out = out*F.sigmoid(G)
        out = self.fc_O(out)
        
        return out

@torch.jit.script
def outer_product_multiplication(a: Tensor, b: Tensor, W: Tensor, bias: Tensor) -> Tensor:
    '''
    a of shape (B, N, d)
    b of shape (B, M, d)
    W of shape (d, d, d2) # Weight of the linear layer
    b of shape (d2,) # Bias of the linear layer
    out_iju = sum_k sum_l a_ik b_jl W_klu + B_u
    if symmetric, then W_klu = W_lku
    '''
    out = torch.einsum("bik,bjl,klu->biju", a, b, W) + bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    return out
   
class OuterProduct(nn.Module):
    def __init__(self, d_atom, d_pair, d_hid=32, memory_efficient=False,symmetric=False):
        super(OuterProduct, self).__init__()

        self.d_atom = d_atom
        self.d_pair = d_pair
        self.d_hid = d_hid

        self.linear_a = nn.Linear(d_atom, d_hid)
        self.linear_b = nn.Linear(d_atom, d_hid) if not symmetric else self.linear_a
        self.outer_product_weight = nn.Parameter(torch.randn(d_hid, d_hid, d_pair))
        self.outer_product_bias = nn.Parameter(torch.randn(d_pair))
        self.act = nn.GELU()
        self._memory_efficient = memory_efficient
        self.symmetric = symmetric

    def apply_memory_efficient(self):
        self._memory_efficient = True

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        a = self.linear_a(x1)
        b = self.linear_b(x2)
        if self.symmetric:
            W = self.outer_product_weight + self.outer_product_weight.transpose(0,1)
        else:
            W = self.outer_product_weight
        if self._memory_efficient and torch.is_grad_enabled():
            z = checkpoint(outer_product_multiplication, a, b, W, self.outer_product_bias)
        else:
            z = outer_product_multiplication(a, b, W, self.outer_product_bias)
        return z

@torch.jit.script
def multiplication_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bkid,bkjd->bijd", a, b)

@torch.jit.script
def multiplication_cols(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bikd,bjkd->bijd", b, a)
    
class TriangleMultiplication(nn.Module):
    def __init__(self, d_pair, d_hid, mode = 'symmetric'):
        super(TriangleMultiplication, self).__init__()

        self.linear_ab_p = nn.Linear(d_pair, d_hid * 2)
        self.linear_ab_g = nn.Linear(d_pair, d_hid * 2)
        self.linear_g = nn.Linear(d_pair, d_pair)
        self.linear_out = nn.Linear(d_hid, d_pair)
        self.mode = mode
        assert mode in ['symmetric', 'row', 'col'], "mode should be either 'symmetric', 'row' or 'col'"
        
        self.norm = nn.LayerNorm(d_hid)

    def forward(self, E: torch.Tensor, mask_edges: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        g = torch.sigmoid(self.linear_g(E))
        if mask_edges is None:
            ab = self.linear_ab_p(E) * torch.sigmoid(self.linear_ab_g(E))
        else:
            mask_edges = mask_edges.unsqueeze(-1)
            ab = self.linear_ab_p(E) * (~mask_edges) * torch.sigmoid(self.linear_ab_g(E))
        a, b = torch.chunk(ab, 2, dim=-1)
        
        del ab
        
        if self.mode == 'row':
            E = multiplication_rows(a, b) 
        elif self.mode == 'col':
            E = multiplication_cols(a, b)
        else:
            E = multiplication_rows(a, b) + multiplication_cols(a, b)        
        E = g*self.linear_out(self.norm(E))
    
        return E


if __name__=='__main__':
    batchsize = 16
    n_heads = 8
    n_nodes = 5    
    model_dim = 32  
    
    # information flow 
    
    '''
    # Test: information flow triangular_self_attention1
    i = 1
    j = 3
    masked = 4
    mask_nodes = torch.tensor([0,0,0,0,0], dtype=torch.bool)
    mask_nodes[masked] = True
    mask_edges = mask_nodes_to_edges(mask_nodes)
    mask_edges = mask_edges.unsqueeze(0).repeat(batchsize,1,1)
    mask_edges = mask_edges.unsqueeze(1).repeat(1,n_heads,1,1)
    
    Q = torch.rand(batchsize, n_heads, n_nodes, n_nodes, model_dim, requires_grad=True)
    K = torch.rand(batchsize, n_heads, n_nodes, n_nodes, model_dim, requires_grad=True)
    V = torch.rand(batchsize, n_heads, n_nodes, n_nodes, model_dim, requires_grad=True)
    out = triangular_self_attention1(Q, K, V, trgl_biais=None, mask_edges = mask_edges)
    
    out[0,0,i,j,:].sum().backward()
    print(Q.grad[0,0,:,:,0]) # Should be zero everywhere except for i,j     
    print(K.grad[0,0,:,:,0]) # Should be zero everywhere except for ith row  (cols for v2)
    print(V.grad[0,0,:,:,0]) # Should be zero everywhere except for ith row  (cols for v2)
    '''
    '''
    # Test: information flow MultiHeadSelfAttentionEdges
    masked = 4
    mask_nodes = torch.tensor([0,0,0,0,0], dtype=torch.bool)
    mask_nodes[masked] = True
    mask_edges = mask_nodes_to_edges(mask_nodes)
    mask_edges = mask_edges.unsqueeze(0).repeat(batchsize,1,1)
    E = torch.rand(batchsize, n_nodes, n_nodes, model_dim, requires_grad=True)
    model = MultiHeadSelfAttentionEdges(model_dim, n_heads, start_or_end_node='end')
    out = model(E, mask_edges=mask_edges)
    out[0,i,j,:].sum().backward()
    information = (E.grad[0,:,:,0] != 0)
    print(information)'''
    
    # Test: information flow MultiHeadAttention
 
    i = 2
    masked1 = 3
    masked2 = 4
    mask_nodes = torch.tensor([0,0,0,0,0], dtype=torch.bool)
    mask_nodes[masked1] = True
    mask_nodes = mask_nodes.unsqueeze(0).repeat(batchsize,1)
    attn_bias = torch.zeros(batchsize, n_nodes, n_nodes, dtype=torch.float)
    attn_bias[:,i,masked2] = -float('inf')
    X = torch.rand(batchsize, n_nodes, model_dim, requires_grad=True)
    model = MultiHeadAttention(model_dim, model_dim, model_dim, n_heads)
    out = model(X, X, X, mask = mask_nodes,attn_bias=attn_bias)
    out[0,i,:].sum().backward()
    information = (X.grad[0,:,0] != 0)
    print(information)
    
    print(type(scaled_dot_product_attention))
