from torch import Tensor
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional    
from .utils import *
from torch.utils.checkpoint import checkpoint

# Note: Blocks always have a skip connection + normalization (can be before or after)
    
class FFB(nn.Module):
    '''
    Feed Forward Block
    '''
    def __init__(self,in_dim,out_dim,hidden_dim=None,dropout_mlp=0.0,dropout_skip=0.0,norm_post_or_pre='pre'):
        '''
        If hidden_dim is None, No hidden layer is used. There will still be a non linearity.
        '''
        super().__init__()
        self.norm_post_or_pre = norm_post_or_pre
        self.norm = nn.LayerNorm(out_dim)
        self.mlp = MLP(in_dim,hidden_dim,out_dim,dropout_mlp)
        self.dropout = nn.Dropout(dropout_skip)  
        
    def forward(self, X):
        if self.norm_post_or_pre == 'pre':
            X2 = self.norm(X)
            X2 = self.mlp(X2)
            X = X + self.dropout(X2)
        else:
            X2 = self.mlp(X)
            X = X + self.dropout(X2)
            X = self.norm(X)
        return X 
    
class SAB(nn.Module):
    '''
    Self Attention Block
    '''
    def __init__(self, model_dim, n_heads, dropout_attn = 0.0,dropout_skip=0.0, norm_post_or_pre='pre'):
        
        super().__init__()

        self.MHA = MultiHeadAttention(model_dim, model_dim, model_dim, n_heads, dropout_attn)
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_skip)
        self.norm_post_or_pre = norm_post_or_pre
        
    def forward(self, X, mask=None, attn_bias=None):
        
        if self.norm_post_or_pre == 'pre':
            X2 = self.norm(X)
            X2 = self.MHA(X2,X2,X2, mask = mask, attn_bias = attn_bias)
            X = X + self.dropout(X2)
            
        if self.norm_post_or_pre == 'post':
            X2 = self.MHA(X,X,X, mask = mask, attn_bias = attn_bias)
            X = X + self.dropout(X2)
            X = self.norm(X)
            
        return X
    
class CAB(nn.Module):
    '''
    Cross Attention Block
    '''
    def __init__(self, src_dim, trgt_dim, n_heads, dropout_attn = 0.0,dropout_skip=0.0, norm_post_or_pre='pre'):
        
        super().__init__()

        self.MHA = MultiHeadAttention(trgt_dim, src_dim, src_dim, n_heads, dropout_attn)
        self.norm = nn.LayerNorm(trgt_dim)
        self.dropout = nn.Dropout(dropout_skip)
        self.norm_post_or_pre = norm_post_or_pre
        
    def forward(self, src, trgt, mask_src=None, attn_bias=None):
        
        if self.norm_post_or_pre == 'pre':
            trgt2 = self.norm(trgt)
            trgt2 = self.MHA(trgt2,src,src, mask = mask_src, attn_bias = attn_bias)
            trgt = trgt + self.dropout(trgt2)
            
        if self.norm_post_or_pre == 'post':
            trgt2 = self.MHA(trgt,src,src, mask = mask_src, attn_bias = attn_bias)
            trgt = trgt + self.dropout(trgt2)
            trgt = self.norm(trgt)

        return trgt

class OPB(nn.Module):
    '''
    Outer Product Block
    '''
    def __init__(self, node_dim, edge_dim, hidden_dim=32, dropout_skip=0.0, norm_post_or_pre='pre',symmetric=False):
        
        super().__init__()
        self.norm_post_or_pre = norm_post_or_pre
        self.OP = OuterProduct(node_dim, edge_dim, hidden_dim,symmetric=symmetric)
        self.norm = nn.LayerNorm(edge_dim)
        self.dropout = nn.Dropout(dropout_skip)
        
    def forward(self, X1, X2, E):
        
        if self.norm_post_or_pre == 'pre':
            E = self.norm(E)
            E2 = self.OP(X1, X2)
            E = E + self.dropout(E2)
            
        if self.norm_post_or_pre == 'post':
            E2 = self.OP(X1, X2)
            E = E + self.dropout(E2)
            E = self.norm(E)

        return E

class TAB(nn.Module):
    '''
    Triangular Attention Block
    '''
    def __init__(self, model_dim, n_heads, dropout_attn = 0.0, dropout_skip=0.0, norm_post_or_pre='pre', mode = 'symmetric'):
        
        super().__init__()
        self.norm_post_or_pre = norm_post_or_pre
        self.MultiHeadTriangularAttention = MultiHeadSelfAttentionEdges(model_dim, n_heads, dropout = dropout_attn, mode = mode)
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_skip)
        
    def forward(self, E, mask_edges):
        
        if self.norm_post_or_pre == 'pre':
            E2 = self.norm(E)
            E2 = self.MultiHeadTriangularAttention(E2, mask_edges = mask_edges)
            E = E + self.dropout(E2)
            
        if self.norm_post_or_pre == 'post':
            E2 = self.MultiHeadTriangularAttention(E, mask_edges = mask_edges)
            E = E + self.dropout(E2)
            E = self.norm(E)
            
        return E
    
class TMB(nn.Module):
    '''
    Triangular Multiplication Block
    '''
    
    def __init__(self, model_dim, hidden_dim, dropout_skip=0.0, norm_post_or_pre='pre', mode = 'symmetric'):
        
        super().__init__()
        self.norm_post_or_pre = norm_post_or_pre
        self.TriangleMultiplication = TriangleMultiplication(model_dim, hidden_dim,mode=mode)
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_skip)
        
    def forward(self, E, mask_edges):
        
        if self.norm_post_or_pre == 'pre':
            E2 = self.norm(E)
            E2 = self.TriangleMultiplication(E2, mask_edges = mask_edges)
            E = E + self.dropout(E2)
            
        if self.norm_post_or_pre == 'post':
            E2 = self.TriangleMultiplication(E, mask_edges = mask_edges)
            E = E + self.dropout(E2)
            E = self.norm(E)
            
        return E
    
if __name__ == '__main__':
    
    src_dim = 64
    trgt_dim = 64
    trgt_size = 20
    n_heads = 8
    batchsize = 72
    input_size = 10
    i = 4
    
    X_src = torch.rand(batchsize, input_size, src_dim, requires_grad=True)
    mask_src = torch.zeros(batchsize, input_size, dtype=torch.bool)
    mask_src[:,i] = 1
    X_trgt = torch.rand(batchsize, trgt_size, trgt_dim)
    
    CAB_block = CAB(src_dim, trgt_dim, n_heads)
    out = CAB_block(X_src, X_trgt, mask_src=mask_src)
    out[0,0,:].sum().backward()
    information = (X_src.grad[0,:,0] != 0)
    print(information)