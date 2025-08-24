import torch
from math import sqrt

def bmv(Matrices,Vectors):
    return torch.einsum('bij,bj->bi', Matrices, Vectors)

def bop(vectors1,vectors2):
    return torch.einsum('bi,bj->bij', vectors1, vectors2)

###### ------------------ Usefull functions to build cost matrices ------------------ ######

def squared_norm(X1,X2):
    dim = X1.shape[-1]
    return torch.sum((X1-X2)**2,dim=-1)/sqrt(dim)

def pairwise_squared_norm(X1,X2):
    dim = X1.shape[-1]
    norm1 = torch.sum(X1**2,dim=2)
    norm2 = torch.sum(X2**2,dim=2)
    dot = torch.bmm(X1,X2.permute(0,2,1))
    loss = norm1[:,:,None] + norm2[:,None,:] - 2*dot
    loss = loss/sqrt(dim)
    return loss

def pairwise_L1_norm(X1,X2):
    dim = X1.shape[-1]
    return torch.sum(torch.abs(X1[:,:,None]-X2[:,None,:]),dim=-1)/sqrt(dim)

def pairwise_BCE_loss(logits,targets):
    batchsize, M = logits.shape
    logits = logits.unsqueeze(-1).expand(-1,-1,M)
    targets = targets.unsqueeze(-2).expand(-1,M,-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits,targets,reduction='none')
    return loss

def pairwise_CE_loss(logits,targets):
    return -torch.log_softmax(logits,dim=-1)@torch.permute(targets,(0,2,1)) 

def softmax_to_one_hot(logits):
    classes = torch.argmax(logits,dim=-1,keepdim=True)
    one_hot = torch.zeros_like(logits).scatter_(-1, classes, 1)
    return one_hot 