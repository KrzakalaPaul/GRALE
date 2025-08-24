import torch
from torch.nn.functional import logsigmoid
from torch.special import entr as entr_torch
from .utils import *

class LinearLoss():
    '''
    Compute the loss L(T,C1,C2) = sum_ik T_ik L(F_i, F_k)
    For C1 of shape (B,N1,D), C2 of shape (B,N2,D2) and T of shape (B,N1,N2)
    
    It is also possible to add weights to the loss:
    L(T,C1,C2) = sum_ik T_ik L(F_i, F_k) w1_i w2_k
    '''
    
    def pairwise_loss(self, F1, F2):
        '''
        Build cost matrix C_ij = L(F1_i, F2_j)
        '''
        return ...
    
    def pointwise_loss(self, F1, F2):
        '''
        Compute pointwise loss C_i = L(F1_i, F2_i)
        '''
        return ...
    
    def forward(self, T, F1, F2, weight_1=None, weight_2=None):
        cost_matrix = self.pairwise_loss(F1,F2)
        if weight_1 is not None:
            cost_matrix = cost_matrix * weight_1[:,:,None]
        if weight_2 is not None:
            cost_matrix = cost_matrix * weight_2[:,None,:]
        return torch.sum(T*cost_matrix, dim=(1,2))
    
    def forward_aligned(self, F1, F2, weight = None):
        cost = self.pointwise_loss(F1,F2)
        cost = cost * weight if weight is not None else cost
        return torch.sum(cost, dim=1)
    
class LinearL2(LinearLoss):
    
    def pairwise_loss(self, F1, F2):
        '''
        Build cost matrix C_ij = L(F1_i, F2_j)
        '''
        return pairwise_squared_norm(F1,F2)
    
    def pointwise_loss(self, F1, F2):
        '''
        Compute pointwise loss C_i = L(F1_i, F2_i)
        '''
        return squared_norm(F1,F2)
    
class LinearBCE(LinearLoss):
    
    def pairwise_loss(self, logits, targets):
        return pairwise_BCE_loss(logits,targets)
    
    def pointwise_loss(self, logits, targets):
        return torch.nn.BCEWithLogitsLoss(reduction='none')(logits,targets) 
    
class LinearBinaryAccuracy(LinearLoss):
    
    def pairwise_loss(self, predictions, targets):
        preds = torch.where(predictions>0.5,1,0)
        cost = torch.where(preds[:,:,None]==targets[:,None,:],0,1)
        return cost
     
    def pointwise_loss(self, logits, targets):
        preds = torch.where(logits>0.5,1,0)
        cost = torch.where(preds == targets, 0 , 1)
        return cost
     
class LinearCE(LinearLoss):
    
    def pairwise_loss(self, logits, targets):
        return pairwise_CE_loss(logits,targets)
    
    def pointwise_loss(self, logits, targets):
        return -(torch.log_softmax(logits,dim=-1)*targets).sum(dim=-1)
    
class LinearAccuracy(LinearLoss):
    
    def pairwise_loss(self, predictions, targets):
        preds = softmax_to_one_hot(predictions)
        cost = 1 - torch.bmm(preds, targets.transpose(1,2))
        return cost
    
    def pointwise_loss(self, predictions, targets):
        preds = softmax_to_one_hot(predictions)
        cost = 1 - (preds*targets).sum(-1)
        return cost