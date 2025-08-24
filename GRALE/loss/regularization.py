import torch
from torch.special import entr

class MarginalLoss():
    '''
    Compute a loss on the marginals of T 
    Promotes T.sum(1) = 1, T.sum(2) = 1
    '''
    
    def marginal_loss(self, marginal):
        '''
        Compute the loss on a marginal
        '''
        raise NotImplementedError()
    
    def forward(self, T):
        marg1 = T.sum(1)
        marg2 = T.sum(2)
        return self.marginal_loss(marg1) + self.marginal_loss(marg2)
    
class MarginalKL(MarginalLoss):
    def marginal_loss(self, marginal):
        loss = (-torch.log(marginal) + marginal - 1).sum(-1)
        return loss
    
class EntropyCols():
    '''
    Return entropy of the cols 
    - \sum_j \sum_i T_ij log T_ij
    Masked version ignores masked cols
    - \sum_j h_j \sum_i T_ij log T_ij where m = sum h_j 
    Promotes a sparse transport plan T 
    (as close as possible to a permutation matrix)
    '''
    def __call__(self, T, weight_2 = None):
        H = entr(T)
        H_cols = H.sum(1)
        if weight_2 is not None:
            H_cols = H_cols*weight_2
        H_cols = H_cols.mean(-1)
        return H_cols
