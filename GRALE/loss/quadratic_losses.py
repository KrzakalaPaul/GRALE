import torch
from torch.nn.functional import logsigmoid
from torch.special import entr as entr_torch
from .utils import *

class QuadraticLoss():
    '''
    Compute the loss L(T,C1,C2) = sum_ijkl T_ik T_jl L(C1_ij, C2_kl)
    For C1 of shape (B,N1,N1,D), C2 of shape (B,N2,N2,D2) and T of shape (B,N1,N2)
    Assumes that L(a,b) = f1(a) + f2(b) - < h1(a), h2(b) >
    
    It is also possible to add weights to the loss:
    L(T,C1,C2) = sum_ijkl T_ik T_jl L(C1_ij, C2_kl) W1_ij W2_kl
    
    For instance, setting mask_self_loops = True will set:
         W1 = 1 - diag(1) 
         W2 = 1 - diag(1)
    Providing weight_1 and weight_2 of shape (B,N1) and (B,N2) will set:
        W1_ij = weight_1_i weight_1_j 
        W2_kl = weight_2_k weight_2_l
    '''
    
    def __init__(self, mask_self_loops = False):
        self.mask_self_loops = mask_self_loops
    
    def f1(self, C1):
        return ...
    
    def f2(self, C2):
        return ...
    
    def h1(self, C1):
        return ...

    def h2(self, C2):
        return ...
    
    def forward(self, T, C1, C2, weight_1=None, weight_2=None):
        L = self.tensor_product(T,self.f1(C1),self.f2(C2),self.h1(C1),self.h2(C2),weight_1,weight_2,self.mask_self_loops)
        return torch.sum(L*T, dim=(1,2))
    
    def forward_aligned(self, C1, C2, weight=None):
        L = self.f1(C1) + self.f2(C2) - torch.einsum('bijd,bijd->bij', self.h1(C1), self.h2(C2))
        if weight is not None:
            L = L * weight[:,:,None] * weight[:,None,:]
        if self.mask_self_loops:
            L = L * (1 - torch.eye(L.shape[1], device = L.device)).unsqueeze(0)
        return torch.sum(L, dim=(1,2))
    
    def tensor_product(self,T,f1,f2,h1,h2,weight_1=None,weight_2=None,mask_self_loops=False):
        '''
        Compute Tensor_ik = sum_jl T_jl L(C1_ij, C2_kl) W1_ij W2_kl
        where:
            L is the lost function that decomposes as L(a,b) = f1(a) + f2(b) - < h1(a), h2(b) >
            W1 and W2 are masks defined as:
                W = mask_self_loops * weight
                mask_self_loops = 1 - diag(1) if mask_self_loops = True else 1
                mask_nodes = mask * mask.t() if mask is not None else 1
        '''
        
        B, N1, N2 = T.shape
            
        w1 = weight_1 if weight_1 is not None else torch.ones(B,N1, device = T.device)
        w2 = weight_2 if weight_2 is not None else torch.ones(B,N2, device = T.device)
        
        # Initialize W1 with as few operations as possible (try to avoid multipliLation by 1)
        if mask_self_loops and weight_1 is not None:
            W1 = 1 - torch.eye(N1, device = T.device).unsqueeze(0).repeat(B,1,1)
            W1 = W1 * w1[:,None,:] * w1[:,:,None]
        elif mask_self_loops:
            W1 = 1 - torch.eye(N1, device = T.device).unsqueeze(0).repeat(B,1,1)
        elif weight_1 is not None:
            W1 = w1[:,None,:] * w1[:,:,None]
        
        # Initialize W2 with as few operations as possible (try to avoid multipliLation by 1)
        if mask_self_loops and weight_2 is not None:
            W2 = 1 - torch.eye(N2, device = T.device).unsqueeze(0).repeat(B,1,1)
            W2 = W2 * w2[:,None,:] * w2[:,:,None]
        elif mask_self_loops:
            W2 = 1 - torch.eye(N2, device = T.device).unsqueeze(0).repeat(B,1,1)
        elif weight_2 is not None:
            W2 = w2[:,None,:] * w2[:,:,None]

        # Compute U1 = f1 * W1 and V1 = h1 * W1
        U1 = f1
        V1 = h1
        if mask_self_loops or weight_1 is not None:
            U1 = U1 * W1
            V1 = V1 * W1.unsqueeze(-1)
            
        # Compute U2 = f2 * W2 and V2 = h2 * W2
        U2 = f2
        V2 = h2
        if mask_self_loops or weight_2 is not None:
            U2 = U2 * W2
            V2 = V2 * W2.unsqueeze(-1)
            
        # Compute La = U1@T@W2^T, this can be done faster if W2 = w2 w2^T
        if mask_self_loops: # No acceleration possible
            La = torch.bmm(U1, torch.bmm(T, W2.transpose(1,2)))
        else: # compute only matrix/vector products + an outer product
            La = bmv(T,w2)
            La = bmv(U1,La)
            La = bop(La,w2)
        
        # Compute Lb = W1@T@U2^T, this can be done faster if W1 = w1 w1^T
        if mask_self_loops:
            Lb = torch.bmm(W1, torch.bmm(T, U2.transpose(1,2)))
        else:
            Lb = bmv(T.transpose(1,2),w1)
            Lb = bmv(U2,Lb)
            Lb = bop(w1,Lb)

        # Compute Lc = sum_d (V1@T@V2^T)_d, no acceleration possible
        Lc = torch.einsum('bijd,bjl,bkld->bik', V1, T, V2)
        
        L = La + Lb - Lc
        
        return L
    
class QuadraticL2(QuadraticLoss):
    
    def f1(self, C1):
        if C1.ndim == 4:
            return torch.sum(C1**2,dim=-1)
        else:
            return C1**2

    def f2(self, C2):
        if C2.ndim == 4:
            return torch.sum(C2**2,dim=-1)
        else:
            return C2**2
    
    def h1(self, C1):
        if C1.ndim == 3:
            C1 = C1.unsqueeze(-1)
        return 2*C1

    def h2(self, C2):
        if C2.ndim == 3:
            C2 = C2.unsqueeze(-1)
        return C2
        
class QuadraticBCE(QuadraticLoss):
    '''
    L(a,b) = KL(sigmoid(a),b) + KL(1-sigmoid(a),1-b)
    Expect C1 to be given in logits (pre-sigmoid) and C2 to be given in probabilities
    '''
    
    def f1(self, logits):
        return -logsigmoid(logits)

    def f2(self, targets):                        
        return-entr_torch(targets)-entr_torch(1-targets)

    def h1(self, logits):
        return -logits.unsqueeze(-1)

    def h2(self, targets):
        return (1-targets).unsqueeze(-1)
    
class QuadraticCE(QuadraticLoss):
    '''
    L(a,b) = KL(softmax(a),b)
    Expect C1 to be given in logits (pre-softmax) and C2 to be given in probabilities
    '''
    
    def f1(self, logits):
        return torch.logsumexp(logits,dim=-1)
    
    def f2(self, targets):
        return -entr_torch(targets).sum(dim=-1)
        
    def h1(self, logits):
        return logits
    
    def h2(self, targets):
        return targets

class QuadraticAccuracy(QuadraticLoss):
    '''
    L(a,b) = 1[argmax(a)!=b] = 1 - <one_hot(a),b>
    Expect C1 to be a softmax and C2 to be a one hot encoded target
    '''
    
    def f1(self, logits):
        B, N, _, D = logits.shape
        return torch.ones((B,N,N), device=logits.device) 
    
    def f2(self, targets):
        B, N, _, D = targets.shape
        return torch.zeros((B,N,N), device=targets.device) 
        
    def h1(self, logits):
        return softmax_to_one_hot(logits)
    
    def h2(self, targets):
        return targets
    
class QuadraticBinaryAccuracy(QuadraticAccuracy):
    '''
    L(a,b) = 1[ (a>0.5) == b ]
    Can be computed by first defining A = [a, 1-a] and B = [b, 1-b] then
    L(a,b) = Accuracy(A,B)
    '''

    def transform(self, C1,C2):
        C1 = torch.stack([C1,1-C1], dim=-1)
        C2 = torch.stack([C2,1-C2], dim=-1)
        return C1, C2
    
    def forward(self, T, C1, C2, weight_1=None, weight_2=None):
        C1, C2 = self.transform(C1, C2)
        return super().forward(T, C1, C2, weight_1=weight_1, weight_2=weight_2)
        
    
    def forward_aligned(self, C1, C2, weight=None):
        C1, C2 = self.transform(C1, C2)
        return super().forward_aligned(C1, C2, weight = weight)
    
 
    
if __name__ == '__main__':
    from matching import batched_hungarian_projection, permutations_list_to_matrices, batched_sinkhorn_projection
    
    batchsize = 16
    N = 3
    symmetric = False
    mask_self_loops = True
    
    def get_A():
        A = torch.randn(batchsize,N,N, device='cuda')
        if symmetric:
            A = (A + A.transpose(1,2))/2
        A = torch.where(A>0,torch.tensor(1.,device='cuda'),torch.tensor(0.,device='cuda'))
        return A
    
    def get_P():
        T = torch.rand(batchsize,N,N, device='cuda')
        T, _ = batched_hungarian_projection(T)
        T = permutations_list_to_matrices(T)
        return T
    
    def get_T():
        T = torch.rand(batchsize,N,N, device='cuda')
        T, _ = batched_sinkhorn_projection(T)
        return T

    # Test 1: recover 0 error when comparing the same matrix up to permutation
    loss_fct = QuadraticL2(mask_self_loops = mask_self_loops)
    A_pred = get_A()
    T = get_P()
    A_trgt = torch.bmm(T.transpose(1,2),torch.bmm(A_pred,T))
    
    loss = loss_fct.forward(T,A_pred,A_trgt)    
    assert torch.allclose(loss,torch.tensor(0.,device='cuda'))

    loss = loss_fct.forward_aligned(A_pred,A_pred)
    assert torch.allclose(loss,torch.tensor(0.,device='cuda'))
    
    # Test 2: when T is a permutation forward and permute+forward_aligned should be the same
    loss_fct = QuadraticL2(mask_self_loops = mask_self_loops)
    A_pred = get_A()
    A_trgt = get_A()
    T = get_P()
    
    loss1 = loss_fct.forward(T,A_pred,A_trgt)    
    A_pred = torch.bmm(T.transpose(1,2),torch.bmm(A_pred,T))
    loss2 = loss_fct.forward_aligned(A_pred,A_trgt)
    assert torch.allclose(loss1,loss2)

    # Test 3: if T is a transport plan we only have an inequality
    loss_fct = QuadraticL2(mask_self_loops = False) # Does not hold for mask_self_loops = True
    A_pred = get_A()
    A_trgt = get_A()
    T = get_T()
    
    loss1 = loss_fct.forward(T,A_pred,A_trgt)    
    A_pred = torch.bmm(T.transpose(1,2),torch.bmm(A_pred,T))
    loss2 = loss_fct.forward_aligned(A_pred,A_trgt)
    
    assert torch.all(loss2 <= loss1)
    
    # Test 4: BCE loss
    A_trgt = get_A()
    A_pred = 100*(A_trgt-0.5)
    loss_fct = QuadraticBCE(mask_self_loops=mask_self_loops)
    loss = loss_fct.forward_aligned(A_pred,A_trgt)
    assert torch.allclose(loss,torch.tensor(0.,device='cuda'))
    
    # Test 4: CE loss
    A_trgt = get_A()
    A_trgt = torch.stack([1-A_trgt,A_trgt],dim=-1)
    A_pred = 100*(A_trgt-0.5)
    loss_fct = QuadraticCE(mask_self_loops=mask_self_loops)
    loss = loss_fct.forward_aligned(A_pred,A_trgt)
    assert torch.allclose(loss,torch.tensor(0.,device='cuda'))

    # Test 5: Accuracy 
    loss_fct_mask = QuadraticBinaryAccuracy(mask_self_loops=True)
    loss_fct = QuadraticBinaryAccuracy(mask_self_loops=False)
    A_pred = get_A()
    T = get_P()
    A_trgt = torch.bmm(T.transpose(1,2),torch.bmm(A_pred,T))
    
    loss = loss_fct.forward(T, A_pred, A_trgt)
    assert torch.allclose(loss,torch.tensor(0.,device='cuda'))
    
    A_trgt[0,0,0] = 1 - A_trgt[0,0,1]
    loss = loss_fct_mask.forward(T, A_pred, A_trgt)
    assert loss[0] == 0
    loss = loss_fct.forward(T, A_pred, A_trgt)
    assert loss[0] == 1
    