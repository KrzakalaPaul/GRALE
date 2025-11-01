from GRALE.loss.linear_losses import *
from GRALE.loss.quadratic_losses import *
from GRALE.loss.regularization import *
from GRALE.loss.matching import batched_sinkhorn_projection, batched_hungarian_projection, batched_log_sinkhorn_projection

class PMFGW_solver():
    def __init__(self, alpha, config: dict, mask_self_loops=False):
        self.Mmax = config['n_nodes_max']
        self.alpha = alpha
        self.mask_self_loops = mask_self_loops
        self.symmetric = True
        self.h_objective = LinearBCE()
        self.node_features_objective = LinearL2()
        self.node_labels_objective = LinearCE()
        self.adjacency_objective = QuadraticBCE(mask_self_loops=mask_self_loops)
        self.edge_features_objective = QuadraticL2(mask_self_loops=mask_self_loops)
        self.edge_labels_objective = QuadraticCE(mask_self_loops=mask_self_loops)
        self.marginal_objective = MarginalKL()
        self.max_iter = config['max_iter_solver']
        self.tol = config['tol_solver']
        self.max_iter_inner = config['max_iter_inner_solver']
    
    def __call__(self, outputs, targets, hard=False):
        
        alpha_h = self.alpha['h']
        alpha_node_features = self.alpha['node_features']
        alpha_node_labels = self.alpha['node_labels']
        alpha_edge_features = self.alpha['edge_features']
        alpha_edge_labels = self.alpha['edge_labels']
        alpha_adjacency = self.alpha['adjacency']
        alpha_marginals = self.alpha['marginals']
        
        h_logits = outputs.h
        F_feats_preds = outputs.nodes.features
        F_logits = outputs.nodes.labels
        A_logits = outputs.edges.adjacency
        
        h = targets.h.to(h_logits.dtype)
        F_feats = targets.nodes.features
        F = targets.nodes.labels
        A = targets.edges.adjacency
        
        # Init Cost matrix h
        M_weight = self.h_objective.pairwise_loss(h_logits,h)
        M_weight = alpha_h*M_weight
        
        # Init Cost matrix F
        weights = self.Mmax*h/torch.sum(h,1,keepdim=True)
        M_F = alpha_node_labels*self.node_labels_objective.pairwise_loss(F_logits,F)*weights[:,None,:]
        M_F_fd = alpha_node_features*self.node_features_objective.pairwise_loss(F_feats_preds,F_feats)*weights[:,None,:]
    
        M =  M_weight + M_F + M_F_fd
            
        # Init Cost matrix A
        L = init_matrix_quad_batch(A_logits=A_logits,A=A,w=weights,alpha=alpha_adjacency,mask_self_loops=self.mask_self_loops)
        if self.symmetric:
            Lt = None
        else:
            Lt = init_matrix_quad_batch(A_logits=A_logits.transpose(1,2),A=A.transpose(1,2),w=weights,alpha=alpha_adjacency,mask_self_loops=self.mask_self_loops)
            
        # Get OT plan 
        T,log_solver = solver_quad_batch(M=M,L=L,Lt=Lt,max_iter=self.max_iter,tol=self.tol,max_iter_inner=self.max_iter_inner,mask_self_loops=self.mask_self_loops,log=True)
        T = T.to(device=M.device, dtype=M.dtype)
        T = T * self.Mmax
        if hard:
            permutations_list, _  = batched_hungarian_projection(T, metric = 'F')
            return permutations_list, log_solver
        return T, log_solver


import torch
from ot.optim import solve_1d_linesearch_quad
from torch.special import entr as entr_torch
from torch.nn.functional import logsigmoid
import numpy as np
from scipy.optimize import linear_sum_assignment

def emd(M):
    n = len(M)
    _, permutation = linear_sum_assignment(M.T)
    T = np.eye(n)[:,permutation]/n
    return T

def bmv(Matrices,Vectors):
    return torch.einsum('bij,bj->bi', Matrices, Vectors)


def init_matrix_quad_batch(A_logits,A,w,alpha,mask_self_loops=False):
    
    B,n,_ = A_logits.shape

    def f1(a):
        return -logsigmoid(a)

    def f2(b):
        return -entr_torch(b)-entr_torch(1-b)

    def h1(a):
        return -a

    def h2(b):
        return (1-b)
        
    if mask_self_loops:
        
        ww = w[:,:,None]*w[:,None,:]
        mask = (1-torch.eye(n,device=A_logits.device)).repeat(B, 1, 1)
            
        U1 = alpha*f1(A_logits)*mask
        U2 = alpha*f2(A)*ww*mask
        
        V1 = alpha*h1(A_logits)*mask
        V2 = h2(A)*ww*mask
        
        L = U1, U2, V1, V2, w
        
    else:
        
        hC1 = h1(A_logits)
        ww = w[:,:,None]*w[:,None,:]
        hC2 = alpha*h2(A)*ww
            
        constC = alpha*w*bmv(f2(A).transpose(1,2),w)/n
        constC  = constC[:,None,:]
        
        fC1 = alpha*f1(A_logits)
        
        L = constC, fC1, hC1, hC2, w
        
    return L 



def tensor_product_quad_batch(L,Gs,mask_self_loops=False):
    
    if mask_self_loops:
        U1, U2, V1, V2, w = L
        U1TW2 = torch.bmm(U1, bmv(Gs,w)[:,:,None]*w[:,None,:] - (w**2)[:,None,:]*Gs)
        W1TU2 = torch.bmm( torch.sum(Gs,1,keepdim=True) - Gs,torch.permute(U2,dims=(0,2,1)))
        V1TV2 = torch.bmm( torch.bmm(V1,Gs), torch.permute(V2,dims=(0,2,1)) )
        return U1TW2 + W1TU2 - V1TV2
    
    else:
        constC, fC1, hC1, hC2, w = L
        return constC + bmv(fC1, bmv(Gs,w))[:,:,None]*w[:,None,:] - torch.bmm( torch.bmm(hC1,Gs), torch.permute(hC2,dims=(0,2,1)) )

def tensor_product_quad(L,G,mask_self_loops=False):
    if mask_self_loops:
        U1, U2, V1, V2, w = L
        U1TW2 = np.dot(U1,np.outer(np.dot(G,w),w)- (w**2)[None,:]*G)
        W1TU2 = np.dot( np.sum(G,0,keepdims=True) - G, U2.T)
        V1TV2 = np.dot(np.dot(V1, G), V2.T)
        return U1TW2 + W1TU2 - V1TV2
    else:
        constC, fC1, hC1, hC2, w = L
        return constC + np.outer(np.dot(fC1, np.dot(G,w)),w) - np.dot(np.dot(hC1, G), hC2.T)
    
def line_search(M, L, Gprev, G, costprev, Lt=None, mask_self_loops=False, symmetric=True):
    delta_G = G-Gprev
    if Lt is None:
        dot = tensor_product_quad(L,delta_G,mask_self_loops=mask_self_loops)
    else:
        dot = 0.5*(tensor_product_quad(L,delta_G,mask_self_loops=mask_self_loops)+tensor_product_quad(Lt,delta_G,mask_self_loops=mask_self_loops))
    a = np.sum(delta_G*dot)
    b = np.sum(M * delta_G) + 2*np.sum(dot * Gprev)
    t = solve_1d_linesearch_quad(a,b)
    G  = t*G+(1-t)*Gprev
    cost = costprev + a*t**2 + b*t
    #print(t)
    return G,cost

def solver_linear_batch(M,max_iter_inner,log=True):
    
    B,n,m = M.shape
    M = M.cpu().detach().numpy()
    Gs = []
    
    for k in range(B):
        G = emd(M=M[k])
        Gs.append(G)
        
    Gs = np.stack(Gs,0)
    Gs = torch.tensor(Gs)
    log_solver = {'avg_cg_iter':0}
    
    return Gs,log_solver
            
def solver_quad(M, L, cost, max_iter, tol, max_iter_inner, Lt=None, mask_self_loops=False, log=False):
    
    n,m = M.shape
    G = np.ones((n,m),dtype=np.float32)/(n*m)  
        
    for ii in range(max_iter):
        Gprev = G
        costprev = cost
        
        if Lt is None:
            M_ii = 2*tensor_product_quad(L,G,mask_self_loops=mask_self_loops) + M
        else:
            M_ii = tensor_product_quad(L,G,mask_self_loops=mask_self_loops) + tensor_product_quad(Lt,G,mask_self_loops=mask_self_loops) + M
        
        G = emd(M=M_ii)

        G,cost = line_search(M,L,Gprev,G,costprev,mask_self_loops=mask_self_loops,Lt=Lt)
        
        if abs(costprev-cost) < tol:
            break
        
    lg = {'n_cg_iter':ii+1,'cost':cost}
    if log:
        return G,lg
    return G 

def solver_quad_batch(M, L, max_iter, tol, max_iter_inner, Lt=None, mask_self_loops=False, log=False):
    
    B,n,m = M.shape
    
    T_init = torch.ones((B,n,m),device=M.device)/(n*m)    
    LT = tensor_product_quad_batch(L,T_init,mask_self_loops=mask_self_loops)
    M_init  = M + LT
    cost_init = torch.sum(T_init*M_init,dim=(1,2)) 
    
    M = M.cpu().detach().numpy()
    L = [ matrix.cpu().detach().numpy() for matrix in L]
    Lt = [ matrix.cpu().detach().numpy() for matrix in Lt] if Lt is not None else None
    cost_init = cost_init.cpu().detach().numpy()
    
    Gs = []
    avg_cg_iter = 0
    for k in range(B):
        Mk = M[k]
        Lk = [matrix[k] for matrix in L]
        Lkt = [matrix[k] for matrix in Lt] if Lt is not None else None
        if log:
            G,log = solver_quad(Mk,Lk,cost_init[k],max_iter,tol,max_iter_inner, mask_self_loops=mask_self_loops, log=log,Lt=Lkt)
            avg_cg_iter += log['n_cg_iter']
        else:
            G = solver_quad(Mk,Lk,cost_init[k],max_iter,tol,max_iter_inner, mask_self_loops=mask_self_loops, log=log,Lt=Lkt)
        Gs.append(G)
    Gs = np.stack(Gs,0)
    Gs = torch.tensor(Gs)
    if log:
        return Gs,{'avg_cg_iter':avg_cg_iter/B}
    return Gs



