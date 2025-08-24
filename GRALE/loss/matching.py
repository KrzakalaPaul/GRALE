from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed
import torch
import numpy as np

######################### Permutations list vs matrices #########################u

# Convention: permutation list = numpy cpu
#             permutation matrices = torch cuda

# Prediction can be aligned with target using prediction[permutation]
#                                             permutation_matrix.T @ prediction 

def permutations_list_to_matrices(permutations_list, device = 'cuda'):
    B,Mmax = len(permutations_list),len(permutations_list[0])
    permutations_matrices = torch.stack([torch.eye(Mmax, device=device)[:,permutation] for permutation in permutations_list]) 
    return permutations_matrices                                    

def permutations_matrices_to_list(permutations_matrices):
    permutations_list = permutations_matrices.argmax(dim=1)
    permutations_list = permutations_list.cpu().detach().numpy()
    permutations_list = [permutation for permutation in permutations_list]
    # Check that it is a permutation
    for i,permutation in enumerate(permutations_list):
        assert len(permutation) == len(set(permutation)), f'Permutation matrix cannot be converted to list, try to approximate with Hungarian'
    return permutations_list

######################### HUNGARIAN #########################

def hungarian_solver(cost):
    _, permutation = linear_sum_assignment(cost.T)
    return permutation

def batched_hungarian_solver(cost, use_joblib = False):
    '''
    Compute optimal permutation matrices for each batch element
    '''

    cost = cost.detach().cpu().numpy()
    if use_joblib:
        permutations_list = Parallel(n_jobs = 8, backend = 'threading')(delayed(hungarian_solver)(cost_i) for cost_i in cost)   
    else:
        permutations_list = [hungarian_solver(cost_i) for cost_i in cost]
    log_solver = {} # Placeholder for now
    
    return permutations_list, log_solver

def batched_hungarian_projection(K, metric = 'KL'):
    '''
    Project a batch of positive matrices onto the set of permutations
    If metric = 'KL': return argmin_P KL(P||K) = argmin_P -<P,log(K)>
    If metric = 'F': return argmin_P ||P-K||_F = argmin_P -<P,K>
    '''
    if metric == 'KL':
        K = torch.clamp(K,1e-10,1)
        return batched_hungarian_solver(-K.log())
    elif metric == 'F':
        return batched_hungarian_solver(-K)

######################### SINKHORN #########################

def batched_sinkhorn_projection(K, max_iter = 10000, tol = 1e-5, last_iter_grad = False, fixed_n_iters = False):
    '''
    Project a batch of positive matrices onto the set of doubly stochastic matrices using Sinkhorn algorithm
    return argmin_T KL(T||K) = argmin_T -<T,log(K)>
    '''

    B, m, n = K.shape
    
    assert m == n, "Cost matrix must be square"

    # Weights [n,] and [m,]
    a = torch.ones(B,n, device = K.device, dtype=K.dtype)
    b = torch.ones(B,n, device = K.device, dtype=K.dtype)

    # Initialize the iteration with the change of variable
    f = torch.ones_like(a)
    g = torch.ones_like(b)
    
    with torch.set_grad_enabled(not last_iter_grad):
        
        n_iters = 0
        for _ in range(max_iter):
            
            f_prev = f
            g_prev = g
            
            summand_f = (K*g[:,None,:]).sum(dim=2)
            f = a / summand_f
            
            summand_g = (K*f[:,:,None]).sum(dim=1)
            g = b / summand_g
            
            n_iters += 1
            
            if not fixed_n_iters:
                max_err_u = torch.max(torch.abs(f_prev-f))
                max_err_v = torch.max(torch.abs(g_prev-g))
                if max_err_u < tol and max_err_v < tol:
                    break
            
    if last_iter_grad:
        summand_f = (K*g[:,None,:]).sum(dim=2)
        f = a / summand_f
        
        summand_g = (K*f[:,:,None]).sum(dim=1)
        g = b / summand_g

    P = (K * f[:,:,None] * g[:,None,:])
    summand_f = P.sum(dim=2)
    summand_g = P.sum(dim=1)
    max_err_a = torch.amax(torch.abs(a - summand_f), dim=-1).mean()
    max_err_b = torch.amax(torch.abs(b - summand_g), dim=-1).mean()
    log_solver = {'n sinkhorn iters': n_iters, 'sinkhorn marginal error (rows)': max_err_a, 'sinkhorn marginal error (cols)': max_err_b} 

    return P, log_solver

def batched_log_sinkhorn_projection(K, max_iter = 10000, tol = 1e-5, last_iter_grad = False, fixed_n_iters = False):
    '''
    Project exp(K) onto the set of doubly stochastic matrices, using log-sum-exp trick and Sinkhorn algorithm
    return argmin_T KL(T||exp(K)) = argmin_T -<T,exp(K)>
    '''

    B, m, n = K.shape
    
    assert m == n, "Cost matrix must be square"

    # Weights [n,] and [m,]
    a = torch.ones(B,n, device = K.device, dtype=K.dtype)
    b = torch.ones(B,n, device = K.device, dtype=K.dtype)

    log_a = torch.log(a)  # [n]
    log_b = torch.log(b)  # [m]

    # Initialize the iteration with the change of variable
    u = torch.zeros_like(a)
    v = torch.zeros_like(b)
    
    with torch.set_grad_enabled(not last_iter_grad):
        
        n_iters = 0
        for _ in range(max_iter):
            
            u_prev = u
            v_prev = v

            summand_u = (K + v[:,None,:]).logsumexp(dim=2).squeeze() 
            u = (log_a - summand_u)

            summand_v = (K + u[:,:,None]).logsumexp(dim=1).squeeze()
            v = (log_b - summand_v)
            
            n_iters += 1
            
            if not fixed_n_iters:
                max_err_u = torch.max(torch.abs(u_prev-u))
                max_err_v = torch.max(torch.abs(v_prev-v))
                if max_err_u < tol and max_err_v < tol:
                    break
            
    if last_iter_grad:
        summand_u = (K + v[:,None,:]) 
        u = (log_a - summand_u.logsumexp(dim=2).squeeze())

        summand_v = (K + u[:,:,None]) 
        v = (log_b - summand_v.logsumexp(dim=1).squeeze())

    log_P = (K + u[:,:,None] + v[:,None,:])
    P = log_P.exp()
    
    summand_f = P.sum(dim=2)
    summand_g = P.sum(dim=1)
    max_err_a = torch.amax(torch.abs(a - summand_f), dim=-1).mean()
    max_err_b = torch.amax(torch.abs(b - summand_g), dim=-1).mean()
    log_solver = {'n sinkhorn iters': n_iters, 'sinkhorn marginal error (rows)': max_err_a.item(), 'sinkhorn marginal error (cols)': max_err_b.item()} 

    return P, log_P, log_solver

def batched_sinkhorn_solver(cost, epsilon = None, max_iter = 10000, tol = 1e-5, last_iter_grad = False, log_solver = True):
    cost = cost/torch.abs(cost).sum(dim=(1,2),keepdim=True)
    K =  -cost/epsilon 
    if log_solver:
        P, log_P, log = batched_log_sinkhorn_projection(K, max_iter = max_iter, tol = tol, last_iter_grad = last_iter_grad)
        return P, log
    else:
        return batched_sinkhorn_projection(K.exp(), max_iter = max_iter, tol = tol, last_iter_grad = last_iter_grad)
    


if __name__ == '__main__':
    
    torch.set_printoptions(precision=2, sci_mode=False)
    
    K = torch.eye(7)
    K += 0.1*torch.rand_like(K)
    K = len(K)*K/K.sum()
    K = K.unsqueeze(0)
    
    T, log = batched_sinkhorn_projection(K, max_iter = 10000, tol = 1e-5)
    T = T.squeeze(0)
    print(f'Sinkhorn:')
    print(T)
    
    permutations, log = batched_hungarian_projection(K, metric='KL')
    P = permutations_list_to_matrices(permutations)
    P = P.squeeze(0)
    print(f'Hungarian (KL):')
    print(P)
    
    permutations, log = batched_hungarian_projection(K, metric='F')
    P = permutations_list_to_matrices(permutations)
    P = P.squeeze(0)
    print(f'Hungarian (F):')
    print(P)
    