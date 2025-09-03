import torch
from torch import Tensor
from typing import List,Dict,Union
import numpy as np

### ---------------------------- Data ---------------------------- ###

class DenseData():
        def __init__(self, size, h: Union[Tensor,None] = None, nodes: Dict[str,Tensor] = {}, edges: Dict[str,Tensor] = {}):
            self.size = size
            self.h = h if h is not None else torch.ones(size, dtype = torch.bool)
            self.is_valid(nodes = nodes, edges=edges)
            self.nodes = EasyDict(nodes)
            self.edges = EasyDict(edges)
            
        def is_valid(self, nodes = {}, edges = {}):
            for X in nodes.values():
                assert X.size(0) == self.size, 'X must have the same size as the number of nodes'
            for E in edges.values():
                assert E.size(0) == self.size and E.size(1) == self.size, 'E must have the same size as the number of nodes in the first two dimensions'
                
        def pad_(self, new_size):
            assert new_size >= self.size, 'The new size must be greater than the current size'
            padding_size = new_size - self.size
            self.h = torch.nn.functional.pad(self.h, (0, padding_size))
            for key, X in self.nodes.items():
                # put dimension to pad at the end
                X = X.transpose(0,-1)
                # pad the last dimension
                X = torch.nn.functional.pad(X, (0, padding_size))
                # transpose back
                self.nodes[key] = X.transpose(0,-1)

            for key, E in self.edges.items():
                # put dimensions to pad at the end
                E = E.transpose(0,-1)
                E = E.transpose(1,-2)
                # pad the last two dimensions
                E = torch.nn.functional.pad(E, (0, padding_size, 0, padding_size))
                # transpose back
                E = E.transpose(0,-1)
                E = E.transpose(1,-2)
                self.edges[key] = E
            self.size = new_size
        
        def clone(self):
            h = self.h.clone()
            nodes = {key: self.nodes[key].clone() for key in self.nodes.keys()}
            edges = {key: self.edges[key].clone() for key in self.edges.keys()}
            return DenseData(self.size, h, nodes, edges)
                
class BatchedDenseData():
    def __init__(self, h: Tensor, nodes: Dict[str, Tensor] = {}, edges: Dict[str, Tensor] = {}):
        self.batchsize = h.shape[0]
        self.size = h.shape[1]
        self.h = h
        self.is_valid(nodes=nodes, edges=edges)
        self.nodes = EasyDict(nodes)
        self.edges = EasyDict(edges)

    def is_valid(self, nodes={}, edges={}):
        for X in nodes.values():
            assert X.size(0) == self.batchsize, 'X must have the same batch size'
            assert X.size(1) == self.size, 'X must have the same size as the number of nodes'
        for E in edges.values():
            assert E.size(0) == self.batchsize, 'E must have the same batch size'
            assert E.size(1) == self.size and E.size(2) == self.size, 'E must have the same size as the number of nodes in the first two dimensions'

    def pad_(self, new_size):
        assert new_size >= self.size, 'The new size must be greater than the current size'
        padding_size = new_size - self.size
        self.h = torch.nn.functional.pad(self.h, (0, padding_size))
        for key, X in self.nodes.items():
            # put dimension to pad at the end
            X = X.transpose(1, -1)
            # pad the last dimension
            X = torch.nn.functional.pad(X, (0, padding_size))
            # transpose back
            self.nodes[key] = X.transpose(1, -1)

        for key, E in self.edges.items():
            # put dimensions to pad at the end
            E = E.transpose(1, -1)
            E = E.transpose(2, -2)
            # pad the last two dimensions
            E = torch.nn.functional.pad(E, (0, padding_size, 0, padding_size))
            # transpose back
            E = E.transpose(1, -1)
            E = E.transpose(2, -2)
            self.edges[key] = E
        self.size = new_size
        
    def to(self, device, non_blocking=False):
        self.h = self.h.to(device, non_blocking=non_blocking)
        for key, X in self.nodes.items():
            self.nodes[key] = X.to(device, non_blocking=non_blocking)
        for key, E in self.edges.items():
            self.edges[key] = E.to(device, non_blocking=non_blocking)
        return self
    
    def align_(self, permutation_list):
        permutation_batch  = torch.tensor(np.array(permutation_list), device=self.h.device)
        self.h = torch.take_along_dim(self.h, permutation_batch, dim=1)
        for key in self.nodes.keys():
            self.nodes[key] = torch.take_along_dim(self.nodes[key], permutation_batch[:,:,None], dim=1)
        for key in self.edges.keys():
            is_tensor = (self.edges[key].ndim == 4)
            if is_tensor:
                self.edges[key] = torch.take_along_dim(self.edges[key], permutation_batch[:,:,None,None], dim=1)
                self.edges[key] = torch.take_along_dim(self.edges[key], permutation_batch[:,None,:,None], dim=2)
            else:
                self.edges[key] = torch.take_along_dim(self.edges[key], permutation_batch[:,:,None], dim=1)
                self.edges[key] = torch.take_along_dim(self.edges[key], permutation_batch[:,None,:], dim=2)
         
    def permute_(self, permutation_matrices):
        self.h = torch.einsum('bki,bk->bi',permutation_matrices,self.h)
        for key in self.nodes.keys():
            self.nodes[key] = torch.einsum('bki,bkd->bid',permutation_matrices,self.nodes[key])
        for key in self.edges.keys():
            is_tensor = (self.edges[key].ndim == 4)
            if is_tensor:
                self.edges[key] = torch.einsum('bki,blj,bkld->bijd',permutation_matrices,permutation_matrices,self.edges[key])  
            else:
                self.edges[key] = torch.einsum('bki,blj,bkl->bij',permutation_matrices,permutation_matrices,self.edges[key])  
    
    @classmethod
    def from_list(cls, list: List[DenseData]):
        size = max([data.size for data in list])
        for data in list:
            data.pad_(size)
        h = torch.stack([data.h for data in list])
        nodes = {key: torch.stack([data.nodes[key] for data in list]) for key in list[0].nodes.keys()}
        edges = {key: torch.stack([data.edges[key] for data in list]) for key in list[0].edges.keys()}
        return cls(h, nodes, edges)
    
    def clone(self):
        h = self.h.clone()
        nodes = {key: self.nodes[key].clone() for key in self.nodes.keys()}
        edges = {key: self.edges[key].clone() for key in self.edges.keys()}
        return BatchedDenseData(h, nodes, edges)
    
    def __len__(self):
        return self.h.shape[0]
    
    def __getitem__(self, idx):
        h = self.h[idx]
        nodes = {key: self.nodes[key][idx] for key in self.nodes.keys()}
        edges = {key: self.edges[key][idx] for key in self.edges.keys()}
        return DenseData(self.size, h, nodes, edges)

##### ---------------------------- EasyDicts ---------------------------- #####
# Adapted from https://pypi.org/project/easydict/

class EasyDict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        else:
            d = dict(d)        
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in list(self.__class__.__dict__): # Instead of "#for k in self.__class__.__dict__.keys():" to avoid RuntimeError: dictionary changed size during iteration
            if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = type(value)(self.__class__(x)
                     if isinstance(x, dict) else x for x in value)
        elif isinstance(value, dict) and not isinstance(value, EasyDict):
            value = EasyDict(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(EasyDict, self).pop(k, d)
