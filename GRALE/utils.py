import torch
from torch import Tensor
from typing import List,Dict,Union
import os 
import pickle
import lmdb
from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch.optim import AdamW
import json

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

class LMDBDataset:
    def __init__(self, db_path, split="train"):
        self.db_path = os.path.join(db_path, split+".lmdb")
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        env = self.connect_db(self.db_path)
        self.dataset_size = int(env.begin().get("size".encode("ascii")).decode("ascii")) 
        node_labels = eval(env.begin().get("node_labels".encode("ascii")))
        self.n_node_labels = len(node_labels)
        edge_labels = eval(env.begin().get("edge_labels".encode("ascii")))
        self.n_edge_labels = len(edge_labels)

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return self.dataset_size
    
    def load_idx(self, idx):
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(f"{idx}".encode("ascii"))
        data = pickle.loads(datapoint_pickled)
        return data

    @lru_cache()
    def __getitem__(self, idx):
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(f"{idx}".encode("ascii"))
        data = pickle.loads(datapoint_pickled)
        if isinstance(data, tuple):
            graph, target = data
        else:
            graph = data

        node_labels = graph["node_labels"]              # matrix of shape n_nodes (ints)
        adjacency_matrix = graph["adjacency_matrix"]    # sparse matrix of shape n_nodes x n_nodes (binary)
        edge_labels = graph["edge_labels"]              # sparse matrix of shape n_nodes x n_nodes (ints)
        SP_matrix = graph["SP_matrix"]                  # matrix of shape n_nodes x n_nodes (ints)
        
        node_labels = torch.nn.functional.one_hot(torch.tensor(node_labels, dtype=torch.long), num_classes=self.n_node_labels).to(torch.float)
        adjacency_matrix = torch.tensor(adjacency_matrix.todense(), dtype=torch.float)
        edge_labels = torch.nn.functional.one_hot(torch.tensor(edge_labels.todense(), dtype=torch.long), num_classes=self.n_edge_labels).to(torch.float)
        SP_matrix = torch.tensor(SP_matrix, dtype=torch.float)
        
        input = DenseData(size=len(node_labels), nodes={'labels': node_labels}, edges={'adjacency': adjacency_matrix, 'labels': edge_labels, 'SP': SP_matrix})
            
        return idx, input, data
    
    def collate_fn_autoencoder(self, batch):
        _, inputs, _ = zip(*batch)
        return BatchedDenseData.from_list(inputs)
    
    def collate_fn_all(self, batch):
        indices, inputs, datas = zip(*batch)
        return indices, BatchedDenseData.from_list(inputs), datas

def collate_fn_autoencoder(batch):
        _, inputs, _ = zip(*batch)
        return BatchedDenseData.from_list(inputs)
    
### ---------------------------- Optimization ---------------------------- ###
    
class EarlyStopper:
    def __init__(self, patience=5, tol=1e-4, rel_tol=0.05 ):
        self.patience = patience
        self.tol = tol
        self.rel_tol = rel_tol
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.tol and validation_loss < self.min_validation_loss*(1-self.rel_tol):
            self.min_validation_loss = validation_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class ScheduledAdamW:
    def __init__(self, base_lr, num_step_warmup, num_step_total, decay_mode,
                 params, betas=(0.9, 0.999), eps=1e-9, weight_decay=0):
        '''
        base_lr: the base learning reate
        num_step_warmup: number of linear warmup step before reaching the base_lr
        num_step_total: number of total steps
        decay_mode: ['linear','sqrt','none'] mode for decaying the lr after the warmup phase
        '''
        assert decay_mode in ['linear','sqrt','cosine','none'], 'decay_mode must be in ["linear","sqrt","cosine","none"]'
        self.optimizer = AdamW(params, lr=0, betas=betas, eps=eps, weight_decay=weight_decay)
        self._step = 0
        self.num_step_warmup = num_step_warmup if num_step_warmup is not None else 0
        self.num_step_post_warmup = num_step_total - num_step_warmup
        self.base_lr = base_lr
        self.decay_mode = decay_mode
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def rate(self, step=None):
        if step is None:
            step = self._step
        if self.decay_mode == 'none':
            return self.base_lr
        if step < self.num_step_warmup:
            return self.base_lr * step / self.num_step_warmup
        if self.decay_mode == 'linear':
            progress = (step - self.num_step_warmup) / self.num_step_post_warmup
            return self.base_lr * (1 - progress)
        elif self.decay_mode == 'sqrt':
            progress = step / self.num_step_warmup
            return self.base_lr * (progress)  ** (-0.5)
        elif self.decay_mode == 'cosine':
            progress = (step - self.num_step_warmup) / self.num_step_post_warmup
            return self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        else:
            return self.base_lr
        
    def save(self, path):
        scheduler_state = {'step': self._step,
                           'rate': self._rate,
                           'num_step_warmup': self.num_step_warmup,
                           'num_step_post_warmup': self.num_step_post_warmup,
                           'base_lr': self.base_lr,
                           'decay_mode': self.decay_mode}
        torch.save(self.optimizer.state_dict(), path+'.opt')   
        with open(path + '.json', 'w') as f:
            json.dump(scheduler_state, f)
            
    def load(self, path, params):
        with open(path + '.json', 'r') as f:
            scheduler_state = json.load(f)
        self._step = scheduler_state['step']
        self._rate = scheduler_state['rate']
        self.num_step_warmup = scheduler_state['num_step_warmup']
        self.num_step_post_warmup = scheduler_state['num_step_post_warmup']
        self.base_lr = scheduler_state['base_lr']
        self.decay_mode = scheduler_state['decay_mode']
        self.optimizer = AdamW(params, lr=0, betas=self.optimizer.param_groups[0]['betas'], eps=self.optimizer.param_groups[0]['eps'], weight_decay=self.optimizer.param_groups[0]['weight_decay'])
        self.optimizer.load_state_dict(torch.load(path + '.opt'))
        
### ---------------------------- Plots ---------------------------- ###

def softmax_to_one_hot(logits):
    classes = torch.argmax(logits,dim=-1,keepdim=True)
    one_hot = torch.zeros_like(logits).scatter_(-1, classes, 1)
    return one_hot 

def plot_graph(node_labels,A,ax=None,pos='kamada',frame=False,edge_weight=False, node_size=200, colors=None):
    
    if isinstance(node_labels, torch.Tensor):
        node_labels = node_labels.detach().cpu().numpy()
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    
    if colors is None:
        colors = np.array([[1/4,1/4,3/4],
                            [1/4,3/4,1/4],
                            [3/4,1/4,1/4],
                            [1,4/5,2/5]])
    
    if not(edge_weight):
        A = (A > 0.5).astype(np.float32)
    
    if ax is None:
        ax = plt.gca()
        
    graph_nx = nx.from_numpy_array(A)
    
    if pos == 'kamada':
        pos = nx.kamada_kawai_layout(graph_nx)
    elif pos == 'spring':
        pos = nx.spring_layout(graph_nx)

    color_map = [colors[l.argmax()] for l in node_labels]
    
    nx.draw_networkx_nodes(graph_nx,node_color="k",ax=ax, pos=pos)
    nx.draw_networkx_nodes(graph_nx, pos, node_size=node_size, node_color=color_map,ax=ax,alpha=1)
    [nx.draw_networkx_edges(graph_nx,pos=pos,edgelist=[(u,v)],alpha=A[u,v],width=2,ax=ax) for u,v in graph_nx.edges] #loop through edges and draw the
        
    ax.axis('equal')
    if frame:
        pass
    else:
        ax.axis('off')
        
    return pos
        
def plot_img(img,ax=None,frame=False):
    if ax is None:
        ax = plt.gca()
    img = img.astype(np.float32)/255
    ax.imshow(np.transpose(img,(1,0,2)),vmin=0,vmax=1,origin='lower')
    if frame:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_axis_off()
        
        
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
