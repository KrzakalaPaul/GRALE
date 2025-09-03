import networkx as nx 
import numpy as np
import imageio
import torch
import matplotlib.pyplot as plt
from networkx import graph_edit_distance
from GRALE.vizualization.plots import atomic_num_colormap
from GRALE.data.dataset import custom_collate_fn

def interpolate(start, end, model, device='cuda', n_timesteps=5):
    inputs = custom_collate_fn([start, end])
    inputs.to(device)
    embeddings = model.encode(inputs)
    embedding1 = embeddings[0]
    embedding2 = embeddings[1]

    # Linear interpolation
    T = torch.linspace(0,1,n_timesteps,device=device)
    embedding_interpolated = torch.lerp(embedding1,embedding2,T.unsqueeze(1).unsqueeze(1))

    # Flatten if needed
    if len(embedding_interpolated.shape) == 2:
        embedding_interpolated = embedding_interpolated.unsqueeze(1)

    # Decode interpolated embeddings
    predictions = model.decode(embedding_interpolated, logits = False)
    
    return predictions

import networkx as nx
import numpy as np

def get_pos_with_none(h,A):
    '''
    Returns a list of positions for each node in the graph.
    Set to None for inactive nodes.
    '''
    active = h>0.5
    A_active = A[active][:,active]
    A_active = (A_active > 0.5).astype(np.float32)
    graph_nx = nx.from_numpy_array(A_active)
    pos = nx.kamada_kawai_layout(graph_nx)
    pos_all = []
    i_active = 0
    for i in range(len(h)):
        if active[i]:
            pos_all.append(pos[i_active])
            i_active += 1
        else:
            pos_all.append(None)
    return pos_all

def interpolate_pos_with_none(pos1, pos2, t):
    '''
    Interpolates between two lists of positions, where some positions may be None.
    '''
    pos = []
    for p1,p2 in zip(pos1,pos2):
        if p1 is not None and p2 is not None:
            pos.append((1-t)*p1 + t*p2)
        elif p1 is not None:
            pos.append(p1)
        elif p2 is not None:
            pos.append(p2)
        else:
            pos.append(None)
    return pos

def get_matrices(graph):
    '''
    Returns the node feature matrix and adjacency matrix of a graph.
    '''
    h = graph.h.detach().cpu().numpy()
    F = graph.nodes.labels.detach().cpu().numpy()
    A = graph.edges.adjacency.detach().cpu().numpy()
    return h, F, A

def plot_graph_interpolation(pos,h,F,A,ax=None,frame=False,edge_weight=False, node_size=200, colors=atomic_num_colormap):
    '''
    pos = position of the atom or None if the atom should not be displayed
    h = probability of being an atom (1 for real atoms, 0 for padding), displayed using an atom of size h*node_size
    F = node features (one-hot encoded)
    A = adjacency matrix (values between 0 and 1)
    ax = matplotlib axis to plot on
    frame = whether to display the frame of the plot
    edge_weight = whether to display the edge weights (A values) or just the presence/absence of edges
    node_size = size of the nodes
    '''
    active = [pos is not None for pos in pos]
    pos = [p for p in pos if p is not None]
    node_labels = F[active]
    A = A[active][:,active]
    h = h[active] 
    
    A = A * h[:,None] * h[None,:]
    node_size = node_size * h

    if not(edge_weight):
        A = (A > 0.5).astype(np.float32)
        
    graph_nx = nx.from_numpy_array(A)
    
    if ax is None:
        ax = plt.gca()

    color_map = [colors[f.argmax()] for f in node_labels]
    
    nx.draw_networkx_nodes(graph_nx,node_color="k",ax=ax, pos=pos, node_size=node_size*1.5)
    nx.draw_networkx_nodes(graph_nx, pos, node_size=node_size, node_color=color_map,ax=ax,alpha=1)
    [nx.draw_networkx_edges(graph_nx,pos=pos,edgelist=[(u,v)],alpha=A[u,v],width=2,ax=ax) for u,v in graph_nx.edges] #loop through edges and draw the
        
    ax.axis('equal')
    if frame:
        pass
    else:
        ax.axis('off')