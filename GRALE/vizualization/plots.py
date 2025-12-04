import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch

# Define a colors maps for the elements of the period table
# Define atomic number list
atomic_num_list = [6, 8, 7, 17, 16, 9, 35, 15, 53, 14, 11, 33, 80, 50, 5, 20, 19, 30, 26, 34, 13, 29, 12, 82, 24, 27, 28, 56, 78, 25, 52, "ukn"]

# Predefined colors for specific atomic numbers
predefined_colors = {
    6: "black",  # Carbon
    8: "blue"    # Oxygen
}

# Generate colormap
cmap = plt.get_cmap("tab20")  # Use a categorical colormap
num_colors = len(atomic_num_list)
colors = [cmap(i % cmap.N) for i in range(num_colors)]

# Create atomic_num_colormap dictionary
atomic_num_colormap = {}
for idx, atomic_num in enumerate(atomic_num_list):
    if isinstance(atomic_num, int) and atomic_num in predefined_colors:
        atomic_num_colormap[idx] = predefined_colors[atomic_num]
    else:
        atomic_num_colormap[idx] = colors[idx]
        

def plot_graph(node_labels,A,ax=None,pos='kamada',frame=False,edge_weight=False, node_size=200, colors=atomic_num_colormap):
    
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

import rdkit.Chem as Chem

valid_atomic_nums = [6, 8, 7, 17, 16, 9, 35, 15, 53, 14, 11, 33, 80, 50, 5, 20, 19, 30, 26, 34, 13, 29, 12, 82, 24, 27, 28, 56, 78, 25, 52, "ukn"]
valid_bond_types = ["NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "ukn"]
bond_map = {
    "SINGLE": Chem.BondType.SINGLE,
    "DOUBLE": Chem.BondType.DOUBLE,
    "TRIPLE": Chem.BondType.TRIPLE,
    "AROMATIC": Chem.BondType.AROMATIC
}

def graph_to_mol(node_labels, edge_labels, sanitize=True):
    mol = Chem.RWMol()
    atom_indices = []

    # --- add atoms ---
    for label in node_labels:
        atomic_num = valid_atomic_nums[label]
        if atomic_num == "ukn":
            atomic_num = 6  # default to carbon
        atom = Chem.Atom(int(atomic_num))
        idx = mol.AddAtom(atom)
        atom_indices.append(idx)

    n = len(node_labels)

    # --- add bonds ---
    for i in range(n):
        for j in range(i+1, n):  # upper triangle
            bond_label = edge_labels[i][j]
            bond_label = valid_bond_types[bond_label]
            if bond_label in bond_map:
                mol.AddBond(atom_indices[i], atom_indices[j], bond_map[bond_label])
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            return None
    return mol

def graph_to_smiles(node_labels, edge_labels):
    mol = graph_to_mol(node_labels, edge_labels, sanitize=True)
    smiles = Chem.MolToSmiles(mol)
    return smiles