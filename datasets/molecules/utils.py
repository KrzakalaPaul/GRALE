import numpy as np
from rdkit.Chem import MolFromSmiles
from rdkit import RDLogger
from scipy.sparse import csgraph

def safe_index(lst, item):
    '''
    If item in list: return idx of item
    Elif "ukn" in list: return idx of "ukn"
    Else: return None
    '''
    if item in lst:
        return lst.index(item)
    elif "ukn" in lst:
        return lst.index("ukn")
    else:
        return None

def smiles2graph(smiles, n_max_nodes, valid_atomic_nums, valid_bond_types, verbose=False):
    
    RDLogger.DisableLog('rdApp.*')
    
    mol = MolFromSmiles(smiles)
    if mol is None or  len(mol.GetAtoms()) > n_max_nodes:
        if verbose:
            print(f"Invalid molecule or too many atoms: {smiles}")
        return None
    
    node_labels = np.zeros((n_max_nodes), dtype=np.uint8)
    node_mask = np.ones((n_max_nodes), dtype=bool)
    
    for i, atom in enumerate(mol.GetAtoms()):
        label = safe_index(valid_atomic_nums, atom.GetAtomicNum())
        if label is None:
            if verbose:
                print(f"Invalid atom: {atom.GetAtomicNum()} in {smiles}")
            return None
        node_labels[i] = label
        node_mask[i] = False
    
    edge_labels = np.zeros((n_max_nodes,n_max_nodes), dtype=np.uint8)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bondtype = str(bond.GetBondType())
        label = safe_index(valid_bond_types, bondtype)
        if label is None:
            if verbose:
                print(f"Invalid bond: {bondtype} in {smiles}")
            return None
        edge_labels[i, j] = label
        edge_labels[j, i] = label

    adjacency_matrix = (edge_labels > 0).astype(int)
    adjacency_matrix = csgraph.csgraph_from_dense(adjacency_matrix)
    SP_matrix = csgraph.shortest_path(adjacency_matrix, directed=False, unweighted=True)
    SP_matrix[SP_matrix == np.inf] = n_max_nodes
    SP_matrix = SP_matrix.astype(np.uint8)

    graph = {
        'node_mask': node_mask,
        'node_labels': node_labels,
        'edge_labels': edge_labels,
        'SP_matrix': SP_matrix
    }
        
    return graph
