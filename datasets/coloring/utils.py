import numpy as np
import networkx as nx
from random import choice
from functools import partial
from math import sqrt
from scipy.ndimage import gaussian_filter
from scipy.sparse import csgraph

# Initialize colors
n_colors = 4
colors =  np.array([[1/4,1/4,3/4],
                    [1/4,3/4,1/4],
                    [3/4,1/4,1/4],
                    [1,4/5,2/5]])

def color_graph(graph, n_colors):
    valid_coloring = False
    n_nodes = graph.number_of_nodes()
    while not(valid_coloring):
        # Try a new random coloring
        coloring = {}
        valid_coloring = True
        for m in range(n_nodes):
            valid_colors = set([k for k in range(n_colors)])
            for n in graph.neighbors(m):
                if n in coloring:
                    valid_colors.discard(coloring[n])
            if len(valid_colors) == 0:
                valid_coloring = False
                # No valid color found, restart
                break
            else:
                coloring[m] = choice(list(valid_colors))
    return coloring

def coloring_sample(n_min_nodes=2,
                    n_max_nodes=10,
                    n_pixels=32,
                    sigma_filter=0.4,
                    sigma_noise=0.02,
                    node_weight_treshold=None,
                    edge_weight_treshold=None,
                    precompute_shortest_paths=False):
    
    # To allow for recursion
    restart = partial(coloring_sample,
                      n_min_nodes=n_min_nodes,
                      n_max_nodes=n_max_nodes,
                      n_pixels=n_pixels,
                      sigma_filter=sigma_filter,
                      sigma_noise=sigma_noise,
                      node_weight_treshold=node_weight_treshold,
                      edge_weight_treshold=edge_weight_treshold,
                      precompute_shortest_paths=precompute_shortest_paths)

    # Sample the size of the graph
    n_nodes = np.random.randint(n_min_nodes,n_max_nodes+1)
    
    # Compute normalizations 
    node_weight_normalization = n_pixels**2
    edge_weight_normalization = n_pixels # sqrt(shape[0]**2+shape[1]**2) large diagonal as the maximum possible edge weight

    # Random centroids
    centroids = np.random.uniform(0,1,(2,n_nodes))
    
    # Assign each pixel to the closest centroid
    X = np.linspace(0,1,n_pixels)
    Y = np.linspace(0,1,n_pixels)
    dists = np.abs(X[:,None,None] - centroids[None,0,:]) + np.abs(Y[None,:,None] - centroids[1,None,:])
    closest = np.argmin(dists,axis=2)

    # Count pixels per node
    unique, counts = np.unique(closest, return_counts=True)
    
    # If some node is empty, resample
    if len(unique) != n_nodes:
        print('Discarding because empty node')
        return restart()
        
    # Init the graph with nodes
    graph = nx.Graph()
    graph.add_nodes_from([(n,{'pos':centroids[:,n], 'weight':counts[n]/node_weight_normalization}) for n in range(n_nodes)])
    if node_weight_treshold != None:
        print('Using node weight treshold:', node_weight_treshold)
        for n in range(n_nodes):
            if graph.nodes[n]['weight'] < node_weight_treshold:
                print('Discarding because node too small')
                return restart()

    # Add the edges based on pixel adjacency
    for i in range(n_pixels):
        for j in range(n_pixels):
            if j+1<n_pixels:
                u = closest[i,j+1]
                v = closest[i,j]
                if u!=v:
                    if graph.has_edge(u,v):
                        graph[u][v]['weight']+=1/edge_weight_normalization
                    else:
                        graph.add_edge(u, v, weight = 1/edge_weight_normalization)

            if i+1<n_pixels:
                u = closest[i+1,j]
                v = closest[i,j]
                if u!=v:
                    if graph.has_edge(u,v):
                        graph[u][v]['weight']+=1/edge_weight_normalization
                    else:
                        graph.add_edge(u, v, weight = 1/edge_weight_normalization)

    # if using a edge weight, we ensure it is in [0,1]
    if edge_weight_treshold != None:
        for (u,v) in graph.edges:
            if graph[u][v]['weight'] > 1:
                print('Discarding because edge too big')
                return restart()
            
    # Coloring the graph
    coloring = color_graph(graph, n_colors)
    nx.set_node_attributes(graph, coloring, 'color')
    
    # Convert to matrices
    node_positions = np.array([graph.nodes[n]['pos'] for n in range(n_nodes)])
    node_weights = np.array([graph.nodes[n]['weight'] for n in range(n_nodes)])
    node_colors = np.array([graph.nodes[n]['color'] for n in range(n_nodes)])
    edge_weights = np.zeros((n_nodes, n_nodes))
    for u, v, data in graph.edges(data=True):
        edge_weights[u, v] = data['weight']
        edge_weights[v, u] = data['weight']
    adjacency_matrix = (edge_weights>edge_weight_treshold).astype(float) if edge_weight_treshold is not None else (edge_weights>0).astype(float)
    if precompute_shortest_paths:
        adjacency_matrix_ = csgraph.csgraph_from_dense(adjacency_matrix)
        SP_matrix = csgraph.shortest_path(adjacency_matrix_, directed=False, unweighted=True)
        SP_matrix = SP_matrix.astype(np.uint8)
    else:
        SP_matrix = None
        
    graph = {'node_positions': node_positions,
             'node_weights': node_weights,
             'node_colors': node_colors,
             'edge_weights': edge_weights,
             'adjacency_matrix': adjacency_matrix,
             'SP_matrix': SP_matrix}
        
    # Convert to image
    node_colors = np.vstack([colors[coloring[n]] for n in range(n_nodes)])
    img = np.take(node_colors,closest,axis=0)
    img = gaussian_filter(img,sigma_filter, mode = 'nearest')
    img += np.random.normal(0,sigma_noise,size=img.shape)
    img = np.clip(img,0,1)
    img = (img*255).astype(np.uint8)
        
    return img, graph

def plot_coloring_img(img,ax,frame=False):
    image = img.astype(np.float32)/255
    ax.imshow(np.transpose(image,(1,0,2)),vmin=0,vmax=1,origin='lower')
    if frame:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_axis_off()

def plot_coloring_graph(graph,ax,frame=False):

    # Reconstruct NetworkX graph from adjacency matrix
    adjacency = graph['adjacency_matrix']
    graph_nx = nx.from_numpy_array(adjacency)
    
    # Set node positions and colors
    pos = {i: graph['node_positions'][i] for i in range(graph_nx.number_of_nodes())}
    color_map = [colors[c] for c in graph['node_colors']] 

    nx.draw_networkx_nodes(graph_nx,node_color="k",ax=ax, pos=pos)
    nx.draw_networkx_nodes(graph_nx, pos, node_size=200, node_color=color_map,ax=ax,alpha=1)
    [nx.draw_networkx_edges(graph_nx,pos=pos,edgelist=[(u,v)],alpha=1,width=2,ax=ax) for u,v in graph_nx.edges] #loop through edges and draw the

    ax.axis('equal')
    if frame:
        pass
    else:
        ax.axis('off')
    

if __name__ == "__main__":
    img, graph = coloring_sample(n_min_nodes=2,
                                n_max_nodes=8,
                                n_pixels=64,
                                sigma_filter=0.4,
                                sigma_noise=0.02,
                                node_weight_treshold=None,
                                edge_weight_treshold=None)

    import matplotlib.pyplot as plt
    fig, (ax_img, ax_graph) = plt.subplots(1,2,figsize=(8,4))
    plot_coloring_img(img, ax_img)
    plot_coloring_graph(graph, ax_graph)
    plt.show()