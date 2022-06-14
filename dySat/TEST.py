import networkx as nx
import numpy as np

G = nx.Graph()
#with np.load("data/myBara/graphs.npz") as data:
with np.load("baraz.npz") as data:
    for key, arr in data.items():
        print(key,arr)
        #"""
        if key == 'nodes':
            G.add_nodes_from(arr)
        if key == 'edges':
            G.add_edges_from(arr)
        #"""
print("G:")
print(G)
print(G.edges())
