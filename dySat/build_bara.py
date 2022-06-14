# just for look up

import networkx as nx
import pathlib
import sys
import numpy as np

def delete_node_edges(graph, removed_node: int):
    # first check if node is in graph
    assert(removed_node in graph.nodes().copy())
    # and that node has edges
    assert(len([1 for edge in graph.edges() if removed_node in edge]) > 0)
    print("nach assert")
    neighbors = []
    """
    for edge in self._edges:
        if removed_node in edge:
            neighbors.append(edge[0] if edge[1] == removed_node else edge[1]) 
    print(neighbors)
    """
    # define new edges
    new_edges = list(filter(lambda edge: edge[0] != removed_node and edge[1] != removed_node, graph.edges()))
    #print("new edges",new_edges)
    
    # nodes stay the same
    new_nodes = graph.nodes().copy()
    #print("new_nodes",new_nodes)
    new_graph = nx.Graph()
    new_graph.add_nodes_from(new_nodes)
    new_graph.add_edges_from(new_edges)
    #print("new_graph",new_graph.nodes())
    return new_graph


print(int(sys.argv[1]))
barabasi = nx.generators.random_graphs.barabasi_albert_graph(int(sys.argv[1]),5,1)
barabasi1 = barabasi.copy()
barabasi1 = delete_node_edges(barabasi1,0)
print("barabasi")
print(barabasi)
print("barabasi1")
print(barabasi1)
nx.write_edgelist(barabasi,("bara"+(sys.argv[1])),data=False)
nx.write_edgelist(barabasi1,("bara1"+(sys.argv[1])),data=False)
# to .npz
print(type(barabasi))
#np.savez("bara100",nodes=barabasi.nodes(),edges=barabasi.edges())
#np.savez("bara1001",nodes=barabasi1.nodes(),edges=barabasi1.edges())
print("hi")
print(type(barabasi.edges()))
print(type(barabasi1.edges()))
#np.savez("baraz",graph=barabasi.nodes())
