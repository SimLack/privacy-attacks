import sys
import os
import json
import numpy as np
import time
import datetime

#sys.path.insert(0, config_gem_path + "/DNE/src/utils")
#import env
#from data_handler import DataHandler as dh
#from utils.env import *
#from utils.data_handler import DataHandler as dh

def init(params, metric, output_path, number_of_nodes, config_gem_path):
    sys.path.insert(0, config_gem_path + "/DNE/src/utils")
    import env
    from data_handler import DataHandler as dh
    # load graph structure
    def load_data(params,number_of_nodes,config_gem_path):
        # load data-path from changed file 
        print("config_gem_path is:",config_gem_path)
        in_file = open(config_gem_path + 'DNE/src/utils/env.py', 'r')
        lines = in_file.readlines()
        data_path = lines[5].split("= ")[1].replace('"','').replace("\n","")
        print("line:",data_path)
        params["network_file"] = os.path.join(data_path, params["network_file"])
        print("params networkfile")
        print(params["network_file"])
        print("params func:",params["func"])
        print("params are:",params)
        G = getattr(dh, params["func"])(params,number_of_nodes)
        return G

    print("[] Loading data...")
    G = load_data(params["load_data"],number_of_nodes, config_gem_path)
    print(type(G))
    print("[+] Loaded data!")
    print("#nodes:",len(G.nodes()))

    """
    neighbors = []
    for edge in G.edges():
        if edge == 
        if removed_node in edge:
            neighbors.append(edge[0] if edge[1] == removed_node else edge[1]) 
    print(neighbors)
    """
    #print(G.node[0])
    #print("G.graph degree")
    #print(G.graph["degree"])
    print("[] Initializing embeddings with the original network...")
    module_embedding = __import__(
            "init_embedding." + params["init_train"]["func"], fromlist = ["init_embedding"]).NodeEmbedding
    ne = module_embedding(params["init_train"], G)
    embeddings, weights = ne.train()
    print("[+] Finished initializing!")

    #"""
    with open(output_path + "_init", "w") as f:
        f.write(json.dumps({"embeddings": embeddings.tolist(), "weights": weights.tolist()}))
    #metric(embeddings)
    #"""
    return G, embeddings, weights

