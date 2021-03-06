import sys
import os
import json
import numpy as np
import time
import datetime
from queue import PriorityQueue as pq
import os
sys.path.insert(0, str(os.getcwd())+"/../../DNE/src/utils") 
from env import *
from data_handler import DataHandler as dh

def loop(params, G, embeddings, weights, metric, output_path, graph_without_nodes=[]):

    mapp = list(range(G.number_of_nodes()))
    rmapp = list(range(G.number_of_nodes()))


    params_dynamic = params["dynamic_embedding"]
    K = params_dynamic["num_sampled"]

    def cal_delta(num_new):
        num_pre = G.number_of_nodes() - num_new
        for u, v in G.edges():
            if u >= num_pre or v >= num_pre:
                continue
            um, vm = mapp[u], mapp[v]
            delta_real = np.matmul(embeddings[[um]], weights[[vm]].T)[0, 0]
            G[u][v]['delta'] = delta_real - np.log(
                    float(G[u][v]['weight'] * G.graph['degree']) / float(G.node[u]['in_degree'] * G.node[v]['out_degree'])) + np.log(K)


    def rank_nodes(num_new):
        num_pre = G.number_of_nodes() - num_new  ## we don't delete nodes, only edges, so 0
        num_modify = params_dynamic['num_modify']
        if num_modify == 0:
            return

        nodes_of_edges = []
        for edge in G.edges():
            nodes_of_edges.append(edge[0])
            nodes_of_edges.append(edge[1])
        nodes_of_edges = list(set(nodes_of_edges))
        nodes_not_in_edges = [x for x in range(G.number_of_nodes()) if x not in nodes_of_edges]
        print("nodes not in edges:",nodes_not_in_edges)

        delta_list = [0.0] * num_pre
        for u, v in G.edges():
            if u >= num_pre or v >= num_pre:
                continue
            delta_list[u] += float(G[u][v]['weight']) * abs(G[u][v]['delta'])
            delta_list[v] += float(G[u][v]['weight']) * abs(G[u][v]['delta'])

        for u in G:
            if u >= num_pre:
                continue
            if u not in nodes_not_in_edges:
                delta_list[u] /= (G.node[u]['in_degree'] + G.node[u]['out_degree'])

        q = pq()
        for u in G:
            if u not in nodes_not_in_edges:
                if u >= num_pre:
                    continue
                if q.qsize() < num_modify:
                    q.put_nowait((delta_list[u], u))
                    continue
                items = q.get_nowait()
                if items[0] < delta_list[u]:
                    q.put_nowait((delta_list[u], u))
                else:
                    q.put_nowait(items)

        idx = num_pre - 1
        while not q.empty():
            u = q.get_nowait()[1]
            um = mapp[u]
            v = rmapp[idx]
            mapp[u] = idx
            rmapp[idx] = u
            mapp[v] = um
            rmapp[um] = v
            embeddings[[um, idx], :] = embeddings[[idx, um], :]
            weights[[um, idx], :] = weights[[idx, um], :]
            idx -= 1

    def reset(num_new):
        num_modify = params_dynamic['num_modify']
        num_pre = G.number_of_nodes() - num_new
        embeddings[[range(num_pre)], :] = embeddings[rmapp, :]
        weights[[range(num_pre)], :] = weights[rmapp, :]


    module_dynamic_embedding = __import__(
            "dynamic_embedding." + params_dynamic["func"],
            fromlist = ["dynamic_embedding"]).NodeEmbedding

    time_path = output_path + "_time"
    dynamic_embeddings = []
    
    num_new = 0
    cal_delta(num_new)
    rank_nodes(num_new)
    ne = module_dynamic_embedding(params_dynamic, embeddings, weights, G, mapp, rmapp, num_new)
    embeddings, weights = ne.train()

    return embeddings, weights
