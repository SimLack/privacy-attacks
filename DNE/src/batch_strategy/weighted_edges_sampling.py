import sys
import os
import re
import networkx as nx
import random
import numpy as np

from src.batch_strategy.alias_table_sampling import AliasTable as at

class BatchStrategy(object):
    # G is a DiGraph with edge weights
    def __init__(self, G, params = None):
        self.edges = G.edges()
        #print("G edges in weightedEdgesSampling:",G.edges())
        probs = []
        for it in G.edges():
            probs.append(G[it[0]][it[1]]['weight'])
        self.sampling_handler = at(probs)

    def get_batch(self, batch_size):
        batch_x = []
        batch_y = []
        for _ in range(batch_size):
            idx = self.sampling_handler.sample()
            #print("self edges idx 0")
            #print("idx:",idx)
            #print(list(self.edges))
            #print(list(self.edges)[idx][0])
            batch_x.append((self.edges)[idx][0])
            batch_y.append([(self.edges)[idx][1]])
        return np.array(batch_x, dtype = np.int32), np.array(batch_y, dtype = np.int32)

if __name__ == "__main__":
    G = nx.DiGraph()
    G.add_edge(1, 2, weight = 1)
    G.add_edge(3, 4, weight = 2)
    bs = BatchStrategy(G)
    batch_x, batch_y = bs.get_batch(1000)
    m = {}
    for it in batch_x:
        if it in m:
            m[it] += 1
        else:
            m[it] = 1
    print(m)
    #print batch_x
    #print batch_y
