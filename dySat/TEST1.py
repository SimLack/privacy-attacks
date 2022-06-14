import networkx as nx
import numpy as np

with np.load("data/Enron_new/graphs.npz",allow_pickle=True,encoding='bytes') as data:
    for key,arr in data.items():
        print(key,arr)
        print(1 in arr[0])
