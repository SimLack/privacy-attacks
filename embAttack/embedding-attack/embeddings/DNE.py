import sys
import config
sys.path.insert(0, config.GEM_PATH + "embAttack/embedding-attack/embeddings")
from link_prediction import create_data_splits

sys.path.insert(0, config.GEM_PATH + "DNE/src/init")
sys.path.insert(0, config.GEM_PATH + "DNE/src/dynamic_loop")
sys.path.insert(0, config.GEM_PATH + "DNE/")

import embeddings.embedding
import abc
import networkx as nx
import graphs.graph_class as gc
import memory_access as sl
import pandas as pd
import numpy as np

from dne_init import init
from dne_loop import loop
from src.utils.data_handler import DataHandler as dh
import typing
import pathlib
import os
import json
#from src.utils.env import *

sys.path.insert(0, config.GEM_PATH + "dySat/")
# for link pred
from eval.link_prediction import evaluate_classifier, write_to_csv
import scipy.sparse as sp
from time import time

class DNE(embeddings.embedding.Embedding):  # metaclass=abc.ABCMeta

    def __init__(self,dim:int=128):
        self.dim=dim
        #self.__is_static = is_static
    
    @staticmethod
    def init_DNE(dim:int=128):
        return DNE(dim)

    def __str__(self):
        return f'DNE-dim={self.dim}'

    def short_name(self):
        return "DNE" 

    def train_embedding(self, graph: gc.Graph, save_info: sl.MemoryAccess, removed_nodes: [int], graph_without_nodes: [int],
            num_of_embeddings: int, dataset_is=None, savefilesuffix:str=None):
        # change file of DNE config to have proper file locations 
        in_file = open(config.GEM_PATH + 'DNE/src/utils/env.py', 'r')
        lines = in_file.readlines()
        change = 'DATA_PATH = "'+str(pathlib.Path().resolve().joinpath(save_info.BASE_PATH+'temp_graphs'))+'"\n'
        lines[5] = change
        in_file.close()
        out_file = open(config.GEM_PATH + 'DNE/src/utils/env.py','w')
        for line in lines:
            out_file.write(line)
        out_file.close()

        nx_g = graph.to_networkx()
        np.testing.assert_array_equal(nx_g.nodes(), graph.nodes())

        # Adjust myConf.json for proper number of nodes
        in_file = open(config.GEM_PATH +'embAttack/embedding-attack/myConf.json', 'r')
        data_file = in_file.read()
        data = json.loads(data_file)
        data['num_nodes'] = len(graph.nodes())
        data['init']['init_train']['num_nodes'] = len(graph.nodes())
        in_file.close()
        out_file = open(config.GEM_PATH +'embAttack/embedding-attack/myConf.json','w')
        out_file.write(json.dumps(data))
        out_file.close()

        # write graphs correctly as list
        nx.write_edgelist(G=nx_g,path=str(pathlib.Path().resolve().joinpath(save_info.BASE_PATH+"/temp_graphs/nx_gOLD")))
        f = open(str(pathlib.Path().resolve().joinpath(save_info.BASE_PATH+"/temp_graphs/nx_gOLD")),'r')
        temp = f.read()
        f.close()
        f = open(str(pathlib.Path().resolve().joinpath(save_info.BASE_PATH+"/temp_graphs/nx_gOLD")), 'w')
        f.write("{len(nx_g.nodes())}\n")
        f.write(temp)
        f.close()
        infile = str(pathlib.Path().resolve().joinpath(save_info.BASE_PATH+"/temp_graphs/nx_gOLD"))
        outfile = str(pathlib.Path().resolve().joinpath(save_info.BASE_PATH+"/temp_graphs/nx_g"))
        delete_list = ["{}"]
        with open(infile) as fin, open(outfile, "w+") as fout:
            for line in fin:
                for word in delete_list:
                    line = line.replace(word, "")
                fout.write(line)

        for iteration in range(num_of_embeddings):
            if save_info.has_embedding(removed_nodes=removed_nodes,iteration=iteration,graph_without_nodes=graph_without_nodes):
                print("embedding already trained")
                continue
            print(f"embedding iteration number:{iteration+1}/{num_of_embeddings}.")
            Y,weights = self._learn_embedding(iteration=iteration,removed_nodes=removed_nodes,graph_without_nodes=graph_without_nodes,retrain=False,graph=nx_g,save_info=save_info)
            emb = pd.DataFrame(Y, index=graph.nodes())
            save_info.save_embedding(removed_nodes=removed_nodes, iteration=iteration, embedding=emb, graph_without_nodes=graph_without_nodes)

    def _learn_embedding(self, iteration:int,removed_nodes:[int],graph_without_nodes:[int], save_info, retrain:bool=False,graph=None):
        params = dh.load_json_file(os.path.join(config.CONFIG_DIR,"myConf.json"))
        if not retrain:
            _,emb,weights = init(params["init"],"",str(pathlib.Path().resolve().joinpath(save_info.BASE_PATH+"/temp_weights/"))+"_"+str(removed_nodes)+"_"+str(iteration),len(graph.nodes()),config.GEM_PATH)

        else:
            paramsTemp = params['init']['load_data']
            # load data-path from changed file 
            in_file = open(config.GEM_PATH + 'DNE/src/utils/env.py', 'r')
            lines = in_file.readlines()
            data_path = lines[5].split("= ")[1].replace('"','').replace('\n','')
            paramsTemp['network_file'] = os.path.join(data_path, paramsTemp["network_file"])
            embeddings = dh.load_json_file(str(pathlib.Path().resolve().joinpath(save_info.BASE_PATH+"/temp_weights/"))+"_"+str(removed_nodes)+"_"+str(iteration)+"_init")["embeddings"]
            weights = dh.load_json_file(str(pathlib.Path().resolve().joinpath(save_info.BASE_PATH+"/temp_weights/"))+"_"+str(removed_nodes)+"_"+str(iteration)+"_init")["weights"]
            embeddings = np.asarray(embeddings)

            weights = np.asarray(weights)
            G = dh.load_unweighted_digraph(paramsTemp,len(graph.nodes()))
            emb,weights = loop(params["main_loop"], G, embeddings, weights, "",str(pathlib.Path().resolve().joinpath(save_info.BASE_PATH+"/temp_weights/"))+"_"+str(removed_nodes)+"_"+str(iteration)+"_retrained",graph_without_nodes=graph_without_nodes)
            with open(paramsTemp['network_file'],'r') as f:
                tmp = f.readlines()
            with open(paramsTemp['network_file'],'w') as f:
                f.writelines(tmp[:-len(graph_without_nodes)])


        ## sanity check: link-pred with hadamard and trained classifier
        #"""
        print("graph is:",type(graph))
        print("shape emb is:",np.shape(emb))

        print(len(graph.edges()))
        t = time()
        train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = create_data_splits(nx.adjacency_matrix(graph),nx.adjacency_matrix(graph))
        print("create train/test edges needs:",time()-t)
        val_results, test_results, _, _ = evaluate_classifier(train_edges, train_edges_false, val_edges,val_edges_false, test_edges, test_edges_false, emb, emb)
        if not retrain:
            print(">>> val train <<< :",val_results['HAD'][0])
            print(">>> test train <<< :",test_results['HAD'][0])
        else:
            print(">>> val retrain <<< :",val_results['HAD'][0])
            print(">>> test retrain <<< :",test_results['HAD'][0])

        #"""
        return emb,weights



    # retrain with deleted node(-edges)
    def retrain_embedding(self, graph: gc.Graph, save_info: sl.MemoryAccess, removed_nodes: [int], graph_without_nodes: [int],
            num_of_embeddings: int,loadfilesuffix:str=None,retraining:bool=True):
        nx_g = graph.to_networkx()
        np.testing.assert_array_equal(nx_g.nodes(), graph.nodes())
        nx.write_edgelist(G=nx_g,path=str(pathlib.Path().resolve().joinpath(save_info.BASE_PATH+"/temp_graphs/nx_gOLD")))
        f = open(str(pathlib.Path().resolve().joinpath(save_info.BASE_PATH+"/temp_graphs/nx_gOLD")),'r')
        temp = f.read()
        f.close()
        f = open(str(pathlib.Path().resolve().joinpath(save_info.BASE_PATH+"/temp_graphs/nx_gOLD")), 'w')
        f.write("{len(nx_g.nodes())}\n")
        f.write(temp)
        f.close()
        infile = str(pathlib.Path().resolve().joinpath(save_info.BASE_PATH+"/temp_graphs/nx_gOLD"))
        outfile = str(pathlib.Path().resolve().joinpath(save_info.BASE_PATH+"/temp_graphs/nx_g"))
        delete_list = ["{}","{'weight': 1.0}"]
        with open(infile) as fin, open(outfile, "w+") as fout:
            for line in fin:
                for word in delete_list:
                    line = line.replace(word, "")
                fout.write(line)


        for iteration in range(num_of_embeddings):
            if save_info.has_embedding(removed_nodes=removed_nodes,iteration=iteration,graph_without_nodes=graph_without_nodes):
                print("embedding already trained")
                continue

            print(f"(retraining) embedding iteration number:{iteration+1}/{num_of_embeddings}.")
            if retraining:
                Y, weights = self._learn_embedding(iteration=iteration,removed_nodes=removed_nodes,graph_without_nodes=graph_without_nodes,retrain=True,graph=nx_g,save_info=save_info)
                emb = pd.DataFrame(Y, index=graph.nodes())
            else:
                emb = save_info.load_embedding_to_df(removed_nodes=removed_nodes, iteration=iteration,graph_without_nodes=removed_nodes)

            save_info.save_embedding(removed_nodes=removed_nodes, iteration=iteration, embedding=emb, graph_without_nodes=graph_without_nodes)


    def load_embedding(self, graph: gc.Graph, removed_nodes: [int], save_info, iteration: int,
                       load_neg_results: bool = False):
        pass

    def continue_train_embedding(self, graph: gc.Graph,
                                 save_info, removed_nodes: [int],
                                 num_of_embeddings: int, model, emb_description: str = None,
                                 graph_description: str = None):
        pass

    def is_static(self):
        return self.__is_static


