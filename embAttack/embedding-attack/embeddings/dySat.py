import sys
import config
import subprocess

sys.path.insert(0, config.GEM_PATH + "embAttack/embedding-attack/embeddings")
from link_prediction import create_data_splits

sys.path.insert(0, config.GEM_PATH + "dySat/")
import embeddings.embedding
import abc
import networkx as nx
import graphs.graph_class as gc
import memory_access as sl
import pandas as pd
import numpy as np
# for dySat
import typing
import pathlib
import os
import json

## for sanity check:
#sys.path.insert(0, config.GEM_PATH + "dySat/")
from eval.link_prediction import evaluate_classifier, write_to_csv
import scipy.sparse as sp

class DySat(embeddings.embedding.Embedding):  # metaclass=abc.ABCMeta

    def __init__(self,dim:int=128):
        pass 
        #self.__is_static = is_static
    
    @staticmethod
    def init_dySat():
        return DySat()

    def __str__(self):
        return f'DySat'

    def short_name(self):
        return "DySat" 


    def train_embedding(self, graph: gc.Graph, save_info: sl.MemoryAccess, removed_nodes: [int], graph_without_nodes: [int],
            num_of_embeddings: int,savefilesuffix:str=None):
        nx_g = graph.to_networkx()
        np.testing.assert_array_equal(nx_g.nodes(), graph.nodes())
        # save graph as .npz for DySat
        graph_name = "nx_g_rm_"+str(removed_nodes)+"_gwn_"+str(graph_without_nodes)
        print("save temp graph in :",save_info.BASE_PATH+"temp_graphs/")
        np.savez(save_info.BASE_PATH+"temp_graphs/"+graph_name.replace(", ]","]").replace(", ","x"),nodes=nx_g.nodes(),edges=nx_g.edges())

        for iteration in range(num_of_embeddings):
            if save_info.has_embedding(removed_nodes=removed_nodes,iteration=iteration,graph_without_nodes=graph_without_nodes):
                continue
            print(f"embedding iteration number:{iteration+1}/{num_of_embeddings}.")
            Y = self._learn_embedding(iteration=iteration,removed_nodes=removed_nodes,graph_without_nodes=graph_without_nodes,retrain=False,graph_name=graph_name, graph=nx_g, save_info=save_info)
            emb = pd.DataFrame(Y, index=graph.nodes())
            save_info.save_embedding(removed_nodes=removed_nodes, iteration=iteration, embedding=emb, graph_without_nodes=graph_without_nodes)

    def _learn_embedding(self,graph:gc.Graph,iteration:int,removed_nodes:[int],graph_without_nodes:[int], save_info, retrain:bool=False,graph_name:str=None,command=None):

        if not retrain:
            print("graphname:",graph_name.replace(", ]","]").replace(", ","x"))

            if command==None:
                command = f'python run_script.py --dataset myBara --base_model IncSAT --batch_size 256 --min_time 2 --max_time 2 --epochs 125 --neg_sample_size 20 --walk_len 5 --learning_rate 0.01 --weight_decay 0.00001 --spatial_drop 0.2 --graph1 '+ str(graph_name.replace(", ]","]").replace(", ","x") + " --save_graphs_embs_in "+ save_info.BASE_PATH)
            print(command)
            current_directory = os.getcwd()
            os.chdir(config.GEM_PATH + "dySat")
            subprocess.call(command,shell=True)
            os.chdir(current_directory)
            emb = np.load(save_info.BASE_PATH+"temp_embeddings/_1_"+graph_name.replace(", ]","]").replace(", ","x")+".npy")


            if len(graph_without_nodes) > 0:
                dlnode = graph_without_nodes[-1]
                emb[dlnode,:] = 0

            # change for linkpred to
            #emb = np.load(config.GEM_PATH + "embAttack/embedding-attack/link_prediction/link_pred/my_embs_dySat/_1_"+graph_name.replace(", ]","]").replace(", ","x")+".npy")
        else:
            graph_name_original = ("".join(graph_name.rsplit(str(graph_without_nodes[-1]),1)))
            command = f'python run_script.py --dataset myBara --base_model IncSAT --batch_size 256 --min_time 2 --max_time 3 --epochs 125 --neg_sample_size 20 --walk_len 5 --learning_rate 0.01 --weight_decay 0.00001 --spatial_drop 0.2 --graph1 '+ str(graph_name_original.replace(", ]","]").replace(", ","x"))+f' --graph2 '+ str(graph_name.replace(", ]","]").replace(", ","x") + " --save_graphs_embs_in "+ save_info.BASE_PATH)
            current_directory = os.getcwd()
            os.chdir(config.GEM_PATH + "dySat")
            subprocess.call(command,shell=True)
            os.chdir(current_directory)
            emb = np.load(save_info.BASE_PATH+"temp_embeddings/_1_"+graph_name_original.replace(", ]","]").replace(", ","x")+"_2_"+graph_name.replace(", ]","]").replace(", ","x")+".npy")

            dlnode = graph_without_nodes[-1]
            emb[dlnode,:] = 0


            # change for linkpred to
            #emb = np.load(config.GEM_PATH + "embAttack/embedding-attack/link_prediction/link_pred/my_embs_dySat/_1_"+graph_name_original.replace(", ]","]").replace(", ","x")+"_2_"+graph_name.replace(", ]","]").replace(", ","x")+".npy")
        ## sanity check: link-pred with hadamard and trained classifier
        #"""
        print("graph is:",type(graph))
        print("shape emb is:",np.shape(emb))

        print(len(graph.edges()))
        train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = create_data_splits(nx.adjacency_matrix(graph),nx.adjacency_matrix(graph))
        val_results, test_results, _, _ = evaluate_classifier(train_edges, train_edges_false, val_edges,val_edges_false, test_edges, test_edges_false, emb, emb)
        if not retrain:
            print(">>> val train <<< :",val_results['HAD'][0])
            print(">>> test train <<< :",test_results['HAD'][0])
        else:
            print(">>> val retrain <<< :",val_results['HAD'][0])
            print(">>> test retrain <<< :",test_results['HAD'][0])

        #"""
        return emb




    # retrain with deleted node(-edges)
    def retrain_embedding(self, graph: gc.Graph, save_info: sl.MemoryAccess, removed_nodes: [int], graph_without_nodes: [int],
            num_of_embeddings: int,loadfilesuffix:str=None,retraining:bool=True):
        nx_g = graph.to_networkx()
        # save graph as .npz for DySat
        graph_name = "nx_g_rm_"+str(removed_nodes)+"_gwn_"+str(graph_without_nodes)
        np.savez(save_info.BASE_PATH+"temp_graphs/"+graph_name.replace(", ]","]").replace(", ","x"),nodes=nx_g.nodes(),edges=nx_g.edges())
        for iteration in range(num_of_embeddings):
            if save_info.has_embedding(removed_nodes=removed_nodes,iteration=iteration,graph_without_nodes=graph_without_nodes):
                continue
            print(f"(retraining) embedding iteration number:{iteration+1}/{num_of_embeddings}.")
            if retraining:
                Y = self._learn_embedding(iteration=iteration,removed_nodes=removed_nodes,graph_without_nodes=graph_without_nodes,retrain=True,graph_name=graph_name,graph=nx_g, save_info=save_info)
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


