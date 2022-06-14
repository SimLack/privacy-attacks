import sys
import config
sys.path.insert(0, config.GEM_PATH + "embAttack/embedding-attack/embeddings")
from link_prediction import create_data_splits
sys.path.insert(0, config.GEM_PATH + "DHPE/")
import embeddings.embedding
import abc
import networkx as nx
import graphs.graph_class as gc
import memory_access as sl
import pandas as pd
import numpy as np
import typing
import pathlib
import os
import json
import subprocess
from tqdm import tqdm
from datetime import datetime
from scipy import sparse
import scipy.sparse.linalg as lg
import csv
import scipy.io

## for sanity check:
sys.path.insert(0, config.GEM_PATH + "dySat/")
from eval.link_prediction import evaluate_classifier, write_to_csv
import scipy.sparse as sp
from time import time


class DHPE(embeddings.embedding.Embedding):  # metaclass=abc.ABCMeta

    def __init__(self,dim:int=0):
        self.dim=dim
        #self.__is_static = is_static
    
    @staticmethod
    def init_DHPE(dim:int=0):
        return DHPE(dim)

    def __str__(self):
        return f'DHPE-dim={self.dim}'

    def short_name(self):
        return "DHPE" 


    def train_embedding(self, graph: gc.Graph, save_info: sl.MemoryAccess, removed_nodes: [int], graph_without_nodes: [int],
            num_of_embeddings: int,savefilesuffix:str=None,eng=None,beta=None):
        nx_g = graph.to_networkx()
        np.testing.assert_array_equal(nx_g.nodes(), graph.nodes())

        for iteration in range(num_of_embeddings):
            if save_info.has_embedding(removed_nodes=removed_nodes,iteration=iteration,graph_without_nodes=graph_without_nodes):
                print("embedding already trained")
                continue
            print(f"embedding iteration number:{iteration+1}/{num_of_embeddings}.")
            Y = self._learn_embedding(graph=nx_g,iteration=iteration,removed_nodes=removed_nodes,graph_without_nodes=graph_without_nodes,retrain=False,eng=eng,save_info=save_info)
            emb = pd.DataFrame(Y, index=graph.nodes())
            save_info.save_embedding(removed_nodes=removed_nodes, iteration=iteration, embedding=emb, graph_without_nodes=graph_without_nodes)

    def _learn_embedding(self, graph, iteration:int,removed_nodes:[int],graph_without_nodes:[int],retrain:bool=False,eng=None,save_info: sl.MemoryAccess=None):
        if not retrain:
            my_dim = int(self.dim)
            nx_g = graph
            np_adj = nx.to_numpy_matrix(nx_g)
            np.save(save_info.BASE_PATH+"/temp_embeddings/"+"_rm_"+str(removed_nodes)+"_iter_"+str(iteration)+"_gwn_"+str(graph_without_nodes)+"_np_adj_",np_adj)
            np_adj_str = save_info.BASE_PATH+"/temp_embeddings/"+"_rm_"+str(removed_nodes)+"_iter_"+str(iteration)+"_gwn_"+str(graph_without_nodes)
            pd.DataFrame(np_adj).to_csv(np_adj_str+"_np_adj_")
            np.save(save_info.BASE_PATH+"/temp_embeddings/"+"_rm_"+str(removed_nodes)+"_iter_"+str(iteration)+"_gwn_"+str(graph_without_nodes)+"_my_dim_",my_dim)
            static_start = datetime.now()
            # beta only set in embed_static.m, here only placeholder
            beta = 0 
            (eng.embed_static(np_adj_str, my_dim, beta, nargout=0))
            U = np.array(list(csv.reader(open(save_info.BASE_PATH+"/temp_embeddings/"+"_rm_"+str(removed_nodes)+"_iter_"+str(iteration)+"_gwn_"+str(graph_without_nodes)+"_U_.csv", "r"), delimiter=","))).astype("float")
            Sigma = np.array(list(csv.reader(open(save_info.BASE_PATH+"/temp_embeddings/"+"_rm_"+str(removed_nodes)+"_iter_"+str(iteration)+"_gwn_"+str(graph_without_nodes)+"_Sigma_.csv", "r"), delimiter=","))).astype("float")
            V = np.array(list(csv.reader(open(save_info.BASE_PATH+"/temp_embeddings/"+"_rm_"+str(removed_nodes)+"_iter_"+str(iteration)+"_gwn_"+str(graph_without_nodes)+"_V_.csv", "r"), delimiter=","))).astype("float")
            Ma = np.array(list(csv.reader(open(save_info.BASE_PATH+"/temp_embeddings/"+"_rm_"+str(removed_nodes)+"_iter_"+str(iteration)+"_gwn_"+str(graph_without_nodes)+"_Ma_.csv", "r"), delimiter=","))).astype("float")
            Mb = np.array(list(csv.reader(open(save_info.BASE_PATH+"/temp_embeddings/"+"_rm_"+str(removed_nodes)+"_iter_"+str(iteration)+"_gwn_"+str(graph_without_nodes)+"_Mb_.csv", "r"), delimiter=","))).astype("float")
            
            U = np.asarray(U)
            V = np.asarray(V)
            Sigma = np.diag(Sigma)
            emb = np.dot(U,(np.diag(np.sqrt(Sigma))))

            SKatz = np.dot(np.linalg.inv(Ma), Mb)
            p_d_p_t = np.dot(U, np.dot(np.diag(Sigma),V.T))
            eig_err = np.linalg.norm(p_d_p_t - SKatz)
            print('SVD error (low rank): %f' % eig_err)
            np.save(save_info.BASE_PATH+"/temp_embeddings/"+"_rm_"+str(removed_nodes)+"_iter_"+str(iteration)+"_gwn_"+str(graph_without_nodes)+"_emb_",emb)
        else: # retraining = True
            print("Retraining is chosen")
            nx_g = graph
            np_adj_old = np.load(save_info.BASE_PATH+"/temp_embeddings/"+"_rm_"+str(removed_nodes)+"_iter_"+str(iteration)+"_gwn_"+str(removed_nodes)+"_np_adj_.npy")
            np_adj = (nx.to_numpy_matrix(nx_g))
            ## (new - old) should be best choice; alternatively abs(new-old) or (old-new), but it is not stated in the paper
            np_adj_delta = (np_adj - np_adj_old)
            #np_adj_delta = abs(np_adj - np_adj_old)
            #np_adj_delta = (np_adj_old - np_adj)
            np_adj_delta_str = save_info.BASE_PATH+"/temp_embeddings/"+"_rm_"+str(removed_nodes)+"_iter_"+str(iteration)+"_gwn_"+str(removed_nodes)
            pd.DataFrame(np_adj_delta).to_csv(np_adj_delta_str+"_np_adj_delta_")
            dim = np.load(save_info.BASE_PATH+"/temp_embeddings/"+"_rm_"+str(removed_nodes)+"_iter_"+str(iteration)+"_gwn_"+str(removed_nodes)+"_my_dim_.npy")
            dim = int(dim)
            # use matlab for the computations
            eng.embed_update(np_adj_delta_str, dim, nargout=0)

            # matlab saves U,S,V in files which we read now
            U = np.array(list(csv.reader(open(save_info.BASE_PATH+"/temp_embeddings/"+"_rm_"+str(removed_nodes)+"_iter_"+str(iteration)+"_gwn_"+str(removed_nodes)+"_nU_.csv", "r"), delimiter=","))).astype("float")
            Sigma = np.array(list(csv.reader(open(save_info.BASE_PATH+"/temp_embeddings/"+"_rm_"+str(removed_nodes)+"_iter_"+str(iteration)+"_gwn_"+str(removed_nodes)+"_nSigma_.csv", "r"), delimiter=","))).astype("float")
            V = np.array(list(csv.reader(open(save_info.BASE_PATH+"/temp_embeddings/"+"_rm_"+str(removed_nodes)+"_iter_"+str(iteration)+"_gwn_"+str(removed_nodes)+"_nV_.csv", "r"), delimiter=","))).astype("float")
            U = np.asarray(U)
            V = np.asarray(V)
            Sigma = np.asarray(Sigma)
            Sigma = np.diag(Sigma)
            # compute n x d Embedding = [sqrt(sigma1*v^l_1,...] as Eq. 3 in paper
            emb = np.dot(U,(np.diag(np.sqrt(np.sqrt(Sigma)))))
            # compute new Ma, Mb to generate SKatz matrix for computing SVD error
            # use same beta as before
            beta = scipy.io.loadmat(save_info.BASE_PATH+"/temp_embeddings/"+"_rm_"+str(removed_nodes)+"_iter_"+str(iteration)+"_gwn_"+str(removed_nodes)+"_beta_.mat")['beta'][0][0]
            MaNew = np.eye(len(graph.nodes())) - beta * np_adj
            MbNew = beta * np_adj
            SKatz = np.dot(np.linalg.inv(MaNew), MbNew)
            tmp = np.dot(U, np.diag(Sigma))
            p_d_p_t = np.dot(tmp,V.T)
            eig_err = np.linalg.norm(p_d_p_t - SKatz)
            print('SVD error (low rank): %f' % eig_err)


        ## sanity check: link-pred with hadamard and trained classifier
        #"""
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
            num_of_embeddings: int,loadfilesuffix:str=None, retraining:bool=True,eng=None):
        nx_g = graph.to_networkx()
        np.testing.assert_array_equal(nx_g.nodes(), graph.nodes())
        for iteration in range(num_of_embeddings):
            if save_info.has_embedding(removed_nodes=removed_nodes,iteration=iteration,graph_without_nodes=graph_without_nodes):
                print("embedding already trained")
                continue
            print(f"(retraining) embedding iteration number:{iteration+1}/{num_of_embeddings}.")

            # if retraining is activated, save resulting embedding
            # if it is deactivated, load and save the original embedding in the same way
            if retraining:
                Y = self._learn_embedding(graph=nx_g,iteration=iteration,removed_nodes=removed_nodes,graph_without_nodes=graph_without_nodes,retrain=True,eng=eng, save_info=save_info)
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



