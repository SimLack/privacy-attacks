import memory_access as sl
from graphs import graph_class as gc
from embeddings.embedding import Embedding
from experiments import exp_calc_features_from_embedding as cf
from classifier import train_sklearn_predictions as te
import my_utils as utils
from features import feature_type as ft
import config
import logging
import os
import numpy as np
import networkx as nx
from time import time
import sys
from pathlib import Path
import shutil

def train_embedding_per_graph(graph: gc.Graph, embedding: Embedding, save_info: sl.MemoryAccess,
                              num_of_embeddings: int = 30, num_of_test_evaluations_per_degree_level: int = 5,
                              num_of_training_graphs: int = 10, num_of_bins_for_tf: [int] = None,
                              run_experiments_on_embedding: bool = True,
                              feature_type: ft.FeatureType = ft.FeatureType.DIFF_BIN_WITH_DIM,args=None):

    assert(args != None)
    print("number graph edges:",len(graph.edges()))
    original_graph_nodes = graph.nodes()
    print("embedding in mydyn is:",str(embedding).split("-")[0])
    if str(embedding).split("-")[0] == "DHPE":
        import matlab.engine
        print(matlab.engine)
        eng = matlab.engine.start_matlab()
    # add folder for saving temporary embeddings:
    Path(save_info.BASE_PATH+"/temp_embeddings").mkdir(parents=True, exist_ok=True)
    Path(save_info.BASE_PATH+"/temp_graphs/").mkdir(parents=True, exist_ok=True)
    # only for DynGEM / DNE
    Path(save_info.BASE_PATH+"/temp_weights").mkdir(parents=True, exist_ok=True)

    retraining = save_info.retraining
    print("retraining is:",retraining)
    assert (num_of_embeddings == save_info.get_num_iterations())
    if num_of_bins_for_tf is None:
        num_of_bins_for_tf = [10]
    elif isinstance(num_of_bins_for_tf, int):
        num_of_bins_for_tf = [num_of_bins_for_tf]
    time_start = time()
    nx_g = graph.to_networkx()
    np.testing.assert_array_equal(nx_g.nodes(), graph.nodes())
    print("Train first embedding E....")
    if args.DHPE:
        embedding.train_embedding(graph=graph, save_info=save_info, removed_nodes=[],num_of_embeddings=num_of_embeddings, graph_without_nodes=[],eng=eng)
    else:
        embedding.train_embedding(graph=graph, save_info=save_info, removed_nodes=[],num_of_embeddings=num_of_embeddings, graph_without_nodes=[])


    first_started_embedding = save_info.get_list_of_available_embeddings(graph=graph, find_started_trainings=True)
    #"""
    tested_nodes = utils.sample_low_avg_high_degree_nodes(graph=graph,
                                                          quantity=num_of_test_evaluations_per_degree_level,
                                                          init_range=2, pref_list=first_started_embedding)

    # used to sample first and second nodes only
    """
    def get_sampled_nodes_early():
        second_nodes = {}
        for first_node in tested_nodes:
            graph_removed_one = graph.delete_node_edges(first_node)
            second_completed_diffs = save_info.get_list_of_available_embeddings(graph=graph_removed_one,
                                                                                      removed_first_node=first_node,
                                                                                      find_started_trainings=False,
                                                                                      graph_without_nodes=[])

            second_started_embedding = save_info.get_list_of_available_embeddings(graph=graph_removed_one,
                                                                                  removed_first_node=first_node,
                                                                                  find_started_trainings=True,
                                                                                  graph_without_nodes=[])

            #print(len(graph_removed_one.delete_node(first_node).nodes()))
            #print(len(list(filter(lambda x : x != first_node, graph_removed_one.nodes()))))
            second_tested_nodes = utils.sample_randomly_with_pref_list_without_splitting_nodes(
                #graph=graph_removed_one.delete_node_edges(first_node), pref_list=second_completed_diffs,
                graph=graph_removed_one.copy().delete_node(first_node), pref_list=second_completed_diffs,
                secondary_pref_list=second_started_embedding,
                all_list=list(filter(lambda x : x != first_node, graph_removed_one.nodes())),
                #all_list=graph_removed_one.nodes(),
                quantity=num_of_training_graphs)
            second_nodes[first_node] = second_tested_nodes
        print(f"tested_nodes = {list(tested_nodes)}")
        print(f"second_nodes = {second_nodes}")
        del graph_removed_one
        return second_nodes

    second_nodes = get_sampled_nodes_early()
    print("GET SAMPLED NODES EARLY done.")
    print("tested_nodes =",tested_nodes)
    print("second_nodes=",second_nodes)
    exit()
    #"""

    # here, fixed first and second nodes can be set.
    if args.fixed_nodes:
        # nodes for bara
        if args.BarabasiDataset:
            tested_nodes = [19, 13, 18, 12, 9, 17, 826, 5, 739, 10, 984, 934, 731, 1, 16]
            second_nodes = {19: [862, 604, 657, 829, 150, 120, 882, 626, 975, 469], 13: [428, 118, 809, 702, 982, 526, 539, 661, 672, 205], 18: [874, 278, 67, 238, 518, 712, 472, 567, 888, 360], 12: [747, 593, 917, 828, 871, 566, 642, 8, 511, 144], 9: [101, 643, 759, 833, 792, 823, 451, 957, 552, 216], 17: [62, 689, 204, 826, 643, 546, 616, 542, 990, 185], 826: [925, 266, 636, 834, 361, 915, 563, 447, 786, 648], 5: [689, 877, 332, 694, 851, 471, 563, 235, 910, 946], 739: [45, 783, 887, 467, 430, 552, 296, 731, 938, 578], 10: [401, 394, 204, 595, 581, 245, 118, 250, 674, 406], 984: [85, 841, 959, 818, 544, 396, 677, 395, 378, 537], 934: [445, 915, 701, 231, 750, 50, 72, 316, 725, 596], 731: [684, 930, 319, 481, 591, 891, 54, 334, 839, 209], 1: [151, 296, 146, 281, 287, 546, 570, 699, 424, 44], 16: [929, 747, 579, 460, 870, 65, 444, 981, 626, 156]}
            #pass
        # nodes for face
        if args.FacebookDataset:
            pass
        # nodes for hams
        if args.HamstersterDataset:
            pass 
        # nodes for dblp 
        if args.DBLPDataset:
            pass 

    print("first_started_embedding:",first_started_embedding)
    print(f"\nTrain Embeddings for nodes: {tested_nodes}")
    nodes_for_training_embedding = {}

    for index, first_node in enumerate(tested_nodes):
        print(f"Start training embedding #{index+1}/{len(tested_nodes)} for first_node={first_node}.")
        graph_removed_one = graph.delete_node_edges(first_node)
        nx_g_removed_one = graph_removed_one.to_networkx()
        np.testing.assert_array_equal(nx_g_removed_one.nodes(),graph_removed_one.nodes())

        # retrain embedding Epsilon on graph G' (without removed node, here: removing = zero edge weights)
        if retraining:
            graph_removed_one_dynamic = graph.delete_node_edges(first_node)
        else:
            graph_removed_one_dynamic = graph
        nx_g_removed_one_dynamic = graph_removed_one_dynamic.to_networkx()
        np.testing.assert_array_equal(nx_g_removed_one_dynamic.nodes(),graph_removed_one.nodes())
        print(f"\nRetrain embedding E with graph G' with removed node {first_node}....")
        if args.DHPE:
            embedding.retrain_embedding(graph=graph_removed_one_dynamic,save_info=save_info,removed_nodes=[],num_of_embeddings=num_of_embeddings, graph_without_nodes=[first_node],retraining=retraining,eng=eng)
        else:
            embedding.retrain_embedding(graph=graph_removed_one_dynamic,save_info=save_info,removed_nodes=[],num_of_embeddings=num_of_embeddings, graph_without_nodes=[first_node],retraining=retraining)


        print(f"\nTrain embedding E' with graph G' with removed node {first_node}....")
        if args.DHPE:
            embedding.train_embedding(graph=graph_removed_one, save_info=save_info, removed_nodes=[first_node], num_of_embeddings=num_of_embeddings, graph_without_nodes=[first_node],eng=eng)
        else:
            embedding.train_embedding(graph=graph_removed_one, save_info=save_info, removed_nodes=[first_node], num_of_embeddings=num_of_embeddings, graph_without_nodes=[first_node])

        if not args.fixed_nodes:
        # sample if we don't have fixed first/second nodes
            if num_of_training_graphs:
                second_completed_diffs = save_info.get_list_of_available_embeddings(graph=graph_removed_one,
                                                                                      removed_first_node=first_node,
                                                                                      find_started_trainings=False,
                                                                                      graph_without_nodes=[])

                second_started_embedding = save_info.get_list_of_available_embeddings(graph=graph_removed_one,
                                                                                      removed_first_node=first_node,
                                                                                      find_started_trainings=True,
                                                                                      graph_without_nodes=[])
                second_tested_nodes = utils.sample_randomly_with_pref_list_without_splitting_nodes(
                    graph=graph_removed_one.copy().delete_node(first_node), pref_list=second_completed_diffs,
                    secondary_pref_list=second_started_embedding,
                    all_list=list(filter(lambda x : x != first_node, graph_removed_one.nodes())),
                    quantity=num_of_training_graphs)
            else:
                second_tested_nodes = graph_removed_one.nodes()
        else:
            second_tested_nodes = second_nodes[first_node]
        print("second nodes (v_j) are:")
        print(second_tested_nodes)
        nodes_for_training_embedding[first_node] = second_tested_nodes
        print(f"\nTrain embeddings for removed node {first_node} and {second_tested_nodes}")
        for index2, second_node in enumerate(second_tested_nodes):
            print(f"Start train embedding ##{index2+1}/{len(second_tested_nodes)} for second node:{second_node} for #{index+1}/{len(tested_nodes)} of first node:{first_node}.")
            # retrain embedding Epsilon' on graph G'' (without removed node, here, removing = zero edge weights)
            if retraining:
                graph_removed_two_dynamic = graph_removed_one.delete_node_edges(second_node)
            else:
                graph_removed_two_dynamic = graph_removed_one
            nx_g_removed_two_dynamic = graph_removed_two_dynamic.to_networkx()
            print(f"\nRetrain embedding E' with graph G'' with removed nodes 1st: {first_node}, 2nd: {second_node}....")
            if args.DHPE:
                embedding.retrain_embedding(graph=graph_removed_two_dynamic,save_info=save_info,removed_nodes=[first_node],num_of_embeddings=num_of_embeddings, graph_without_nodes=[first_node,second_node],retraining=retraining,eng=eng)
            else:
                embedding.retrain_embedding(graph=graph_removed_two_dynamic,save_info=save_info,removed_nodes=[first_node],num_of_embeddings=num_of_embeddings, graph_without_nodes=[first_node,second_node],retraining=retraining)

            graph_removed_two = graph_removed_one.delete_node_edges(second_node)
            nx_g_removed_two = graph_removed_two.to_networkx()
            np.testing.assert_array_equal(nx_g_removed_two.nodes(),graph_removed_two.nodes())
            print(f"\nTrain embedding E'' with graph G'' with removed first node:{first_node} and second node:{second_node}.")
            if args.DHPE:
                embedding.train_embedding(graph=graph_removed_two, save_info=save_info, removed_nodes=[first_node, second_node], num_of_embeddings=num_of_embeddings, graph_without_nodes=[first_node,second_node],eng=eng)
            else:
                embedding.train_embedding(graph=graph_removed_two, save_info=save_info, removed_nodes=[first_node, second_node], num_of_embeddings=num_of_embeddings, graph_without_nodes=[first_node,second_node])
    time_end = time()
    print(f"time for training: {round(((time_end-time_start)/60),2)} minutes ({round(((time_end-time_start)),2)} seconds).")
    # delete tmp data
    shutil.rmtree(save_info.BASE_PATH+"/temp_embeddings")
    shutil.rmtree(save_info.BASE_PATH+"/temp_weights")
    shutil.rmtree(save_info.BASE_PATH+"/temp_graphs")

    # create features
    if run_experiments_on_embedding:
        for num_bins in num_of_bins_for_tf:
            # try:
            print("compute_training_features...")
            cf.compute_training_features(save_info=save_info, graph=graph, num_of_bins=num_bins,
                                         list_nodes_to_predict=tested_nodes,
                                         nodes_to_train_on=nodes_for_training_embedding, feature_type=feature_type)
            print("test with sklearn...")
            te.test(save_info=save_info, graph=graph, feature_type=feature_type, num_of_bins=num_bins,
                    limit_num_training_graphs=num_of_training_graphs, list_nodes_to_predict=tested_nodes,
                    nodes_to_train_on=nodes_for_training_embedding)
    return tested_nodes, nodes_for_training_embedding
