#####test
import sys
import config
import argparse

from graphs import graph_class as gc
import embeddings.embedding as embeds
import embeddings.GEM_embeddings as gemEmbed
import memory_access as sl
import my_utils
import features.diff_type as dt
import networkx as nx
from graphs.graph_class import Graph as graph_gen
from time import time
import json
import os

def loadGraphFromEdgeListTxt(file_name, directed=True):
    print("dataset is:",file_name)
    with open(file_name, 'r') as f:
        if directed:
            print("directed")
            G = nx.DiGraph()
        else:
            print("undirected")
            G = nx.Graph()
        for line in f:
            edge = line.strip().split()
            if len(edge) == 3:
                w = float(edge[2])
            else:
                w = 1.0
            G.add_edge(int(edge[0]), int(edge[1]), weight=w)
    G = G.to_undirected()
    return G

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-re","--Retraining",action="store_true")
    parser.add_argument("-mf","--DHPE",action="store_true")
    parser.add_argument("-ae","--DynGEM",action="store_true")
    parser.add_argument("-sg","--DNE",action="store_true")
    parser.add_argument("-gnn","--DySat",action="store_true")
    parser.add_argument("-bara","--BarabasiDataset",action="store_true",help="Dataset with 2000 nodes and todo edges.")
    parser.add_argument("-face","--FacebookDataset",action="store_true",help="Dataset with 2000 nodes and 14251 edges.")
    parser.add_argument("-hams","--HamstersterDataset",action="store_true",help="Dataset with 1788 nodes and 12476 edges.")
    parser.add_argument("-DBLP","--DBLPDataset",action="store_true",help="Dataset with 2000 nodes and 7036 edges.")
    parser.add_argument("-f","--fixed_nodes",action="store_true")
    args = parser.parse_args()

    print("retraining is:",args.Retraining)
    if not args.DHPE and not args.DynGEM and not args.DNE and not args.DySat and not args.Netwalk and not args.NormalHope:
        print("One algorithm and one dataset must be chosen! You did not specify an algorithm.")
        exit()
    if not args.BarabasiDataset and not args.FacebookDataset and not args.HamstersterDataset and not args.DBLPDataset:
        print("One algorithm and one dataset must be chosen! You did not specify a dataset.")
        exit()

    # num iterations is fixed to 1 here (can be increased)
    num_iterations=1

    print("init Graph...")
    dataset_is = ""

    if args.BarabasiDataset:
        dataset_is = "bara"
        my_loaded_graph = nx.generators.random_graphs.barabasi_albert_graph(1000,5,1)

    if args.FacebookDataset:
        dataset_is = "face"
        my_loaded_graph = loadGraphFromEdgeListTxt("../data/facebook_wosn_snowball_sampled_2000.edgelist")
    if args.HamstersterDataset:
        dataset_is = "hams"
        my_loaded_graph = loadGraphFromEdgeListTxt("../data/hamsterster_cc.edgelist")

    if args.DBLPDataset:
        dataset_is = "dblp"
        my_loaded_graph = loadGraphFromEdgeListTxt("../data/DBLP_graph_moderate_homophily_snowball_sampled_2000.edgelist")

    print("args:",args)
    print("my loaded graph edges and #edges:",len(my_loaded_graph.edges()))
    #print(my_loaded_graph.edges())
    print(type(my_loaded_graph))
    print("first 100 of my loaded graph:",my_loaded_graph.edges()[:100])
    my_graph = graph_gen.init_from_networkx(my_loaded_graph)
    print("first 100 of my graph:",list(my_graph.edges())[:100])
    print(f"graph has {len(my_graph.nodes())} nodes and {len(my_graph.edges())} edges.")

    print("init embedding...")

    print("start training...")
    start_time = time()

    if args.DHPE:
        sys.path.insert(0, config.GEM_PATH + "DHPE/")
        import myDyn
        import embeddings.DHPE as DHPEEmbed
        myEmbed = DHPEEmbed.DHPE(embeds.Embedding).init_DHPE(dim=128)
        myMemAcc = sl.MemoryAccess(graph=my_graph,embedding_type=DHPEEmbed,num_iterations=num_iterations,diff_type = dt.DiffType.DIFFERENCE,retraining=args.Retraining,dataset_name=dataset_is)
        myDyn.train_embedding_per_graph(graph=my_graph,embedding=myEmbed,save_info=myMemAcc,num_of_embeddings=num_iterations,num_of_test_evaluations_per_degree_level=5,num_of_training_graphs=10,num_of_bins_for_tf=[10],args = args)

    if args.DynGEM:
        import myDyn
        import embeddings.GEM_embeddings as gemEmbed
        myEmbed = gemEmbed.GEM_embedding(embeds.Embedding).init_dynamicSDNE(dim=128,n_batch=500,n_iter=30, n_iter_subs=5, n_units=[400,200,],node_frac=2,n_walks_per_node=20,len_rw=1) #n_iter=#epochs #n_iter_subs=#RetrainEpochs
        myMemAcc = sl.MemoryAccess(graph=my_graph,embedding_type=gemEmbed,num_iterations=num_iterations,diff_type = dt.DiffType.DIFFERENCE,retraining=args.Retraining,dataset_name=dataset_is)
        myDyn.train_embedding_per_graph(graph=my_graph,embedding=myEmbed,save_info=myMemAcc,num_of_embeddings=num_iterations,num_of_test_evaluations_per_degree_level=5,num_of_training_graphs=10,num_of_bins_for_tf=[10],args = args)

    if args.DNE:
        import myDyn
        import embeddings.DNE as DNEEmbed
        myEmbed = DNEEmbed.DNE(embeds.Embedding).init_DNE()
        myMemAcc = sl.MemoryAccess(graph=my_graph,embedding_type=DNEEmbed,num_iterations=num_iterations,diff_type = dt.DiffType.DIFFERENCE,retraining=args.Retraining,dataset_name=dataset_is)

        # adjust myConf.json depending on dataset, as params are crucial here
        in_file = open(config.GEM_PATH +'embAttack/embedding-attack/myConf.json', 'r')
        data_file = in_file.read()
        data = json.loads(data_file)
        if args.BarabasiDataset:
            data['init']['init_train']['learn_rate'] = 0.05
            data['main_loop']['dynamic_embedding']['learn_rate'] = 0.05
            data['init']['init_train']['iteration_num'] = 100000 
            data['main_loop']['dynamic_embedding']['iteration_num'] = 30000
            data['init']['init_train']['batch_size'] = 100
            data['main_loop']['dynamic_embedding']['batch_size'] = 300
        elif args.FacebookDataset:
            data['init']['init_train']['learn_rate'] = 0.0005
            data['main_loop']['dynamic_embedding']['learn_rate'] = 0.0005
            data['init']['init_train']['iteration_num'] = 30000
            data['main_loop']['dynamic_embedding']['iteration_num'] = 10000
            data['init']['init_train']['batch_size'] = 200
            data['main_loop']['dynamic_embedding']['batch_size'] = 600
        elif args.HamstersterDataset:
            data['init']['init_train']['learn_rate'] = 0.0005
            data['main_loop']['dynamic_embedding']['learn_rate'] = 0.0005
            data['init']['init_train']['iteration_num'] = 30000
            data['main_loop']['dynamic_embedding']['iteration_num'] = 10000
            data['init']['init_train']['batch_size'] = 100
            data['main_loop']['dynamic_embedding']['batch_size'] = 300
        elif args.DBLPDataset:
            data['init']['init_train']['learn_rate'] = 0.0005
            data['main_loop']['dynamic_embedding']['learn_rate'] = 0.0005
            data['init']['init_train']['iteration_num'] = 30000
            data['main_loop']['dynamic_embedding']['iteration_num'] = 10000
            data['init']['init_train']['batch_size'] = 400
            data['main_loop']['dynamic_embedding']['batch_size'] = 1200
        in_file.close()
        out_file = open(config.GEM_PATH +'embAttack/embedding-attack/myConf.json','w')
        out_file.write(json.dumps(data))
        out_file.close()

        myDyn.train_embedding_per_graph(graph=my_graph,embedding=myEmbed,save_info=myMemAcc,num_of_embeddings=num_iterations,num_of_test_evaluations_per_degree_level=5,num_of_training_graphs=10,num_of_bins_for_tf=[10],args = args)

    if args.DySat:
        sys.path.insert(0, config.GEM_PATH + "dySat/")
        import myDyn
        import embeddings.dySat as dySatEmbed
        myEmbed = dySatEmbed.DySat(embeds.Embedding).init_dySat()
        myMemAcc = sl.MemoryAccess(graph=my_graph,embedding_type=dySatEmbed,num_iterations=num_iterations,diff_type = dt.DiffType.DIFFERENCE,retraining=args.Retraining,dataset_name=dataset_is)
        myDyn.train_embedding_per_graph(graph=my_graph,embedding=myEmbed,save_info=myMemAcc,num_of_embeddings=num_iterations,num_of_test_evaluations_per_degree_level=5,num_of_training_graphs=5,num_of_bins_for_tf=[10],args = args)


    print("done, needed:",round(time()-start_time,2),"seconds")
    print("retraining was:",args.Retraining)
