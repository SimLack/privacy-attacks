import os
from dynamicgem.evaluation import visualize_embedding as viz
from dynamicgem.embedding.ae_static import AE
from dynamicgem.graph_generation import dynamic_SBM_graph as sbm
from time import time
from dynamicgem.utils import graph_util as gu
import networkx as nx

def main():
    # Parameters for Stochastic block model graph
    # Todal of 1000 nodes
    node_num = 500
    # Test with two communities
    community_num = 2
    # At each iteration migrate 10 nodes from one community to the another
    node_change_num = 10
    # At each iteration remove 10 nodes
    node_to_remove_number = 1
    # Length of total time steps the graph will dynamically change
    length = 3
    # output directory for result
    outdir = './output'
    intr = './intermediate'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if not os.path.exists(intr):
        os.mkdir(intr)
    testDataType = 'sbm_cd'
    outdir = outdir + '/' + testDataType
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    outdir = outdir + '/staticAE'
    if not os.path.exists(outdir):
        os.mkdir(outdir)


    # Generate the dynamic graph
    dynamic_sbm_series = list(sbm.get_community_diminish_series_v2(node_num,
                                                                   community_num,
                                                                   length,
                                                                   1,  # comminity ID to perturb
                                                                   node_change_num))

    # Generate usual graph instead of sbm
    dynamic_graph_series = []
    myGraph = gu.loadGraphFromEdgeListTxt("DBLP_graph_moderate_homophily_snowball_sampled_2000.edgelist")
    dynamic_graph_series.append(myGraph)
    print("dynamic_graph_series:")
    print(dynamic_graph_series)
    print(myGraph.nodes())
    myGraph1 = myGraph.copy()
    myGraph1.remove_node(398)
    print("myGraph1:")
    print(myGraph1.nodes())
     

    print("dynamic_sbm_series:")
    print(dynamic_sbm_series)

    graphs = [g[0] for g in dynamic_sbm_series]
    print("graph:")
    print(type(graphs[0]))
    print("graphs0 nodes:")
    print(graphs[0].nodes())
    # parameters for the dynamic embedding
    # dimension of the embedding
    dim_emb = 128
    print("dim_emb:",dim_emb)
    lookback = 2

    # AE Static
    embedding = AE(d=dim_emb,
                   beta=5,
                   nu1=1e-6,
                   nu2=1e-6,
                   K=3,
                   n_units=[500, 300],
                   n_iter=10,
                   xeta=1e-4,
                   n_batch=100,
                   modelfile=['./intermediate/enc_modelsbm.json',
                              './intermediate/dec_modelsbm.json'],
                   weightfile=['./intermediate/enc_weightssbm.hdf5',
                               './intermediate/dec_weightssbm.hdf5'],
                   savefilesuffix='mySavedModel',
                             )
    embs = []
    t1 = time()
    # ae static
    for temp_var in range(length):
        emb, _ = embedding.learn_embeddings(graphs[temp_var])
        embs.append(emb)
    print("embs:")
    print(embs)  
    print(embedding._method_name + ':\n\tTraining time: %f' % (time() - t1))
    viz.plot_static_sbm_embedding(embs[-4:], dynamic_sbm_series[-4:])


if __name__ == '__main__':
    main()
