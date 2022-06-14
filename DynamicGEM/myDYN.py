import os
import matplotlib.pyplot as plt
from dynamicgem.embedding.dynAE import DynAE
from dynamicgem.graph_generation import dynamic_SBM_graph as sbm
from dynamicgem.visualization import plot_dynamic_sbm_embedding
from time import time
from dynamicgem.utils import graph_util as gu


def main():
    # Parameters for Stochastic block model graph
    # Todal of 1000 nodes
    node_num = 1000
    # Test with two communities
    community_num = 2
    # At each iteration migrate 10 nodes from one community to the another
    node_change_num = 10
    # Length of total time steps the graph will dynamically change
    # At each iteration remove 10 nodes
    node_to_remove_number = 1
    length = 3
    # output directory for result
    outdir = './output'
    intr = './intermediate'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if not os.path.exists(intr):
        os.mkdir(intr)
    testDataType = 'sbm_cd'
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
     


    graphs = [g[0] for g in dynamic_sbm_series]
    print("GRAPHSSSSSS")
    print(graphs[:2])
    testarr = [1,2,3]
    print("testarr",testarr[:2])
    # parameters for the dynamic embedding
    # dimension of the embedding
    dim_emb = 128
    lookback = 2

    embedding = DynAE(d=dim_emb,
                      beta=5,
                      n_prev_graphs=lookback,
                      nu1=1e-6,
                      nu2=1e-6,
                      n_units=[500, 300, ],
                      rho=0.3,
                      n_iter=50,
                      xeta=1e-4,
                      n_batch=100,
                      modelfile=['./intermediate/enc_model_dynAE.json',
                                 './intermediate/dec_model_dynAE.json'],
                      weightfile=['./intermediate/enc_weights_dynAE.hdf5',
                                  './intermediate/dec_weights_dynAE.hdf5'],
                      savefilesuffix="testing")
    embs = []
    t1 = time()
    for temp_var in range(lookback + 1, length + 1):
        emb, _ = embedding.learn_embeddings(graphs[:temp_var])
        embs.append(emb)
    print(embedding._method_name + ':\n\tTraining time: %f' % (time() - t1))
    plt.figure()
    plt.clf()
    plot_dynamic_sbm_embedding.plot_dynamic_sbm_embedding_v2(embs[-5:-1], dynamic_sbm_series[-5:])
    plt.show()


if __name__ == '__main__':
    main()
