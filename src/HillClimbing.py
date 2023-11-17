import bnlearn as bn
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def hill_climbing(file_path):
    df = pd.read_csv(file_path, encoding='utf-8')
    model_hc_bic = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
    # model_hc_bic = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic', max_indegree=3)


    # print(model_hc_bic)
    # G = bn.plot(model_hc_bic)
    
    # nx.draw_shell(G, with_labels=True, font_weight='bold')
    # plt.show()
    # DAG = bn.parameter_learning.fit(model_hc_bic, df, methodtype='bayes')
    DAG = bn.parameter_learning.fit(model_hc_bic, df, methodtype='maximum_likelihood')
    nx.draw_shell(DAG, with_labels=True, font_weight='bold')

    # for node in nx.topological_sort(model_hc_bic):
    #     print(node,list(model_hc_bic.predecessors(node)))
    

    # Convert the adjacency matrix into a networkx graph
    G = nx.from_pandas_adjacency(model_hc_bic['adjmat'])

    nx.draw_shell(G, with_labels=True, font_weight='bold')
    plt.show()

hill_climbing('data/discreet/cardiovascular_data-discretized-train.csv')