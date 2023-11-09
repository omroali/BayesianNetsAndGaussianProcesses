from logging import config
import matplotlib
from py import test
from sklearn import naive_bayes
from sympy import Chi
from ConditionalIndependence import ConditionalIndependence
from CPT_Generator import CPT_Generator
from ModelEvaluator import ModelEvaluator
import BayesNetUtil as bnu
import networkx as nx
import matplotlib.pyplot as plt
import utils

def Task1a(train_data = 'data/lung_cancer-train.csv', test_args = 'I(Smoking,Coughing|Lung_cancer)'):
    ci = ConditionalIndependence(train_data, 'gsq')
    Vi, Vj, parents_i = ci.parse_test_args(test_args)
    ci.compute_pvalue(Vi, Vj, parents_i)
    
def Task2ScoringFunctionsAndModelPerfoemance():
    # Part a: what is the Log-likelihood of the Naive Bayes model (config-lungcancer.txt)
    config_path = 'config/config-lungcancer.txt'
    train_data = 'data/lung_cancer-train.csv'
    test_data = 'data/lung_cancer-test.csv'
    
    me = ModelEvaluator(config_path, train_data, test_data)
    '''
    CARRYING-OUT probabilistic inference on test data...
    
    COMPUTING performance of classifier_type on test data:
    Balanced Accuracy=0.7691328279563574
    F1 Score=0.8647887323943662
    Area Under Curve=0.8787878787878788
    Brier Score=0.13440325637483394
    KL Divergence=114.57835666078407
    Training Time=0.002000093460083008 secs.
    Inference Time=0.011887311935424805 secs.

    CALCULATING LL and BIC on training data...
    LL score=-10009.090540458645
    BIC score=-10177.294609361721
    '''
    # print(me.calculate_log_lilelihood()) need to pass in an a naive_baye classifier as input
    # print(me.calculate_bic()) need to pass in an a naive_baye classifier as input
    
def Task3DetectingLoops():
    G_slide25 = [('A', 'B'), ('A', 'D'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'B')] 
    print(G_slide25)
    bnu.has_cycles(G_slide25)
    
def Task4LlAndBicofTask1():
    config_path = 'config/config-lungcancer-w1.txt'
    train_data = 'data/lung_cancer-train-w1.csv'
    test_data = 'data/lung_cancer-test-w1.csv'
    
    # manually updated the structure to match the one at the top of the workshop 3 file
    # ran the CPT generator to generate the CPTs
    CPT_Generator('config/config-lungcancer-w1.txt', 'data/lung_cancer-train-w1.csv')
    
    me = ModelEvaluator(config_path, train_data, test_data)
    
    '''
    COMPUTING performance of classifier_type on test data:
    Balanced Accuracy=0.7691328279563574
    F1 Score=0.8647887323943662
    Area Under Curve=0.8787878787878788
    Brier Score=0.13440325637483394
    KL Divergence=114.57835666078407
    Training Time=0.0019681453704833984 secs.
    Inference Time=0.005972385406494141 secs.

    CALCULATING LL and BIC on training data...
    LL score=-10153.583748892535
    BIC score=-10321.787817795612
    '''
    # very nedative score on the LL and BIC score
    
def testingNetworkX():
    G = nx.Graph() 
    # TG = nx.DiGraph() <- directed
    # G = nx.MultiGraph() <- multiple edges
    # G = nx.MultiDiGraph() <- multiple Directed 
    node1 = 1
    node2 = 2
    node3 = 3
    
    
    
    # G.add_edge(1,2, weight=0.1)
    # G.add_edge(2,3, weight=0.7)
    # G.add_edge('A','B')
    # G.add_edge('B','B')
    # G.add_node('C')
    # G.add_node(print)
    
    edge_list = [('A','C'), ('B','C'), ('C','D'), ('C','E')]
    # G = nx.from_edgelist(edge_list)
    G.add_edges_from(edge_list)
    
    # nx.topological sort to <- dimi tip 
    
    nx.draw_spring(G, with_labels=True)
    plt.show()
    
    
def buildingPCA():
    TG = nx.DiGraph()
    node_list = ['A', 'B', 'C', 'D', 'E']
    edge_list = [('A','C'), ('B','C'), ('C','D'), ('C','E')]
    
    TG.add_edges_from(edge_list)
    
    G = nx.Graph()
    G = utils.fullyConnectedGraph(node_list)
    
    plt.subplot(2,2,1)
    nx.draw_spring(TG, with_labels=True)
    
    plt.subplot(2,2,2)
    nx.draw_spring(G, with_labels=True)
    
    severed_edges = utils.removeConnectionsWithNoDirectionalPaths(G, TG)
    print(severed_edges)
    
    plt.subplot(2,2,3)
    nx.draw_spring(G, with_labels=True)
    
    
    plt.show()


def testingIndependence():
    data = utils.ChiIndependenceTest('data/lung_cancer-train.csv', 'I(Smoking,Anxiety|Coughing)')
    
    
    
if __name__ == "__main__":
    # train_data = 'data/lung_cancer-train.csv'
    # Task1a(train_data, 'I(Smoking,Coughing|Lung_cancer)',)
    # Task1a(train_data, 'I(Smoking,Car_Accident|Lung_cancer)')
    # Task1a(train_data, 'I(Anxiety,Fatigue|Lung_cancer)')
    # Task1a(train_data, 'I(Anxiety,Attention_Disorder|Lung_cancer)')
    # Task1a(train_data, 'I(Allergy,Fatigue|Lung_cancer)')
    # '''
    # Results:
    # Chi test:
    # X2test: Vi=Smoking, Vj=Coughing, pa_i=['Lung_cancer'], p=0.7671012800093508
    # X2test: Vi=Smoking, Vj=Car_Accident, pa_i=['Lung_cancer'], p=2.2223338414565005e-06
    # X2test: Vi=Anxiety, Vj=Fatigue, pa_i=['Lung_cancer'], p=0.1588073812200198
    # X2test: Vi=Anxiety, Vj=Attention_Disorder, pa_i=['Lung_cancer'], p=0.08855281302696125
    # X2test: Vi=Allergy, Vj=Fatigue, pa_i=['Lung_cancer'], p=3.055680781825919e-19

    # gsq test:
    # X2test: Vi=Smoking, Vj=Coughing, pa_i=['Lung_cancer'], p=0.7726106454381072
    # X2test: Vi=Smoking, Vj=Car_Accident, pa_i=['Lung_cancer'], p=3.706744827645817e-07
    # X2test: Vi=Anxiety, Vj=Fatigue, pa_i=['Lung_cancer'], p=0.15671922762370036
    # X2test: Vi=Anxiety, Vj=Attention_Disorder, pa_i=['Lung_cancer'], p=0.08976816134783991
    # X2test: Vi=Allergy, Vj=Fatigue, pa_i=['Lung_cancer'], p=2.1212849341311613e-19
    # '''
    
    # Task2ScoringFunctionsAndModelPerfoemance()
    
    # Task3DetectingLoops()
    
    # Task4LlAndBicofTask1()
    
    # G = nx.complete_graph(5)
    # plt.plot(100,100, G)
    # testingNetworkX()
    
    # buildingPCA()
    testingIndependence()
