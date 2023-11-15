from ctypes import util
from logging import config
from py import test
# import utils
import os
from matplotlib import pyplot as plt
import networkx as nx

from pcstable import PCStable as pcs
from CPT_Generator import CPT_Generator
from NB_Classifier import NB_Classifier as nbc
from BayesNetInference import BayesNetInference as bni
from GaussianProcess import GaussianProcess as gp

run_path = os.path.dirname(os.path.realpath(__file__))

def pc_stable():
    # PCStable = PCStable('data/lung_cancer-train.csv')
    pcs_test = pcs('data/cardiovascular_data-discretized-train.csv', method='chisq', independence_threshold=0.05)
    pcs_test.evaluate_skeleton(with_plots= True,log_level=1)
    pcs_test.evaluate_immoralities()
    print('something cool')
    # pcs_test.create_directional_edge_using_immorality()
    # print(len(pcs_test.get_graph_edges))
    
    rad_dag = pcs_test.randomised_directed_graph()
    nx.draw_shell(rad_dag, with_labels=True)
    plt.show()
    print('something cool')
    # nx.topological_sort(rand_dag)

    # # Step 2: Identify the immoralities (v-structures) and orient them
    # self.identifyImmoralities()
    
    # # Step 3: Identify the remaining edges and orient them
    # self.identifyQualifyingEdges()
    
    # produce config file with the structure of the graph
    # util.formatIntoConfigStructureFile() <- check if this will work?

    # run inference 
    
def nb_classifier(train_file, test_file):

    nb_fitted = nbc(train_file)
    if test_file is not None:
        nb_tester = nbc(test_file, nb_fitted)
        return nb_tester
    return nb_fitted
    # we assume that all the data is independent of each other because it is a naive bayes classifier
    

def bayes_net_exact_inference(config_file, prob_query, num_samples):
    # run this after creating the config file for the specific bayes net
    # we can only run an inference method with parents of the variable we are interested in?
    exact_inference = bni(
        alg_name='InferenceByEnumeration',
        file_name=config_file,
        prob_query=prob_query, 
        num_samples=num_samples
    )
    return exact_inference

def rejection_sampling(config_file, prob_query, num_samples):
    # run this after creating the config file for the specific bayes net
    # we can only run an inference method with parents of the variable we are interested in?
    rejection_sampling = bni(
        alg_name='RejectionSampling',
        file_name=config_file,
        prob_query=prob_query, 
        num_samples=num_samples
    )
    return rejection_sampling
    
    
def cpt_generator():
    # data_path = f"{run_path}/../docs/workshops/w2/data/play_tennis-train.csv"
    # evaluation_variable = 'PT'
    # config_file_path = utils.formatIntoConfigStructureFile("PT",data_path)
    
    config_file_path = "config/config-pt-created.txt"
    training_data = "data/play_tennis-train.csv"
    CPT_Generator(config_file_path, training_data)
    
    
def gaussian_process(train_data, test_data):
    gaussian_process = gp(train_data, test_data)
    return gaussian_process
    
       

if __name__ == "__main__":
    train_data = 'data/diabetes_data-discretized-train.csv'
    test_data = 'data/diabetes_data-discretized-test.csv'
    config_file = 'config/config-lungcancer.txt'

    
    pc_stable()
    
    # run_nb_classifier = nb_classifier('data/diabetes_data-discretized-train.csv', 'data/diabetes_data-discretized-test.csv')
    
    # # we can only run an inference method with parents of the variable we are interested in?
    # exact_inference = bayes_net_exact_inference(config_file, 'P(Yellow_Fingers|Lung_cancer=0)', 1000)   
     
     
    # # gauss_proc = gaussian_process('data/diabetes_data-original-train.csv', 'data/diabetes_data-original-test.csv')
    
    # rej = rejection_sampling('config/config-lungcancer.txt', "P(Yellow_Fingers|Lung_cancer=1,Anxiety=1)", 1000)
    # rej = rejection_sampling('config/config-alarm.txt', "P(B|J=true,M=true)", 100)

    print('end')

    
    
