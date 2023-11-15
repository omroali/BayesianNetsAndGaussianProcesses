from time import sleep
# from pgmpy import config
from py import test
import utils
import os
from matplotlib import pyplot as plt
import networkx as nx

from pcstable import PCStable as pcs
from CPT_Generator import CPT_Generator
from NB_Classifier import NB_Classifier as nbc
from BayesNetInference import BayesNetInference as bni
from GaussianProcess import GaussianProcess as gp
from ModelEvaluator import ModelEvaluator as me

run_path = os.path.dirname(os.path.realpath(__file__))

def get_naive_bayes_struct(data_train: str, data_test: str, target_value: str):
    # data_train = 'data/diabetes_data-discretized-train.csv'
    # target_value = 'Outcome'
    training_data = utils.read_training_data(data_train, target_value)
    
    
    # nb_fitted = nbc(data_train)
    # nb_tester = nbc(data_test, nb_fitted)
    
    # get the structure of the naive bayes classifier
    get_naive_bayes_struct = utils.independent_probability_structure(training_data)
    structure_data, structure_array = get_naive_bayes_struct
    print(structure_array)
    
    # generate config file with structure
    config_path = utils.config_structure_file(structure_array, 'nb-diabetes-structure', 'run_test')
    return config_path

    

def get_pc_stable_structure(data_train, method='chisq', independence_threshold=0.05):
    pcs_test = pcs(data_train, method, independence_threshold)
    pcs_test.evaluate_skeleton(with_plots= False,log_level=1)
    # pcs_test.evaluate_immoralities()
    # pcs_test.create_directional_edge_using_immorality()
    
    rand_dag = pcs_test.randomised_directed_graph()
    nx.draw_shell(rand_dag, with_labels=True)
    plt.show()
    
    # # creating the config file
    data = utils.topological_sort_for_structure(rand_dag)
    config_path = utils.config_structure_file(data, 'pc-diabetes-structure', 'run_test')
    
    return config_path

    
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
    
def gaussian_process(train_data, test_data):
    gaussian_process = gp(train_data, test_data)
    return gaussian_process
    
       

if __name__ == "__main__":
    train_data = 'data/diabetes_data-discretized-train.csv'
    test_data = 'data/diabetes_data-discretized-test.csv'
    # config_file = 'config/config-lungcancer.txt'
    target_value = 'Outcome'

    #############################
    #### Structure Generator ####
    #############################
    
    # config_file = get_naive_bayes_struct(train_data,test_data,target_value)
    config_file = get_pc_stable_structure(train_data, method='chisq', independence_threshold=0.05)
    
    
    
    #############################
    ###### CPT Generator ########
    #############################
    CPT_Generator(config_file, train_data)
    
    
    
    # run_nb_classifier = nb_classifier(train_data, test_data)
    # print(run_nb_classifier.estimate_probabilities)
    
    
    #############################
    ###### Model Evaluator ######
    #############################
    # config_file = 'config/config-pc-diabetes-structure-run_test-11-15_16:36:13.txt'
    # config_file = 'config/config-nb-diabetes-structure-run_test-11-15_17:43:09.txt'
    evaluator = me(config_file, train_data,test_data, True)
    
    ###################################
    ###### Inference ##################
    ###################################
    
    # # we can only run an inference method with parents of the variable we are interested in?
    exact_inference = bayes_net_exact_inference(config_file, "P(Outcome|Glucose=2,BMI=2,Age=2)", 1000)   

     
    # # gauss_proc = gaussian_process('data/diabetes_data-original-train.csv', 'data/diabetes_data-original-test.csv')
    
    rej = rejection_sampling(config_file, "P(Outcome|Glucose=2,BMI=2,Age=2)", 1000)
    # rej = rejection_sampling('config/config-alarm.txt', "P(B|J=true,M=true)", 100)

    print('end')

    
    
