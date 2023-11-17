import datetime
from importlib.metadata import distribution
import random
import time
# from sklearn.metrics import accuracy_score, balanced_accuracy_score
# from pgmpy import config
from py import test
from scipy import rand
from tqdm import tqdm
import utils
import os
from matplotlib import pyplot as plt
import networkx as nx
from sklearn import metrics
import pandas as pd

from pcstable import PCStable as pcs
from CPT_Generator import CPT_Generator
from NB_Classifier import NB_Classifier as nbc
from BayesNetInference import BayesNetInference as bni
from ModelEvaluator import ModelEvaluator as me

run_path = os.path.dirname(os.path.realpath(__file__))

def get_naive_bayes_struct(data_train: str,input_data_name, data_test: str, target_value: str):
    # data_train = 'data/diabetes_data-discretized-train.csv'
    # target_value = 'Outcome'
    training_data = utils.read_training_data(data_train, target_value)
    
    
    # nb_fitted = nbc(data_train)
    # nb_tester = nbc(data_test, nb_fitted)
    
    # get the structure of the naive bayes classifier
    get_naive_bayes_struct = utils.independent_probability_structure(training_data)
    structure_data, structure_array = get_naive_bayes_struct
    ## print(structure_array)
    
    # generate config file with structure
    config_path = utils.config_structure_file(structure_array, f'nb-{input_data_name}-structure', 'run_test')
    return config_path

def get_pc_stable_structure(data_train, insert_dataset_name, method='chisq', independence_threshold=0.05):
    pcs_test = pcs(data_train, method, independence_threshold)
    pcs_test.evaluate_skeleton(with_plots= False,log_level=1)
    # pcs_test.evaluate_immoralities()
    # pcs_test.create_directional_edge_using_immorality()
    
    rand_dag = pcs_test.randomised_directed_graph()
    nx.draw_shell(rand_dag, with_labels=True)
    plt.show()
    
    # # creating the config file
    data = utils.topological_sort_for_structure(rand_dag)
    config_path = utils.config_structure_file(data, f'pc-{insert_dataset_name}-structure', 'run_test')
    
    return config_path


def find_best_pc_structure(data_train, insert_file_name, method='chisq', independence_threshold=0.05, max_iterations=10000):
    iteration = 0
    structures = []
    best_accuracy = 0
        
    pcs_test = pcs(data_train, method, independence_threshold)
    pcs_test.evaluate_skeleton(with_plots= False,log_level=1)
    
    pbar = tqdm(total=max_iterations)
    pbar.set_description(f'Evaluating PC using {method} with independence threshold {independence_threshold}')
    # randomising colour for pbar 
    colour = random.choice(['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'])
    pbar.format_meter(0, max_iterations, 0, colour=colour)
                      
    while(iteration < max_iterations):
        start_time = time.time()
        rand_dag = pcs_test.randomised_directed_graph()
        rand_dag_time = time.time() - start_time
        
        data = utils.topological_sort_for_structure(rand_dag)
        config_path = utils.config_structure_file(data, f'pc-automated/config-pc-{insert_file_name}-structure', 'best-structure')
        
        start_time = time.time()
        CPT_Generator(config_path, train_data)
        cpt_generator_time = time.time() - start_time
        
        evaluator = me(config_path, train_data, test_data)
        computed_performance = evaluator.computed_performance
        
        evaluation_data = {
            'balanced_accuracy': computed_performance['bal_acc'],
            'structure': data,
            'config_path': config_path,
            'rand_dag_time': rand_dag_time,
            'cpt_generator_time': cpt_generator_time,
        }
        
        if computed_performance['bal_acc'] > best_accuracy:
                best_accuracy = computed_performance['bal_acc']
                best_structure = evaluation_data
                structures.append(evaluation_data)
        
        iteration += 1
        pbar.update(1)
        pbar.set_postfix_str(f'Best Accuracy: {best_accuracy}')
    return structures
# {'best_accuracy': best_accuracy, 'best_structure': best_structure, 'best_config_path': best_config_path}
    
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


def evaluating_pc_stable(train_data, structure_name, method='chisq', independence_threshold=0.01, max_iterations=100):
    best_structures = find_best_pc_structure(train_data, structure_name, method='gsq', independence_threshold=independence_threshold, max_iterations=max_iterations)
    config_file = best_structures[-1]
    cpt_generator_time = best_structures[-1]
    print(config_file)
    
    # inference_query = "P(Outcome|Glucose=4,BMI=1,Age=5)"
    # config_file = 'config/nb-cardiovascular-structure-run_test-11-16_12:09:49.txt' # Naive Bayes cardiovascular
    evaluator = me(config_file, train_data,test_data)
    computed_performance = evaluator.computed_performance
    
    print('\n')
    print("------------------------------------------------------------")
    print(f'Method: {method}')
    print(f'Independence Threshold: {independence_threshold}')
    print(f'Max Iterations: {max_iterations}')
    print("------------------------------------------------------------")
    print(f'**Config**: {config_file}')
    print(f'Brier Score: {computed_performance["brier"]}')
    print(f'KL Divergence: {computed_performance["kl_div"]}')
    print(f'Training Time: {cpt_generator_time}')
    print(f'Inference Time: {computed_performance["inference_time"]}')
    print(f'Balanced Accuracy: {computed_performance["bal_acc"]}')
    print(f'F1 Score: {computed_performance["f1"]}')
    print(f'Area Under Curve: {computed_performance["auc"]}')
    print("\n--------------------")
    print(f'Structure: {best_structures[-1]["structure"]}')
    print("\n--------------------")
    with open(f'best_structures_{method}_{independence_threshold}-real.txt', 'w ') as f:
        f.write('\n')
        f.write("------------------------------------------------------------")
        f.write(f'Method: {method}')
        f.write(f'Independence Threshold: {independence_threshold}')
        f.write(f'Max Iterations: {max_iterations}')
        f.write("------------------------------------------------------------")
        f.write(f'**Config**: {config_file}')
        f.write(f'Brier Score: {computed_performance["brier"]}')
        f.write(f'KL Divergence: {computed_performance["kl_div"]}')
        f.write(f'Training Time: {cpt_generator_time}')
        f.write(f'Inference Time: {computed_performance["inference_time"]}')
        f.write(f'Balanced Accuracy: {computed_performance["bal_acc"]}')
        f.write(f'F1 Score: {computed_performance["f1"]}')
        f.write(f'Area Under Curve: {computed_performance["auc"]}')
        f.write("\n--------------------")
        f.write(f'Structure: {best_structures[-1]["structure"]}')
        f.write("\n--------------------")
    
    for i in range(5):
        if len(best_structures) == 0:
            break
        print(best_structures.pop())


if __name__ == "__main__":
    # train_data = 'data/discreet/diabetes_data-discretized-train.csv'
    # test_data = 'data/discreet/diabetes_data-discretized-test.csv'
    train_data = 'data/discreet/cardiovascular_data-discretized-train.csv'
    test_data = 'data/discreet/cardiovascular_data-discretized-test.csv'
    # config_file = 'config/pc-automated/config-pc-diabetes-structure-best-structure-11-16_01:24:08.txt'
    structure_name = 'cardiovascular-discrete'
    target_value = 'target'
    
    # evaluating_pc_stable(train_data, structure_name, method='chisq', independence_threshold=0.01, max_iterations=20)
    evaluating_pc_stable(train_data, structure_name, method='chisq', independence_threshold=0.05, max_iterations=20)
    # evaluating_pc_stable(train_data, structure_name, method='gsq', independence_threshold=0.01, max_iterations=20)
    # evaluating_pc_stable(train_data, structure_name, method='gsq', independence_threshold=0.05, max_iterations=20)
    # evaluating_pc_stable(train_data, structure_name, method='chisq', independence_threshold=0.01, max_iterations=20)
    # evaluating_pc_stable(train_data, structure_name, method='chisq', independence_threshold=0.05, max_iterations=20)
    # evaluating_pc_stable(train_data, structure_name, method='gsq', independence_threshold=0.01, max_iterations=20)
    # evaluating_pc_stable(train_data, structure_name, method='gsq', independence_threshold=0.05, max_iterations=20)
    
    # #############################
    # #### Structure Generator ####
    # #############################
    # config_file = get_naive_bayes_struct(train_data,structure_name,test_data,target_value)
    # config_file = get_pc_stable_structure(train_data, structure_name, method='fisherz', independence_threshold=0.05)
    
    # get_metrics(train_data, 'cardiovascular', max_iterations=100)
    # best = find_best_pc_structure(train_data, 'cardiovascular-discrete', method='chisq', independence_threshold=0.05, max_iterations=1000)
    # 'config/pc-automated/config-pc-diabetes-structure-best-structure-11-16_01:30:09.txt'
    # 'config/pc-automated/config-pc-diabetes-structure-best-structure-11-16_01:30:09.txt' 0.70 
    # {'best_accuracy': 0.7, 'best_structure': {'random_variables': 'BloodPressure(BloodPressure);SkinThickness(SkinThickness);Age(Age);Insulin(Insulin);BMI(BMI);Pregnancies(Pregnancies);Glucose(Glucose);Outcome(Outcome)', 'structure': 'P(BloodPressure);P(SkinThickness);P(Age|BloodPressure);P(Insulin|SkinThickness);P(BMI|BloodPressure,SkinThickness);P(Pregnancies|BloodPressure,SkinThickness,Age);P(Glucose|Insulin);P(Outcome|Glucose)'}, 'best_config_path': 'config/pc-automated/config-pc-diabetes-structure-best-structure-11-16_01:30:09.txt', 'rand_dag_time': 0.00015664100646972656, 'cpt_generator_time': 0.039079904556274414}
        
    # print('\n\n\nBEST STRUCTURE---')
    # print('best_accuracy:', best['best_accuracy'])
    # print('best_structure:', best['best_structure'])
    # print('best_config_path:', best['best_config_path'])
    # print('best_config_path:', best['rand_dag_time'])
    # print('rand_dag_time:', best['cpt_generator_time'])
    
    #############################
    ###### CPT Generator ########
    #############################
    # config_file = 'config/pc-automated/config-pc-diabetes-structure-best-structure-11-16_22:32:00.txt'
    # start_time = time.time()
    # CPT_Generator(config_file, train_data)
    # cpt_generator_time = time.time() - start_time
    
    
    # # run_nb_classifier = nb_classifier(train_data, test_data)
    # # ## print(run_nb_classifier.estimate_probabilities)
    
    
    # #############################
    # ###### Model Evaluator ######
    # #############################
    
    
    
    # best_structure, best_structures = find_best_pc_structure(train_data, structure_name, method='gsq', independence_threshold=0.05, max_iterations=10000)
    # config_file = best_structure['config_path']
    # cpt_generator_time = best_structure['cpt_generator_time']
    # print(config_file)
    
    # inference_query = "P(Outcome|Glucose=4,BMI=1,Age=5)"
    # # config_file = 'config/nb-cardiovascular-structure-run_test-11-16_12:09:49.txt' # Naive Bayes cardiovascular
    # evaluator = me(config_file, train_data,test_data)
    # computed_performance = evaluator.computed_performance
    # print("------------------------------------------------------------")
    # print('**Config**:', config_file)
    # print('Training Time:', cpt_generator_time)
    # print('Inference Time:', computed_performance['inference_time'])
    # print('Balanced Accuracy:', computed_performance['bal_acc'])
    # print('F1 Score:', computed_performance['f1'])
    # print('Area Under Curve:', computed_performance['auc'])
    # print('Brier Score:', computed_performance['brier'])
    # print('KL Divergence:', computed_performance['kl_div'])
    # print("\n--------------------")
    
    # print(best_structures.pop())
    # print(best_structures.pop())
    # print(best_structures.pop())
    # print(best_structures.pop())
    # print(best_structures.pop())
        
        
    # best_structure, best_structures = find_best_pc_structure(train_data, structure_name, method='gsq', independence_threshold=0.05, max_iterations=10000)
    # config_file = best_structure['config_path']
    # cpt_generator_time = best_structure['cpt_generator_time']
    # print(config_file)
    
    # inference_query = "P(Outcome|Glucose=4,BMI=1,Age=5)"
    # # config_file = 'config/nb-cardiovascular-structure-run_test-11-16_12:09:49.txt' # Naive Bayes cardiovascular
    # evaluator = me(config_file, train_data,test_data)
    # computed_performance = evaluator.computed_performance
    # print("------------------------------------------------------------")
    # print('**Config**:', config_file)
    # print('Training Time:', cpt_generator_time)
    # print('Inference Time:', computed_performance['inference_time'])
    # print('Balanced Accuracy:', computed_performance['bal_acc'])
    # print('F1 Score:', computed_performance['f1'])
    # print('Area Under Curve:', computed_performance['auc'])
    # print('Brier Score:', computed_performance['brier'])
    # print('KL Divergence:', computed_performance['kl_div'])
    # print("\n--------------------")
    
    # print(best_structures.pop())
    # print(best_structures.pop())
    # print(best_structures.pop())
    # print(best_structures.pop())
    # print(best_structures.pop())
        
    
    # # # # we can only run an inference method with parents of the variable we are interested in?
    # print("**Query**:", inference_query)
    # exact_inference = bayes_net_exact_inference(config_file, inference_query, 1000)   
    # rej = rejection_sampling(config_file, inference_query, 1000)
    # rej = rejection_sampling(config_file, inference_query, 10000)
    # rej = rejection_sampling(config_file, inference_query, 100000)
    # rej = rejection_sampling(config_file, inference_query, 200000)
    # print("\n------------------------------------------------------------")


    # exact_inference = bayes_net_exact_inference(config_file, "P(Outcome|Glucose=2,BMI=2,Age=2)", 1000)   

    
    # # # gauss_proc = gaussian_process('data/diabetes_data-original-train.csv', 'data/diabetes_data-original-test.csv')
    
    # # rej = rejection_sampling('config/config-alarm.txt', "P(B|J=true,M=true)", 100)
    
    
    
    
    
    ################################
    ###### Gaussian Processes ######
    ################################
    
    
    ## print('end')

    
    
