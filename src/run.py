import datetime
import json
import time

import random
import numpy as np
from pgmpy import config
from sympy import plot

from tqdm import tqdm
from colorama import Fore, Back, Style

import utils
import os
from matplotlib import pyplot as plt
import networkx as nx

from pcstable import PCStable as pcs
from CPT_Generator import CPT_Generator
from NB_Classifier import NB_Classifier as nbc
from BayesNetInference import BayesNetInference as bni
from ModelEvaluator import ModelEvaluator as me

run_path = os.path.dirname(os.path.realpath(__file__))
colour_dict = {
    "red": Fore.RED,
    "green": Fore.GREEN,
    "yellow": Fore.YELLOW,
    "blue": Fore.BLUE,
    "magenta": Fore.MAGENTA,
    "cyan": Fore.CYAN,
    "white": Fore.WHITE,
}
colour = random.choice(list(colour_dict.keys()))


def get_naive_bayes_struct(data_train: str, input_data_name, target_value: str):
    """
    method to generate the naive bayes structure
    """
    training_data = utils.read_training_data(data_train, target_value)

    # get the structure of the naive bayes classifier
    get_naive_bayes_struct = utils.independent_probability_structure(training_data)
    structure_data, structure_array = get_naive_bayes_struct

    # generate config file with structure
    config_path = utils.config_structure_file(
        structure_array, f"nb-{input_data_name}-structure", "run_test"
    )
    return config_path


def get_pc_stable_structure(
    data_train, insert_dataset_name, method="chisq", independence_threshold=0.05
):
    """
    Method to generate the PC Stable structure
    """
    pcs_test = pcs(data_train, method, independence_threshold)

    # Evaluating the skelton
    pcs_test.evaluate_skeleton(with_plots=False, log_level=0)

    # Evaluating the collider nodes aka (immoralities)
    # pcs_test.evaluate_immoralities()
    # pcs_test.create_directional_edge_using_immorality()

    # Generating a random DAG
    rand_dag = pcs_test.randomised_directed_graph()

    ## Display the DAG
    nx.draw_shell(rand_dag, with_labels=True)
    plt.show()

    # Creating the config file in preparation for the CPT (or PDF) Generator
    data = utils.topological_sort_for_structure(rand_dag)
    config_path = utils.config_structure_file(
        data, f"pc-{insert_dataset_name}-structure", "run_test"
    )

    return config_path


def find_best_pc_structure(
    data_train,
    insert_file_name,
    method="chisq",
    independence_threshold=0.05,
    max_iterations=10000,
):
    """
    Finds the best structure for a PC algorithm based on the given training data.
    over the given number of iterations.

    Parameters:
    data_train: The training data for the PC algorithm.
    insert_file_name (str): The name of the file where the resulting structure will be inserted.
    method (str, optional): The statistical test method to use for independence tests. Defaults to "chisq" (Chi-square test).
    independence_threshold (float, optional): The p-value threshold for the independence tests. If the p-value is above this threshold, the variables are considered independent. Defaults to 0.05.
    max_iterations (int, optional): The maximum number of iterations for the PC algorithm. Defaults to 10000.

    Returns:
    array of best structures evaluated by the PC algorithm
    """
    iteration = 0
    structures = []
    best_accuracy = 0

    pcs_test = pcs(data_train, method, independence_threshold)
    pcs_test.evaluate_skeleton(with_plots=False, log_level=1)

    pbar = tqdm(
        total=max_iterations,
        bar_format="{l_bar}%s{bar}%s{r_bar}" % (colour_dict[colour], Style.RESET_ALL),
    )
    pbar.set_description(
        f"Evaluating PC using {method} with independence threshold {independence_threshold}"
    )

    while iteration < max_iterations:
        start_time = time.time()
        rand_dag = pcs_test.randomised_directed_graph()
        rand_dag_time = time.time() - start_time

        data = utils.topological_sort_for_structure(rand_dag)
        config_path = utils.config_structure_file(
            data,
            f"pc-automated/config-pc-{insert_file_name}-{iteration}-structure",
            "best-structure",
        )

        start_time = time.time()
        CPT_Generator(config_path, train_data)
        cpt_generator_time = time.time() - start_time

        evaluator = me(config_path, train_data, test_data)
        computed_performance = evaluator.computed_performance

        evaluation_data = {
            "balanced_accuracy": computed_performance["bal_acc"],
            "structure": data,
            "config_path": config_path,
            "rand_dag_time": rand_dag_time,
            "cpt_generator_time": cpt_generator_time,
            "iteration": iteration,
        }

        if computed_performance["bal_acc"] > best_accuracy:
            best_accuracy = computed_performance["bal_acc"]
            best_structure = evaluation_data
            print("-----------------")
            print("New Best Structure")
            print("Accuracy:", best_accuracy)
            print(best_structure)
            structures.append(evaluation_data)

        iteration += 1
        pbar.update(1)
        pbar.set_postfix_str(f"Best Accuracy: {best_accuracy}")
    return structures


def evaluating_pc_stable(
    train_data, structure_name, method, independence_threshold, max_iterations=100
):
    """
    This method makes use of the find_best_pc_structure but also prints and stored the best data found into
    a file for later use. Primarily created for metrics generation.
    """
    best_structures = find_best_pc_structure(
        train_data,
        structure_name,
        method,
        independence_threshold=independence_threshold,
        max_iterations=max_iterations,
    )
    config_file = best_structures[-1]["config_path"]
    cpt_generator_time = best_structures[-1]["cpt_generator_time"]
    evaluator = me(config_file, train_data, test_data)
    computed_performance = evaluator.computed_performance
    now = datetime.datetime.now()
    generated_at = now.strftime("%m-%d_%H:%M:%S%f") + now.strftime("%f")[:3]
    eval_data = f"""
------------------------------------------------------------
Method: {method}
Independence Threshold: {independence_threshold}
Max Iterations: {max_iterations}
Iteration: {best_structures[-1]["iteration"]}
Completed at: {generated_at}
--------------------
Config: {config_file}
Brier Score: {computed_performance["brier"]}
KL Divergence: {computed_performance["kl_div"]}
Training Time: {cpt_generator_time}
Inference Time: {computed_performance["inference_time"]}
Balanced Accuracy: {computed_performance["bal_acc"]}
F1 Score: {computed_performance["f1"]}
Area Under Curve: {computed_performance["auc"]}
--------------------
Iteration: {best_structures[-1]["iteration"]}

random_variables: {best_structures[-1]["structure"]["random_variables"]}

structure: {best_structures[-1]["structure"]["structure"]}
------------------------------------------------------------
"""
    print(eval_data)
    file_name = str(
        f"best_structures-{structure_name}-{method}-{independence_threshold}.txt"
    )
    with open(file_name, "a") as f:
        f.write(eval_data)

    for i in range(5):
        if len(best_structures) == 0:
            break
        print(best_structures.pop())


def evaluate_effect_of_changing_independence_val(
    data_train,
    method="chisq",
    ind_thresh_min=0.04,
    ind_thresh_max=0.05,
    iterations_per_thresh=5,
    threshold_increment=0.005,
):
    """
    Evaluates the effect of changing the independence threshold on the performance of the algorithm.

    This function runs the algorithm multiple times with different independence thresholds and compares the results.
    The aim is to understand how changing the independence threshold affects the performance and accuracy of the algorithm.

    TODO: This function is not complete. It currently is found stuck in a loop when generating the random directional edges.

    Parameters:
    data_train: The training data file for the algorithm.
    method (str, optional): The statistical test method to use for independence tests. Defaults to "chisq" (Chi-square test).
    ind_thresh_min (float, optional): The minimum independence threshold to start with. Defaults to 0.04.
    ind_thresh_max (float, optional): The maximum independence threshold to end with. Defaults to 0.05.
    iterations_per_thresh (int, optional): The number of iterations to run the algorithm for each threshold. Defaults to 5.
    threshold_increment (float, optional): The increment to apply to the threshold after each set of iterations. Defaults to 0.005.

    Returns:
    a dict of the evaluation data and also saves the data to a file.
    """
    if ind_thresh_min > ind_thresh_max:
        raise ValueError("ind_thresh_min must be less than ind_thresh_max")

    total_array_size = (
        int((ind_thresh_max - ind_thresh_min) / threshold_increment)
        * iterations_per_thresh
    )
    plot_data = {}
    plot_data["training_data"] = data_train
    plot_data["method"] = method
    plot_data["threshold_min"] = ind_thresh_min
    plot_data["threshold_maxx"] = ind_thresh_max
    plot_data["independence_threshold"] = {}
    threshold = ind_thresh_min
    now = datetime.datetime.now()
    generation_started_at = now.strftime("%m-%d_%H:%M:%S%f") + now.strftime("%f")[:1]
    file_name = str(f"independence-evaluation-{method}-{generation_started_at}.txt")

    pbar = tqdm(
        total=total_array_size,
        bar_format="{l_bar}%s{bar}%s{r_bar}" % (colour_dict[colour], Style.RESET_ALL),
    )
    for i in range(int((ind_thresh_max - ind_thresh_min) / threshold_increment)):  #
        print(i)
        pbar.set_description(
            f"Getting data for {method} with independence threshold {threshold})"
        )
        plot_data["independence_threshold"][threshold] = []
        pcs_test = pcs(data_train, method, threshold)
        pcs_test.evaluate_skeleton(with_plots=False, log_level=1)
        for j in range(iterations_per_thresh):
            print(j)
            # FIXME: THIS IS STUCK IN SOME LOOP FOR SOME REASON
            start_time = time.time()
            rand_dag = pcs_test.randomised_directed_graph()
            rand_dag_time = time.time() - start_time

            data = utils.topological_sort_for_structure(rand_dag)
            config_path = utils.config_structure_file(
                data,
                "metrics-configs/effect-of-independence-overwritable-structure",
                "best-structure",
            )

            start_time = time.time()
            CPT_Generator(config_path, train_data)
            cpt_generator_time = time.time() - start_time

            evaluator = me(config_path, train_data, test_data)
            computed_performance = evaluator.computed_performance

            plot_data["independence_threshold"][threshold].append(
                {
                    "method": method,
                    "time_generated": int(datetime.datetime.now().timestamp()),
                    "balanced_accuracy": computed_performance["bal_acc"],
                    "f1_score": computed_performance["f1"],
                    "brier_score": computed_performance["brier"],
                    "kl_divergence": computed_performance["kl_div"],
                    "area_under_curve": computed_performance["auc"],
                    "structure": data,
                    "config_path": config_path,
                    "rand_dag_time": rand_dag_time,
                    "cpt_generator_time": cpt_generator_time,
                }
            )
        threshold += threshold_increment
        pcs_test = None
        with open(f"metrics/{file_name}.json", "a") as f:
            json.dump(plot_data, f)
    return plot_data


def nb_classifier(train_file, test_file):
    """
    Performs Naive Bayes classification on the given training data.
    """
    nb_fitted = nbc(train_file)
    if test_file is not None:
        nb_tester = nbc(test_file, nb_fitted)
        return nb_tester
    return nb_fitted


def bayes_net_exact_inference(config_file, prob_query, num_samples):
    """
    Performs exact inference on a Bayesian network.

    This function uses the configuration file to construct the Bayesian network,
    then performs exact inference to calculate the probability of the query.
    """
    exact_inference = bni(
        alg_name="InferenceByEnumeration",
        file_name=config_file,
        prob_query=prob_query,
        num_samples=num_samples,
    )
    return exact_inference


def rejection_sampling(config_file, prob_query, num_samples):
    """
    Performs exact Rejection Sampling on a Bayesian network.

    This function uses the configuration file to construct the Bayesian network,
    then performs rejection sampling to calculate the probability of the query.
    """
    rejection_sampling = bni(
        alg_name="RejectionSampling",
        file_name=config_file,
        prob_query=prob_query,
        num_samples=num_samples,
    )
    return rejection_sampling


if __name__ == "__main__":
    #### Diabetes Data
    # train_data = "data/discreet/diabetes_data-discretized-train.csv"
    # test_data = "data/discreet/diabetes_data-discretized-test.csv"
    # structure_name = "diabetes-discrete"
    # target_value = "Outcome"

    # config_file = "config/structure-learn/config-nb-diabetes-structure-run_test-11-15_17:36:25.txt"  # Naive Bayes
    # config_file = "config/pc-automated/config-pc-diabetes-structure-best-structure-11-16_16:58:08.txt"  # Chisq 0.05
    # config_file = '' # Chisq 0.01
    # config_file = "config/pc-automated/config-pc-diabetes-structure-best-structure-11-16_17:46:58-best.txt"  # Gsq 0.05
    # config_file = '' # Gsq 0.01

    #### Cardiovascular Data
    # train_data = "data/discreet/cardiovascular_data-discretized-train.csv"
    # test_data = "data/discreet/cardiovascular_data-discretized-test.csv"
    # structure_name = "pc-chisq-0.01-cardiovascular-discrete"
    # target_value = "target"

    # config_file = "config/nb-cardiovascular-structure.txt"  # Naive Bayes
    # config_file = "config/pc-automated/config-pc-pc-chisq-0.05-cardiovascular-discrete-structure-best-structure-11-18_20:03:24.txt"  # chi 0.05
    # config_file = "config/pc-automated/config-pc-pc-chisq-0.01-cardiovascular-discrete-5-structure-best-structure-11-18_21:27:07.txt"  # chi 0.01
    # config_file = "config/pc-automated/config-pc-cardiovascular-discrete-structure-best-structure-11-17_13:42:53.txt"  # gsq 0.05
    # config_file = "config/pc-automated/config-pc-cardiovascular-discrete-structure-best-structure-11-17_13:01:48.txt"  # gsq 0.01

    # config_file = get_naive_bayes_struct(train_data,structure_name,test_data,target_value) ??

    #########################################
    ###### Manually running PC stable #######
    #########################################

    # pcs_test = pcs(train_data, "chisq", 0.05)
    # pcs_test.evaluate_skeleton(with_plots=False, log_level=2)
    # pcs_test.evaluate_immoralities()
    # pcs_test.create_directional_edge_using_immorality()
    # rand_dag = pcs_test.randomised_directed_graph()
    # nx.draw_shell(rand_dag, with_labels=True)
    # plt.show()

    # ###############################################
    # #### Finding the Best PC Stable Structure  ####
    # ###############################################

    # evaluating_pc_stable(train_data, structure_name, method='chisq', independence_threshold=0.05, max_iterations=10)

    # configs = find_best_pc_structure(
    #     train_data,
    #     structure_name,
    #     method="chisq",
    #     independence_threshold=0.01,
    #     max_iterations=50,
    # )
    # print(configs)

    ###############################################
    ###### Config Structure File Evaluation #######
    ###############################################

    # start_time = time.time()
    # CPT_Generator(config_file, train_data)
    # cpt_generator_time = time.time() - start_time

    # evaluator = me(config_file, train_data, test_data)
    # computed_performance = evaluator.computed_performance
    # print("------------------------------------------------------------")
    # print("**Config**:", config_file)
    # # print("Training Time:", cpt_generator_time)
    # print("Inference Time:", computed_performance["inference_time"])
    # print("Balanced Accuracy:", computed_performance["bal_acc"])
    # print("F1 Score:", computed_performance["f1"])
    # print("Area Under Curve:", computed_performance["auc"])
    # print("Brier Score:", computed_performance["brier"])
    # print("KL Divergence:", computed_performance["kl_div"])
    # print("\n--------------------")

    #############################
    ###### Run Inference ########
    #############################
    # inference_query = "P(Outcome|Glucose=4,BMI=1,Age=5)"  # diabetes
    # inference_query = "P(Outcome|Glucose=2,BMI=2,Age=2)"  # diabetes
    # inference_query = "P(target|height=1,weight=5,ap_hi=2,ap_lo=2,gluc=1,smoke=0,alco=0)"  # cardiovascular
    # inference_query = (
    #     "P(target|height=2,weight=2,ap_hi=3,ap_lo=2,gluc=1,smoke=0,alco=0)"
    # )
    # inference_query = (
    #     "P(target|height=4,weight=2,ap_hi=2,ap_lo=2,gluc=1,smoke=0,alco=0)"
    # )

    # print("**Query**:", inference_query)
    # exact_inference = bayes_net_exact_inference(config_file, inference_query, 1000)
    # rej = rejection_sampling(config_file, inference_query, 1000)
    # rej = rejection_sampling(config_file, inference_query, 10000)
    # rej = rejection_sampling(config_file, inference_query, 100000)
    # rej = rejection_sampling(config_file, inference_query, 200000)
    # print("\n------------------------------------------------------------")
