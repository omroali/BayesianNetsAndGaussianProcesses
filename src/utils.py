from datetime import datetime   
from platform import node
from typing import final
import numpy as np
import csv

from sympy import im, topological_sort

# from BayesNetInference import BayesNetInference
# from ConditionalIndependence import ConditionalIndependence

# import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
from itertools import combinations

def read_csv(filePath, target: str):
    '''
    @param filePath is the path to the csv file
    @param target is the random variable to be evaluated
    @return a dictionary of the data
    '''
    read_data = np.genfromtxt(filePath, delimiter=",", dtype="<U32")
    response_data = {}
    headers = {}
    header_data = []
    
    
    if target:
        header_data = read_data[0]
        if target not in header_data:
            raise ValueError(f'{target} must be a random variable in the dataset')
        
        [[idx_result]] = np.where(header_data == target)
        # // find index of training
        headers["result"] = header_data[idx_result]
        headers["random_variables"] = np.delete(header_data, idx_result)
        response_data["output"] = read_data[1:, idx_result]  # i.e. the last column
        response_data["input_variables"] = np.delete(read_data, idx_result, axis=1)[1:]
        print(f'Training data formatted to evaluate {headers["result"]}')
    else:
        headers["random_variables"] = header_data
        response_data["input_variables"] = read_data
        print("Testing data formatted")

    return {
        "type": "training" if target else "testing",
        "headers": headers,
        "variables": response_data,
    }


# storing
def processData(
    returnData, key: str, values: list[str] | str
) -> dict[str, str | list[str]]:
    if key == "name":
        returnData["name"] = values
        return returnData

    if key not in returnData:
        returnData[key] = []

    # TODO: should i say the random_variables are the abbreviation? random_variable name is the full name?
    if key == "random_variables":
        for random_variable in values:
            returnData[key].append(random_variable)
        return returnData

    if key == "structure":
        for struct in values:
            returnData[key].append(struct)
        return returnData

    if "CPT" in key:
        # returnData["CPT"] = processCPT(returnData, key, values)
        # CPT obj
        #   random_variable: eg B
        #   Table:

        return returnData

    return returnData

def splitData(arrayData, delimitter):
    return list(filter(None, arrayData.split(delimitter)))


def histogram(data):
    # csv
    # x0,x1,x2,x3,x4,x5
    #  c, l, o, s, e, _
    return 0


def getHeaders(data):
    return data[0].split(",")


def getBasicStructure(filePath):
    data = read_csv(filePath, True)
    inputs = data["headers"]["input"]
    output = data["headers"]["output"]
    structure = []

    for input in inputs:
        structure.append(f"{output}|{input}")

    return structure


def read_training_data(trainingDataPath, training: str):
    if trainingDataPath is None:
        return "training data path or data values must be provided"
    data = read_csv(trainingDataPath, training)
    type, random_variables, result, input_data, output_data = (
        data["type"],
        data["headers"]["random_variables"],
        data["headers"]["result"],
        data["variables"]["input_variables"],
        data["variables"]["output"],
    )
    return type, random_variables, result, input_data, output_data

def calculateMarginalProbabilities(data):
    if data is not None:
        type, random_variables, result, input_data, output_data = data
    else:
        return "training data values must be provided"

    if type != "training":
        return "this should be training data type"

    marginal_probability = {}
    data_rows, data_cols = input_data.shape

    unique_inputs = np.unique(input_data)
    for variable in np.concatenate((random_variables, [result])):
        marginal_probability[variable] = {}

    #  get variable count for the number for the random variable counts
    for row in input_data:
        for idx, variable in enumerate(random_variables):
            if row[idx] in marginal_probability[variable]:
                marginal_probability[variable][row[idx]] += 1 / data_rows
            else:
                marginal_probability[variable][row[idx]] = 1 / data_rows

    unique_outputs, output_counts = np.unique(output_data, return_counts=True)
    for idx, output in enumerate(unique_outputs):
        marginal_probability[result][output] = output_counts[idx] / data_rows

    return marginal_probability

def allConditionalProbabilities(data):
    if data is not None:
        type, random_variables, result, input_data, output_data = data
    else:
        return "training data values must be provided"

    conditional_probabilities = {}
    for var in random_variables:
        conditional_probabilities[result + "|" + var] = calculateConditionalProbability(
            data, result, var
        )

    return conditional_probabilities

def calculateConditionalProbability(data, prior, evidence):
    """
    Method to calculate the conditional probabilies of the given dataset
    """
    # validation
    if data is not None:
        type, random_variables, result, input_data, output_data = data
    else:
        return "training data path or data values must be provided"

    if type != "training":
        return "this should be training data type"

    variables = np.append(random_variables, result)
    variable_data = np.append(input_data, np.vstack(output_data), axis=1)

    # find unique values in data column with index idx
    idx_prior, prior_data = getUniqueValues(variables, variable_data, prior)

    idx_evidence, evidence_data = getUniqueValues(variables, variable_data, evidence)

    conditional_probabilities = {}

    # initialising all the possible random variables with given outcomes
    for condition in prior_data:
        conditional_probabilities[condition] = {}
        for possible_outputs in evidence_data.keys():
            conditional_probabilities[condition][possible_outputs] = 0

    # looping through all rows and adding a count to determine the conditional probabilities
    for row in variable_data:
        curr_prior = row[idx_prior]
        curr_evidence = row[idx_evidence]
        conditional_probabilities[curr_prior][curr_evidence] += (
            1 / evidence_data[curr_evidence]
        )

    return conditional_probabilities


def getUniqueValues(random_variables, variables_data, variable):
    """
    @param random_variables is all the variables in the dataset
    @param variables_data is all the data in the dataset
    @param variable is the specifific variable we want to get the index and
           count of
    """
    [[idx_rand_vars]] = np.where(random_variables == variable)
    [variables, counts] = np.unique(
        variables_data[:, idx_rand_vars], return_counts=True
    )
    unique_values = {}
    for idx, variable in enumerate(variables):
        unique_values[variable] = counts[idx]
    return idx_rand_vars, unique_values


def independent_probability_structure(data):
    """
    evaluating all combinations of random variables for Naive Bayes
    """
    # reading data
    type, random_variables, result, input_data, output_data = data

    # step1 get all    
    marProb = calculateMarginalProbabilities(data)
    print(marProb[result])
    

    # step 2: get all possible conditional probabilities assuming all are independent
    condProb = allConditionalProbabilities(data)

    structure_data = condProb.copy()
    structure_data[result] = marProb[result]
    
    edges = []
    for struct in structure_data:
        edges.append(tuple(struct.split('|')))
        
    graph = nx.DiGraph()
    edges_list = []
    node_list = []
    for edge in edges:
        if len(edge) == 2: edges_list.append(edge)
        if len(edge) == 1: node_list.append(edge[0])
    graph.add_edges_from(edges_list)
    graph.add_nodes_from(node_list)

    structure_array = topological_sort_for_structure(list(node_list) ,graph)

    return structure_data, structure_array 


def formatIntoConfigStructureFile(evalVar, filePath, structureType="independent"):
    acceptable_struct_types = ['independant']
    data = read_training_data(filePath, evalVar)
    type, random_variables, result, input_data, output_data = data
    variables = np.append(random_variables, result)

    if evalVar not in variables:
        return f"{evalVar} was not found in the list of random_variables\
        \nPlease selct from {variables}"

    # section to check the type of structure that will be used for the cpt
    if structureType not in acceptable_struct_types:
        return "structureType must be provided"
        
    if structureType == "independent":
        structureData = independent_probability_structure(data)
        
    
    newFilePath = "config/config-" + evalVar.lower() + "-created.txt"
    with open(newFilePath, "w") as file:
        file.write(f"name:{evalVar}({evalVar})")

        file.write("\n\nrandom_variables:")
        vars = []
        for var in np.append(random_variables, result):
            vars.append(f'P({var})') 
               
        file.write(';'.join(vars))
        # how to split array into string with ; sperator
        # file.write(random_variables.join(";"))
        file.write("\n\nstructure:")
        structs = []
        for struct in structureData:
            structs.append(f'P({struct})')
        file.write(';'.join(structs))
    
    return newFilePath



def config_structure_file(node_structure: dict, file_name: str, unique: str = '') -> str:
    '''
    @param: node_structure = {
            'random_variables':'var1(var1);...;varn(varn)',
            'structure':'P(var1);P(var1|var2);...;P(varn|varm,varo)'
        }
    
    '''
    random_variables, structure = node_structure.values()
    if unique != '': unique = '-' + unique + '-' + datetime.utcnow().strftime("%m-%d_%H:%M:%S")
    new_file_path = f'config/config-{file_name}{unique}.txt'
    with open(new_file_path, "w") as file:
        file.write(f"name:{file_name}")
        file.write("\n\nrandom_variables:" + random_variables)
        file.write("\n\nstructure:" + structure)
    return new_file_path

def topological_sort_for_structure(di_graph: nx.DiGraph):
    output_struct = []
    random_variables = []

    struct_list = [(node,list(di_graph.predecessors(node))) for node in nx.topological_sort(di_graph)]
    for [node, parents] in struct_list:
        if len(parents) > 0:
            output_struct.append(f'P({node}|{",".join(parents)})')
        else:
            output_struct.append(f'P({node})')

    for node in list(nx.topological_sort(di_graph)):
        random_variables.append(f'{node}({node})')

    return {'random_variables': ';'.join(random_variables), 'structure': ';'.join(output_struct)}


#########################################
############### Inference  ##############
#########################################
# def InferenceByEnumeration():
#     alg_name = 'InferenceByEnumeration'
#     # file_name =
#     # prob_query = 
#     # num_samples = 
#     BayesNetInference(alg_name, file_name, prob_query, num_samples)
#     return 0

# def RejectionSampling():
#     alg_name = 'RejectionSampling'
#     # file_name =
#     # prob_query = 
#     # num_samples = 
#     BayesNetInference(alg_name, file_name, prob_query, num_samples)
#     return 0

#########################################
########## Independece Testing ##########
#########################################
'''
for more independence tests, check the city.py file as it contains many tests
/home/krono/.local/lib/python3.10/site-packages/causallearn/utils/cit.py
'''

########### Discrete Tests ###########
def ChiIndependenceTest(train_data_path = '../data/cardiovascular_data-discretized-train.csv', test_args = 'I(weight,gluc|target)'):
    ci = ConditionalIndependence(train_data_path, 'chisq')
    Vi, Vj, parents_i = ci.parse_test_args(test_args)
    p = ci.compute_pvalue(Vi, Vj, parents_i)
    return p
    
def GsqIndependenceTest(train_data_path = '../data/cardiovascular_data-discretized-train.csv', test_args = 'I(weight,gluc|target)'):
    ci = ConditionalIndependence(train_data_path, 'gsq')
    Vi, Vj, parents = ci.parse_test_args(test_args)
    p = ci.compute_pvalue(Vi, Vj, parents)
    return p

########### Continuous Tests ###########
def fisherzIndependenceTest(train_data_path, test_args = 'I(Smoking,Coughing|Lung_cancer)'):
    # TODO:
    return 0 

#########################################
## PC Algorithm for Structure Learning ##
#########################################  

#

def ConditionNodes(G: nx.Graph, TrueGraph: nx.DiGraph) -> tuple[nx.Graph, list[str]]:
    edges = list(G.edges())
    nodes = list(G.nodes())
    conditioning_nodes = []
    
    for edge in edges:
        for node in nodes:
            if immortalityTest(TrueGraph, edge, node):
                conditioning_nodes.append(node)
                logging('EDGE REMOVAL', f'removing edge {edge} as it is immoral of {node}')
                removeEdge(G, edge)
    return G, conditioning_nodes


#########################################
############# Other Tools ###############
#########################################  

def logging(info:str, message):
    types = [
        'GRAPH NOTICE', 
        'CHECKING', 
        'IMMORAILITY TEST', 
        'IMMORALITY DETECTED', 
        # 'EDGE REMOVAL'
        # 'SHORTEST NODE PATHS', 
    ]
    
    if info in types:
        return False
    
    return print(f'{info}: {message}')

import multiprocessing.pool
import functools

def timeout(max_timeout):
    """Timeout decorator, parameter in seconds."""
    def timeout_decorator(item):
        """Wrap the original function."""
        @functools.wraps(item)
        def func_wrapper(*args, **kwargs):
            """Closure for function."""
            pool = multiprocessing.pool.ThreadPool(processes=1)
            async_result = pool.apply_async(item, args, kwargs)
            # raises a TimeoutError if execution exceeds max_timeout
            return async_result.get(max_timeout)
        return func_wrapper
    return timeout_decorator


#######Validation stuff 
# def validate_path_evaluation(func):
#     def validate(Graph: nx.Graph | nx.DiGraph, node_1: str, node_2: str, *cond_node: str, method = 'dijkstra'):
#         if node_1 not in Graph.nodes() or node_2 not in Graph.nodes():
#             raise ValueError("node_1 and node_2 must be in Graph")
#         if cond_node and cond_node not in Graph.nodes():
#             print(cond_node)
#             raise ValueError("conditional node must be in Graph")
#         if method not in ['dijkstra', 'bellman-ford']:
#             raise ValueError("method must be 'dijkstra' or 'bellman-ford'")
#         return func(Graph, node_1, node_2, method)
#     return validate