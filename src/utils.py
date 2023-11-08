import numpy as np
import csv

from sympy import im

from BayesNetInference import BayesNetInference
from ConditionalIndependence import ConditionalIndependence

# import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
from itertools import combinations


def readCsv(filePath, *training: str):
    '''
    @param filePath is the path to the csv file
    @param training is the random variable to be evaluated
    @return a dictionary of the data
    '''
    read_data = np.genfromtxt(filePath, delimiter=",", dtype="<U11")
    # read_data = pd.read_csv(filePath)
    # header_data = list(read_data.columns)
    response_data = {}
    headers = {}
    
    
    if training:
        header_data = read_data[0]
        if training not in header_data:
            return "Must be a random variable in the dataset"
        [[idx_result]] = np.where(header_data == training)

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
        "type": "training" if training else "testing",
        "headers": headers,
        "variables": response_data,
    }

### Consider using dataclass instead for this?
class BayesianNetworkStructure:
    def __init__(self, name, structure, rand_vars, cpt_vals=None):
        # Validation
        # if (cpt_vals or rand_vars) is None:
        #     return ValueError(
        #         "There are no cpt values or\
        #                     random variables avaiblabe"
        #     )
        # if (cpt_vals and rand_vars) is not None:
        #     return ValueError(
        #         "Both cpt values and random\
        #                     variables are populated"
        #     )

        # Creating the Class structure
        self.name = name
        self.structure = np.array(structure, dtype="S")
        self.rand_vards = self.setRandomVariableStructure()

        if cpt_vals:
            self.setUpCptStructure()

    #
    def setUpCptStructure(self):
        return 0

    def setRandomVariableStructure(self):
        return 0


def readBayesianNetworkStructure(filePath):
    values = None
    returnData = {}

    with open(filePath, newline="") as structure_data:
        struct = csv.reader(structure_data, delimiter=" ", quotechar="|")
        key = ""
        values = []
        # thinking for each row check if there is a ':'
        # if there is then assume the remaining data in that row belongs to it,

        for row in struct:
            if len(row) == 0:
                continue

            rowData = row[0]
            keyDelim = rowData.find(":")

            if keyDelim != -1:  # ie found a :
                colonSplitRowData = splitData(rowData, ":")
                key = colonSplitRowData[0]  # eg it it now structure
                if len(colonSplitRowData) > 1:
                    values = colonSplitRowData[1]
            else:
                values = rowData

            # if there are no values
            if values and len(values) == 0:
                continue

            dataDelim = rowData.find(";")
            if dataDelim != -1:  # ie found a ;
                values = splitData(values, ";")

            returnData = processData(returnData, key, values)
            # print(returnData)
    return returnData


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

    #                data = rowData[1]  # eg now P(B);P(E);P(A|B,E);P(J|A);P(M|A) dataDelim = data.find(';'); if dataDelim !== -1:  # ie found a ;
    #                for data.split(':'):

    # figure out what key is (key stays the same until it's changed)

    # for idx, row in enumerate(struct):
    #     if len(row) == 0:
    #         continue
    #
    #     # splitting data name: Alarm to array ['name', 'Alarm']
    #     keys = ["name", "random_variables", "structure", "CPT"]
    #     rowData = row[0].split(":")
    #
    #     # print(rowData)
    #     if rowData[0] == "name":  # TODO: possibly bug if values have key name
    #         name = rowData[1]
    #     if rowData[0] == "random_variables":
    #         random_variables = rowData[1]
    #     if rowData[0] == "structure":
    #         structure = rowData[1]
    #     if idx == 10:
    #         idx = 32
    #     print(idx)
    #
    # print(name)
    # print(random_variables)
    # print(structure)

    # read name
    # random_variables
    # structure

    # are there CPTs
    # CPTs

    # BayesianNetworkStructure(name, structure, rand_vars)


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
    data = readCsv(filePath, True)
    inputs = data["headers"]["input"]
    output = data["headers"]["output"]
    structure = []

    for input in inputs:
        structure.append(f"{output}|{input}")

    return structure


def readTrainingData(trainingDataPath, training: str):
    if trainingDataPath is None:
        return "training data path or data values must be provided"
    data = readCsv(trainingDataPath, training)
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
        conditional_probabilities[var + "|" + result] = calculateConditionalProbability(
            data, var, result
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


def independantProbabilityStructure(data):
    """
    evaluting all combinations of random vaiables
    """
    # reading data
    type, random_variables, result, input_data, output_data = data

    # step1 get all    
    marProb = calculateMarginalProbabilities(data)
    print(marProb[result])
    

    # step 2: get all possible conditional probabilities assuming all are independant
    condProb = allConditionalProbabilities(data)

    structure = condProb.copy()
    structure[result] = marProb[result]
    return structure


def formatIntoConfigStructureFile(evalVar, filePath, structureType="independant"):
    acceptable_struct_types = ['independant']
    data = readTrainingData(filePath, evalVar)
    type, random_variables, result, input_data, output_data = data
    variables = np.append(random_variables, result)

    if evalVar not in variables:
        return f"{evalVar} was not found in the list of random_variables\
        \nPlease selct from {variables}"

    # section to check the type of structure that will be used for the cpt
    if structureType not in acceptable_struct_types:
        return "structureType must be provided"
        
    if structureType == "independant":
        structureData = independantProbabilityStructure(data)
        
    
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

#########################################
############### Inference  ##############
#########################################
def InferenceByEnumeration():
    alg_name = 'InferenceByEnumeration'
    # file_name =
    # prob_query = 
    # num_samples = 
    BayesNetInference(alg_name, file_name, prob_query, num_samples)
    return 0

def RejectionSampling():
    alg_name = 'RejectionSampling'
    # file_name =
    # prob_query = 
    # num_samples = 
    BayesNetInference(alg_name, file_name, prob_query, num_samples)
    return 0

#########################################
########## Independece Testing ##########
#########################################
'''
for more independence tests, check the city.py file as it contains many tests
/home/krono/.local/lib/python3.10/site-packages/causallearn/utils/cit.py
'''

########### Discrete Tests ###########
def ChiIndependenceTest(train_data_path, test_args = 'I(Smoking,Coughing|Lung_cancer)'):
    ci = ConditionalIndependence(train_data_path, 'chisq')
    Vi, Vj, parents_i = ci.parse_test_args(test_args)
    ci.compute_pvalue(Vi, Vj, parents_i)
    return 0
    
def GsqIndependenceTest(train_data_path, test_args = 'I(Smoking,Coughing|Lung_cancer)'):
    ci = ConditionalIndependence(train_data_path, 'gsq')
    Vi, Vj, parents_i = ci.parse_test_args(test_args)
    ci.compute_pvalue(Vi, Vj, parents_i)
    return 0 

########### Continuous Tests ###########
def fisherzIndependenceTest(train_data_path, test_args = 'I(Smoking,Coughing|Lung_cancer)'):
    # TODO:
    return 0 




#########################################
## PC Algorithm for Structure Learning ##
#########################################    
def fullyConnectedGraph(nodes: list[str], G = nx.Graph()):
    # using the combinations method from itertools to create all possible edges
    edges = combinations(nodes, 2)
    
    # setting up the graphs
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    return G
    
def removeEdge(G, edge: list[str]):
    G.remove_edge(edge[0], edge[1])
    return G

def immortalityTest(parentNodes: list[str], conditioningNode: str, TrueGraph: nx.DiGraph):
    immoralEdgeCheck = [(parent, conditioningNode) for parent in parentNodes]
    print(immoralEdgeCheck)
    TrueGraphEdges = TrueGraph.edges()
    print(TrueGraphEdges)
    for edge in immoralEdgeCheck:
        if edge not in TrueGraphEdges:
            return False
    return True
    
def independenceOnNoCondition(G: nx.Graph, TrueGraph: nx.DiGraph) -> nx.Graph:
    
    edges = list(G.edges())
    nodes = list(G.nodes())
    
    for edge in edges:
        for node in nodes:
            if immortalityTest(edge, node, TrueGraph):
                removeEdge(G, edge)
    return G

def independenceOnCondition(G: nx.Graph, TrueGraph: nx.DiGraph, conditioningSet: list[str]) -> nx.Graph:
    # for all other pairs of variables (not the conditioningSet) 
        # detect the ones that are independent
    
    for condition in conditioningSet:    
        edges = list(G.edges())
    return G

    