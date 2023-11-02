import numpy as np
import csv

# import pandas as pd


def readCsv(filePath, training=False):
    read_data = np.genfromtxt(filePath, delimiter=",", dtype="<U11")
    header_data = read_data[0]
    # read_data = pd.read_csv(filePath)
    # header_data = list(read_data.columns)
    response_data = {}
    headers = {}

    if training:
        headers["random_variables"] = header_data[:-1]
        headers["result"] = header_data[-1]
        # reponse_data['data'] = read_data.iloc[:, :-1]
        # reponse_data['result'] = read_data.iloc[:, -:-1]
        response_data["input_variables"] = read_data[1:, :-1]
        response_data["output"] = read_data[1:, -1]  # i.e. the last column
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


### Consider using dataclass instead for this?
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


"""
assumed structure
name: ...
random_variables = ...
structure = ...
CPTs = ...
"""


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


def calculateMarginalProbabilities(trainDataPath):
    data = readCsv(trainDataPath, True)
    type, random_variables, result, input_data, output_data = (
        data["type"],
        data["headers"]["random_variables"],
        data["headers"]["result"],
        data["variables"]["input_variables"],
        data["variables"]["output"],
    )

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


def calculateConditionalProbability(trainingData, prior, evidence):
    # P(O|PT) = P(O)*P(PT)/P(PT)
    data = readCsv(trainingData, True)
    type, random_variables, result, input_data, output_data = (
        data["type"],
        data["headers"]["random_variables"],
        data["headers"]["result"],
        data["variables"]["input_variables"],
        data["variables"]["output"],
    )

    variables = np.append(random_variables, result)
    variable_data = np.append(input_data, np.vstack(output_data), axis=1)

    # prior = 'O = sunny', evidence = 'PT = yes'
    # get index of prior

    # TODO: think about how to process nested situations, ie if
    # if prior is ('PT ^ O')
    # or evidence is ('W ^ T')

    # find unique values in data column with index idx
    idx_prior, prior_outputs, prior_counts = getUniqueValues(
        variables, variable_data, prior
    )

    idx_evidence, evidence_outputs, evidence_counts = getUniqueValues(
        variables, variable_data, prior
    )

    # for variable_data in variable_data[idx_prior]

    # conditional_probabilities = {}
    # result_header = data["headers"]["result"]
    # for random_variable in data["headers"]["random_variables"]:
    #     conditional_probabilities[random_variable] = (
    #         P[random_variable] * P[result_header]
    #     ) / P[result_header]

    # return conditional_probabilities


# @param random_variables is all the variables in the dataset
# @param variables_data is all the data in the dataset
# @param variable is the specifific variable we want to get the index and count of
def getUniqueValues(random_variables, variables_data, variable):
    [[idx]] = np.where(random_variables == variable)
    outputs, counts = np.unique(variables_data[:, idx], return_counts=True)
    return idx, outputs, counts
