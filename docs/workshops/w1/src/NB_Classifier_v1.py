############################################################################
# NB_Classifier.py
#
# Implements the Naive Bayes classifier for simple probabilistic inference.
# It assumes the existance of data in CSV format, where the first line contains
# the names of random variables -- the last being the variable to predict.
# This implementation aims to be agnostic of the data (no hardcoded vars/probs)
#
# Version: 1.0, Date: 03 October 2022
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys
import math


class NB_Classifier:
    rand_vars = []
    rv_key_values = {}
    rv_all_values = []
    predictor_variable = None
    num_data_instances = 0
    default_missing_count = 0.000001
    probabilities = {}
    log_probabilities = False

    def __init__(self, file_name, fitted_model=None):
        if file_name is None:
            return
        else:
            self.read_data(file_name)

        self.read_data(file_name)
        if fitted_model is None:
            self.estimate_probabilities()
        else:
            self.rv_key_values = fitted_model.rv_key_values
            self.probabilities = fitted_model.probabilities
            self.test_learnt_probabilities()

    '''
    1 
    method to read and format all the relevant data in preparation for training or testing of the model
    '''
    def read_data(self, data_file):
        print("\nREADING data file %s..." % (data_file))
        print("---------------------------------------")

        self.rand_vars = [] # this is the array of key data (the random variables we are testing)
            # rand_vars = [
            #  [1,0,1,..],
            #  [0,1,1,...]
            # ]
        self.rv_key_values = {}  # the keys of the data with the possible outcome values
            # rv_key_values = [
            #   'Smoking' => [0,1],
            #   'Yellow_Fingers' => [],
            #   'Anxiety' => [1,0],
            #   ...
            # ]
        self.rv_all_values = [] #

        with open(data_file) as csv_file:
            for line in csv_file:
                line = line.strip() #removes spaces on either end of the data line 

                # getting the keys in the data
                if len(self.rand_vars) == 0:
                    
                    self.rand_vars = line.split(',')

                    # initialising the rv_key_values array with no array data
                    for variable in self.rand_vars:
                        self.rv_key_values[variable] = [] 
                else:
                    values = line.split(',') # [0,1,0,...]
                    self.rv_all_values.append(values) # stores array of arrays of the model training data
                    self.update_variable_key_values(values)
                    self.num_data_instances += 1 # counter for how many instance we have

        self.predictor_variable = self.rand_vars[len(self.rand_vars)-1] # last variabe, ie 'Lung_cancer'

        print("RANDOM VARIABLES=%s" % (self.rand_vars))
        print("VARIABLE KEY VALUES=%s" % (self.rv_key_values))
        print("VARIABLE VALUES=%s" % (self.rv_all_values))
        print("PREDICTOR VARIABLE=%s" % (self.predictor_variable))
        print("|data instances|=%d" % (self.num_data_instances))

    '''
    2
    method to evaluate all the possible outcomes found in the training data 
    ie if only 0s and 1s are found in the dataset for i = 3, 'Anxiety' then rv_key_values['Anxiety'] = [0,1]
    '''
    def update_variable_key_values(self, values):
        for i in range(0, len(self.rand_vars)):
            variable = self.rand_vars[i] # when i = 3, variable = 'Anxiety'
            key_values = self.rv_key_values[variable] # => key_values = rv_key_values['Anxiety] = [1,0]
            value_in_focus = values[i] # i = 3 --- values line 4  --->  value_in_focus = 1
            if value_in_focus not in key_values: # rv_key_values['Anxiety] = [1,0] only
                self.rv_key_values[variable].append(value_in_focus)

    '''
    3
    method to create an estimate of the probabilities
    '''
    def estimate_probabilities(self):
        countings = self.estimate_countings()
        prior_counts = countings[self.predictor_variable] # getting the counts for Lung_cancer

        ## TODO: CURRENTLY UP TO HERE NOW FURTHER EXPLOR 
        ## TEMPTED TO MOVE ALL THE PROCESSING METHODS INTO IT'S OWN UTIL FILE 
        ## TO MAKE IT SIMPLER TO UNDERSTAND
        print("\nESTIMATING probabilities...")
        for variable, counts in countings.items():
            prob_distribution = {}
            for key, val in counts.items():
                variables = key.split('|')

                if len(variables) == 1:
                    # prior probability
                    probability = float(val/self.num_data_instances)
                else:
                    # conditional probability
                    probability = float(val/prior_counts[variables[1]])

                if self.log_probabilities is False:
                    prob_distribution[key] = probability
                else:
                    # convert probability to log probability
                    prob_distribution[key] = math.log(probability)

            self.probabilities[variable] = prob_distribution

        for variable, prob_dist in self.probabilities.items():
            prob_mass = 0
            for value, prob in prob_dist.items():
                prob_mass += prob
            print("P(%s)=>%s\tSUM=%f" % (variable, prob_dist, prob_mass))



    '''
    4
    method to count how many symtoms point towardste liklyhood of outcome

    output structure
    countings = {
        #   ...,
        #   'Anxiety':{'0|0': 16,'1|1': 20,'0|1': 14,'1|0': 12},
        #   ...,
        #   'Lung_cancer':{'0': 15,'1': 7}
        # }
    '''
    def estimate_countings(self):
        print("\nESTIMATING countings...")

        countings = {}

        # [Smoking,Yellow_Fingers,Anxiety,Peer_Pressure,...,Lung_cancer]
        for variable_index in range(0, len(self.rand_vars)):
            variable = self.rand_vars[variable_index] # 'Anxiety' -> index 3

            if variable_index == len(self.rand_vars)-1: #last index ie 'Lung_cancer' if 3 == 11
                # prior counts
                countings[variable] = self.get_counts(None) # use the predictor_index (see inside method)
            else:
                # conditional counts
                countings[variable] = self.get_counts(variable_index) 
        
        #countings = {
        #   ...,
        #   'Anxiety':{'0|0': 16,'1|1': 20,'0|1': 14,'1|0': 12},
        #   ...,
        #   'Lung_cancer':{'0': 15,'1': 7}
        # }

        print("countings="+str(countings))
        return countings

    '''
    5
    method to evaluate the count of given index in the rv_all_values array
    ie look for how many times 'Anxiety' returns 1,0 and how many times that corresponds to
    Lung_cancer being 1,0 
    '''
    def get_counts(self, variable_index):
        counts = {}
        predictor_index = len(self.rand_vars)-1 #last index

        # counting true/false for all values for a given index
        for values in self.rv_all_values:
            if variable_index is None:
                # case: prior probability
                value = values[predictor_index] 
            else:
                # case: conditional probability
                # value bool for index eg 'Anxiety' 
                # and Lung_cancer bool for that row
                value = values[variable_index]+"|"+values[predictor_index]

            try:
                counts[value] += 1
            except Exception:
                counts[value] = 1

            # {
            #   '0|0': 5,
            #   '1|1': 1,
            #   '0|1': 2,
            #   '1|0': 2
            # }

        # verify countings by checking missing prior/conditional counts
        if variable_index is None:
            counts = self.check_missing_prior_counts(counts)
        else:
            counts = self.check_missing_conditional_counts(counts, variable_index)

        return counts

    def check_missing_prior_counts(self, counts):
        for var_val in self.rv_key_values[self.predictor_variable]:
            if var_val not in counts:
                print("WARNING: missing count for variable=" % (var_val))
                counts[var_val] = self.default_missing_count

        return counts

    def check_missing_conditional_counts(self, counts, variable_index):
        variable = self.rand_vars[variable_index]
        for var_val in self.rv_key_values[variable]:
            for pred_val in self.rv_key_values[self.predictor_variable]:
                pair = var_val+"|"+pred_val
                if pair not in counts:
                    print("WARNING: missing count for variables=%s" % (pair))
                    counts[pair] = self.default_missing_count

        return counts


    def test_learnt_probabilities(self):
        print("\nEVALUATING on test data...")

        # iterate over all instances in the test data
        for instance in self.rv_all_values:
            distribution = {}
            print("Input vector=%s" % (instance))

            # iterate over all values in the predictor variable
            for predictor_value in self.rv_key_values[self.predictor_variable]:
                prob_dist = self.probabilities[self.predictor_variable]
                prob = prob_dist[predictor_value]

                # iterate over all instance values except the predictor var.
                for value_index in range(0, len(instance)-1):
                    variable = self.rand_vars[value_index]
                    value = instance[value_index]
                    prob_dist = self.probabilities[variable]
                    cond_prob = value+"|"+predictor_value

                    if self.log_probabilities is False:
                        prob *= prob_dist[cond_prob]
                    else:
                        prob += prob_dist[cond_prob]

                distribution[predictor_value] = prob

            normalised_dist = self.get_normalised_distribution(distribution)
            print("UNNORMALISED DISTRIBUTION=%s" % (distribution))
            print("NORMALISED DISTRIBUTION=%s" % (normalised_dist))
            print("---")

    def get_normalised_distribution(self, distribution):
        normalised_dist = {}
        prob_mass = 0
        for var_val, prob in distribution.items():
            prob = math.exp(prob) if self.log_probabilities is True else prob
            prob_mass += prob

        for var_val, prob in distribution.items():
            prob = math.exp(prob) if self.log_probabilities is True else prob
            normalised_prob = prob/prob_mass
            normalised_dist[var_val] = normalised_prob

        return normalised_dist


if len(sys.argv) != 3:
    print("USAGE: NB_Classifier.py [train_file.csv] [test_file.csv]")
    exit(0)
else:
    file_name_train = sys.argv[1] #'lung_cancer-train.csv'  #sys.argv[1] 
    file_name_test = sys.argv[2] #'lung_cancer-test.csv' #sys.argv[2] 
    nb_fitted = NB_Classifier(file_name_train) # trained model
    nb_tester = NB_Classifier(file_name_test, nb_fitted) #testing the fitted model
