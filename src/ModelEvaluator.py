#############################################################################
# ModelEvaluator.py
#
# Implements the following scoring functions and performance metrics:
# Log Likelihood (LL), Bayesian Information Criterion (BIC).
# Balanced Accuracy, F1 Score, Area Under Curve (AUC), 
# Brier Score, Kulback-Leibler Divergence (KLL), training/test times.
#
# IMPORTANT: This program currently makes use of two instantiations of
# NB_Classifier: one for training and one for testing. If you want this
# program to work for any arbitrary Bayes Net, the constructor (__init__) 
# needs to be updated to support a trainer (via CPT_Generator) and a
# tester (e.g., via BayesNetExactInference) -- instead of Naive Bayes models.
#
# This implementation also assumes that normalised probability distributions
# of predictions are stored in an array called "NB_Classifier.predictions".
# Performance metrics need such information to do the required calculations.
#
# This program has been tested for Binary classifiers. Minor extensions are
# needed should you wish this program to work for non-binary classifiers.
#
# Version: 1.0, Date: 03 October 2022, basic functionality
# Version: 1.1, Date: 15 October 2022, extended with performance metrics
# Version: 1.2, Date: 18 October 2022, extended with LL and BIC functions
# Version: 1.3, Date: 08 October 2023, refactored for increased reusability 
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys
import math
import time
import random
import numpy as np
from sklearn import metrics

import BayesNetUtil as bnu
from BayesNetReader import BayesNetReader
from NB_Classifier import NB_Classifier


class ModelEvaluator(BayesNetReader):
    verbose = False

    def __init__(self):
        print("EMPTY!!!")
        pass

    def __init__(self, configfile_name, datafile_train, datafile_test):
        # load Bayesian network stored in configfile_name
        super().__init__(configfile_name)
        # load Naive Bayes classifiers with training and test data
        nb_fitted = NB_Classifier(datafile_train)
        nb_tester = NB_Classifier(datafile_test, nb_fitted)
        true, pred, prob = self.get_true_and_predicted_targets(nb_tester)
        self.compute_performance(nb_tester, true, pred, prob)
        self.calculate_scoring_functions(nb_fitted)

    def calculate_scoring_functions(self, nbc):
        print("\nCALCULATING LL and BIC on training data...")
        LL = self.calculate_log_lilelihood(nbc)
        BIC = self.calculate_bayesian_information_criterion(LL, nbc)
        print("LL score="+str(LL))
        print("BIC score="+str(BIC))

    def calculate_log_lilelihood(self, nbc):
        LL = 0
 
        # iterate over all instances in the training data
        for instance in nbc.rv_all_values:
            predictor_value = instance[len(instance)-1]

            # iterate over all random variables except the predictor variable
            for value_index in range(0, len(instance)-1):
                variable = nbc.rand_vars[value_index]
                value = instance[value_index]
                parent = bnu.get_parents(variable, self.bn)
				###############################################
				## the following line should be updated in   ##
				## the case of multiple parents -- currently ##
				## only one parent is taken into account.    ##
				###############################################
                evidence = {parent: predictor_value}
                prob = bnu.get_probability_given_parents(variable, value, evidence, self.bn)
                LL += math.log(prob)

            # accumulate the log prob of the predictor variable
            variable = nbc.predictor_variable
            value = predictor_value
            prob = bnu.get_probability_given_parents(variable, value, {}, self.bn)
            LL += math.log(prob)
			
            if self.verbose == True:
                print("LL: %s -> %f" % (instance, LL))

        return LL

    def calculate_bayesian_information_criterion(self, LL, nbc):
        penalty = 0

        for variable in nbc.rand_vars:
            num_params = bnu.get_number_of_probabilities(variable, self.bn)
            local_penalty = (math.log(nbc.num_data_instances)*num_params)/2
            penalty += local_penalty

        BIC = LL - penalty
        return BIC

    def get_true_and_predicted_targets(self, nbc):
        Y_true = []
        Y_pred = []
        Y_prob = []

        # obtain vectors of categorical and probabilistic predictions
        for i in range(0, len(nbc.rv_all_values)):
            target_value = nbc.rv_all_values[i][len(nbc.rand_vars)-1]
            if target_value == 'yes': Y_true.append(1)
            elif target_value == 'no': Y_true.append(0)
            elif target_value == '1': Y_true.append(1)
            elif target_value == '0': Y_true.append(0)
            elif target_value == 1: Y_true.append(1)
            elif target_value == 0: Y_true.append(0)

            predicted_output = nbc.predictions[i][target_value]
            if target_value in ['no', '0', 0]:
                predicted_output = 1-predicted_output
            Y_prob.append(predicted_output)

            best_key = max(nbc.predictions[i], key=nbc.predictions[i].get)
            if best_key == 'yes': Y_pred.append(1)
            elif best_key == 'no': Y_pred.append(0)
            elif best_key == '1': Y_pred.append(1)
            elif best_key == '0': Y_pred.append(0)
            elif best_key == 1: Y_pred.append(1)
            elif best_key == 0: Y_pred.append(0)

        for i in range(0, len(Y_prob)):
            if np.isnan(Y_prob[i]):
                Y_prob[i] = 0

        return Y_true, Y_pred, Y_prob

    def compute_performance(self, nbc, Y_true, Y_pred, Y_prob):
        P = np.asarray(Y_true)+0.00001 # constant to avoid NAN in KL divergence
        Q = np.asarray(Y_prob)+0.00001 # constant to avoid NAN in KL divergence

        #print("Y_true="+str(Y_true))
        #print("Y_pred="+str(Y_pred))
        #print("Y_prob="+str(Y_prob))

        # calculate metrics to measure performance
        bal_acc = metrics.balanced_accuracy_score(Y_true, Y_pred)
        f1 = metrics.f1_score(Y_true, Y_pred)
        fpr, tpr, _ = metrics.roc_curve(Y_true, Y_prob, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        brier = metrics.brier_score_loss(Y_true, Y_prob)
        kl_div = np.sum(P*np.log(P/Q))
        print("\nCOMPUTING performance on test data...")
        print("Balanced Accuracy="+str(bal_acc))
        print("F1 Score="+str(f1))
        print("Area Under Curve="+str(auc))
        print("Brier Score="+str(brier))
        print("KL Divergence="+str(kl_div))
		
        if nbc != None:
            print("Training Time="+str(nbc.training_time)+" secs.")
            print("Inference Time="+str(nbc.inference_time)+" secs.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("USAGE: ModelEvaluator.py [config_file.txt] [training_file.csv] [test_file.csv]")
        print("EXAMPLE> ModelEvaluator.py config-lungcancer.txt lung_cancer-train.csv lung_cancer-test.csv")
        exit(0)
    else:
        configfile = sys.argv[1]
        datafile_train = sys.argv[2]
        datafile_test = sys.argv[3]
        ModelEvaluator(configfile, datafile_train, datafile_test)
