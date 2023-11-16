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
# Version: 1.3, Date: 21 October 2023, refactored for increased reusability 
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys
import math
import time
import random
import numpy as np
import os.path
from sklearn import metrics

import BayesNetUtil as bnu
from NB_Classifier import NB_Classifier
from BayesNetInference import BayesNetInference


class ModelEvaluator(BayesNetInference):
    verbose = False 
    useBayesNet = True # False uses NaiveBayes, True uses BayesNet
    inference_time = None

    def __init__(self, configfile_name, datafile_train, datafile_test):
        if os.path.isfile(configfile_name):
            # loads Bayesian network stored in configfile_name, where
            # the None arguments prevent any inference at this time
            super().__init__(None, configfile_name, None, None)
            self.inference_time = time.time()
        else:
            self.useBayesNet = False

        if self.useBayesNet is True:
            # reads test data in datafile_test using NB_Classifier
            nb_fitted = None # because a Bayes net will be used, not Naive Bayes
            nb_tester = NB_Classifier(None)
            nb_tester.read_data(datafile_test)

        else:
            # loads Naive Bayes classifiers with training and test data
            nb_fitted = NB_Classifier(datafile_train)
            nb_tester = NB_Classifier(datafile_test, nb_fitted)

        # generates performance results from the predictions above  
        self.inference_time = time.time()
        true, pred, prob = self.get_true_and_predicted_targets(nb_tester)
        self.inference_time = time.time() - self.inference_time
        self.compute_performance(nb_tester, true, pred, prob)
        self.calculate_scoring_functions(nb_fitted)

    # calculates scores based on metrics Log Likelihood (LL) and
    # Bayesian Information Criterion (BIC). Note that BIC extends 
    # LL with a penalty factor.
    def calculate_scoring_functions(self, nbc):
        if not self.bn: return
        print("\nCALCULATING LL and BIC on training data...")
        LL = self.calculate_log_lilelihood(nbc)
        BIC = self.calculate_bayesian_information_criterion(LL, nbc)
        print("LL score="+str(LL))
        print("BIC score="+str(BIC))

    # calculates LL scores but only for Naive Bayes classifiers.
    # You need to provide support for this to work with Bayes nets,
    # see comment below for a hint on what is needed.
    def calculate_log_lilelihood(self, nbc):
        LL = 0

        if self.useBayesNet:
            print("===============================================================")
            print("WARNING: This method currently works for discrete Naive Bayes! ")
            print(" You need to extend it to be able to work with Bayes nets.     ")
            print("===============================================================")
            return None
 
        # iterates over all data points (instances) in the training data
        for instance in nbc.rv_all_values:
            predictor_value = instance[len(instance)-1]

            # iterates over all random variables except the predictor variable
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

            # accumulates the log prob of the predictor variable
            variable = nbc.predictor_variable
            value = predictor_value
            prob = bnu.get_probability_given_parents(variable, value, {}, self.bn)
            LL += math.log(prob)
			
            if self.verbose == True:
                print("LL: %s -> %f" % (instance, LL))

        return LL

    def calculate_bayesian_information_criterion(self, LL, nbc):
        if LL is None:
            return

        penalty = 0
        for variable in nbc.rand_vars:
            num_params = bnu.get_number_of_probabilities(variable, self.bn)
            local_penalty = (math.log(nbc.num_data_instances)*num_params)/2
            penalty += local_penalty

        BIC = LL - penalty
        return BIC

    def get_true_and_predicted_targets(self, nbc):
        print("\nCARRYING-OUT probabilistic inference on test data...")
        Y_true = []
        Y_pred = []
        Y_prob = []

        # obtains vectors of categorical and probabilistic predictions
        for i in range(0, len(nbc.rv_all_values)):
            data_point = nbc.rv_all_values[i]
            target_value = data_point[len(nbc.rand_vars)-1]
            if target_value == 'yes': Y_true.append(1)
            elif target_value == 'no': Y_true.append(0)
            elif target_value == '1': Y_true.append(1)
            elif target_value == '0': Y_true.append(0)
            elif target_value == 1: Y_true.append(1)
            elif target_value == 0: Y_true.append(0)

            # obtains a probability distribution of predictions as a dictionary 
            # either from a Bayesian Network or from a Naive Bayes classifier.
            # example prob_dist={'1': 0.9532340821183165, '0': 0.04676591788168346}
            if self.useBayesNet:
                prob_dist = self.get_predictions_from_BayesNet(data_point, nbc)
            else:
                prob_dist = nbc.predictions[i]

            # retrieves the probability of the target_value and adds it to
            # the vector of probabilistic predictions referred to as 'Y_prob'
            try:
                predicted_output = prob_dist[target_value]
            except Exception:
                predicted_output = prob_dist[float(target_value)]
            if target_value in ['no', '0', 0]:
                predicted_output = 1-predicted_output
            Y_prob.append(predicted_output)

            # retrieves the label with the highest probability, which is
            # added to the vector of hard (non-probabilistic) predictions Y_pred
            best_key = max(prob_dist, key=prob_dist.get)
            if best_key == 'yes': Y_pred.append(1)
            elif best_key == 'no': Y_pred.append(0)
            elif best_key == '1': Y_pred.append(1)
            elif best_key == '0': Y_pred.append(0)
            elif best_key == 1: Y_pred.append(1)
            elif best_key == 0: Y_pred.append(0)

        # verifies that probabilities are not NaN (not a number) values -- 
        # in which case are replaced by 0 probabilities
        for i in range(0, len(Y_prob)):
            if np.isnan(Y_prob[i]):
                Y_prob[i] = 0

        return Y_true, Y_pred, Y_prob

    # returns a probability distribution using Inference By Enumeration
    def get_predictions_from_BayesNet(self, data_point, nbc):
        # forms a probabilistic query based on the predictor variable,
        # the evidence (non-predictor variables), and the values of
        # the current data point (test instance) given as argument
        evidence = ""
        for var_index in range(0, len(nbc.rand_vars)-1):
            evidence += "," if len(evidence)>0 else ""
            evidence += nbc.rand_vars[var_index]+'='+str(data_point[var_index])
        prob_query = "P(%s|%s)" % (nbc.predictor_variable, evidence)
        self.query = bnu.tokenise_query(prob_query, False)

        # sends query to BayesNetInference and get probability distribution
        self.prob_dist = self.enumeration_ask()
        normalised_dist = bnu.normalise(self.prob_dist)
        if self.verbose: print("%s=%s" % (prob_query, normalised_dist))

        return normalised_dist

    # prints model performance according to the following metrics:
    # balanced accuracy, F1 score, AUC, Brier score, KL divergence,
    # and training and test times. But note that training time is
    # dependent on model training externally to this program, which
    # is the case of Bayes nets trained via CPT_Generator.py	
    def compute_performance(self, nbc, Y_true, Y_pred, Y_prob):
        P = np.asarray(Y_true)+0.00001 # constant to avoid NAN in KL divergence
        Q = np.asarray(Y_prob)+0.00001 # constant to avoid NAN in KL divergence

        #print("Y_true="+str(Y_true))
        #print("Y_pred="+str(Y_pred))
        #print("Y_prob="+str(Y_prob))

        bal_acc = metrics.balanced_accuracy_score(Y_true, Y_pred)
        f1 = metrics.f1_score(Y_true, Y_pred)
        fpr, tpr, _ = metrics.roc_curve(Y_true, Y_prob, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        brier = metrics.brier_score_loss(Y_true, Y_prob)
        kl_div = np.sum(P*np.log(P/Q))

        if nbc != None: 
            classifier_type = "Bayes net" if self.useBayesNet else "Naive Bayes" 
            print("\nCOMPUTING performance of %s on test data:" % ("classifier_type"))
        else:
            print("\nCOMPUTING performance on test data...")

        print("Balanced Accuracy="+str(bal_acc))
        print("F1 Score="+str(f1))
        print("Area Under Curve="+str(auc))
        print("Brier Score="+str(brier))
        print("KL Divergence="+str(kl_div))
		
        try:
            if nbc != None and not self.useBayesNet:
                print("Training Time="+str(nbc.training_time)+" secs.")
                print("Inference Time="+str(nbc.inference_time)+" secs.")
            else:
                print("Training Time=this number should come from the CPT_Generator!")
                print("Inference Time="+str(self.inference_time)+" secs.")
        except Exception:
            pass


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
