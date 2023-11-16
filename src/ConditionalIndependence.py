#############################################################################
# ConditionalIndependence.py
#
# Implements functionality for conditional independence tests via the
# library causal-learn (https://github.com/cmu-phil/causal-learn), which
# can be used to identify edges to keep or remove in a graph given a dataset.
# The flag 'chi_square_test' can be used to change tests between X^2 and G^2.
#
# This requires installing the following (at Uni-Lincoln computer labs):
# 1. Type Anaconda Prompt in your Start icon
# 2. Open your terminal as administrator
# 3. Execute=> pip install causal-learn
#
# USAGE instructions to run this program can be found at the bottom of this file.
#
# Version: 1.0, Date: 19 October 2022 (first version)
# Version: 1.1, Date: 07 October 2023 (minor revision)
# Version: 1.2, Date: 03 November 2023 (support for continuous data)
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys  
import numpy as np
from causallearn.utils.cit import CIT, Chisq_or_Gsq
from causallearn.utils.cit import Chisq_or_Gsq as Chisq_or_Gsq

# from CIT import Chisq_or_Gsq, FisherZ


class ConditionalIndependence:
    chisq_obj = None
    rand_vars = []
    rv_all_values = []
    use_continuous_data = False
    continuous_data_tests = ['fisherz']
    discrete_data_tests = ['chisq', 'gsq']
    all_tests = continuous_data_tests + discrete_data_tests
    test = ''

    def __init__(self, file_name, test='chisq'):
        data = self.read_data(file_name)
        if test not in self.all_tests:
            raise ValueError(f'ERROR: Unknown test type {test} please use one of {self.all_tests}')
        self.test = test
        self.use_continuous_data = True if self.test in self.continuous_data_tests else False
        if not self.use_continuous_data:
            self.chisq_obj = Chisq_or_Gsq(data, self.test)
        else:
            raise NotImplementedError('Continuous data not yet supported')
            # self.chisq_obj = CIT(data, self.test)

    def read_data(self, data_file):
        ## print("\nREADING data file %s..." % (data_file))
        ## print("---------------------------------------")

        with open(data_file) as csv_file:
            for line in csv_file:
                line = line.strip()
                if len(self.rand_vars) == 0:
                    self.rand_vars = line.split(',')
                else:
                    values = line.split(',')
                    values = list(map(float, values)) if self.use_continuous_data else values
                    self.rv_all_values.append(values)

        ## print("RANDOM VARIABLES=%s" % (self.rand_vars))
        ## print("VARIABLE VALUES (first 10)=%s" % (self.rv_all_values[:10])+"\n")
        returnVals = np.array(self.rv_all_values) if self.use_continuous_data else self.rv_all_values
        return returnVals

    def parse_test_args(self, test_args) -> tuple[int, int, list[int]] :
        main_args = test_args[2:len(test_args)-1]
        if (main_args.find('|')):
            variables = main_args.split('|')[0].split(',')
            parents = main_args.split('|')[1].split(',')
        else:
            variables = main_args.split(',')
            parents = []
        
        if len(variables) > 2:
            raise Exception("ERROR: Only 2 variables Vi and Vj are supported")
        return variables[0], variables[1], parents
    
    def compute_pvalue(self, var_i, var_j, parents):
        var_i = self.rand_vars.index(var_i)
        var_j = self.rand_vars.index(var_j)
        pars = [self.rand_vars.index(par) for par in parents]
        
        if self.chisq_obj is None:
            raise Exception("ERROR: chisq_obj is None")
        
        p = self.chisq_obj(var_i, var_j, pars)
        # ## print(f'X2test: Vi={self.rand_vars[var_i]}, Vj={self.rand_vars[var_j]}, pa_i={parents}, p={p}')
        return p
    
    @staticmethod
    def ChiIndependenceTest(train_data_path: str, var_i: str, var_j: str, parents: list[str] = []):
        ci = ConditionalIndependence(train_data_path, 'chisq')
        p = ci.compute_pvalue(var_i, var_j, parents)
        return p
    
    @staticmethod
    def GsqIndependenceTest(train_data_path: str, var_i: str, var_j: str, parents: list[str] = [], test = 'gsq'):
        if test not in ConditionalIndependence.all_tests:
            raise ValueError(f'ERROR: Unknown test type {test} please use one of {ConditionalIndependence.all_tests}')
        ci = ConditionalIndependence(train_data_path, 'gsq')
        p = ci.compute_pvalue(var_i, var_j, parents)
        return p

    ########### Continuous Tests ###########
    @staticmethod
    def fisherzIndependenceTest(train_data_path, test_args = 'I(Smoking,Coughing|Lung_cancer)'):
        # TODO:
        raise NotImplementedError('Continuous data not yet supported')
    

# if __name__=="__main":
#     if len(sys.argv) != 3:
#         ## print("USAGE: ConditionalIndepencence.py [train_file.csv] [I(Vi,Vj|parents)]")
#         ## print("EXAMPLE1: python ConditionalIndependence.py lang_detect_train.csv \"I(X1,X2|Y)\“")
#         ## print("EXAMPLE2: python ConditionalIndependence.py lang_detect_train.csv \"I(X1,X15|Y)\“")
#         exit(0)
#     else:
#         data_file = sys.argv[1]
#         test_args = sys.argv[2]

#         ci = ConditionalIndependence(data_file)
#         V, parents_i = ci.parse_test_args(test_args)
#         ci.compute_pvalue(V, parents_i)
