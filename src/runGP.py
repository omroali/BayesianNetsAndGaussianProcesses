from GaussianProcess import GaussianProcess as gp
from ModelEvaluator import ModelEvaluator as me
import utils
import PDF_Generator as pdf

def model_evaluator(config_file ,data_file, test_file):
    model_evaluation = me(config_file , data_file, test_file)

def main():
    #Task 1
    training_data = 'data/diabetes_data-original-train.csv'
    testing_data = 'data/diabetes_data-original-test.csv'
    config_file = 'config/config-nb-diabetes-original.txt'
    # gaussian_process = gp(datafile_train, datafile_test)
    
    # get_naive_bayes_struct = utils.independent_probability_structure(training_data)
    # structure_data, structure_array = get_naive_bayes_struct
    # config_path = utils.config_structure_file(config_file ,structure_array, 'nb-diabetes-structure-original', 'run_test')

    
    model_evaluator(config_file, training_data, testing_data)

    
    # utils.config_structure_file(datafile_train, datafile_test)
    
    
    
    
    
if __name__ == "__main__":
    main()