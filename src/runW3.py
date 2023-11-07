from py import test
from sympy import Chi
from ConditionalIndependence import ConditionalIndependence
from ModelEvaluator import ModelEvaluator


def Task1a(train_data = 'data/lung_cancer-train.csv', test_args = 'I(Smoking,Coughing|Lung_cancer)'):
    ci = ConditionalIndependence(train_data, 'gsq')
    Vi, Vj, parents_i = ci.parse_test_args(test_args)
    ci.compute_pvalue(Vi, Vj, parents_i)
    
def Task2ScoringFunctionsAndModelPerfoemance():
    # Part a: what is the Log-likelihood of the Naive Bayes model (config-lungcancer.txt)
    config_path = 'config/config-lungcancer.txt'
    train_data = 'data/lung_cancer-train.csv'
    test_data = 'data/lung_cancer-test.csv'
    
    ModelEvaluator(config_path, train_data, test_data)
    
    
    
if __name__ == "__main__":
    train_data = 'data/lung_cancer-train.csv'
    Task1a(train_data, 'I(Smoking,Coughing|Lung_cancer)',)
    Task1a(train_data, 'I(Smoking,Car_Accident|Lung_cancer)')
    Task1a(train_data, 'I(Anxiety,Fatigue|Lung_cancer)')
    Task1a(train_data, 'I(Anxiety,Attention_Disorder|Lung_cancer)')
    Task1a(train_data, 'I(Allergy,Fatigue|Lung_cancer)')
    '''
    Results:
    Chi test:
    X2test: Vi=Smoking, Vj=Coughing, pa_i=['Lung_cancer'], p=0.7671012800093508
    X2test: Vi=Smoking, Vj=Car_Accident, pa_i=['Lung_cancer'], p=2.2223338414565005e-06
    X2test: Vi=Anxiety, Vj=Fatigue, pa_i=['Lung_cancer'], p=0.1588073812200198
    X2test: Vi=Anxiety, Vj=Attention_Disorder, pa_i=['Lung_cancer'], p=0.08855281302696125
    X2test: Vi=Allergy, Vj=Fatigue, pa_i=['Lung_cancer'], p=3.055680781825919e-19

    gsq test:
    X2test: Vi=Smoking, Vj=Coughing, pa_i=['Lung_cancer'], p=0.7726106454381072
    X2test: Vi=Smoking, Vj=Car_Accident, pa_i=['Lung_cancer'], p=3.706744827645817e-07
    X2test: Vi=Anxiety, Vj=Fatigue, pa_i=['Lung_cancer'], p=0.15671922762370036
    X2test: Vi=Anxiety, Vj=Attention_Disorder, pa_i=['Lung_cancer'], p=0.08976816134783991
    X2test: Vi=Allergy, Vj=Fatigue, pa_i=['Lung_cancer'], p=2.1212849341311613e-19
    '''