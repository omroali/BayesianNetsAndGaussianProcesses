
from pgmpy.estimators import PC,HillClimbSearch,ExhaustiveSearch,BicScore,K2Score,BDeuScore
from pgmpy.models import BayesianNetwork
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork


# Sample data
data = pd.read_csv(r'data/discreet/cardiovascular_data-discretized-test.csv')


#start=[('BMI', 'BloodPressure'), ('BMI', 'SkinThickness'), ('Outcome', 'Glucose'), ('Age', 'BloodPressure'), ('Pregnancies', 'Age')]

# Initialize the PC algorithm
estimator = HillClimbSearch(data)
 

#Structure learning algorithm
model = estimator.estimate() #(data))

#Print the learned model structure
print("Learned Bayesian Network Structure:")
nodes =model.edges()
print(nodes)
graph = nx.DiGraph(nodes)

store_structure_complete = ''
store_structure = []
for node in nodes:
    store_structure.append(f"P("+str(node[0])+"|"+str(node[1])+")")
store_structure_complete = ';'.join(store_structure)


# construct the tree graph structure
model = BayesianNetwork(nodes)
nx.draw_circular(
    model, with_labels=True, arrowsize=30, node_size=800, alpha=0.3, font_weight="bold"
)
plt.show()



