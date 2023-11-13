import re
from pcstable import PCStable as pcs
from matplotlib import pyplot as plt
import networkx as nx


def main():
    
    # PCStable = PCStable('data/lung_cancer-train.csv')
    pcs_test = pcs('data/diabetes_data-discretized-train.csv', method='chisq' ,independence_threshold=0.05)
    pcs_test.evaluate_skeleton(with_plots= True,log_level=0)
    pcs_test.evaluate_immoralities()
    print('something cool')
    pcs_test.create_directional_edge_using_immorality()
    
    # Step 1: get all the edges with no parents
    # self.identifySkeleton()

    # # Step 2: Identify the immoralities (v-structures) and orient them
    # self.identifyImmoralities()
    
    # # Step 3: Identify the remaining edges and orient them
    # self.identifyQualifyingEdges()

if __name__ == '__main__':
    main()