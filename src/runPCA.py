import re
from pcstable import PCStable as pcs
from matplotlib import pyplot as plt
import networkx as nx


def main():
    
    # PCStable = PCStable('data/lung_cancer-train.csv')
    pcs_test = pcs('data/diabetes_data-discretized-train.csv', method='chisq' ,independence_threshold=0.05)

    # pcs_test.identifySkeleton(with_subplots=True) 

    severless_count = 0
    iterations = 0
    # pcs_test.subplot(1)
    while (severless_count < 5):
        
        # plt.subplot(1,2,1)
        # nx.draw_shell(pcs_test.graph, with_labels=True)
        
        # pcs_test.subplot(1) 
        print(f'\n--iteration {iterations}--')
        sever = False
        connected_nodes = pcs_test.getting_connected_nodes_for_path_length(iterations)
        testing_nodes = pcs_test.getting_nodes_parent_sets_for_independence_testing(iterations)

        # for connected in connected_nodes: print(connected)
        # if len(connected_nodes) == 0: break #no more paths of this length exist
        
        edges = pcs_test.create_valid_path_set(testing_nodes)
        remove_edges = []
        for edge in edges:
            if edge.is_conditionally_independent[0]:
                remove_edges.append(edge)
                sever = True
                severless_count = 0
        
        for edge in remove_edges:
            if pcs_test.graph.has_edge(edge.var_i,edge.var_j):
                pcs_test.graph.remove_edge(edge.var_i, edge.var_j)
                print(f'Severed: {edge.var_i} {edge.var_j} | {edge.parents} || p = {edge.is_conditionally_independent[1]}')
                
        
        if sever == False:
            severless_count += 1
        iterations += 1
        
        # plt.subplot(1,2,2)
        # nx.draw_shell(pcs_test.graph, with_labels=True)
        # plt.show()
    
    pcs_test.drawPlot()


    # print('ending skeleton')
    # pcs_test.drawPlot()                    
    
    
    # Step 0: setup a completely undirected graph
    # self.graph
    
    # Step 1: get all the edges with no parents
    # self.identifySkeleton()

    # # Step 2: Identify the immoralities (v-structures) and orient them
    # self.identifyImmoralities()
    
    # # Step 3: Identify the remaining edges and orient them
    # self.identifyQualifyingEdges()

if __name__ == '__main__':
    main()