from pcstable import PCStable as pcs

def main():
    
    # PCStable = PCStable('data/lung_cancer-train.csv')
    pcs_test = pcs('data/diabetes_data-discretized-train.csv', method='chisq' ,independence_threshold=0.05)

    # pcs_test.identifySkeleton(with_subplots=True) 

    severless_count = 0
    iterations = 0
    while (severless_count < 5):
        pcs_test.drawPlot()
        print(f'iteration {iterations}')
        sever = False
        connected_nodes = pcs_test.getting_connected_nodes_for_path_length_new_new(iterations)
        # if len(connected_nodes) == 0: break #no more paths of this length exist
        
        edges = pcs_test.create_valid_path_set(connected_nodes)
        for edge in edges:
            if pcs_test.graph.has_edge(edge.var_i,edge.var_j) and not edge.is_conditionally_independent:
                pcs_test.graph.remove_edge(edge.var_i, edge.var_j)
                print(f'Severed: {edge.var_i} {edge.var_j}')
                sever = True
                severless_count = 0
        
        if sever == False:
            severless_count += 1
        iterations += 1

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