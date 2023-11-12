from pcstable import PCStable as pcs
import networkx as nx

def main():
    
    # PCStable = PCStable('data/lung_cancer-train.csv')
    pcs_test = pcs('data/diabetes_data-discretized-train.csv')

    # pcs_test.identifySkeleton(with_subplots=True)
    
    #IDENTIFY SKELETON ALGO
    # iterations = 3
    # with_subplots = True
    #TODO: i think iterations should not really apply and should auto reach a skelton
    # Part 1: get all the edges with no parents
    # pcs_test.initialise_edges()

    for i in range(5):
        connected_nodes = pcs_test.getting_connected_nodes_for_path_length(i+1)
        edges = pcs_test.create_valid_path_set(connected_nodes)
        print(f'iteration{i}')
        for edge in edges:
            if pcs_test.graph.has_edge(edge.var_i,edge.var_j) and edge.is_conditionally_independent:
                pcs_test.graph.remove_edge(edge.var_i, edge.var_j)
                print(f'Severed: {edge.var_i} {edge.var_j}')
                
        pcs_test.drawPlot()   
    

    # connected_nodes = pcs_test.getting_connected_nodes_for_path_length(2)
    # edges = pcs_test.create_valid_path_set(connected_nodes)
    # for edge in edges:
    #     # check if nodes are directly connected, if so detach
    #     if pcs_test.graph.has_edge(edge.var_i,edge.var_j) and edge.is_conditionally_independent:
    #         pcs_test.graph.remove_edge(edge.var_i, edge.var_j)
    # pcs_test.subplot(3)
    # print(f'hello 3')
    
    
    
    
    
    
    
    pcs_test.draw()

    
    
    
    
    
    # get all nodes
    # nodes = pcs_test.get_graph_nodes()

    # Part 2: independence on parents = []
    connections = 3
    
    # con = []
    
    # getting all the connected nodes for of 
    print('getting connected nodes')
    
    # output = []
    # # output2 = {}
    # for v_i in nodes:
#     for v_j in nodes:
    #         if v_j == v_i: continue
    #         print(f'v_i = {v_i}, v_j = {v_j}')
    #         paths = list(nx.all_simple_edge_paths(pcs_test.graph, source = v_i, target = v_j, cutoff=connections))
    #         for path in paths:
    #             if len(path) == connections:
    #                 parents = set(node for edge in path for node in edge if node not in [v_i, v_j])
    #                 # print(f'path = {v_i} -> {list(parents)} -> {v_j}')
    #                 output.append([v_i, v_j ,parents])
                    
    #                 # # ALTERNATIVE????
    #                 # par_str = ','.join(list(parents))
    #                 # if par_str not in output2.keys():
    #                 #     output2[par_str] = []
    #                 # output2[par_str].append([v_i,v_j])
                    
    # connected_nodes = pcs_test.getting_connected_nodes_for_path_length(1)
    # edges = pcs_test.create_valid_path_set(connected_nodes)
    # pcs_test.subplot(1)
    # pcs_test.draw()
    # testing node independence
                       
    print('end')
            # i = 0
            # for i in range(len(con)): 
            #     path = list(con)[i]
            #     if len(path) > 0: print(f'v_i = {v_i}, v_j = {v_j} path_{i} = {list(list(con)[i])}')

            # vi_connections  
            # if 
        # cond = []
            
        

    
    
    
    
    # part 2.1 get connected nodes of size 1
    
    
    
    # for i in range(iterations):
    #     pcs_test.independence_test(i)
    #     if with_subplots: pcs_test.subplot(i+2)
    
    
    
    
    
    # Step 0: setup a completely undirected graph
    # self.graph
    
    # Step 1: get all the edges with no parents
    # self.identifySkeleton()

    # # Step 2: Identify the immoralities (v-structures) and orient them
    # self.identifyImmoralities()
    
    # # Step 3: Identify the remaining edges and orient them
    # self.identifyQualifyingEdges()
    
    # conditioning_variables = ['a','b']
    # edges = [['a','f'],['b','d'],['e','g']] 
    # for edge in edges:
    #     for con in conditioning_variables:
    #         if con not in edge:
    #             print(f'edge: {edge}, cond: {con}')
    #         else:
    #             break
    #     print(' ')
    
if __name__ == '__main__':
    main()