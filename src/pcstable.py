from matplotlib import pyplot as plt
import networkx as nx
from itertools import combinations

from sklearn import neighbors

from ConditionalIndependence import ConditionalIndependence as ci
from edge import Edge

class PCStable:
    graph = nx.Graph()
    
    def __init__(self, data_path, method = 'chisq', independence_threshold = 0.01) -> None:
        # initialise empty fully connected graph
        self.ci = ci(data_path, method)
        self.variables = self.ci.rand_vars
        self.graph = PCStable.fully_connected_graph(self.variables, self.graph)
        self.immoralNodes = []
        self.markovChains = []
        self.independence_threshold = independence_threshold
        self.all_nodes = self.graph.nodes()
        
        print(f'Using conditional test: {method}')
        print(f'Test threshold: {independence_threshold}')
                
    def addImmorality(self, node: str | list[str], edge: list[str]) -> None:
        self.immoralNodes.append([node, edge])
        
    def getImmoralities(self) -> list[list[str]]:
        return self.immoralNodes
    
    def addMarkovChain(self, vi: str, vj: str, parents: list[str]) -> None:
        self.markovChains.append([vi, vj, parents])
    
    def getMarkovChains(self) -> list[list[str]]:
        return self.markovChains    
        
    # def get_graph_nodes(self) -> list[str]:
    #     return list(self.graph.nodes())

    def getGraphEdges(self) -> list[str]:
        for edge in self.graph.edges():
            Edge(edge[0], edge[1], [], self.ci, [], threshold=self.independence_threshold)
        return list(self.graph.edges())
    
    def drawPlot(self):
        nx.draw_shell(self.graph, with_labels=True)
        plt.show()

    def create_valid_path_set(self, paths) -> list[Edge]:
        '''
        create edge<[vi,vj, [list[parents]]> structure so long as edge is not 
        contained in the parents 
        '''
        edges = []
        for path in paths:
            v_i = path[0]
            v_j = path[1]
            parents = path[2]
            if v_i in parents or v_j in parents:
                raise ValueError('v_i or v_j values should not be in parent list')
            
            edges.append(Edge(v_i, v_j, parents, conditional_independence_test=self.ci, threshold = self.independence_threshold))
        return edges
    
    # def independence_test(self):
    #     # TODO: SOME_CONDITION while the independence test continues to keep running
        
    #     # parent_combinations = list(combinations(self.get_graph_nodes(), iterations))
    #     # all_edges = list(combinations(self.get_graph_nodes(), 2))
        
    #     for conditionals in parent_combinations:
    #         conditionals = list(conditionals)
    #         # conditionals = self.all_adjacent_nodes         
    #         # self.all_paths_nodes(3)     

    #         # edges_set_2 = self.initialise_edges(conditionals)
    #         # print(self.listNodeParents(edges_set_2))
    #         for edge in edges_set_2:
    #             if edge.is_conditionally_independent:
    #                 self.graph.remove_edge(edge.var_i, edge.var_j)
    #             # if iterations > 0 and edge.is_immoral:
    #             #     self.addImmorality(conditionals, [edge.var_i, edge.var_j]) 
    #     # iterations += 1          
    
    def getting_connected_nodes_for_path_length(self, connections = 1):
        if connections == 0:
            raise ValueError("Connections must have a value larger than 1")
        output = []
        nodes = list(self.all_nodes)
        for v_i in nodes:
            v_i = nodes.pop(0)
            for v_j in nodes:
                if v_j == v_i: continue
                paths = list(nx.all_simple_edge_paths(self.graph, source = v_i, target = v_j, cutoff=connections))
                for path in paths:
                    if len(path)-1 == connections:
                        parents = set(node for edge in path for node in edge if node not in [v_i, v_j])
                        output.append([v_i, v_j ,parents])
        return output
    
    def getting_connected_nodes_for_path_length_new(self, connections = 0):
        # connections = path_cutoff or 1
        output = []
        nodes = list(self.all_nodes)
        
        for v_i in nodes:
            # v_i = nodes.pop(0)
            for v_j in nodes:
                if v_j == v_i: continue
                if connections == 0:
                    output.append([v_i, v_j ,[]])
                    continue
                
                for node in nodes:
                    paths = list(nx.all_simple_edge_paths(self.graph, source = v_i, target = node, cutoff=connections))
                    for path in paths:
                        if len(path) != connections: continue
                        parents = list(set(node for edge in path for node in edge if node not in [v_i, v_j]))
                        if len(parents) != connections: continue
                        output.append([v_i, v_j ,parents])

        return output
    
    def all_paths_nodes(self, connections):# -> list[list[str]]:
        all_edges = list(combinations(self.all_nodes(), 2))
        parent_nodes = []
        for edge in all_edges:
            parent_nodes = list(nx.all_simple_edge_paths(self.graph, source = edge[0], target = edge[1], cutoff=connections))
        print(parent_nodes)
    
    def getting_nodes_parent_sets_for_independence_testing(self, connections = 0):
        # connections = path_cutoff or 1
        output = []
        nodes = list(self.all_nodes)
        
        for v_i in nodes:
            for v_j in nodes:
                if v_j == v_i: continue
                if connections == 0:
                    output.append([v_i, v_j ,[]])
                    continue
                
                # for node in nodes:
                adjacent_sets = self.all_adjacent_nodes(v_i, connections)
                for set in adjacent_sets:
                    if v_i in set or v_j in set: continue
                    output.append([v_i, v_j, set])
        return output
    
    def all_adjacent_nodes(self, node, connections):# -> list[list[str]]:        
        # for node in self.all_nodes:
        neighbors = list(self.graph.neighbors(node))
        parents_combinations = set(combinations(neighbors, connections))
        return [list(combo) for combo in parents_combinations]


        
        # Magic sauce
        
        # set(list(combinations(adjacent_nodes_list[0][1].keys(), 2)))
        # for adjacent_nodes in adjacent_nodes_list:
        #     for node in adjacent_nodes[1].keys():
        #         continue
        #         # adjacent_nodes_output[adjacent_nodes[0]] = f
        #         # 1. find node in adjacent_nodes_list 
        #         # 2. create a pair but if on iteration 3, do it again (recursively)
        return adjacent_nodes_list
        
    def identifySkeleton(self, with_subplots = False, iterations = 3):
    #     '''
    #     get's the core skeleton shape given the independence conditions
    #     '''
    #     #TODO: i think iterations should not really apply and should auto reach a skelton
    #     # Part 1: get all the edges with no parents
    #     edges = self.initialise_edges()

    #     # Part 2: get the skeleton graph - I still don't trust
    #     for i in range(iterations):
    #         self.independence_test(i)
    #         if with_subplots: self.subplot(i+2)
    #     plt.show()
        raise NotImplementedError
      
        
    def subplot(self, subplot = 1):
        plt.subplot(2,2,subplot)
        nx.draw_spring(self.graph, with_labels=True)
        
    def draw(self):
        plt.show()
        
        
    
    @staticmethod
    def listNodeParents(edges):
        return [[edge.var_i, edge.var_j, edge.parents] for edge in edges]
            
    def identifyImmoralities(self):
        '''
        get's the immoralities given the independence conditions and the edge orientations
        '''
        raise NotImplementedError
    
    def identifyQualifyingEdges(self):
        '''
        get's the edges that qualify for orientation given the collider (immoral) nodes
        '''
        raise NotImplementedError
    
    ###################################################
    ############### STATIC METHODS ####################
    ###################################################
    
    @staticmethod 
    def listFullyConnectedEdges(nodes: list[str]) -> list[tuple[str, str]]:
        edges = combinations(nodes, 2)
        return list(edges)
    
    @staticmethod
    def fully_connected_graph(nodes: list[str], G = nx.Graph()) -> nx.Graph:
        # using the combinations method from itertools to create all possible edges
        edges = PCStable.listFullyConnectedEdges(nodes)
        # setting up the graphs
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        return G
    
    @staticmethod
    def removeEdge(G: nx.Graph | nx.DiGraph, edge: list[str]) -> nx.Graph | nx.DiGraph:
        '''very redundant method'''
        G.remove_edge(edge[0], edge[1])
        return G
    
    @staticmethod
    def validate_path_evaluation(func):
        def validate(Graph: nx.Graph | nx.DiGraph, node_1: str, node_2: str, method = 'dijkstra'):
            if node_1 not in Graph.nodes() or node_2 not in Graph.nodes():
                raise ValueError("node_1 and node_2 must be in Graph")
            if method not in ['dijkstra', 'bellman-ford']:
                raise ValueError("method must be 'dijkstra' or 'bellman-ford'")
            return func(Graph, node_1, node_2, method)
        return validate
    
    @staticmethod
    @validate_path_evaluation
    def getAllPathsBetweenNodes(Graph: nx.Graph | nx.DiGraph, node_1: str, node_2: str, method = 'dijkstra') -> list[tuple[str, ...]]:
        try:
            paths = [tuple(path) for path in list(nx.all_shortest_paths(Graph, node_1, node_2, method=method))]
        except:
            paths = []
        return paths
    
    @staticmethod
    def getAllPathsWithoutConditionalNode(Graph: nx.Graph | nx.DiGraph, node_1: str, node_2: str, cond_node: str, method = 'dijkstra') -> list[tuple[str, ...]]:
        all_shortest_paths = PCStable.getAllPathsBetweenNodes(Graph, node_1, node_2, method)
        paths = [path for path in all_shortest_paths if cond_node not in path]
        return paths

    @staticmethod
    def getSkeleton():
        '''
        gives us the structure of the graph
        '''
        raise NotImplementedError
    
    @staticmethod
    def isMarkovEquivalent():
        '''
        1 check if 2 graphs have the same immoralities
        2 check if the graphs have the same skeleton (v-structures)
        '''
        raise NotImplementedError    
    
       
###################################################
################## OTHER ##########################
###################################################

# def buildingPCStable():
#     TG = nx.DiGraph()
#     node_list = ['A', 'B', 'C', 'D', 'E']
#     edge_list = [('A','C'), ('B','C'), ('C','D'), ('C','E')]
    
#     TG.add_edges_from(edge_list)
    
#     G = nx.Graph()
#     G = utils.fully_connected_graph(node_list)
    
#     plt.subplot(2,2,1)
#     nx.draw_spring(TG, with_labels=True)
    
#     plt.subplot(2,2,2)
#     nx.draw_spring(G, with_labels=True)
    
#     severed_edges = utils.removeConnectionsWithNoDirectionalPaths(G, TG)
#     print(severed_edges)
    
#     plt.subplot(2,2,3)
#     nx.draw_spring(G, with_labels=True)
    
    
#     plt.show()
    
    
if __name__ == '__main__':

    # PCStable = PCStable('data/lung_cancer-train.csv')
    pcs = PCStable('data/lung_cancer-train.csv')
    pcs.identifySkeleton()
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
    
        
