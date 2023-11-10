from matplotlib import pyplot as plt
import networkx as nx
from itertools import combinations

from ConditionalIndependence import ConditionalIndependence as ci
from edge import Edge

class PCStable:
    graph = nx.Graph()
    
    def __init__(self, data_path, method = 'chisq') -> None:
        # initialise empty fully connected graph
        self.ci = ci(data_path, method)
        self.variables = self.ci.rand_vars
        self.graph = PCStable.fullyConnectedGraph(self.variables, self.graph)
        self.immoralNodes = []
        self.markovChains = []
                
    def addImmorality(self, node: str | list[str], edge: list[str]) -> None:
        self.immoralNodes.append([node, edge])
        
    def getImmoralities(self) -> list[list[str]]:
        return self.immoralNodes
    
    def addMarkovChain(self, vi: str, vj: str, parents: list[str]) -> None:
        self.markovChains.append([vi, vj, parents])
    
    def getMarkovChains(self) -> list[list[str]]:
        return self.markovChains    
        
    def getGraphNodes(self) -> list[str]:
        return list(self.graph.nodes())

    def getGraphEdges(self) -> list[str]:
        for edge in self.graph.edges():
            Edge(edge[0], edge[1], [], self.ci, [])
        return list(self.graph.edges())
    
    def drawPlot(self):
        nx.draw_spring(self.graph, with_labels=True)
        plt.show()
    
    def initialiseEdges(self, conditioning_variables = []) -> list[Edge]:
        edges = []
        for edge in self.getGraphEdges():
            if len(conditioning_variables) > 0:
                for condition in conditioning_variables:
                    # TODO: figuring out why edge is still being added even though condition not met
                    if condition not in list(edge):
                        edges.append(Edge(edge[0], edge[1], conditioning_variables, self.ci, []))
                    continue 
            else:
                edges.append(Edge(edge[0], edge[1], [], self.ci, []))
        return edges
    
    
    def identifySkeleton(self):
        '''
        get's the core skeleton shape given the independence conditions
        '''
        # Part 1: get all the edges with no parents
        edges = self.initialiseEdges()
        
        self.drawPlot()
        
        # first pass, conditioning sets size 0
        for edge in edges:
            # independence test (vi, vj, [])
            if edge.is_conditionally_independent:
                # remove the edge
                self.graph = PCStable.removeEdge(self.graph, [edge.var_i, edge.var_j])

        self.drawPlot()
        
        # Part 2: get all the edges with 1 parent
        for node in self.getGraphNodes():
            edges_set_1 = self.initialiseEdges([node])
            for edge in edges_set_1:
                if edge.is_conditionally_independent:
                    # remove the edge
                    self.graph = PCStable.removeEdge(self.graph, [edge.var_i, edge.var_j])
                if edge.is_immoral:
                    self.addImmorality(node, [edge.var_i, edge.var_j])
                    
        # self.drawPlot()            
        # print('holup')
        
        pair_combinations = list(combinations(self.getGraphNodes(), 2))
        for conditionals in pair_combinations:
            conditionals = list(conditionals)
            edges_set_2 = self.initialiseEdges(conditionals)
            for edge in edges_set_2:
                if edge.is_conditionally_independent:
                    # remove the edge
                    self.graph = PCStable.removeEdge(self.graph, [edge.var_i, edge.var_j])
                if edge.is_immoral:
                    self.addImmorality(conditionals, [edge.var_i, edge.var_j])           
        
        # self.drawPlot()
        # print('holup2')
        
        
        
        # # Step 1: get all the edges with no parents
        # raise NotImplementedError

        # # Step 1.1: get all the edges
        # raise NotImplementedError

        
        
        # edges = list(self.graph.edges())
        
        # for edge in edges:
            # independence test chi (vi, vj)
            # copilot suggestion
            # # get all possible paths between node and all other nodes
            # paths = PCStable.getAllPathsBetweenNodes(self.graph, node, None)
            # # get all possible paths without the conditional node
            # paths_without_cond = PCStable.getAllPathsWithoutConditionalNode(self.graph, node, None, None)
            
             # if there are paths without the conditional node, then we have a v-structure
            # if len(paths) != len(paths_without_cond):
            #     # get the edges that are not in the paths without the conditional node
            #     edges = [edge for edge in paths if edge not in paths_without_cond]
            #     # remove the edges that are not in the paths without the conditional node
            #     for edge in edges:
            #         self.graph = PCStable.removeEdge(self.graph, edge)
            #         # store the immoralities
            #         self.storeImmoralities(node, edge)
        # raise NotImplementedError
    
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
    def fullyConnectedGraph(nodes: list[str], G = nx.Graph()) -> nx.Graph:
        # using the combinations method from itertools to create all possible edges
        edges = PCStable.listFullyConnectedEdges(nodes)
        # setting up the graphs
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        return G
    
    @staticmethod
    def removeEdge(G: nx.Graph | nx.DiGraph, edge: list[str]) -> nx.Graph | nx.DiGraph:
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
#     G = utils.fullyConnectedGraph(node_list)
    
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
    
conditioning_variables = ['a','b']
edges = ['a','b','c','d','e'] 
for edge in edges:
    for con in conditioning_variables:
        if con not in edge:
            print(edge)
        else:
            break
    
        
