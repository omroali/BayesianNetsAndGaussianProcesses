from matplotlib import pyplot as plt
import networkx as nx
from itertools import combinations
import random

from ConditionalIndependence import ConditionalIndependence as ci
from edge import Edge
import BayesNetUtil as bnu


class PCStable:
    graph = nx.Graph()

    def __init__(self, data_path, method="chisq", independence_threshold=0.01) -> None:
        # initialise empty fully connected graph
        self.ci = ci(data_path, method)
        self.variables = self.ci.rand_vars
        self.graph = PCStable.fully_connected_graph(self.variables, self.graph)
        self.independence_threshold = independence_threshold
        self.all_nodes = self.graph.nodes()
        self.severed_edges = []
        self.immoral_nodes = []
        self.non_immoral_nodes = []
        self.directional_graph = []
        # self.markovChains = []

        ## print(f'Using conditional test: {method}')
        ## print(f'Test threshold: {independence_threshold}')

    def addImmorality(self, node: str | list[str], edge: list[str]) -> None:
        self.immoral_nodes.append([node, edge])

    def getImmoralities(self) -> list[list[str]]:
        return self.immoral_nodes

    def evaluate_skeleton(self, with_plots=False, log_level=0):
        """
        Method to get the basic skeleton structure algorithm
        """
        severless_count = 0
        iteration = 0
        log_removals = False
        if log_level == 2:
            log_removals = True

        while severless_count < 10:
            severed = False

            # if log_level >= 1: ## print(f'\n--iteration {iteration}--')
            edge_parents_list_for_independence_test = (
                self.get_combinations_of_adjacent_nodes(iteration)
            )
            edges = self.create_edges(edge_parents_list_for_independence_test)

            edges_to_remove = []
            for edge in edges:
                if edge.is_conditionally_independent:
                    edges_to_remove.append(edge)
                    severed = True
                    severless_count = 0

            removed_edges = self.remove_edges(edges_to_remove, log_removals)

            # iteration check and logging
            if len(removed_edges) == 0:
                severless_count += 1
            iteration += 1
            if with_plots:
                self.draw_plot()

    def evaluate_immoralities(self):
        """
        Evaluates immoral nodes to introduce initial orientation to graph
        TODO: do i need to iterate?

        """
        moral_nodes = []
        immoral_nodes = []

        evaluation_set = []
        for parent in self.all_nodes:
            child_pairs = self.all_adjacent_nodes(parent, 2)
            for pair in child_pairs:
                evaluation_set.append([pair[0], pair[1], parent])

        for eval in evaluation_set:
            moral_nodes.append(eval)
            for severed in self.get_severed_edges_list:
                # checking if child nodes to check are in a severed edge
                if eval[0] in severed[:2] and eval[1] in severed[:2]:
                    # checking the parent nodes in severed edges
                    if eval[2] not in severed[2]:
                        immoral_nodes.append(eval)
                        moral_nodes.remove(eval)
                        break

        for node in immoral_nodes:
            self.immoral_nodes.append(Edge(node[0], node[1], node[2], self.ci))
            # directional=True))
        for node in moral_nodes:
            self.non_immoral_nodes.append(Edge(node[0], node[1], node[2], self.ci))
            #    , directional=False))

    def create_directional_edge_using_immorality(self):
        # all_graph_edges = self.pcs_test()
        # single_parent_severed_edges = self.get_single_parent_severed_edges
        # immoral_nodes = self.immoral_nodes
        dir_edges = []

        DiG = nx.DiGraph()

        for node in self.immoral_nodes:
            dir_edges.append(
                [tuple([node.var_i, node.parents]), tuple([node.parents, node.var_j])]
            )

        for edges in dir_edges:
            DiG.add_edges_from(edges, directed=True)
            ## print(edges)

        un_dir_edges = []
        for node in self.non_immoral_nodes:
            un_dir_edges.append(
                [tuple([node.var_i, node.parents]), tuple([node.parents, node.var_j])]
            )

        for edges in un_dir_edges:
            DiG.add_edges_from(edges, directed=False)
        # DiG.add_edges_from(dir_edges, directed=False)

        for node in self.graph.nodes():
            if node not in DiG.nodes():
                DiG.add_node(node)

        nx.draw_shell(DiG, with_labels=True)
        plt.show()

    @staticmethod
    def generate_config_file(structure) -> None:
        with open("config.txt", "w") as f:
            f.write(structure)

    def randomised_directed_graph(self):
        attempts = 0
        while attempts < 100000:
            rand_dag = nx.DiGraph()
            for edge in self.graph.edges():
                edge_vars = [edge[0], edge[1]]
                random.shuffle(edge_vars)
                rand_dag.add_edge(edge_vars[0], edge_vars[1])
            if nx.is_directed_acyclic_graph(rand_dag):
                return rand_dag
            attempts += 1
        raise ValueError(f"Could not create a random DAG after {attempts} attempts")

    def create_edges(self, paths) -> list[Edge]:
        """
        create edge<[vi,vj, [list[parents]]> structure so long as edge is not
        contained in the parents
        """
        edges = []
        for path in paths:
            v_i = path[0]
            v_j = path[1]
            parents = path[2]
            if v_i in parents or v_j in parents:
                raise ValueError("v_i or v_j values should not be in parent list")

            edges.append(
                Edge(
                    v_i,
                    v_j,
                    parents,
                    conditional_independence_test=self.ci,
                    threshold=self.independence_threshold,
                )
            )
        return edges

    def getting_connected_nodes_for_path_length(self, connections=1):
        output = []
        nodes = list(self.all_nodes)

        for v_i in nodes:
            for v_j in nodes:
                if v_j == v_i:
                    continue
                if connections == 0:
                    output.append([v_i, v_j, []])
                    continue

                cutoff = connections + 1
                paths = list(
                    nx.all_simple_edge_paths(
                        self.graph, source=v_i, target=v_j, cutoff=cutoff
                    )
                )
                for path in paths:
                    if len(path) != cutoff:
                        continue
                    # ## print(f'v_i = {v_i}, v_j = {v_j}, path = {path}, connections = {connections}')
                    parents = list(
                        set(
                            node
                            for edge in path
                            for node in edge
                            if node not in [v_i, v_j]
                        )
                    )
                    # ## print(f'parents: {parents}, connections: {connections}, valid: {len(parents) == connections}')
                    if len(parents) != connections:
                        continue
                    output.append([v_i, v_j, parents])

        return output

    def getting_connected_nodes_for_path_length_new(self, connections=0):
        output = []
        nodes = list(self.all_nodes)

        for v_i in nodes:
            for v_j in nodes:
                if v_j == v_i:
                    continue
                if not connections:
                    output.append([v_i, v_j, []])
                    continue
                for node in nodes:
                    paths = list(
                        nx.all_simple_edge_paths(
                            self.graph, source=v_i, target=node, cutoff=connections
                        )
                    )
                    for path in paths:
                        if len(path) != connections:
                            continue
                        parents = list(
                            set(
                                node
                                for edge in path
                                for node in edge
                                if node not in [v_i, v_j]
                            )
                        )
                        if len(parents) != connections:
                            continue
                        output.append([v_i, v_j, parents])
        return output

    def all_paths_nodes(self, connections) -> None:
        all_edges = list(combinations(self.all_nodes(), 2))
        parent_nodes = []
        for edge in all_edges:
            parent_nodes = list(
                nx.all_simple_edge_paths(
                    self.graph, source=edge[0], target=edge[1], cutoff=connections
                )
            )
        ## print(parent_nodes)

    def get_combinations_of_adjacent_nodes(
        self, connections=0
    ) -> list[list[str | list[str]]]:
        """
        returns a list of all the 'connection' adjacent node combinations for all nodes in the graph
        """
        output = []
        nodes = list(self.all_nodes)

        for v_i in nodes:
            for v_j in nodes:
                if v_j == v_i:
                    continue
                if connections == 0:
                    output.append([v_i, v_j, []])
                    continue

                adjacent_sets = self.all_adjacent_nodes(v_i, connections)
                for set in adjacent_sets:
                    if v_i in set or v_j in set:
                        continue
                    output.append([v_i, v_j, set])
        return output

    def all_adjacent_nodes(self, node, connections) -> list[list[str]]:
        """
        returns a list of all 'connection' adjacent node combinations for a given node
        """
        neighbors = list(self.graph.neighbors(node))
        parents_combinations = set(combinations(neighbors, connections))
        return [list(combo) for combo in parents_combinations]

    def remove_edges(self, edges: list[Edge], log_removals=True) -> list[Edge]:
        removed_edges = []
        for edge in edges:
            if self.graph.has_edge(edge.var_i, edge.var_j):
                self.graph.remove_edge(edge.var_i, edge.var_j)

                # update class variable and return array
                removed_edges.append(edge)
                self.severed_edges.append(edge)
                ## print(f'Severed: {edge.var_i} {edge.var_j} | {edge.parents} || p = {edge.test_value}')

        return removed_edges

    def draw(self):
        plt.show()

    def draw_plot(self) -> None:
        nx.draw_shell(self.graph, with_labels=True)
        plt.show()

    def identifyQualifyingEdges(self):
        """
        get's the edges that qualify for orientation given the collider (immoral) nodes
        """
        raise NotImplementedError

    ###################################################
    ############### STATIC METHODS ####################
    ###################################################

    @staticmethod
    def listFullyConnectedEdges(nodes: list[str]) -> list[tuple[str, str]]:
        edges = combinations(nodes, 2)
        return list(edges)

    @staticmethod
    def fully_connected_graph(nodes: list[str], G=nx.Graph()) -> nx.Graph:
        # using the combinations method from itertools to create all possible edges
        edges = PCStable.listFullyConnectedEdges(nodes)
        # setting up the graphs
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        return G

    @staticmethod
    def validate_path_evaluation(func):
        def validate(
            Graph: nx.Graph | nx.DiGraph, node_1: str, node_2: str, method="dijkstra"
        ):
            if node_1 not in Graph.nodes() or node_2 not in Graph.nodes():
                raise ValueError("node_1 and node_2 must be in Graph")
            if method not in ["dijkstra", "bellman-ford"]:
                raise ValueError("method must be 'dijkstra' or 'bellman-ford'")
            return func(Graph, node_1, node_2, method)

        return validate

    @staticmethod
    @validate_path_evaluation
    def getAllPathsBetweenNodes(
        Graph: nx.Graph | nx.DiGraph, node_1: str, node_2: str, method="dijkstra"
    ) -> list[tuple[str, ...]]:
        try:
            paths = [
                tuple(path)
                for path in list(
                    nx.all_shortest_paths(Graph, node_1, node_2, method=method)
                )
            ]
        except:
            paths = []
        return paths

    @staticmethod
    def getAllPathsWithoutConditionalNode(
        Graph: nx.Graph | nx.DiGraph,
        node_1: str,
        node_2: str,
        cond_node: str,
        method="dijkstra",
    ) -> list[tuple[str, ...]]:
        all_shortest_paths = PCStable.getAllPathsBetweenNodes(
            Graph, node_1, node_2, method
        )
        paths = [path for path in all_shortest_paths if cond_node not in path]
        return paths

    @staticmethod
    def isMarkovEquivalent():
        """
        1 check if 2 graphs have the same immoralities
        2 check if the graphs have the same skeleton (v-structures)
        """
        raise NotImplementedError

    ###################################################
    ################# PROPERTIES ######################
    ###################################################

    @property
    def get_single_parent_severed_edges_list(self) -> list[Edge]:
        if len(self.severed_edges) == 0:
            return []
        single_parent_severed_edges = []
        for severed_edge in self.severed_edges:
            if severed_edge.has_one_parent:
                single_parent_severed_edges.append(severed_edge.as_list)
        return single_parent_severed_edges

    @property
    def get_severed_edges_list(self) -> list[str | list[str]]:
        if len(self.severed_edges) == 0:
            return []
        severed_edges_list = []
        for severed_edge in self.severed_edges:
            severed_edges_list.append(severed_edge.as_list)
        return severed_edges_list

    @property
    def get_graph_edges(self) -> list[str]:
        return list(self.graph.edges())

    @property
    def nodes(self) -> list[str]:
        return list(self.graph.nodes())


# if __name__ == '__main__':

# PCStable = PCStable('data/lung_cancer-train.csv')
# pcs = PCStable('data/lung_cancer-train.csv')
# pcs.get_skeleton()
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
#             ## print(f'edge: {edge}, cond: {con}')
#         else:
#             break
#     ## print(' ')
