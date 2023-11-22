import networkx as nx
from matplotlib import pyplot as plt

config_file = (
    "config/structure-learn/config-nb-diabetes-structure-run_test-11-15_17:36:25.txt"
)

structure_edge_list = []
node_list = []
with open(config_file) as csv_file:
    for line in csv_file:
        if line.startswith("structure"):
            structure_line = line.strip()  # save the line
            structure_line = structure_line.replace("P(", "").replace(")", "")
            edges_str = structure_line.split(":")[1]
            edges = edges_str.split(";")
            for edge in edges:
                if "|" not in edge:
                    node_list.append(tuple([edge]))
                else:
                    nodes = edge.split("|")
                    structure_edge_list.append(tuple([nodes[1], nodes[0]]))
print(structure_edge_list)


graph = nx.DiGraph()
graph.add_edges_from(structure_edge_list)

nx.draw_spring(graph, with_labels=True, font_weight="bold")
plt.show()
# read_data

# find structure line

# get edges

# pass into draw
