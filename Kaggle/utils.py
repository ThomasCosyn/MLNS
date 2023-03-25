import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import identity
from scipy.sparse.linalg import inv

def adamic_adar(graph, node1, node2):

    c = 0
    gamma_x = set(graph.neighbors(node1))
    gamma_y = set(graph.neighbors(node2))
    for z in gamma_x & gamma_y:
        if len(set(graph.neighbors(z))) > 1:
            c += 1 / np.log(len(set(graph.neighbors(z))))

    return c

def compute_max_flow(row, graph):
    
    if row['node1'] == row['node2']:
        return 0

    flow_value, flow_dict = nx.maximum_flow(graph, _s = row['node1'], _t = row['node2'])
    return flow_value

def katzB_matrix(graph, beta):

    A = nx.adjacency_matrix(graph, nodelist = graph.nodes)
    nodes_index = {list(graph.nodes)[i]: i for i in range(graph.number_of_nodes())}
    nodes_index_reverse = {i: list(graph.nodes)[i] for i in range(graph.number_of_nodes())}
    I = identity(A.shape[0], format = 'csr')
    M = inv(I - beta * A) - I

    return M, nodes_index, nodes_index_reverse

def shortest_path(graph, node1, node2):

    cutoff = 2
    while cutoff < 20:
        sps = list(nx.all_simple_paths(graph, node1, node2, cutoff = cutoff))
        for path in sps:
            if len(path) > 2:
                return len(path)
        cutoff += 1
    return 20


def text_to_graph(filepath):
    """
    Takes a txt filepath as input and returns a nx graph computed from the data contained in the txt
    file.
    """

    # Loads txt file
    train = pd.read_csv(filepath, sep = ' ', names = ['node1', 'node2', 'is_linked'])

    # Creates graph
    graph = nx.Graph()

    # If nodes are linked, adds the edge to the graph
    for _, row in train.iterrows():
        if row['is_linked'] == 1:
            graph.add_edge(row['node1'], row['node2'])

    return graph