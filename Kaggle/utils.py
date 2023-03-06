import networkx as nx
import pandas as pd

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