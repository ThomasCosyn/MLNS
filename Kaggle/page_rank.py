def page_rank(graph, nb_iterations = 50, verbose = False):
    """
    Takes a graph as input
    Returns a dictionary containing the pagerank scores for each node
    """

    # Initialization
    n = graph.number_of_nodes()
    node_values = {node : 1/n for node in list(graph.nodes)}

    # Iteration
    for _ in range(nb_iterations):
        new_values = update_all_values(graph, node_values)
        if verbose:
            print(new_values)
            input()
        node_values = new_values

    return node_values

def update_all_values(graph, node_values):
    """
    Takes as input a graph, the node values associated with the graph
    Updates the node values dictionnary
    """

    # Initializing the dictionnary that will store the results
    new_values = {node : 0 for node in node_values.keys()}

    # Iterating on all nodes
    for node in list(graph.nodes):
        new_values[node] = update_node_value(graph, node, node_values)
    
    return new_values

def update_node_value(graph, node, node_values):
    """
    Takes as input a graph, the node values and the node we update
    Returns the updated values
    """
    neighbors = list(graph.neighbors(node))
    new_value = sum([node_values[v]/len(list(graph.neighbors(v))) for v in neighbors])

    return new_value