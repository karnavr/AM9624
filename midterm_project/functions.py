import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

from networkx.algorithms import community


### GENEERATING/INITIALIZING NETWORKS

def initialize_network(num_caves, cave_size, add_random_ties, p_random):
    """
    Create a caveman network (disconnected or with extra random ties)
    
    Parameters:
      - num_caves: number of clusters (or caves/groups)
      - cave_size: number of nodes per cave
      - add_random_ties: boolean, if True then add random long-range ties
      - p_random: probability for adding a random tie between any pair 
      
    Returns:
      - G: a NetworkX graph object with the caveman structure (and random ties if requested)
    """

    # NOTE: we can also simply use nx.connected_caveman_graph(num_caves, cave_size), which would create a 
    # network that such that there is at least one edge connecting each cave to it's neighbors. This is not 
    # necessarily the case for our implementation below, which is in-line with the Flache and Macy model. 

    # Create a list of complete subgraphs one for each cave.
    subgraphs = [nx.caveman_graph(1, cave_size) for _ in range(num_caves)]
    
    # Combine these complete subgraphs into one graph.
    # The function nx.disjoint_union_all returns a graph that is the disjoint union of all the subgraphs.
    # This means the resulting graph G has each cave as an isolated component.
    G = nx.disjoint_union_all(subgraphs)
    
    # Optionally, add random long-range ties.
    if add_random_ties:
        
        nodes = list(G.nodes())
        
        # Iterate over all pairs of nodes
        for i in nodes:
            for j in nodes:
                # We add a tie if i and j are not already connected (could perhaps also check that i and j belong to different caves)
                if i != j and not G.has_edge(i, j) and random.random() < p_random:
                    G.add_edge(i, j)
                    
    return G









### DYNAMICS



### PLOTTING

def caveman_layout(G, spacing=3):
    """
    Generates a position dictionary for a connected caveman graph 
    where nodes in the same fully connected group are placed together.

    Parameters:
    - G: networkx.Graph
        The connected caveman graph.
    - spacing: float, optional (default=3)
        The horizontal spacing between groups.

    Returns:
    - pos: dict
        A dictionary mapping nodes to positions.
    """
    
    # Detect groups (fully connected cliques)
    groups = list(community.label_propagation_communities(G))
    print(groups)

    pos = {}  # Dictionary to store node positions

    for i, group in enumerate(groups):
        group = list(group)  # Convert set to list
        subgraph = G.subgraph(group)
        
        # Get circular layout for each group
        sub_pos = nx.circular_layout(subgraph, scale=1)
        
        # Shift positions to separate groups
        shift_x = i * spacing
        for node in sub_pos:
            pos[node] = (sub_pos[node][0] + shift_x, sub_pos[node][1])

    return pos, groups


def caveman_layout_positions(G, group_spacing=6, node_spacing=1.5, seed=42):
    """
    Generates a structured layout for a connected caveman network, ensuring that:
    1. Groups remain visually separate but connected.
    2. Groups with strong interconnections stay closer.
    3. Nodes within each group are arranged neatly around a centroid.

    Parameters:
    - G: networkx.Graph
        The connected caveman graph.
    - group_spacing: float, optional (default=6)
        Distance between different groups in the layout.
    - node_spacing: float, optional (default=1.5)
        Distance of nodes around their group centroid.
    - seed: int, optional (default=42)
        Random seed for reproducibility.

    Returns:
    - pos: dict
        A dictionary mapping nodes to positions.
    """
    np.random.seed(seed)  # Ensure reproducibility
    pos = {}  # Dictionary to store node positions

    # Detect groups using label propagation
    from networkx.algorithms import community
    groups = list(community.label_propagation_communities(G))

    # Create a meta-graph where each group is a node
    meta_graph = nx.Graph()
    group_mapping = {}  # Maps each node to its group index

    for i, group in enumerate(groups):
        group = list(group)  # Convert set to list
        meta_graph.add_node(i)  # Add the group as a node in the meta-graph
        for node in group:
            group_mapping[node] = i  # Map each node to its group

    # Add edges between groups if any node from one group connects to another
    for edge in G.edges():
        node1, node2 = edge
        group1, group2 = group_mapping[node1], group_mapping[node2]
        if group1 != group2:  # Only consider inter-group edges
            meta_graph.add_edge(group1, group2)

    # Generate a force-directed layout for the meta-graph
    group_pos = nx.spring_layout(meta_graph, seed=seed, k=group_spacing)

    # Place nodes in a circular arrangement around their group's centroid
    for i, group in enumerate(groups):
        group = list(group)  # Convert set to list
        centroid_x, centroid_y = group_pos[i]  # Get centroid position

        # Arrange nodes in a circular pattern around the centroid
        num_nodes = len(group)
        angle_step = 2 * np.pi / num_nodes
        radius = node_spacing * np.sqrt(num_nodes)  # Adjust radius dynamically

        for j, node in enumerate(group):
            angle = j * angle_step
            pos[node] = (centroid_x + radius * np.cos(angle),
                         centroid_y + radius * np.sin(angle))

    return pos

