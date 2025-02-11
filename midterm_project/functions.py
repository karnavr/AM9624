import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

from networkx.algorithms import community


### GENEERATING/INITIALIZING NETWORKS



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

