import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

from networkx.algorithms import community


### GENERATING/INITIALIZING NETWORKS

def initialize_network(num_caves, cave_size, add_random_ties, p_random):
    """
    Create a caveman network (disconnected caves or with extra random ties)
    
    Parameters:
      - num_caves: number of clusters/groups/caves
      - cave_size: number of agents/nodes per cave
      - add_random_ties: boolean, if True then add random long-range ties with probability p_random
      - p_random: probability for adding a random tie between any pair of agents
      
    Returns:
      - G: a NetworkX graph object with the caveman structure (and random ties if requested)
    """

    # NOTE: we can also simply use nx.connected_caveman_graph(num_caves, cave_size), which would create a 
    # network that such that there is at least one edge connecting each cave to it's neighbors. The Flache and
    # Macy model creates random ties by iterating over all pairs of agents and adding a tie with probability p_random.
    # This can lead to some caves being isolated. We implement this below.

    # Create a list of complete subgraphs one for each cave.
    subgraphs = [nx.caveman_graph(1, cave_size) for _ in range(num_caves)]
    
    # Combine these complete subgraphs into one graph.
    G = nx.disjoint_union_all(subgraphs)
    
    # Optionally, add random long-range ties.
    if add_random_ties:
        
        nodes = list(G.nodes())
        
        # Iterate over all pairs of nodes
        for i in nodes:
            for j in nodes:
                # We add a tie if i and j are not already connected (could perhaps also check that i and j belong to different caves)
                if (i != j) and not (G.has_edge(i, j)) and (random.random() < p_random):
                    G.add_edge(i, j)
                    
    return G

def initialize_opinions(N, K):
    """
    Initialize the opinion matrix S for N agents and K opinion dimensions.
    Each entry is drawn uniformly at random from [-1, 1].
    
    Parameters:
      - N (int): Number of agents.
      - K (int): Number of opinion dimensions.
    
    Returns:
      - S (np.ndarray): An (N x K) array of opinions.
    """
    S = np.random.uniform(-1, 1, size=(N, K))
    return S

def initialize_opinions_Lee(N, K, NS, cave_size):
    """
    Initialize the opinion matrix S for the Lee model.
    
    A fraction NS of the K opinion dimensions (H = int(NS * K)) will be segregated.
    We assign each node to a cave based on its index (assuming nodes are ordered by cave):
        cave_number = i // cave_size.
    Then, we assign each cave alternately to group 0 and group 1 (group = cave_number mod 2).
    
    For segregated dimensions (first H dimensions):
      - If the node belongs to group 0, sample from U(-1, 0).
      - If the node belongs to group 1, sample from U(0, 1).
    For the remaining (K - H) non-segregated dimensions, sample uniformly from U(-1, 1).
    
    Parameters:
      - N (int): Total number of agents.
      - K (int): Number of opinion dimensions.
      - NS (float): Fraction (0 <= NS <= 1) of dimensions that are segregated.
      - cave_size (int): Number of nodes per cave (assumes nodes are ordered by cave).
    
    Returns:
      - S (np.ndarray): An (N x K) opinion matrix.
      - groups (np.ndarray): A vector of length N with the group assignment (0 or 1) for each node.
    """
    S = np.zeros((N, K))
    groups = np.zeros(N, dtype=int)
    
    # Determine the number of segregated dimensions.
    H = int(round(NS * K))
    
    # Iterate over all nodes.
    for i in range(N):

        # Determine the cave membership using integer division. 
        # (relies on nodes being ordered by cave, which is the case for the caveman network created using a disjoint union of complete graphs)
        cave_num = i // cave_size

        # Assign groups alternately based on cave number.
        group = cave_num % 2
        groups[i] = group
        
        for k in range(K):
            if k < H:
                # Segregated dimension: assign based on group membership.
                if group == 0:
                    S[i, k] = np.random.uniform(-1, 0)
                else:
                    S[i, k] = np.random.uniform(0, 1)
            else:
                # Non-segregated dimension: assign uniformly from [-1, 1].
                S[i, k] = np.random.uniform(-1, 1)
    
    return S

def compute_weights(S, allow_negative=True):
    """
    Compute the weight matrix W based on opinion matrix S. To be used when initializing
    the network and setting the initial weights.

    Parameters:
      - S (np.ndarray): opinion matrix of shape (N, K)
      - allow_negative (bool): if True then allow negative weights

    Returns:
      - W (np.ndarray): weight matrix of shape (N, N)
    
    For each pair (i, j):
      If allow_negative is True:
          w_ij = 1 - (1/K) * sum(|s[i,k] - s[j,k]|) (eq. 1 Flache and Macy)
      Otherwise (only nonnegative weights):
          w_ij = 1 - (1/(2*K)) * sum(|s[i,k] - s[j,k]|) (eq. 1a Flache and Macy)
    
    For i == j, we set the weight to 0.
    """
    N, K = S.shape
    W = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i != j:
                avg_diff = np.sum(np.abs(S[i] - S[j])) / K
                if allow_negative:
                    W[i, j] = 1 - avg_diff
                else:
                    W[i, j] = 1 - (avg_diff / 2)
            else:
                W[i, j] = 0
    return W

### DYNAMICS

def update_state_for_agent(i, S, W, graph):
    """
    Update the opinion state of agent i asynchronously.
    
    The rule is:
      \Delta s_{ik} = (1/(2 * N_i)) * sum_{j in neighbors} w_{ij} * (s_{jk} - s_{ik})
      then for each dimension k:
         if s_{ik} > 0: s_{ik} <- s_{ik} + \Delta s_{ik}*(1 - s_{ik})
         else:         s_{ik} <- s_{ik} + \Delta s_{ik}*(1 + s_{ik})
    
    Parameters:
      - i (int): index of the agent to be updated
      - S (np.ndarray): opinion matrix of shape (N, K)
      - W (np.ndarray): weight matrix of shape (N, N)
      - graph (nx.Graph): the NetworkX graph (used to get the neighbors of i)
    """
    neighbors = list(graph.neighbors(i))

    if len(neighbors) == 0:
        return  # No update if agent i has no neighbors.
    
    N_i = len(neighbors)
    K = S.shape[1]
    
    # Compute the change in opinion / influence delta (as a vector of length K)
    delta = np.zeros(K)
    for j in neighbors:
        delta += W[i, j] * (S[j] - S[i])
    delta /= (2 * N_i)
    
    # Update each opinion component with smoothing:
    for k in range(K):
        if S[i, k] > 0:
            S[i, k] += delta[k] * (1 - S[i, k])
        else:
            S[i, k] += delta[k] * (1 + S[i, k])

        # Ensure the updated opinion stays within [-1, 1]
        S[i, k] = np.clip(S[i, k], -1, 1)

def update_state_for_agent_Lee(i, S, W, graph, T):
    """
    Update the opinion state of agent i asynchronously using structural embeddedness.
    
    This function is similar to the original update_state_for_agent, but now for each 
    neighbor j we use the pre-computed tie activation probability T[i,j] to decide 
    whether neighbor j's influence is active.
    
    In particular, for each neighbor j, with probability T[i,j] we include:
         w_{ij}*(S[j] - S[i])
    in the influence sum; otherwise, that neighbor's influence is omitted.
    
    Then, as before, we compute:
        \Delta s_{ik} = (1/(2 * N_i)) * sum_{j in active neighbors} w_{ij}*(s_{jk} - s_{ik})
    and update each component with smoothing.
    
    Parameters:
      - i: index of the focal agent.
      - S: opinion matrix (N x K).
      - W: weight matrix (N x N).
      - graph: the NetworkX graph.
      - T: the pre-computed tie activation probability matrix (N x N).
    """
    neighbors = list(graph.neighbors(i))
    K = S.shape[1]

    if len(neighbors) == 0:
        return  # No update if agent i has no neighbors.
    
    active_delta_sum = np.zeros(K)
    active_neighbors_count = 0  # Count the number of neighbors whose tie is active

    for j in neighbors:

        # For each neighbor j, include its influence only if the tie is activated.
        if random.random() < T[i, j]:
            active_delta_sum += W[i, j] * (S[j] - S[i])
            active_neighbors_count += 1

    if active_neighbors_count == 0:
        return  # No update to S[i] if no neighbor is activated.
    delta = active_delta_sum / (2 * active_neighbors_count)
    
    # Update each opinion dimension with smoothing:
    for k in range(K):
        if S[i, k] > 0:
            S[i, k] += delta[k] * (1 - S[i, k])
        else:
            S[i, k] += delta[k] * (1 + S[i, k])

        # Ensure the updated opinion stays within [-1, 1]
        S[i, k] = np.clip(S[i, k], -1, 1)

def update_weights_for_agent(i, S, W, graph, allow_negative=True):
    """
    Update the weights for agent i's ties. 
    To be used when updating the state of the network.

    Parameters:
      - i (int): index of the agent to update the weights for
      - S (np.ndarray): opinion matrix of shape (N, K)
      - W (np.ndarray): weight matrix of shape (N, N)
      - graph (nx.Graph): the NetworkX graph (used to get the neighbors of i)
      - allow_negative (bool): if True then allow negative weights
    
    For each neighbor j of agent i, update:
      If allow_negative is True:
          w_ij = 1 - (1/K) * sum(|s[i,k] - s[j,k]|)
      Otherwise:
          w_ij = 1 - (1/(2*K)) * sum(|s[i,k] - s[j,k]|)
    """
    neighbors = list(graph.neighbors(i))

    if len(neighbors) == 0:
        return  # No update if agent i has no neighbors.
    
    K = S.shape[1]
    for j in neighbors:
        avg_diff = np.sum(np.abs(S[i] - S[j])) / K
        if allow_negative:
            W[i, j] = 1 - avg_diff
        else:
            W[i, j] = 1 - (avg_diff / 2)

def run_simulation(graph, S, W, num_iterations, allow_negative=True, tie_addition_iter=None, p_random_new=None):
    """
    Run the asynchronous simulation.
    
    In each time step, one agent is chosen at random; then, with 50% chance, 
    either its opinion state is updated or its outgoing weights are updated.
    The total number of time steps is num_iterations * N (N = number of agents).
    
    Additionally, if tie_addition_iter is specified (an integer representing the iteration
    number after which to add new random ties) and p_random_new is provided, then at that
    iteration the function adds new random ties to the graph.
    
    We record the polarization once per iteration (i.e. every N steps).
    
    Parameters:
      - graph: the NetworkX graph
      - S: opinion matrix (N x K)
      - W: weight matrix (N x N)
      - num_iterations: number of iterations (each iteration corresponds to N time steps)
      - allow_negative: Boolean flag for weight computation (default: True)
      - tie_addition_iter: (Optional) iteration number after which to add new ties.
      - p_random_new: (Optional) probability for adding a new tie when tie addition occurs.
      
    Returns:
      - polarization_history: list of polarization values recorded once per iteration.
      - S: final opinion matrix.
      - W: final weight matrix.
    """
    N = graph.number_of_nodes()
    total_steps = num_iterations * N
    polarization_history = []
    
    for t in range(total_steps):

        # Pick an agent at random (with replacement, so no need to remove from list)
        i = random.choice(list(graph.nodes()))
        
        # With probability 0.5, update state; otherwise, update weights.
        if random.random() < 0.5:
            update_state_for_agent(i, S, W, graph)
        else:
            update_weights_for_agent(i, S, W, graph, allow_negative=allow_negative)
        
        # At the end of each iteration (i.e. every N time steps)
        if (t + 1) % N == 0:

            current_iter = (t + 1) // N

            # If tie_addition_iter is specified and matches current iteration, add new ties.
            if (tie_addition_iter is not None) and (current_iter == tie_addition_iter):

                # p_random_new must be provided in this case.
                if p_random_new is not None:
                    add_random_ties_to_graph(graph, p_random_new)
                else:
                    raise ValueError("p_random_new must be provided if tie_addition_iter is specified.")

            # Record polarization at the end of the iteration.
            P_t = compute_polarization(S)
            polarization_history.append(P_t)
    
    return polarization_history, S, W

def run_simulation_Lee(graph, S, W, num_iterations, SE, allow_negative=True, tie_addition_iter=None, p_random_new=None):
    """
    Run the asynchronous simulation for the Lee model.
    
    In each time step, one agent is chosen at random; then, with 50% chance, either:
      - Its opinion state is updated using the new update function that incorporates 
        structural embeddedness (SE) via the pre-computed tie activation matrix T, 
      - OR its outgoing weights are updated.
    
    The total number of time steps is num_iterations * N (where N is the number of agents).
    
    Additionally, if tie_addition_iter is specified (an iteration after which to add new 
    random ties) and p_random_new is provided, then new random ties are added to the graph 
    at that iteration.
    
    Parameters:
      - graph: the NetworkX graph (static access network).
      - S: opinion matrix (N x K), assumed to be produced using initialize_opinions_Lee.
      - W: weight matrix (N x N).
      - num_iterations: number of iterations (each iteration corresponds to N time steps).
      - SE: Structural Embeddedness parameter (0 <= SE <= 1).
      - allow_negative: Boolean flag for weight computation (default: True).
      - tie_addition_iter: (Optional) iteration number after which to add new ties.
      - p_random_new: (Optional) probability for adding a new tie when tie addition occurs.
      
    Returns:
      - polarization_history: list of polarization values recorded once per iteration.
      - S: final opinion matrix.
      - W: final weight matrix.
    """
    # Compute tie activation probability matrix T once at the beginning.
    T = compute_tie_strength(graph, SE)
    
    N = graph.number_of_nodes()
    total_steps = num_iterations * N
    polarization_history = []
    
    for t in range(total_steps):
        # Pick an agent at random (with replacement)
        i = random.choice(list(graph.nodes()))
        
        # With probability 0.5, update the opinion state using the Lee version; otherwise update weights.
        if random.random() < 0.5:
            update_state_for_agent_Lee(i, S, W, graph, T)
        else:
            update_weights_for_agent(i, S, W, graph, allow_negative=allow_negative)
        
        # At the end of each iteration (i.e. every N time steps)
        if (t + 1) % N == 0:
            current_iter = (t + 1) // N
            
            # Optionally, add new random ties if specified.
            if (tie_addition_iter is not None) and (current_iter == tie_addition_iter):
                if p_random_new is not None:
                    add_random_ties_to_graph(graph, p_random_new)
                    T = compute_tie_strength(graph, SE)  # Re-compute T since the network structure has changed.
                else:
                    raise ValueError("p_random_new must be provided if tie_addition_iter is specified.")
            
            # Record polarization at the end of this iteration.
            P_t = compute_polarization(S)
            polarization_history.append(P_t)
    
    return polarization_history, S, W


### HELPER FUNCTIONS

def compute_polarization(S):
    """
    Compute the polarization measure P_t at a given time t, given the opinion matrix S.
    
    For every pair of distinct agents (i, j), we compute:
        d_ij = (1/K) * sum(|S[i, :] - S[j, :]|)
    and then compute the variance of all these d_ij values.
    
    Parameters:
      - S (np.ndarray): opinion matrix of shape (N, K)
    
    Returns:
      - P_t (float): The polarization measure.
    """
    N, K = S.shape
    distances = []
    
    # Loop over all unique pairs (i, j) with i < j to avoid double-counting pairs
    for i in range(N):
        for j in range(i + 1, N): # ensure j > i to avoid redundancy

            # Compute the average absolute difference between opinions of i and j
            d_ij = np.sum(np.abs(S[i] - S[j])) / K
            distances.append(d_ij)
    
    distances = np.array(distances)
    mean_distance = np.mean(distances)

    # Polarization is the variance of these distances.
    # P_t = np.mean((distances - mean_distance) ** 2)
    P_t = 2 * np.sum((distances - mean_distance) ** 2) / (N * (N - 1))

    return P_t

def add_random_ties_to_graph(graph, p_random):
    """
    Add random long-range ties to an existing graph.
    
    For every pair of nodes (i, j) that are not already connected, add an edge with
    probability p_random.
    
    Parameters:
      - graph: a NetworkX graph object (modified in place)
      - p_random (float): probability of adding a tie between any pair
    """
    nodes = list(graph.nodes())
    for i in nodes:
        for j in nodes:
            if (i != j) and (not graph.has_edge(i, j)) and (random.random() < p_random):
                graph.add_edge(i, j)

def compute_topological_overlap(graph, i, j):
    """
    Compute the topological overlap O_ij between nodes i and j.
    
    O_ij = n_ij / ((k_i - 1) + (k_j - 1) - n_ij)
    where:
      - n_ij is the number of common neighbors of i and j,
      - k_i and k_j are the degrees of i and j, respectively.
      
    If the denominator is zero (which may happen for very low-degree nodes),
    we define O_ij = 0.
    
    Parameters:
      - graph: a NetworkX graph
      - i, j: node identifiers
      
    Returns:
      - O_ij (float): the topological overlap between nodes i and j.
    """
    # Get common neighbors between i and j
    common_neighbors = list(nx.common_neighbors(graph, i, j))

    # components of the O_ij formula
    n_ij = len(common_neighbors)
    k_i = graph.degree(i)
    k_j = graph.degree(j)
    
    denominator = (k_i - 1) + (k_j - 1) - n_ij

    if denominator > 0:
        return n_ij / denominator
    else:
        return 0.0
    
def compute_tie_strength(graph, SE):
    """
    Compute and return the tie activation probability matrix T for the given graph,
    based on structural embeddedness (SE).
    
    For each pair of nodes (i,j) that are connected in the graph:
      T_ij = SE * O_ij + (1 - SE)
    For nodes not connected, we set T_ij = 0 (since their tie is never active).
    
    Parameters:
      - graph: a NetworkX graph (static)
      - SE (float): the structural embeddedness parameter, 0 <= SE <= 1.
         SE = 0 recovers the original model (all ties active with probability 1),
         SE = 1 means activation depends fully on topological overlap.
    
    Returns:
      - T (np.ndarray): A (N x N) matrix where T[i,j] is the tie activation probability.
    """
    N = graph.number_of_nodes()
    T = np.zeros((N, N))            # initialize the tie activation probability matrix
    
    nodes = list(graph.nodes())
    for i in nodes:
        for j in nodes:
            if (i != j) and (graph.has_edge(i, j)):
                O_ij = compute_topological_overlap(graph, i, j)
                T[i, j] = SE * O_ij + (1 - SE)
            # For non-neighbors T[i,j] remains 0 (what we initialize it to)
    return T

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