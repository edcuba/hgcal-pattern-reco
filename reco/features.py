import numpy as np
import networkx as nx


def longest_path_from_highest_energy(G):
    """
    Compute the longest path in graph from the point of highest energy
    """
    H = sorted(G.nodes(data="energy"), key=lambda x: x[1])[-1][0]
    return max(nx.shortest_path_length(G, source=H).values())


def mean_edge_length(G):
    """
    Compute mean edge length in the graph
    """
    n_dists = 0
    for edge in G.edges():
        a, b = edge
        node_a = G.nodes[a]
        node_b = G.nodes[b]
        a_coord = np.array(node_a["pos"])
        b_coord = np.array(node_b["pos"])
        n_dists += np.linalg.norm(a_coord - b_coord)
    return np.mean(n_dists)


def mean_edge_energy_gap(G):
    """
    Compute mean edge energy difference in the graph
    """
    e_diffs = 0
    for edge in G.edges():
        a, b = edge
        node_a = G.nodes[a]
        node_b = G.nodes[b]
        e_diffs += abs(node_a["energy"] - node_b["energy"])

    return np.mean(e_diffs)

def mean_degree(G):
    return np.mean([d for _, d in G.degree()])

def mean_degree_centrality(G):
    return np.mean(list(nx.degree_centrality(G).values()))

def mean_clustering_coefficient(G):
    return nx.average_clustering(G)

def get_graph_level_features(G):
    """
        Compute various graph level features out of G
    """
    return [
        mean_degree(G),
        mean_edge_length(G),
        mean_edge_energy_gap(G),
        mean_degree_centrality(G),
        mean_clustering_coefficient(G),
        longest_path_from_highest_energy(G),
    ]

def get_min_max_z_points(vx, vy, vz):
    min_point = np.argmin(vz)
    max_point = np.argmax(vz)
    return (vx[min_point], vy[min_point], vz[min_point]), (vx[max_point], vy[max_point], vz[max_point])