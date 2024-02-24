import random
from collections import OrderedDict
import networkx as nx
#clustering
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import markov_clustering as mc
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import KMeans

#community detection

import networkx.algorithms.community as nx_comm
import community as community_louvain
from collections import defaultdict

import pandas as pd
import numpy as np
import datetime
import time
import os

import re
from collections import OrderedDict
import networkx as nx
import pickle

# Imports required for random matrix generation
import scipy.sparse as sparse
import scipy.stats as stats

# QAOA and circuit cutting specific imports
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import PauliList
from circuit_knitting.cutting import (
    partition_problem,
)
# QP specific imports
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

import utils

# from circuit_knitting_toolbox.circuit_cutting.wire_cutting import cut_circuit_wires

# useful additional packages
#from qiskit.tools.visualization import plot_histogram
#from qiskit.algorithms.minimum_eigensolvers import SamplingVQE, NumPyMinimumEigensolver
#from qiskit.algorithms.optimizers import SPSA
#from qiskit.utils import algorithm_globals

latin_start = 0x0100  # Start of Latin Extended-A block
latin_end = latin_start + 1000  # 1000 characters from the start

labels = ''.join(chr(i) for i in range(latin_start, latin_end))

'''
Methods required to generate random ER graphs
'''


# Erdos Renyi graph
def generate_er_graph(n, p, seed):
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    M = nx.adjacency_matrix(G).todense()
    return M, G


'''
Methods required to generate random sparse matrices
'''


def sprandsym(n, density, seed):
    np.random.seed((seed))
    rvs = stats.poisson(25, loc=10).rvs
    X = sparse.random(n, n, density=density, data_rvs=rvs)
    upper_X = sparse.triu(X)
    result = upper_X + upper_X.T - sparse.diags(X.diagonal())
    return result


def binarize_sparse_matrix(sparse_matrix):
    # create a copy of the sparse matrix to keep the operation non-destructive
    sparse_copy = sparse_matrix.copy()
    # sparse_copy=sparse_copy-sparse.diags(sparse_copy.diagonal())
    # find the coordinates of non-zero elements
    non_zero_coords = sparse_copy.nonzero()
    # set those elements to 1
    sparse_copy[non_zero_coords] = 1
    return sparse_copy


def generate_graph_from_matrix(binarized_sparse_mat):
    G = nx.from_scipy_sparse_array(binarized_sparse_mat)
    return G


# create the quadratic program instance and define the variables
def create_qp_from_qmatrix(Q_matrix):
    max_keys = Q_matrix.shape[0]
    qp = QuadraticProgram('QUBO Matrix Optimization')
    x = qp.binary_var_list(name='x', keys=range(1, max_keys + 1))

    linear_vars = {qp.get_variable(i).name: Q_matrix[i, j]
                   for i in range(max_keys) for j in range(max_keys) if i == j}
    quadratic_vars = {(qp.get_variable(i).name, qp.get_variable(j).name): Q_matrix[i, j]
                      for i in range(max_keys) for j in range(max_keys) if i != j}

    qp.minimize(linear=linear_vars, quadratic=quadratic_vars)
    return qp
    # print(self.qp.prettyprint())


def create_qaoa_ansatz(qp):
    # self.create_qp_from_qmatrix()
    h_qubo, offset = qp.to_ising()
    # print(h_qubo)
    qaoa_ansatz = QAOAAnsatz(cost_operator=h_qubo, reps=1, )
    qaoa_ansatz.entanglement = 'linear'
    params = len(qaoa_ansatz.parameters)
    theta_range = np.linspace(0, np.pi, params)
    qaoa_qc = qaoa_ansatz.bind_parameters(theta_range)
    decomposed_qaoa_ansatz = qaoa_qc.decompose().decompose().decompose().decompose()
    return h_qubo, offset, decomposed_qaoa_ansatz


def get_subgraph_properties1(G):
    cnt = 0
    subgraph = (G.subgraph(c) for c in nx.connected_components(G))
    subgraph_prop = {}
    prop = []
    max_size = []
    max_subgraph_nodes = ''
    for s in subgraph:
        # print(s.nodes())
        n = tuple(s.nodes())
        subgraph_prop[n] = nx.adjacency_matrix(s).todense()
        # print(s.size())
        # print(f'Subgraph {cnt}:: Num of Edges: {s.size()},  Nodes : {s.nodes()}  ')
        cnt += 1
        max_size.append(len(s.nodes()))
        if len(s.nodes) == np.max(max_size):
            max_subgraph_nodes = s.nodes()

    # print(max_subgraph_nodes)
    return cnt, np.max(max_size), subgraph_prop, max_subgraph_nodes


def bfs(q1, max_cluster_size):
    q1_nd = [q[0] for q in sorted(q1.degree, key=lambda x: x[1], reverse=True)]

    q_scidx = {}
    scidx_q = {}
    sc_idx = 0
    visited = []

    # for node in sorted(q1.nodes):
    for node in q1_nd:
        if node not in visited:
            scidx_q.setdefault(sc_idx, [])

            nodes_in_sc = set(list(sum(sorted(list(nx.bfs_tree(q1, source=node, depth_limit=1).edges())), ())))
            # print(f'nodes_in_sc: {nodes_in_sc}')

            for k in nodes_in_sc:
                ## required only if using to find cut wire position*****
                q_scidx.setdefault(k, [])  # q_scidx
                q_scidx[k].append(sc_idx)  # q_scidx
                ## ********

                # print(f'visited: {visited}')
                if (k not in visited) and (len(scidx_q[sc_idx]) < max_cluster_size):
                    scidx_q[sc_idx].append(k)
                    visited.append(k)
            sc_idx += 1

    qsc = OrderedDict(sorted(scidx_q.items()))
    cluster_qubit = {}
    bfs_cluster_labels = []

    for qsc_k, qsc_val in qsc.items():
        for qsc_i in qsc_val:
            cluster_qubit.setdefault(qsc_i, [])
            cluster_qubit[qsc_i].append(qsc_k)
    bfs_cluster_labels = [i for val in (OrderedDict(sorted(cluster_qubit.items()))).values() for i in val]
    return bfs_cluster_labels


def sc(data, n_clusters):
    sc = SpectralClustering(n_clusters, affinity='precomputed', n_init=100)
    sc.fit(data)
    return sc.labels_


def kmeans(data, n_clusters):
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4, random_state=0)
    kmeans.fit(data)
    return kmeans.labels_


def kmeans_pca(data, n_clusters):
    pca = PCA(n_components=n_clusters).fit(data)
    kmeans_pca = KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1)
    kmeans_pca.fit(data)
    return kmeans_pca.labels_


def kmeans_random(data, n_clusters):
    kmeans_rand = KMeans(init="random", n_clusters=n_clusters, n_init=4, random_state=0)
    kmeans_rand.fit(data)
    return kmeans_rand.labels_


def birch(data, n_clusters):
    brc = Birch(n_clusters=n_clusters)
    brc.fit(data)
    birch_labels = brc.predict(data)
    return birch_labels


def agglom(data, n_clusters):
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(data)
    return clustering.labels_


def louivan_community(G, max_size):
    import community as community_louvain
    clabels = community_louvain.best_partition(G)

    # Split communities if they are larger than max_cluster_size
    return clabels  # split_large_communities(clabels, max_size)


def girvan_newman_community(G, max_size):
    communities_generator = nx.community.girvan_newman(G)
    top_level_communities = next(communities_generator)
    partition = {node: i for i, community in enumerate(top_level_communities) for node in community}
    return partition  # split_large_communities(clabels, max_size)


def label_propagation_community(G, max_size):
    partition = nx.community.label_propagation_communities(G)
    partition_dict = {node: i for i, community in enumerate(partition) for node in community}
    return partition_dict  # split_large_communities(partition_dict, max_size)


def leading_eigenvector_community(G, max_size):
    partition = nx.community.leading_eigenvector(G)
    partition_dict = {node: i for i, community in enumerate(partition) for node in community}
    return partition_dict  # split_large_communities(partition_dict, max_size)


def walktrap_community(G, max_size):
    import igraph as ig
    g = ig.Graph.from_networkx(G)
    walktrap = g.community_walktrap()
    communities = walktrap.as_clustering()
    partition_dict = {node: i for i, community in enumerate(communities) for node in community}
    return partition_dict  # split_large_communities(partition_dict, max_size)


def infomap_community(G, max_size):
    from infomap import Infomap
    im = Infomap()
    for e in G.edges():
        im.add_link(*e)
    im.run()

    # Correctly accessing the nodes and their community assignment
    partition = {}
    for node in im.tree:
        if node.is_leaf:
            partition[node.node_id] = node.module_id

    return partition  # split_large_communities(clabels, max_size)


def clique_percolation_community(G, max_size, k=2):
    partition = list(nx.community.k_clique_communities(G, k))
    partition_dict = {node: i for i, community in enumerate(partition) for node in community}
    return partition_dict  # split_large_communities(partition_dict, max_size)


def split_large_communities(partition, max_size):
    new_partition = {}
    new_community_id = max(partition.values()) + 1  # Start from the next available community ID

    for community in set(partition.values()):
        nodes_in_community = [node for node, comm in partition.items() if comm == community]

        if len(nodes_in_community) > max_size:
            # Split the community into smaller ones
            for i in range(0, len(nodes_in_community), max_size):
                for node in nodes_in_community[i:i + max_size]:
                    new_partition[node] = new_community_id
                new_community_id += 1
        else:
            # If the community is within the size limit, keep it unchanged
            for node in nodes_in_community:
                new_partition[node] = community

    return new_partition


def sc2(*argv):
    data = argv[0][0]
    n_clusters = argv[0][1]
    sc = SpectralClustering(n_clusters, affinity='precomputed', n_init=100)
    sc.fit(data)
    # TODO: verify sub cluster size<max_cluster_size. Is this still required??
    return sc.labels_


def kmeans2(*argv):
    data = argv[0][0]
    n_clusters = argv[0][1]

    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4, random_state=0)
    kmeans.fit(data)
    return kmeans.labels_


def birch2(*argv):
    data = argv[0][0]
    n_clusters = argv[0][1]
    brc = Birch(n_clusters=n_clusters)
    brc.fit(data)
    birch_labels = brc.predict(data)
    return birch_labels


def agglom2(*argv):
    data = argv[0][0]
    n_clusters = argv[0][1]
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(data)
    return clustering.labels_


def louivan_community2(*argv):
    G = argv[0][0]
    # print('in louivan')
    # print(G)
    # max_size = argv[0][1]
    # print(max_size)
    import community as community_louvain
    clabels = community_louvain.best_partition(G)


def girvan_newman_community2(*argv):
    G = argv[0][0]
    # max_size = argv[0][1]
    communities_generator = nx.community.girvan_newman(G)
    top_level_communities = next(communities_generator)
    partition = {node: i for i, community in enumerate(top_level_communities) for node in community}
    return partition  # split_large_communities(clabels, max_size)


def label_propagation_community2(*argv):
    G = argv[0][0]
    # max_size = argv[0][1]
    partition = nx.community.label_propagation_communities(G)
    partition_dict = {node: i for i, community in enumerate(partition) for node in community}
    return partition_dict  # split_large_communities(partition_dict, max_size)


def leading_eigenvector_community2(*argv):
    G = argv[0][0]
    # max_size = argv[0][1]
    partition = nx.community.leading_eigenvector(G)
    partition_dict = {node: i for i, community in enumerate(partition) for node in community}
    return partition_dict  # split_large_communities(partition_dict, max_size)


def walktrap_community2(*argv):
    G = argv[0][0]
    # max_size = argv[0][1]

    import igraph as ig
    g = ig.Graph.from_networkx(G)
    walktrap = g.community_walktrap()
    communities = walktrap.as_clustering()
    partition_dict = {node: i for i, community in enumerate(communities) for node in community}
    return partition_dict  # split_large_communities(partition_dict, max_size)


def infomap_community2(*argv):
    G = argv[0][0]
    # max_size = argv[0][1]

    from infomap import Infomap
    im = Infomap()
    for e in G.edges():
        im.add_link(*e)
    im.run()

    # Correctly accessing the nodes and their community assignment
    partition = {}
    for node in im.tree:
        if node.is_leaf:
            partition[node.node_id] = node.module_id

    return partition  # split_large_communities(clabels, max_size)


##TODO: Rest of the partition methods to be updated with *args