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





def ckt(qaoa_decomposed, part_lbl, observables):
    ordered_part_lbl = OrderedDict(sorted(part_lbl.items()))
    partition_labels = ''.join(ordered_part_lbl.values())
    print(f'\nPartition labels for CKT: {partition_labels}')
    start_time = time.time()

    partitioned_problem = partition_problem(circuit=qaoa_decomposed,
                                            partition_labels=partition_labels,
                                            observables=observables)
    ckt_runtime = time.time() - start_time
    bases = partitioned_problem.bases
    sampling_overhead = np.prod([basis.overhead for basis in bases])
    print(f"Sampling overhead: {sampling_overhead}")
    return partition_labels, sampling_overhead, ckt_runtime




def partitioning2(max_cluster_size, qsubgraph_prop, partition_method):
    # cm_part_lbl = {}
    # partitioning_time = {}
    start_time = time.time()
    part_lbl = {}
    max_key_cnt = -1
    for i, key in enumerate(qsubgraph_prop.keys()):
        # print(f'Subgraph nodes : {key}')
        if len(key) > max_cluster_size:
            data = qsubgraph_prop[key]
            n_clusters = int(np.ceil(len(key) / max_cluster_size))
            q1 = nx.from_numpy_array(data)
            algos = {
                'spectral-clustering':
                    {
                        'func': sc2,
                        'params': (data, n_clusters)
                    },
                'kmeans':
                    {
                        'func': kmeans2,
                        'params': (data, n_clusters)
                    },
                # 'kmeans-pca' : { 'func':kmeans_pca2, 'params':(data, n_clusters) },
                # 'kmeans-random' : {'func':kmeans_random2, 'params':(data, n_clusters)},
                'birch':
                    {
                        'func': birch2,
                        'params': (data, n_clusters)
                    },
                'agglom':
                    {
                        'func': agglom2,
                        'params': (data, n_clusters)
                    },
                'louivan':
                    {
                        'func': louivan_community2,
                        'params': (q1, max_cluster_size)},
                'girvan-newman':
                    {
                        'func': girvan_newman_community2,
                        'params': (q1, max_cluster_size)
                    },
                'label-propagation':
                    {
                        'func': label_propagation_community2,
                        'params': (q1, max_cluster_size)},
                'leading-eigenvector':
                    {
                        'func': leading_eigenvector_community2,
                        'params': (q1, max_cluster_size)},
                'walktrap':
                    {
                        'func': walktrap_community2,
                        'params': (q1, max_cluster_size)},
                'infomap':
                    {
                        'func': infomap_community2,
                        'params': (q1, max_cluster_size)},
                # 'clique-percolation' : {'func':clique_percolation_community2, 'params':(q1,max_cluster_size)}
            }

            # pm = ops[partition_method](func_arg[partition_method])
            func = algos[partition_method]['func']
            params = algos[partition_method]['params']

            # call the partition method
            clabels = func(params)

            # increment the sclbl with max_key_cnt so that the next sub-graphs labels are not repeated
            cluster_lbls = [lbl + max_key_cnt + 1 for lbl in clabels]
            for j, k_ in enumerate(key):
                if cluster_lbls[j] < len(labels):
                    part_lbl[k_] = labels[cluster_lbls[j]]
                else:
                    print(
                        f'Error: Index out of range. cluster_lbls[j]: {cluster_lbls[j]}, Length of labels: {len(labels)}')

            max_key_cnt = np.max(cluster_lbls)

        else:
            # Handle smaller subgraphs
            max_key_cnt += 1
            for k in key:
                if max_key_cnt < len(labels):
                    part_lbl[k] = labels[max_key_cnt]
                else:
                    print(f'Error: Index out of range. max_key_cnt: {max_key_cnt}, Length of labels: {len(labels)}')
        # print('done part')
        # cm_part_lbl[cm] = (part_lbl, partitioning_runtime)
        # print(f'Partition labels for method {cm}: {part_lbl}')
    partitioning_time = time.time() - start_time

    return part_lbl, partitioning_time



qaoa_graph_obj_dir = 'qaoa_graph_objects/'
qaoa_output = 'qaoa_output/'
ckt_output = 'ckt_output/'


def ckt_build_qaoa(mat_size, n_times_p, random_seeds, matrix_type):
    '''
    ## Create sparse matrix for a given size and density
    ## Convert the sparse matrix into QUBO, map Ising hamiltonian
    ## Generate QAOA ansatz for QUBO and assign its observables
    '''

    cols = ['n', 'p', 'seed', 'Graph File', 'qaoa Ansatz File',
            'Ising Hamiltonian Runtime', 'Ansatz Building Runtime', 'qubitOp']

    df = pd.DataFrame(columns=cols)
    # df['Graph Prop'] = df['Graph Prop'].astype('object')
    # df['qubitOp'] = df['qubitOp'].astype('object')

    i = 0
    for n in mat_size:
        for p in n_times_p:
            for seed in random_seeds:
                # multi-thread code to go here for a set of seeds??
                try:
                    print(f'\n\nQAOA for size {n}, n*p {p}, seed {seed}')
                    if i == 0:
                        seed1 = seed
                    i += 1
                    # Create sparse matrix for a given size and density
                    # record the matrix creation time
                    start_time = time.time()
                    if 'random_sparse' in matrix_type:
                        M = sprandsym(n, p, seed)
                        M = binarize_sparse_matrix(M)
                        q = generate_graph_from_matrix(M)

                    ## Get adjacency matrix for a random Erdos Renyi Graph
                    elif 'er_graph' in matrix_type:
                        M, q = generate_er_graph(n, p, seed)
                    matrix_creation_time = time.time() - start_time

                    # save graph object to file
                    graph_filename = os.path.join(qaoa_graph_obj_dir, f'{n}_{p}_{seed}_graph.pickle')
                    pickle.dump(q, open(graph_filename, 'wb'))

                    # Get subgraphs' properties
                    # qnum_sub_graphs, largest_subgraph_size, qsubgraph_prop, max_subgraph_nodes = get_subgraph_properties1(q)

                    ## Convert the sparse matrix into QUBO
                    # Timing for QUBO conversion
                    start_time = time.time()
                    qp = create_qp_from_qmatrix(M)
                    qp2qubo = QuadraticProgramToQubo()
                    qubo = qp2qubo.convert(qp)
                    qubitOp, offset = qubo.to_ising()
                    qubo_conversion_time = time.time() - start_time
                    # print(qubitOp)

                    ## Generate and save QAOA ansatz for QUBO
                    # Timing for QAOA
                    start_time = time.time()
                    qaoa = QAOAAnsatz(cost_operator=qubitOp, reps=1)
                    qaoa_observable_pat = '[A-Z]+'
                    observables = (PauliList(re.findall(r"'([A-Z]+)'", str(qubitOp))))
                    # observables = PauliList(re.findall(qaoa_observable_pat, str(qubitOp)))
                    qaoa_time = time.time() - start_time

                    qa = qaoa.decompose().decompose().decompose().decompose()
                    # display(qa.draw(scale=0.4))

                    # pickle the qaoa ansatz
                    qc_file = os.path.join(qaoa_graph_obj_dir, f'{n}_{p}_{seed}_qaoa.qpy')
                    with open(file=qc_file, mode='wb') as qcfile:
                        pickle.dump(qaoa, qcfile)

                    # Populate the df
                    df.loc[i, 'n'] = n
                    df.loc[i, 'p'] = p
                    df.loc[i, 'seed'] = seed
                    df.loc[i, 'Graph File'] = graph_filename
                    df.loc[i, 'qaoa Ansatz File'] = qc_file
                    df.loc[i, 'qubitOp'] = str(qubitOp)
                    df.loc[i, 'Ising Hamiltonian Runtime'] = qubo_conversion_time
                    df.loc[i, 'Ansatz Building Runtime'] = qaoa_time

                except Exception as e:
                    print(f"An error occurred: {e}")
                    # Optionally, save the DataFrame
                    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f'exp_data_up_to_failure_{current_time}.csv'
                    df.to_csv(filename, index=False)
                    # re-raise the exception if you want to stop the loop
                    raise e
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    filename = f'exp_' + str(n) + '_' + str(p) + '_' + str(seed1) + '_' + str(seed) + '_qaoa_' + current_time + '.csv'
    df.to_csv(os.path.join(qaoa_output, filename), index=False)
    return filename


