from mpi4py import MPI
import numpy as np
import random
import pandas as pd
import os
import glob
import datetime
import itertools
from string import ascii_uppercase

import qubo_qutting_utils as qq


from mpi4py import MPI
import numpy as np
import time
import itertools

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n1 = 10
n2 = 10
p1 = 0.2
p2 = 0.2
seed1 = 400
seed2 = 9000
seed_range = np.arange(seed1,seed2,2000)

files = 'exp_' + str(n1) + '_' + str(n2) + '_' + str(p1) + '_' + str(p2) + '_*.csv'
qaoa_files = glob.glob(os.path.join('qaoa_output', files))
print(qaoa_files)

max_cluster_sizes =[3,5]

generated_filenames = []

# Generate all unique combinations of filename,max qubit size, and seed
all_combinations = list(itertools.product(qaoa_files,max_cluster_sizes))
# Generate all combinations

# Split combinations among MPI processes
# Ensure that each process gets a roughly equal number of combinations to work on
combinations_per_process = np.array_split(all_combinations, size)[rank]
#combinations_per_process = [(int(matrix_size),float(density), int(seed)) for matrix_size,density, seed in
combinations_per_process = [(file, int(max_cluster_size)) for file,max_cluster_size in combinations_per_process]

# Extract matrix densities and random seeds for this process
#matrix_sizes_this_process,matrix_densities_this_process, random_seeds_this_process = zip(*combinations_per_process)
file,max_cluster_size = zip(*combinations_per_process)
print(file,max_cluster_size)


if rank == 0:
    start_time = MPI.Wtime()

# Each process executes ckt_build_qaoa with its subset of parameters
for combination in combinations_per_process:
    file, max_cluster_size = combination
    print(file)
    # Ensure max_cluster_size and partition methods are passed as lists if required by ckt_cut_qaoa
    filename = qq.ckt_cut_qaoa([file], [max_cluster_size],
                               ['spectral-clustering','kmeans','agglom'],seed1,seed2)
    #filename=ckt_build_qaoa([matrix_size], [matrix_density], [random_seed], 'er_graph')
    generated_filenames.append(filename)

print(generated_filenames)