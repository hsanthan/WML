


from mpi4py import MPI
import numpy as np
import datetime
import itertools
from string import ascii_uppercase


# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    matrix_sizes = [50, 100]

    matrix_densities = [0.2, 0.4]
    num_of_experiments = 8
    random_seeds = sorted([random.randint(10, 10000) for _ in range(num_of_experiments)])
    max_cluster_sizes = [3]

    generated_filenames = []

    # Generate all unique combinations of matrix size, density, and seed
    all_combinations = list(itertools.product(matrix_sizes, matrix_densities, random_seeds))
    # Generate all combinations

    # Split combinations among MPI processes
    # Ensure that each process gets a roughly equal number of combinations to work on
    combinations_per_process = np.array_split(all_combinations, size)[rank]
    combinations_per_process = [(int(matrix_size), float(density), int(seed)) for matrix_size, density, seed in
                                combinations_per_process]

    # Extract matrix densities and random seeds for this process
    matrix_sizes_this_process, matrix_densities_this_process, random_seeds_this_process = zip(*combinations_per_process)

    if rank == 0:
        start_time = MPI.Wtime()

    # Each process executes ckt_build_qaoa with its subset of parameters
    for combination in combinations_per_process:
        matrix_size, matrix_density, random_seed = combination
        # Ensure matrix_density and random_seed are passed as lists if required by ckt_build_qaoa
        filename = ckt_build_qaoa([matrix_size], [matrix_density], [random_seed], 'er_graph')
        generated_filenames.append(filename)

    all_filenames = comm.gather(generated_filenames, root=0)

    if rank == 0:
        # Optionally, gather results from all processes if needed
        consolidated_df = pd.DataFrame()
        for process_files in all_filenames:
            for filename in process_files:
                temp_df = pd.read_csv(f'qaoa_output/' + filename)
                consolidated_df = pd.concat([consolidated_df, temp_df], ignore_index=True)
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        ms1 = matrix_sizes[0]
        ms2 = matrix_sizes[-1]
        np1 = matrix_densities[0]
        np2 = matrix_densities[-1]
        seed1 = random_seeds[0]
        seed2 = random_seeds[-1]
        filename = f'exp_' + str(ms1) + '_' + str(ms2) + '_' + str(np1) + '_' + str(np2) + '_' + str(seed1) + '_' + str(
            seed2) + '_qaoa_' + current_time + '.csv'
        consolidated_csv_filename = f'qaoa_output/' + filename + '.csv'
        consolidated_df.to_csv(consolidated_csv_filename, index=False)
        print("Total time:", MPI.Wtime() - start_time)

        # Cleanup: Delete the individual CSV files after consolidation
        for process_files in all_filenames:
            for filename in process_files:
                try:
                    os.remove(f"qaoa_output/" + filename)
                    print(f"Deleted file: {filename}")
                except OSError as e:
                    print(f"Error deleting file {filename}: {e.strerror}")
