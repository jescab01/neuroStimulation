import pandas as pd
from mpi4py import MPI
import numpy as np
from stimWfit_parallel import *

"""
Following a tutorial: 
https://towardsdatascience.com/parallel-programming-in-python-with-message-passing-interface-mpi4py-551e3f198053
Synchronization:
https://mpitutorial.com/tutorials/mpi-broadcast-and-collective-communication/

execute in terminal with : mpiexec -n 4 python stimWmpi_main.py
"""

name = "stimWfit_indWP"

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

### STAGE 1: get baseline

## Define param combinations
# Common simulation requirements
# subj_ids = [35, 49, 50, 58, 59, 64, 65, 71, 75, 77]
# subjects = ["NEMOS_0" + str(id) for id in subj_ids]
# subjects.append("NEMOS_AVG")
n_rep = 30
w_space = [0]  # Computing baseline

working_points = [("jr", "NEMOS_035", 6, 23.5),  # JR pass @ 10/06/2022
                  ("jr", "NEMOS_049", 8, 17.5),  # manipulated: original 49, 20.5 - the only wp out of limit cycle
                  ("jr", "NEMOS_050", 17, 24.5),
                  ("jr", "NEMOS_058", 17, 18.5),
                  ("jr", "NEMOS_059", 5, 11.5),
                  ("jr", "NEMOS_064", 5, 24.5),
                  ("jr", "NEMOS_065", 5, 24.5),
                  ("jr", "NEMOS_071", 10, 22.5),
                  ("jr", "NEMOS_075", 6, 24.5),
                  ("jr", "NEMOS_077", 8, 21.5)]

# Individual WP
if 'ind' in name:
    params = [[subj, mode, g, s, r, w] for mode, subj, g, s in working_points for r in range(n_rep) for w in w_space]

# Common WP
elif 'common' in name:
    g, s = 16, 21.5
    params = [[subj, mode, g, s, r, w] for mode, subj, _, _ in working_points for r in range(n_rep) for w in w_space]

params = np.asarray(params, dtype=object)
n = params.shape[0]


## Distribution of task load in ranks
count = n // size  # number of catchments for each process to analyze
remainder = n % size  # extra catchments if n is not a multiple of size

if rank < remainder:  # processes with rank < remainder analyze one extra catchment
    start = rank * (count + 1)  # index of first catchment to analyze
    stop = start + count + 1  # index of last catchment to analyze
else:
    start = rank * count + remainder
    stop = start + count

if rank < size:  # Preventing n_Cores > n_Tasks raising errors

    local_params = params[start:stop, :]  # get the portion of the array to be analyzed by each rank

    local_results = stimWfit_parallel(local_params)  # run the function for each parameter set and rank


if rank > 0:  # WORKERS _send to rank 0
    comm.send(local_results, dest=0, tag=14)  # send results to process 0
    baseline_subj = None

elif rank == 0:  ## MASTER PROCESS _receive, merge and save results

    print("MASTER PROCESS 1")

    baseline_results = np.copy(local_results)  # initialize final results with results from process 0

    for i in range(1, size):

        tmp = comm.recv(source=i, tag=14)  # receive results from the process

        if tmp is not None:  # Sometimes temp is a Nonetype wo/ apparent cause

            baseline_results = np.vstack((baseline_results, tmp))  # add the received results to the final results

    ## Average to obtain baseline
    baseline_results = pd.DataFrame(baseline_results, columns=["Subject", "Mode", "G", "speed", "rep", "w", "IAF", "module", "bModule"])
    baseline_subj = np.asarray(baseline_results.groupby("Subject")[["IAF", "module", "bModule"]].mean().reset_index())

## Synch: wait for all ranks, we cant allow ranks to advance rank0 before being able to bcast baseline.
comm.Barrier()
## send back to ranks: broadcast
baseline_subj = comm.bcast(baseline_subj, root=0)


### STAGE 2: gather datapoints
print("STAGE2")
## Define param combinations
# Common simulation requirements
w_space = np.arange(0, 0.5, 0.01)

# Individual WP
if 'ind' in name:
    params = [[subj, mode, g, s, r, w] for mode, subj, g, s in working_points for r in range(n_rep) for w in w_space]

# Common WP
elif 'common' in name:
    params = [[subj, mode, g, s, r, w] for mode, subj, _, _ in working_points for r in range(n_rep) for w in w_space]

params = np.asarray(params, dtype=object)
n = params.shape[0]

## Distribution of task load in ranks
count = n // size  # number of catchments for each process to analyze
remainder = n % size  # extra catchments if n is not a multiple of size

if rank < remainder:  # processes with rank < remainder analyze one extra catchment
    start = rank * (count + 1)  # index of first catchment to analyze
    stop = start + count + 1  # index of last catchment to analyze
else:
    start = rank * count + remainder
    stop = start + count

local_params = params[start:stop, :]  # get the portion of the array to be analyzed by each rank

local_results = stimWfit_parallel(local_params, baseline_subj)  # run the function for each parameter set and rank

## Synch: wait for all ranks, we cant allow rank 0 to receive before all others have finished.
comm.Barrier()

if rank > 0:  # WORKERS _send to rank 0
    comm.send(local_results, dest=0, tag=14)  # send results to process 0

else:  ## MASTER PROCESS _receive, merge and save results

    final_results = np.copy(local_results)  # initialize final results with results from process 0
    for i in range(1, size):

        tmp = comm.recv(source=i, tag=14)  # receive results from the process

        if (tmp is not None) & (len(tmp) > 0):  # Sometimes temp is a Nonetype wo/ apparent cause

            final_results = np.vstack((final_results, tmp))  # add the received results to the final results

    fResults_df = pd.DataFrame(final_results, columns=["Subject", "Mode", "G", "speed", "rep", "w", "IAF", "module", "bModule"])

    ## Save resutls
    ## Folder structure - Local
    if "Jesus CabreraAlvarez" in os.getcwd():
        wd = os.getcwd()

        main_folder = wd + "\\" + "PSE"
        if os.path.isdir(main_folder) == False:
            os.mkdir(main_folder)
        specific_folder = main_folder + "\\PSEmpi_" + name + "-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")

        if os.path.isdir(specific_folder) == False:
            os.mkdir(specific_folder)

        fResults_df.to_csv(specific_folder + "/stimWmpi_results.csv", index=False)

    ## Folder structure - CLUSTER
    else:
        main_folder = "PSE"
        if os.path.isdir(main_folder) == False:
            os.mkdir(main_folder)

        os.chdir(main_folder)

        specific_folder = "PSEmpi_" + name + "-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
        if os.path.isdir(specific_folder) == False:
            os.mkdir(specific_folder)

        os.chdir(specific_folder)

        fResults_df.to_csv("stimWmpi_results.csv", index=False)