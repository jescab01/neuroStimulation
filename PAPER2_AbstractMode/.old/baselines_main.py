
import pandas as pd
from mpi4py import MPI
import numpy as np
from baselines_parallel import *

"""
Following a tutorial: 
https://towardsdatascience.com/parallel-programming-in-python-with-message-passing-interface-mpi4py-551e3f198053
Synchronization:
https://mpitutorial.com/tutorials/mpi-broadcast-and-collective-communication/

execute in terminal with : mpiexec -n 4 python nmm_main.py
"""

name = "baselines_stimAllConds"

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

## DEFINE SIMULATION PARAMETERS
subj, reps, sigma = "NEMOS_035", 30, 0.11

params = [["isolatedStim_oneNode", subj, 0, 0.11, r] for r in range(reps)] + \
         [["isolatedStim_twoNodes", subj, 27, 0.11, r] for r in range(reps)] + \
         [["isolatedStim_cb", subj, 27, 0.11, r] for r in range(reps)]

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

    local_results = baselines_parallel(local_params)  # run the function for each parameter set and rank


if rank > 0:  # WORKERS _send to rank 0
    comm.send(local_results, dest=0, tag=14)  # send results to process 0


else:  ## MASTER PROCESS _receive, merge and save results

    final_results = np.copy(local_results)  # initialize final results with results from process 0
    for i in range(1, size):

        tmp = comm.recv(source=i, tag=14)  # receive results from the process

        if (tmp is not None) & (len(tmp) > 0):  # Sometimes temp is a Nonetype wo/ apparent cause

            final_results = np.vstack((final_results, tmp))  # add the received results to the final results

    fResults_df = pd.DataFrame(final_results, columns=["mode", "cond", "trial", "node",
                                                       "fpeak", "amplitude_fpeak", "plv"])
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

        fResults_df.to_pickle(specific_folder + "/nmm_results.pkl")

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

        fResults_df.to_pickle("nmm_results.pkl")