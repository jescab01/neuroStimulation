
import pandas as pd
from mpi4py import MPI
import numpy as np
from ThCer_parallel import *

"""
Following a tutorial: 
https://towardsdatascience.com/parallel-programming-in-python-with-message-passing-interface-mpi4py-551e3f198053

execute in terminal with : mpiexec -n 2 python mpi_thcer2.py
"""

name = "JR_v2"

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

## Define param combinations
# Common simulation requirements
subj_ids = [35, 49, 50, 58, 59, 64, 65, 71, 75, 77]
subjects = ["NEMOS_0" + str(id) for id in subj_ids]
# subjects.append("NEMOS_AVG")

coupling_vals = np.arange(0, 120, 1)  # 0.5
noiseth_vals = [0.022, 0.11, 0.22]  #define valor  # [0, 0.022]  # np.logspace(-8, 2, 30)
n_rep = 5

params = [[subj, g, sigma, r] for subj in subjects
          for g in coupling_vals for sigma in noiseth_vals for r in range(n_rep)]

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

local_results = ThCer_parallel(local_params)  # run the function for each parameter set and rank


if rank > 0:  # WORKERS _send to rank 0
    comm.send(local_results, dest=0, tag=14)  # send results to process 0

else:  ## MASTER PROCESS _receive, merge and save results
    final_results = np.copy(local_results)  # initialize final results with results from process 0
    for i in range(1, size):  # determine the size of the array to be received from each process

        # if i < remainder:
        #     rank_size = count + 1
        # else:
        #     rank_size = count
        # tmp = np.empty((rank_size, final_results.shape[1]))  # create empty array to receive results

        tmp = comm.recv(source=i, tag=14)  # receive results from the process

        if tmp is not None:  # Sometimes temp is a Nonetype wo/ apparent cause
            # print(final_results.shape)
            # print(tmp.shape)  # debugging
            # print(i)

            final_results = np.vstack((final_results, tmp))  # add the received results to the final results

    # print("Results")
    # print(final_results)

    fResults_df = pd.DataFrame(final_results, columns=["subject", "coup", "mode", "trial",
                                                       "fpeak", "amplitude_fpeak",
                                                       "rPLV", "plv_mean", "plv_std"])

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

        fResults_df.to_csv(specific_folder + "/results.csv", index=False)

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

        fResults_df.to_csv("results.csv", index=False)