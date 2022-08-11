import pandas as pd
from mpi4py import MPI
import numpy as np
from entSC_parallel import *

"""
Following a tutorial: 
https://towardsdatascience.com/parallel-programming-in-python-with-message-passing-interface-mpi4py-551e3f198053
Synchronization:
https://mpitutorial.com/tutorials/mpi-broadcast-and-collective-communication/

execute in terminal with : mpiexec -n 4 python entSC_main.py
"""

name = "entrainment_bySC_indWPpass"

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#######
## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data2\\"
## Folder structure - CLUSTER
else:
    wd = "/home/t192/t192950/mpi/"
    ctb_folder = wd + "CTB_data2/"


## Define param combinations
stimulation_sites = ["roast_P3P4Model"]  # roast_ACCtarget; roast_P3P4Model;
stim_w = 0.29  # 0.24 = mean; 0.28 = median -- With FFTpeaks instead multitapper. And taking high power wp.

# Define stimulus
stimuli = []
stimulus_type = "sinusoid"
stim_freqs = np.concatenate(([0], np.linspace(4, 18, 60)))

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

params = []
for mode, subj, g, s in working_points:
    conn = connectivity.Connectivity.from_file(ctb_folder + subj + "_AAL2_pass.zip")

    for target in [34, 35, 70, 71]:
        connected2target_rois = np.where(conn.binarized_weights[target])[0]
        max_n2Remove = int(sum(conn.binarized_weights[target])*0.85)  ## Subj1: ACC_L - 50; ACC_R - 46; Pre_L - 106; Pre_R - 108.

        for n_remove in range(max_n2Remove):
            for rep in range(5):
                ## Randomly choose "n_remove" connections and set them to 0
                chosen_rois2remove = np.random.choice(connected2target_rois, size=n_remove, replace=False)

                ######
                for stim_param in stim_freqs:
                    params.append([target, n_remove, chosen_rois2remove, rep, "roast_P3P4Model", stimulus_type, stim_param, mode, subj, g, s, stim_w])

# Individual WP
# if "ind" in name:
#     stim_w = 0.24  # 0.24 = mean; 0.28 = median -- With FFTpeaks instead multitapper. And taking high power wp.
#     params = [[target, n_remove, r, stimulation_site, stimulus_type, stim_params, mode, subj, g, s, stim_w]
#               for target in targets
#               for n_remove in rem_rois
#               for r in range(n_rep)
#               for stimulation_site in stimulation_sites
#               for stimulus_type, stim_params in stimuli
#               for mode, subj, g, s in working_points]

# Common WP
# elif "common" in name:
#     g, s, stim_w = 15, 21.5, 0.15  # 0.15 = average mean
#     params = [[stimulation_site, stimulus_type, stim_params, mode, subj, g, s, stim_w, r]
#               for stimulation_site in stimulation_sites
#               for stimulus_type, stim_params in stimuli
#               for mode, subj, _, _ in working_points
#               for r in range(n_rep)]

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

    local_results = testFreqs_parallel(local_params)  # run the function for each parameter set and rank


if rank > 0:  # WORKERS _send to rank 0
    comm.send(local_results, dest=0, tag=14)  # send results to process 0


elif rank == 0:  ## MASTER PROCESS _receive, merge and save results

    print("MASTER PROCESS...")

    final_results = np.copy(local_results)  # initialize final results with results from process 0

    for i in range(1, size):

        tmp = comm.recv(source=i, tag=14)  # receive results from the process

        if tmp is not None:  # Sometimes temp is a Nonetype wo/ apparent cause

            final_results = np.vstack((final_results, tmp))  # add the received results to the final results

    ### Create DATAFRAME
    # Prepare labels relations
    from tvb.datatypes.connectivity import Connectivity
    # ROIS OF INTEREST for the effect    ###############
    rois = [34, 35, 70, 71]  # rois implicated in the effect: 35-ACCl, 36-AACr, 71-Prl, 72-Prr [note python 0-indexing]
    # Folder structure - Local
    if "LCCN_Local" in os.getcwd():
        ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data2\\"
    # Folder structure - CLUSTER
    else:
        wd = "/home/t192/t192950/mpi/"
        ctb_folder = wd + "CTB_data2/"

    conn = Connectivity.from_file(ctb_folder + "NEMOS_035_AAL2_pass.zip")
    regionLabels = conn.region_labels[rois]

    Results = pd.DataFrame(final_results, columns=["stimulation_site", "stimulus_type", "stim_params", "mode", "subject",
                                                   "g", "speed", "stimW", "rep", "target", "n_remove", "rois_2remove",
                                                   "tracts_left", "removed_tracts"] + list(regionLabels))

    ## Save results
    ## Folder structure - Local
    if "Jesus CabreraAlvarez" in os.getcwd():
        wd = os.getcwd()

        main_folder = wd + "\\" + "PSE"
        if os.path.isdir(main_folder) == False:
            os.mkdir(main_folder)
        specific_folder = main_folder + "\\PSEmpi_" + name + "-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")

        if os.path.isdir(specific_folder) == False:
            os.mkdir(specific_folder)

        Results.to_csv(specific_folder + "/entrainment_bySC_results.csv", index=False)

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

        Results.to_csv("entrainment_bySC_results.csv", index=False)