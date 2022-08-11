import time
import numpy as np
import pandas as pd
import scipy
from mne import time_frequency, filter

from tvb.simulator.lab import *
from mpi4py import MPI
import datetime
import glob

def testFreqs_parallel(params):

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    print("Hello world from rank", str(rank), "of", str(size), '__', datetime.datetime.now().strftime("%Hh:%Mm:%Ss"))

    ## Folder structure - Local
    if "LCCN_Local" in os.getcwd():
        ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data2\\"
        import sys
        sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
        from toolbox.fft import multitapper, FFTpeaks
        from toolbox.fc import PLV
        from toolbox.signals import epochingTool

    ## Folder structure - CLUSTER
    else:
        from toolbox import multitapper, PLV, epochingTool, FFTpeaks
        wd = "/home/t192/t192950/mpi/"
        ctb_folder = wd + "CTB_data2/"

    # Prepare simulation parameters
    simLength = 10 * 1000  # ms
    samplingFreq = 1000  # Hz
    transient = 2000  # ms

    # COMMON SIMULATION PARAMETERS   ###
    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

    mon = (monitors.Raw(),)

    local_results = list()

    for set in params:

        tic = time.time()
        print(set)

        target, n_remove, chosen_rois2remove, r, stimulation_site, stimulus_type, stim_params, mode, emp_subj, g, s, w = set

        # ROIS OF INTEREST for the effect    ###############
        if "cb" in mode:
            rois = [4, 5, 18, 19]
        else:
            rois = [34, 35, 70,
                    71]  # rois implicated in the effect: 35-ACCl, 36-AACr, 71-Prl, 72-Prr [note python 0-indexing]
        ids = [1, 2, 3, 4]  # relations of interest: indices to choose from PLV's upper triangle (no diagonal)


        # STRUCTURAL CONNECTIVITY      #########################################
        if '_pTh' in mode:
            conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2pTh_pass.zip")
        else:
            conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2_pass.zip")
        conn.weights = conn.scaled_weights(mode="tract")

        # Define regions implicated in Functional analysis: remove  Cerebelum, Thalamus, Caudate (i.e. subcorticals)
        cortical_rois = ['Precentral_L', 'Precentral_R', 'Frontal_Sup_2_L',
                         'Frontal_Sup_2_R', 'Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                         'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L',
                         'Frontal_Inf_Tri_R', 'Frontal_Inf_Orb_2_L', 'Frontal_Inf_Orb_2_R',
                         'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L',
                         'Supp_Motor_Area_R', 'Olfactory_L', 'Olfactory_R',
                         'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R',
                         'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R', 'Rectus_L', 'Rectus_R',
                         'OFCmed_L', 'OFCmed_R', 'OFCant_L', 'OFCant_R', 'OFCpost_L',
                         'OFCpost_R', 'OFClat_L', 'OFClat_R', 'Insula_L', 'Insula_R',
                         'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Mid_L',
                         'Cingulate_Mid_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                         'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                         'ParaHippocampal_R', 'Calcarine_L',
                         'Calcarine_R', 'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R',
                         'Occipital_Sup_L', 'Occipital_Sup_R', 'Occipital_Mid_L',
                         'Occipital_Mid_R', 'Occipital_Inf_L', 'Occipital_Inf_R',
                         'Fusiform_L', 'Fusiform_R', 'Postcentral_L', 'Postcentral_R',
                         'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                         'Parietal_Inf_R', 'SupraMarginal_L', 'SupraMarginal_R',
                         'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R',
                         'Paracentral_Lobule_L', 'Paracentral_Lobule_R', 'Heschl_L', 'Heschl_R',
                         'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L',
                         'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R',
                         'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L',
                         'Temporal_Inf_R']

        # load text with FC rois; check if match SC
        FClabs = list(np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/roi_labels_rms.txt", dtype=str))
        FC_cortex_idx = [FClabs.index(roi) for roi in
                         cortical_rois]  # find indexes in FClabs that matches cortical_rois
        SClabs = list(conn.region_labels)
        SC_cortex_idx = [SClabs.index(roi) for roi in cortical_rois]

        ### MODIFY SC      #########
        init_tracts = sum(conn.weights[:, target])

        for roi2remove in chosen_rois2remove:
            conn.weights[roi2remove, target] = 0
            conn.weights[target, roi2remove] = 0

        tracts_left = sum(conn.weights[:, target])
        removed_tracts = init_tracts - tracts_left

        # NEURAL MASS MODEL    #########################################################
        # JANSEN-RIT
        # Parameters from Stefanovski 2019. Good working point at g=33, s=15.5 on AAL2red connectome.
        m = models.JansenRit(A=np.array([3.25]), B=np.array([22]), J=np.array([1]),
                             a=np.array([0.1]), a_1=np.array([135]), a_2=np.array([108]),
                             a_3=np.array([33.75]), a_4=np.array([33.75]), b=np.array([0.06]),
                             mu=np.array([0.1085]), nu_max=np.array([0.0025]), p_max=np.array([0]),
                             p_min=np.array([0]),
                             r=np.array([0.56]), v0=np.array([6]))


        # COUPLING FUNCTION   #########################################
        if "jrd" in mode:
            coup = coupling.SigmoidalJansenRitDavid(a=np.array([g]), w=m.w, e0=m.e0, v0=m.v0, r=m.r)
        else:
            coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                               r=np.array([0.56]))
        conn.speed = np.array([s])


        # STIMULUS ###############################

        if stimulus_type == "sinusoid":

            eqn_t = equations.Sinusoid()
            eqn_t.parameters['amp'] = 1
            eqn_t.parameters['frequency'] = stim_params  # Hz
            eqn_t.parameters['onset'] = 0  # ms
            eqn_t.parameters['offset'] = simLength  # ms
            # if w != 0:
            #     eqn_t.parameters['DC'] = 0.0005 / w

        elif stimulus_type == "noise":

            eqn_t = equations.Noise()
            eqn_t.parameters["mean"] = stim_params
            eqn_t.parameters["std"] = (1 - eqn_t.parameters["mean"]) / 3  # p(mean<x<mean+std) = 0.34 in gaussian distribution [max=1; min=-1]
            eqn_t.parameters["onset"] = 0
            eqn_t.parameters["offset"] = simLength


        ## electric field; electrodes placed @ P3P4 to stimulate precuneus
        # weighting = np.loadtxt(ctb_folder + 'CurrentPropagationModels/' + emp_subj + '-roast_P3P4Model_ef_mag-AAL2red.txt') * w
        ## Focal stimulation on ACC electric field;
        weighting = np.loadtxt(glob.glob(
            ctb_folder + 'CurrentPropagationModels/' + emp_subj + '-efnorm_mag-' + stimulation_site + '*-AAL2.txt')[0],
                               delimiter=",") * w

        stimulus = patterns.StimuliRegion(temporal=eqn_t, connectivity=conn, weight=weighting)

        # Configure space and time
        stimulus.configure_space()
        stimulus.configure_time(np.arange(0, simLength, 1))


        print("Simulating for Coupling factor = %i and speed = %i" % (g, s))

        # Run simulation
        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon, stimulus=stimulus)
        sim.configure()
        output = sim.run(simulation_length=simLength)

        # Extract data: "output[a][b][:,0,:,0].T" where:
        # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
        if "jrd" in mode:
            raw_data = m.w * (output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T) + \
                       (1 - m.w) * (output[0][1][transient:, 3, :, 0].T - output[0][1][transient:, 4, :, 0].T)
        else:
            raw_data = output[0][1][transient:, 0, :, 0].T

        raw_data = raw_data[rois, :]
        raw_time = output[0][0][transient:]

        regionLabels = conn.region_labels[rois]

        # Fourier Analysis plot
        # FFTplot(raw_data, simLength-transient, regionLabels, main_folder, mode="html")
        fft_peaks = FFTpeaks(raw_data, simLength-transient)[0]
        local_results.append([stimulation_site, stimulus_type, stim_params, mode, emp_subj, g, s, w, r, target,
                              n_remove, list(chosen_rois2remove), tracts_left, removed_tracts] + list(fft_peaks))

        print("simulating stim_type = %s | stim_param = %0.2f - round = %i" % (stimulus_type, stim_params, r))
        print("LOOP ROUND REQUIRED %0.4f seconds.\n\n\n\n" % (time.time() - tic,))

    return np.asarray(local_results, dtype=object)
