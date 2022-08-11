
import time
import numpy as np
import pandas as pd
import scipy
from mne import time_frequency, filter

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003
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
    simLength = 24 * 1000  # ms
    samplingFreq = 1000  # Hz
    transient = 4000  # ms

    # COMMON SIMULATION PARAMETERS   ###
    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

    mon = (monitors.Raw(),)

    local_results = list()

    for j, set in enumerate(params):

        tic = time.time()
        print("Rank %i out of %i  ::  %i/%i " % (rank, size, j+1, len(params)))

        print(set)
        stimulation_site, stimulus_type, stim_params, mode, emp_subj, g, s, w, r = set

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
        cingulum_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                         'Insula_L', 'Insula_R',
                         'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                         'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                         'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R',
                         'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                         'Parietal_Inf_R', 'Precuneus_L', 'Precuneus_R',
                         'Thalamus_L', 'Thalamus_R']

        # load text with FC rois; check if match SC
        FClabs = list(np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/roi_labels_rms.txt", dtype=str))
        FC_cortex_idx = [FClabs.index(roi) for roi in
                         cortical_rois]  # find indexes in FClabs that matches cortical_rois
        SClabs = list(conn.region_labels)
        SC_cortex_idx = [SClabs.index(roi) for roi in cortical_rois]


        # Subset for Cingulum Bundle
        if "cb" in mode:
            FC_cb_idx = [FClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cortical_rois
            SC_cb_idx = [SClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cortical_rois
            conn.weights = conn.weights[:, SC_cb_idx][SC_cb_idx]
            conn.tract_lengths = conn.tract_lengths[:, SC_cb_idx][SC_cb_idx]
            conn.region_labels = conn.region_labels[SC_cb_idx]


        # ROIs of interest to measure alpha
        # All occipito-parietal regins w/ 0-indexing in Python.
        occipital_rois = ['Calcarine_L', 'Calcarine_R', 'Cuneus_L', 'Cuneus_R',
                          'Lingual_L', 'Lingual_R', 'Occipital_Sup_L', 'Occipital_Sup_R',
                          'Occipital_Mid_L', 'Occipital_Mid_R', 'Occipital_Inf_L',
                          'Occipital_Inf_R', 'Parietal_Sup_L', 'Parietal_Sup_R',
                          'Parietal_Inf_L', 'Parietal_Inf_R', 'SupraMarginal_L',
                          'SupraMarginal_R', 'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R']

        if "cb" in mode:
            occ_cb = list(set(occipital_rois).intersection(set(cingulum_rois)))
            rois_iaf = [list(conn.region_labels).index(roi) for roi in occ_cb]
        else:
            rois_iaf = [SClabs.index(roi) for roi in occipital_rois]


        # NEURAL MASS MODEL    #########################################################
        if "jrd" in mode:  # JANSEN-RIT-DAVID
            if "_def" in mode:
                sigma_array = 0.022
                p_array = 0.22
            else:  # for jrd_pTh and jrd modes
                sigma_array = np.asarray([0.022 if 'Thal' in roi else 0 for roi in conn.region_labels])
                p_array = np.asarray([0.22 if 'Thal' in roi else 0.13 for roi in conn.region_labels])

            # Parameters edited from David and Friston (2003).
            m = JansenRitDavid2003(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
                                     tau_e1=np.array([10.8]), tau_i1=np.array([22.0]),
                                     He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
                                     tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),

                                     w=np.array([0.8]), c=np.array([135.0]),
                                     c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                                     c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                                     v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                                     p=np.array([p_array]), sigma=np.array([sigma_array]))

            # Remember to hold tau*H constant.
            m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
            m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])

        else:  # JANSEN-RIT
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


        # Run simulation
        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
        sim.configure()
        output = sim.run(simulation_length=12000)

        # Extract data: "output[a][b][:,0,:,0].T" where:
        # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
        if "jrd" in mode:
            raw_data = m.w * (output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T) + \
                       (1 - m.w) * (output[0][1][transient:, 3, :, 0].T - output[0][1][transient:, 4, :, 0].T)
        else:
            raw_data = output[0][1][transient:, 0, :, 0].T

        ## GET IAF and other regions frequency peaks
        # When trying to desynchronize two regions we want two regions behaving differently. On one hand,
        # a **passive region** that keeps its oscillatory regime during stimulation, this can be achieved by
        # network hubs with lost of connections as they are less vulnerable to stimulation. On the other hand,
        # we want an **active region** that will be entrained by stimulation and that will shift its oscillation
        # frequency towards that of the stimulation. With this scheme, in the point where stimulation matches
        # the frequency of the passive region, there will be a hypersynchronization with the active ROI. So,
        # the effects of stimulation have to be considered centered on the passive region natural oscillation frequency.
        # In our experiment, this is the precuneus.
        IAF = np.average(FFTpeaks(raw_data[rois_iaf, :], 12000-transient)[0])
        pre_acc_peak = np.average(FFTpeaks(raw_data[rois[:2], :], 12000 - transient)[0])
        pre_prec_peak = np.average(FFTpeaks(raw_data[rois[2:], :], 12000 - transient)[0])

        print("IAF from occipito-parietal rois @ %0.2f" % IAF)


        ############################################################################
        ## NOW Stimulate
        ############################################################################

        # STIMULUS ###############################

        if stimulus_type == "baseline":
            # No stimulation
            eqn_t = equations.Sinusoid()
            eqn_t.parameters['amp'] = 1
            eqn_t.parameters['frequency'] = 0  # Hz
            eqn_t.parameters['onset'] = 0  # ms
            eqn_t.parameters['offset'] = simLength  # ms

        elif stimulus_type == "sinusoid":
            # tACS
            eqn_t = equations.Sinusoid()
            eqn_t.parameters['amp'] = 1
            eqn_t.parameters['frequency'] = pre_prec_peak + stim_params  # Hz ## Prec peak as passive region
            eqn_t.parameters['onset'] = 0  # ms
            eqn_t.parameters['offset'] = simLength  # ms
            # if w != 0:
            #     eqn_t.parameters['DC'] = 0.0005 / w

        elif stimulus_type == "noise":
            # RNS
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
        if "cb" in mode:
            weighting = weighting[SC_cb_idx]

        if "abstract" in mode:
            if "ACC" in stimulation_site:
                weighting = np.asarray([weight if conn.region_labels[i] in ['Cingulate_Ant_L', 'Cingulate_Ant_R'] else 0 for i, weight in enumerate(weighting)])
            elif "F3F4" in stimulation_site:
                weighting = np.asarray([weight if conn.region_labels[i] in ['Cingulate_Ant_L', 'Cingulate_Ant_R'] else 0 for i, weight in enumerate(weighting)])
            elif "P3P4" in stimulation_site:
                weighting = np.asarray([weight if conn.region_labels[i] in ['Precuneus_L', 'Precuneus_R'] else 0 for i, weight in enumerate(weighting)])

        print(mode)
        print(weighting)

        ## TEMP: test acc indirect influence
        # weighting[34] = 0
        # weighting[35] = 0

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

        # Fourier Analysis plot
        # FFTplot(raw_data, simLength-transient, regionLabels, main_folder, mode="html")
        post_fft_peaks = FFTpeaks(raw_data, simLength-transient)[0]

        ##########
        ### Measure functional connectivity between regions of interest : line 60 - rois

        bands = [["3-alfa"], [(8, 12)]]
        ## [["1-delta", "2-theta", "3-alfa", "4-beta", "5-gamma"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]

        for b in range(len(bands[0])):

            (lowcut, highcut) = bands[1][b]

            # Band-pass filtering
            filterSignals = filter.filter_data(raw_data, samplingFreq, lowcut, highcut)

            # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
            efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals")

            # Obtain Analytical signal
            efPhase = list()
            # efEnvelope = list()

            for i in range(len(efSignals)):
                analyticalSignal = scipy.signal.hilbert(efSignals[i])
                # Get instantaneous phase and amplitude envelope by channel
                efPhase.append(np.unwrap(np.angle(analyticalSignal)))
                # efEnvelope.append(np.abs(analyticalSignal))

            # CONNECTIVITY MEASURES
            ## PLV
            plv = PLV(efPhase)
            # fname = ctb_folder+model_id+"\\"+bands[0][b]+"plv.txt"
            # np.savetxt(fname, plv)

            local_results.append([stimulation_site, stimulus_type, stim_params, mode, emp_subj, g, s, w, r, bands[0][b]]
                                 + list(plv[np.triu_indices(len(rois), 1)][ids]) + list(post_fft_peaks) + [IAF, pre_prec_peak, pre_acc_peak])

        print("simulating stim_type = %s | stim_param = %0.2f - round = %i" % (stimulus_type, stim_params, r))
        print("LOOP ROUND REQUIRED %0.4f seconds.\n\n\n\n" % (time.time() - tic,))

    return np.asarray(local_results, dtype=object)

