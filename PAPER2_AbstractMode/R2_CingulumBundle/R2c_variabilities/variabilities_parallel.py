
import time
import numpy as np
import pandas as pd
from mne import filter
import scipy.signal

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003, JansenRit1995
from mpi4py import MPI
import datetime


def variabilities_parallel(params, baseline_subj=None):

    datapoints = list()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    print("Hello world from rank", str(rank), "of", str(size), '__', datetime.datetime.now().strftime("%Hh:%Mm:%Ss"))

    ## Folder structure - Local
    if "LCCN_Local" in os.getcwd():
        ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_dataOLD2\\"
        import sys
        sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
        from toolbox.fft import PSD
        from toolbox.fc import PLV
        from toolbox.signals import epochingTool
    ## Folder structure - CLUSTER
    else:
        wd = "/home/t192/t192950/mpi/"
        ctb_folder = wd + "CTB_dataOLD2/"
        ctb_folderOLD = wd + "CTB_dataOLD/"

        import sys
        sys.path.append(wd)
        from toolbox.fft import PSD
        from toolbox.fc import PLV
        from toolbox.signals import epochingTool

    # Prepare simulation parameters
    simLength = 25 * 1000  # ms
    samplingFreq = 1000  # Hz
    transient = 4000  # ms

    for set_ in params:
        tic = time.time()
        print(set_)

        emp_subj, mode, g, sigma, r, w = set_

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
            FC_cb_idx = [FClabs.index(roi) for roi in
                         cingulum_rois]  # find indexes in FClabs that matches cortical_rois
            SC_cb_idx = [SClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cortical_rois
            conn.weights = conn.weights[:, SC_cb_idx][SC_cb_idx]
            conn.tract_lengths = conn.tract_lengths[:, SC_cb_idx][SC_cb_idx]
            conn.region_labels = conn.region_labels[SC_cb_idx]

        # NEURAL MASS MODEL  -  JANSEN-RIT  #########################################################
        m = JansenRit1995(He=np.array([3.25]), Hi=np.array([22]),
                          tau_e=np.array([10]), tau_i=np.array([20]),
                          c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                          c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                          p=np.array([0.22]), sigma=np.array([sigma]),
                          e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

        # COUPLING FUNCTION   #########################################
        coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                               r=np.array([0.56]))
        conn.speed = np.array([15])

        # STIMULUS ###############################
        if w != 0:

            # Reconstruct DataFrame
            baseline_subj = pd.DataFrame(baseline_subj, columns=["subject", "fpeak", "amp_fpeak", "amp_fbase"])
            initialPeak = float(baseline_subj.loc[baseline_subj["subject"] == emp_subj].fpeak.values)

            ## Sinusoid input
            eqn_t = equations.Sinusoid()
            eqn_t.parameters['amp'] = 1  # Amplitud diferencial por áreas ajustada en stimWeights
            eqn_t.parameters['frequency'] = initialPeak  # Hz
            eqn_t.parameters['onset'] = 2000  # ms
            eqn_t.parameters['offset'] = simLength  # ms

            # electric field * orthogonal to surface
            weighting = np.loadtxt(
                ctb_folder + 'CurrentPropagationModels/' + emp_subj + '-efnorm_mag-roast_OzCzModel-AAL2.txt',
                delimiter=",") * w

            if "cb" in mode:
                weighting = weighting[SC_cb_idx]

            stimulus = patterns.StimuliRegion(
                temporal=eqn_t,
                connectivity=conn,
                weight=weighting)

            # Configure space and time
            stimulus.configure_space()
            stimulus.configure_time(np.arange(0, simLength, 1))
            # And take a look
            # plot_pattern(stimulus)

        # OTHER PARAMETERS   ###
        # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
        # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
        integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

        mon = (monitors.Raw(),)

        print("Simulating for Coupling factor = %i and speed = %i" % (g, sigma))

        # Run simulation
        if w != 0:
            sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon,
                                      stimulus=stimulus)
        else:
            sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
        sim.configure()
        output = sim.run(simulation_length=simLength)

        # Extract data: "output[a][b][:,0,:,0].T" where:
        # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
        raw_data = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T
        regionLabels = conn.region_labels

        # ROIs of interest to measure alpha peak increase #i
        # All occipito-parietal regins w/ 0-indexing in Python.
        # occipital_rois = ['Calcarine_L', 'Calcarine_R', 'Cuneus_L', 'Cuneus_R',
        #                   'Lingual_L', 'Lingual_R', 'Occipital_Sup_L', 'Occipital_Sup_R',
        #                   'Occipital_Mid_L', 'Occipital_Mid_R', 'Occipital_Inf_L',
        #                   'Occipital_Inf_R', 'Parietal_Sup_L', 'Parietal_Sup_R',
        #                   'Parietal_Inf_L', 'Parietal_Inf_R', 'SupraMarginal_L',
        #                   'SupraMarginal_R', 'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R']
        #
        # if "cb" in mode:
        #     occ_cb = list(set(occipital_rois).intersection(set(cingulum_rois)))
        #     rois = [list(conn.region_labels).index(roi) for roi in occ_cb]
        # else:
        #     rois = [SClabs.index(roi) for roi in occipital_rois]

        ## ROIs implicated in the empirical effect @ 26/04/2022
        empCluster_rois = ['Precentral_L', 'Frontal_Sup_2_L', 'Frontal_Sup_2_R', 'Frontal_Mid_2_L',
                           'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L', 'Frontal_Inf_Tri_R',
                           'Frontal_Inf_Orb_2_L', 'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Frontal_Sup_Medial_L',
                           'Frontal_Sup_Medial_R', 'Rectus_L', 'OFCmed_L', 'Insula_L', 'Insula_R', 'Cingulate_Ant_L', 'Cingulate_Ant_R',
                           'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L', 'ParaHippocampal_R',
                           'Amygdala_L', 'Calcarine_L', 'Calcarine_R', 'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R',
                           'Occipital_Sup_R', 'Occipital_Mid_L', 'Occipital_Mid_R', 'Occipital_Inf_L',
                           'Occipital_Inf_R', 'Fusiform_L', 'Fusiform_R', 'Postcentral_L', 'Parietal_Sup_R',
                           'Parietal_Inf_R', 'Angular_R', 'Precuneus_R', 'Temporal_Sup_L',
                           'Temporal_Sup_R', 'Temporal_Pole_Sup_L', 'Temporal_Pole_Sup_R', 'Temporal_Mid_L',
                           'Temporal_Mid_R', 'Temporal_Pole_Mid_L', 'Temporal_Inf_L', 'Temporal_Inf_R']

        if "cb" in mode:
            occ_cb = list(set(empCluster_rois).intersection(set(cingulum_rois)))
            rois = [list(conn.region_labels).index(roi) for roi in occ_cb]
        else:
            rois = [SClabs.index(roi) for roi in empCluster_rois]

        ## Calculate IAF band power
        spectra, freqs = PSD(raw_data, samplingFreq)


        ##   CLUSTER   ##
        # Pre-Filtering in freq band
        lowcut, highcut = 1, 60
        spectra_filt = spectra[:, (freqs > lowcut) & (freqs < highcut)][rois]
        freqs_filt = freqs[(freqs > lowcut) & (freqs < highcut)]

        # Peak
        fpeak_clus = freqs_filt[np.argmax(spectra_filt, axis=1)]

        # Spectral amplitude at peak
        amp_fpeak05_clus = np.array(
            [sum(spectra_filt[i, (freqs_filt > fp - 0.5) & (freqs_filt < fp + 0.5)])
             for i, fp in enumerate(fpeak_clus)])

        if w != 0:
            # Spectral amplitude at baseline peak
            amp_fbaseline05_clus = np.array(
                [sum(spectra_filt[i, (freqs_filt > initialPeak - 0.5) & (freqs_filt < initialPeak + 0.5)])
                 for i, fp in enumerate(fpeak_clus)])
        else:
            amp_fbaseline05_clus = np.nan

        ##    ALL ROIS   ##
        # Pre-Filtering in freq band
        lowcut, highcut = 1, 60
        spectra_filt = spectra[:, (freqs > lowcut) & (freqs < highcut)]
        freqs_filt = freqs[(freqs > lowcut) & (freqs < highcut)]

        # Peak
        fpeak = freqs_filt[np.argmax(spectra_filt, axis=1)]

        # Spectral amplitude at peak
        amp_fpeak05 = np.array(
            [sum(spectra_filt[i, (freqs_filt > fp - 0.5) & (freqs_filt < fp + 0.5)])
             for i, fp in enumerate(fpeak)])

        if w != 0:
            # Spectral amplitude at baseline peak
            amp_fbaseline05 = np.array(
                [sum(spectra_filt[i, (freqs_filt > initialPeak - 0.5) & (freqs_filt < initialPeak + 0.5)])
                 for i, fp in enumerate(fpeak)])
        else:
            amp_fbaseline05 = np.nan



        ## Calculate PLV in alpha
        bands = [["3-alfa"], [(8, 12)]]

        for b in range(len(bands[0])):
            (lowcut, highcut) = bands[1][b]
            # Band-pass filtering
            filterSignals = filter.filter_data(raw_data, samplingFreq, lowcut, highcut)

            # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
            efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals")

            # Obtain Analytical signal
            efPhase = list()
            for i in range(len(efSignals)):
                analyticalSignal = scipy.signal.hilbert(efSignals[i])
                # Get instantaneous phase and amplitude envelope by channel
                efPhase.append(np.unwrap(np.angle(analyticalSignal)))
                # efEnvelope.append(np.abs(analyticalSignal))

            # CONNECTIVITY MEASURES: PLV
            plv = PLV(efPhase)

            plv_clus = plv[:, rois][rois][np.triu_indices(len(rois), 1)]

        ## Gather results
        datapoints.append((emp_subj, "sigma"+str(sigma), g, r, w,
                           np.average(fpeak_clus), np.average(amp_fpeak05_clus),  np.average(amp_fbaseline05_clus), np.average(plv_clus),
                           fpeak, amp_fpeak05, amp_fbaseline05, plv))

        print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))

    return np.asarray(datapoints, dtype=object)
