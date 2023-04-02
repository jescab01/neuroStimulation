
import time
import numpy as np
import pandas as pd
import scipy
from mne import filter

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRit1995
from mpi4py import MPI
import datetime

def nmm_parallel(params):

    datapoints = list()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    print("Hello world from rank", str(rank), "of", str(size), '__', datetime.datetime.now().strftime("%Hh:%Mm:%Ss"))

    ## Folder structure - Local
    if "LCCN_Local" in os.getcwd():
        ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data3\\"
        import sys
        sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
        from toolbox.fft import FFTpeaks, PSD
        from toolbox.signals import epochingTool
        from toolbox.fc import PLV

    ## Folder structure - CLUSTER
    else:
        wd = "/home/t192/t192950/mpi/"
        ctb_folder = wd + "CTB_data3/"
        ctb_folderOLD = wd + "CTB_dataOLD/"

        import sys
        sys.path.append(wd)
        from toolbox.fft import FFTpeaks, PSD
        from toolbox.signals import epochingTool
        from toolbox.fc import PLV



    # Prepare simulation parameters
    simLength = 50 * 1000  # ms
    samplingFreq = 1000  # Hz
    transient = 4000  # ms

    for ii, set in enumerate(params):

        tic = time.time()
        print("Rank %i - simulation %i / %i" % (rank, ii+1, len(params)))
        print(set)

        mode, emp_subj, g, sigma, r, w, f = set

        # STRUCTURAL CONNECTIVITY      #########################################

        conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2_pass.zip")
        conn.weights = conn.scaled_weights(mode="tract")

        # Define regions implicated in Functional analysis: remove  Cerebelum, Thalamus, Caudate (i.e. subcorticals)
        cingulum_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                         'Insula_L', 'Insula_R',
                         'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                         'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                         'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R',
                         'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                         'Parietal_Inf_R', 'Precuneus_L', 'Precuneus_R',
                         'Thalamus_L', 'Thalamus_R']

        isolated_rois = ['Precuneus_L', 'Precuneus_R']

        # load text with FC rois; check if match SC
        FClabs = list(np.loadtxt(ctb_folder + "FCavg_" + emp_subj + "/roi_labels.txt", dtype=str))
        SClabs = list(conn.region_labels)


        if "cb" in mode:
            FC_cb_idx = [FClabs.index(roi) for roi in
                         cingulum_rois]  # find indexes in FClabs that matches cortical_rois
            SC_cb_idx = [SClabs.index(roi) for roi in
                         cingulum_rois]  # find indexes in FClabs that matches cortical_rois
            conn.weights = conn.weights[:, SC_cb_idx][SC_cb_idx]
            conn.tract_lengths = conn.tract_lengths[:, SC_cb_idx][SC_cb_idx]
            conn.region_labels = conn.region_labels[SC_cb_idx]

        elif "Node" in mode:
            FC_cb_idx = [FClabs.index(roi) for roi in
                         isolated_rois]  # find indexes in FClabs that matches cortical_rois
            SC_cb_idx = [SClabs.index(roi) for roi in
                         isolated_rois]  # find indexes in FClabs that matches cortical_rois
            conn.weights = conn.weights[:, SC_cb_idx][SC_cb_idx]
            conn.tract_lengths = conn.tract_lengths[:, SC_cb_idx][SC_cb_idx]
            conn.region_labels = conn.region_labels[SC_cb_idx]


        # COUPLING FUNCTION   #########################################

        coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                           r=np.array([0.56]))
        conn.speed = np.array([15.5])

        # NEURAL MASS MODEL    #########################################################
        m = JansenRit1995(He=np.array([3.25]), Hi=np.array([22]),
                          tau_e=np.array([10]), tau_i=np.array([20]),
                          c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                          c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                          p=np.array([0.22]), sigma=np.array([sigma]),
                          e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

        # STIMULUS ###############################
        ## Sinusoid input
        eqn_t = equations.Sinusoid()
        eqn_t.parameters['amp'] = w
        eqn_t.parameters['frequency'] = f  # Hz
        eqn_t.parameters['onset'] = 2000  # ms
        eqn_t.parameters['offset'] = simLength  # ms

        # Amplitud diferencial por áreas ajustada with weighting
        if "cb" in mode:
            weighting = np.array([1 if "Precuneus_L" == roi else -1 if ("Precuneus_R"==roi) & ("antiphase" in mode) else 0 for roi in conn.region_labels])

        elif "Node" in mode:
            weighting = np.array([1, 0])  # np.ones(len(conn.weights)) * w


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

        print("Simulating for Coupling factor = %i and sigma = %0.2f" % (g, sigma))

        # Run simulation
        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon,
                                  stimulus=stimulus)

        sim.configure()
        output = sim.run(simulation_length=simLength)

        # Extract data: "output[a][b][:,0,:,0].T" where:
        # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.

        raw_data = output[0][1][transient:, 0, :, 0].T
        regionLabels = conn.region_labels


        ## Calculate IAF band power rise
        spectra, freqs = PSD(raw_data, samplingFreq)

        # Pre-Filtering in freq band
        lowcut, highcut = 1, 60
        spectra_filt = spectra[:, (freqs > lowcut) & (freqs < highcut)]
        freqs_filt = freqs[(freqs > lowcut) & (freqs < highcut)]

        # Peak
        fpeak = freqs_filt[np.argmax(spectra_filt, axis=1)]

        # Spectral amplitude at peak
        # amp_fpeak = np.max(spectra_filt, axis=1)
        amp_fpeak05 = np.array([sum(spectra_filt[i, (freqs_filt > fp - 0.5) & (freqs_filt < fp + 0.5)]) for i, fp in enumerate(fpeak)])

        # Spectral amplitude at fex
        amp_fex = spectra_filt[:, np.argmin(abs(freqs_filt - f))]
        amp_fex05 = np.sum(spectra_filt[:, (freqs_filt > f - 0.5) & (freqs_filt < f + 0.5)], axis=1)

        # Calculate PLV
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


        ## Gather results
        for ii, roi in enumerate(conn.region_labels):

            roi = "stim_Precuneus_L" if roi == "Precuneus_L" else \
                "stimAnti_Precuneus_R" if (roi == "Precuneus_R") & ("antiphase" in mode) else roi

            datapoints.append((mode + "_sigma"+str(sigma), r, roi, w, f, fpeak[ii], amp_fex05[ii], amp_fpeak05[ii], plv[ii, :]))

        print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))

    return np.asarray(datapoints, dtype=object)

