
import time
import numpy as np
import pandas as pd
import scipy
from mne import filter

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRit1995
from mpi4py import MPI
import datetime


def mlr_parallel(params, baseline_subj=None):

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
        from toolbox.fft import FFTpeaks, PSD
        from toolbox.signals import epochingTool
        from toolbox.fc import PLV

    ## Folder structure - CLUSTER
    else:
        wd = "/home/t192/t192950/mpi/"
        ctb_folder = wd + "CTB_dataOLD2/"
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

        emp_subj, mode,  g, sigma, r, w = set

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
        FClabs = list(np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/roi_labels_rms.txt", dtype=str))
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
        if w != 0:

            # Reconstruct DataFrame
            baseline_subj = pd.DataFrame(baseline_subj, columns=["subject", "fpeak", "amp_fpeak", "amp_fbase"])
            initialPeak = float(baseline_subj.loc[baseline_subj["subject"] == emp_subj].fpeak.values)

            ## Sinusoid input
            eqn_t = equations.Sinusoid()
            eqn_t.parameters['amp'] = 1  # Amplitud diferencial por áreas ajustada en stimWeights
            eqn_t.parameters['frequency'] = initialPeak  # Hz
            eqn_t.parameters['onset'] = 16000  # ms
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

        else:
            initialPeak = np.nan

        # OTHER PARAMETERS   ###
        # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
        # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
        integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

        mon = (monitors.Raw(),)

        print("Simulating for Coupling factor = %i and sigma = %0.2f" % (g, sigma))

        # Run simulation
        if w != 0:
            sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon,
                                      stimulus=stimulus)
        else:
            sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)

        sim.configure()
        output = sim.run(simulation_length=simLength)



        ####  BASELINE (pre-stimulation) RESULTS

        # Extract data: "output[a][b][:,0,:,0].T" where:
        # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
        baseline_data = output[0][1][transient:16000, 0, :, 0].T
        regionLabels = conn.region_labels

        ## Calculate IAF band power rise
        spectra, freqs = PSD(baseline_data, samplingFreq)

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
        # amp_fex = spectra_filt[:, np.argmin(abs(freqs_filt - f))]
        # amp_fex05 = np.sum(spectra_filt[:, (freqs_filt > f - 0.5) & (freqs_filt < f + 0.5)], axis=1)

        if w != 0:
            # Spectral amplitude at baseline peak
            amp_fbase05 = np.array(
                [sum(spectra_filt[i, (freqs_filt > initialPeak - 0.5) & (freqs_filt < initialPeak + 0.5)])
                 for i, fp in enumerate(fpeak)])

        else:
            amp_fbase05 = [np.nan]*len(conn.region_labels)

        # Calculate PLV
        bands = [["3-alfa"], [(8, 12)]]
        ## [["1-delta", "2-theta", "3-alfa", "4-beta", "5-gamma"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]

        for b in range(len(bands[0])):

            (lowcut, highcut) = bands[1][b]

            # Band-pass filtering
            filterSignals = filter.filter_data(baseline_data, samplingFreq, lowcut, highcut)

            # Obtain Analytical signal
            efPhase = list()
            # efEnvelope = list()

            for i in range(len(filterSignals)):
                analyticalSignal = scipy.signal.hilbert(filterSignals[i])
                # Get instantaneous phase and amplitude envelope by channel
                efPhase.append(np.unwrap(np.angle(analyticalSignal)))
                # efEnvelope.append(np.abs(analyticalSignal))

        plv = np.ndarray((len(efPhase), len(efPhase)))
        for channel1 in range(len(efPhase)):
            for channel2 in range(len(efPhase)):
                phaseDifference = efPhase[channel1] - efPhase[channel2]
                value = abs(np.average(np.exp(1j * phaseDifference)))
                plv[channel1, channel2] = value

        ## Gather results
        for ii, roi in enumerate(conn.region_labels):

            datapoints.append((mode + "_sigma"+str(sigma), emp_subj, w, initialPeak, r, roi, "baseline",
                               fpeak[ii], amp_fpeak05[ii], amp_fbase05[ii], np.average(plv[ii, plv[ii, :] != 1]), plv[ii, :]))



        ### STIMULATION RESULTS

        # Extract data: "output[a][b][:,0,:,0].T" where:
        # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
        stimulation_data = output[0][1][16000:, 0, :, 0].T

        ## Calculate IAF band power rise
        spectra, freqs = PSD(stimulation_data, samplingFreq)

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
        # amp_fex = spectra_filt[:, np.argmin(abs(freqs_filt - f))]
        # amp_fex05 = np.sum(spectra_filt[:, (freqs_filt > f - 0.5) & (freqs_filt < f + 0.5)], axis=1)

        if w != 0:
            # Spectral amplitude at baseline peak
            amp_fbase05 = np.array(
                [sum(spectra_filt[i, (freqs_filt > initialPeak - 0.5) & (freqs_filt < initialPeak + 0.5)])
                 for i, fp in enumerate(fpeak)])

        else:
            amp_fbase05 = [np.nan]*len(conn.region_labels)

        # Calculate PLV
        bands = [["3-alfa"], [(8, 12)]]
        ## [["1-delta", "2-theta", "3-alfa", "4-beta", "5-gamma"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]

        for b in range(len(bands[0])):

            (lowcut, highcut) = bands[1][b]

            # Band-pass filtering
            filterSignals = filter.filter_data(stimulation_data, samplingFreq, lowcut, highcut)

            # Obtain Analytical signal
            efPhase = list()
            # efEnvelope = list()

            for i in range(len(filterSignals)):
                analyticalSignal = scipy.signal.hilbert(filterSignals[i])
                # Get instantaneous phase and amplitude envelope by channel
                efPhase.append(np.unwrap(np.angle(analyticalSignal)))
                # efEnvelope.append(np.abs(analyticalSignal))

        plv = np.ndarray((len(efPhase), len(efPhase)))
        for channel1 in range(len(efPhase)):
            for channel2 in range(len(efPhase)):
                phaseDifference = efPhase[channel1] - efPhase[channel2]
                value = abs(np.average(np.exp(1j * phaseDifference)))
                plv[channel1, channel2] = value

        ## Gather results
        for ii, roi in enumerate(conn.region_labels):

            datapoints.append((mode + "_sigma"+str(sigma), emp_subj, w, initialPeak, r, roi, "stimulation",
                               fpeak[ii], amp_fpeak05[ii], amp_fbase05[ii], np.average(plv[ii, plv[ii, :] != 1]), plv[ii, :]))

        print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))

    return np.asarray(datapoints, dtype=object)

