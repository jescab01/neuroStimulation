
import time
import numpy as np
import scipy.signal
import scipy.stats
import pandas as pd

from tvb.simulator.lab import *
from mne import filter
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003, JansenRit1995
from mpi4py import MPI
import datetime


def ThCer_parallel(params_):
    result = list()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    print("Hello world from rank", str(rank), "of", str(size), '__', datetime.datetime.now().strftime("%Hh:%Mm:%Ss"))

    ## Folder structure - Local
    if "LCCN_Local" in os.getcwd():
        ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_dataOLD2\\"
        ctb_folderOLD = "E:\\LCCN_Local\PycharmProjects\CTB_dataOLD\\"
        import sys
        sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
        from toolbox.fft import PSD
        from toolbox.signals import epochingTool
        from toolbox.fc import PLV
        from toolbox.dynamics import dynamic_fc, kuramoto_order

    ## Folder structure - CLUSTER
    else:
        wd = "/home/t192/t192950/mpi/"
        ctb_folder = wd + "CTB_dataOLD2/"
        ctb_folderOLD = wd + "CTB_dataOLD/"

        import sys
        sys.path.append(wd)
        from toolbox.fft import PSD
        from toolbox.signals import epochingTool
        from toolbox.fc import PLV
        from toolbox.dynamics import dynamic_fc, kuramoto_order


    # Prepare simulation parameters
    simLength = 12 * 1000  # ms
    samplingFreq = 1000  # Hz
    transient = 2000  # ms

    for ii, set in enumerate(params_):

        tic = time.time()
        print("Rank %i out of %i  ::  %i/%i " % (rank, size, ii + 1, len(params_)))

        print(set)
        emp_subj, g, sigma, r = set

        # STRUCTURAL CONNECTIVITY      #########################################
        # Use "pass" for subcortical (thalamus) while "end" for cortex
        # based on [https://groups.google.com/g/dsi-studio/c/-naReaw7T9E/m/7a-Y1hxdCAAJ]

        conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2_pass.zip")
        conn.weights = conn.scaled_weights(mode="tract")

        # Define regions for the simulations: the Cingulum Bundle
        cingulum_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R', 'Insula_L', 'Insula_R',
                         'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                         'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L', 'ParaHippocampal_R',
                         'Amygdala_L', 'Amygdala_R', 'Parietal_Sup_L', 'Parietal_Sup_R',
                         'Parietal_Inf_L', 'Parietal_Inf_R', 'Precuneus_L', 'Precuneus_R',
                         'Thalamus_L', 'Thalamus_R']

        # load text with FC rois; check if match SC
        FClabs = list(np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/roi_labels_rms.txt", dtype=str))
        FC_cb_idx = [FClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cortical_rois

        SClabs = list(conn.region_labels)
        SC_cb_idx = [SClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cortical_rois
        conn.weights = conn.weights[:, SC_cb_idx][SC_cb_idx]
        conn.tract_lengths = conn.tract_lengths[:, SC_cb_idx][SC_cb_idx]
        conn.region_labels = conn.region_labels[SC_cb_idx]



        ###    NEURAL MASS MODEL  -  JANSEN-RIT         ###########################################
        m = JansenRit1995(He=np.array([3.25]), Hi=np.array([22]),
                          tau_e=np.array([10]), tau_i=np.array([20]),
                          c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                          c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                          p=np.array([0.22]), sigma=np.array([sigma]),
                          e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))


        # COUPLING FUNCTION   #########################################
        coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]), r=np.array([0.56]))
        conn.speed = np.array([15])


        # OTHER PARAMETERS   ###
        # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
        # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
        integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

        mon = (monitors.Raw(),)

        print("Simulating %s (%is)  ||  PARAMS: g%i sigma%0.2f" % (emp_subj, simLength / 1000, g, sigma))

        # Run simulation
        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
        sim.configure()
        output = sim.run(simulation_length=simLength)

        # Extract data: "output[a][b][:,0,:,0].T" where:
        # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.

        raw_data = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T
        raw_time = output[0][0][transient:]
        regionLabels = conn.region_labels


        # Saving in FFT results: coupling value, conduction speed, mean signal freq peak (Hz; module), all signals info.
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

        bands = [["3-alpha"], [(8, 12)]]
        # bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"], [(2, 4), (5, 7), (8, 12), (15, 29), (30, 59)]]

        for b in range(len(bands[0])):
            (lowcut, highcut) = bands[1][b]

            # Band-pass filtering
            filterSignals = filter.filter_data(raw_data, samplingFreq, lowcut, highcut)

            # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
            efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals")

            # Obtain Analytical signal
            efPhase = list()
            efEnvelope = list()
            for i in range(len(efSignals)):
                analyticalSignal = scipy.signal.hilbert(efSignals[i])
                # Get instantaneous phase and amplitude envelope by channel
                efPhase.append(np.angle(analyticalSignal))
                efEnvelope.append(np.abs(analyticalSignal))

            # Check point
            # from toolbox import timeseriesPlot, plotConversions
            # regionLabels = conn.region_labels
            # timeseriesPlot(raw_data, raw_time, regionLabels)
            # plotConversions(raw_data[:,:len(efSignals[0][0])], efSignals[0], efPhase[0], efEnvelope[0],bands[0][b], regionLabels, 8, raw_time)

            # CONNECTIVITY MEASURES
            ## PLV
            plv = PLV(efPhase)

            # Load empirical data to make simple comparisons
            plv_emp = \
                np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/" + bands[0][b] + "_plv_rms.txt", delimiter=',')[:,
                FC_cb_idx][
                    FC_cb_idx]

            # Comparisons
            t1 = np.zeros(shape=(2, len(plv) ** 2 // 2 - len(plv) // 2))
            t1[0, :] = plv[np.triu_indices(len(plv), 1)]
            t1[1, :] = plv_emp[np.triu_indices(len(plv), 1)]
            plv_r = np.corrcoef(t1)[0, 1]


            ## Gather results
            result.append(
                (emp_subj, g, "sigma"+str(sigma), r,
                 np.average(fpeak), np.average(amp_fpeak05),
                 plv_r, np.average(plv[np.triu_indices(len(plv), 1)]), np.std(plv[np.triu_indices(len(plv), 1)])))

        print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))

    return np.asarray(result, dtype=object)
