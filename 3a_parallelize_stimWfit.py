import os
import time

import numpy as np
import pandas as pd

from tvb.simulator.lab import *
from toolbox import FFTpeaks, FFTarray
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

# Set up parallelization
try:
    num_cores = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
except KeyError:
    num_cores = multiprocessing.cpu_count()


weights = np.linspace(0.2, 1, 20)  ## Loop over weights to get a 14% raise in alpha peak


def simulate_parallel(inputs_):

    w = inputs_
    stage = 1

    # Choose a name for your simulation and define the empirical for SC
    model_id = ".1995JansenRit"
    emp_subj = "NEMOS_035"
    (g, s) = (65, 11.5)  # Working point

    n_rep = 50

    ## Folder structure - CLUSTER
    wd = "/home/t192/t192950/pwfit/"
    ctb_folder = wd + "CTB_data/output/"

    main_folder = "PSE"
    if os.path.isdir(main_folder) == False:
        os.mkdir(main_folder)

    os.chdir(main_folder)

    specific_folder = "PSEp_fittingW" + model_id + "-" + emp_subj + "-stage" + str(stage) + "-" + time.strftime("m%md%dy%Y")
    if os.path.isdir(specific_folder) == False:
        os.mkdir(specific_folder)

    os.chdir(specific_folder)

    # ## Folder structure - Local
    # wd = os.getcwd()
    # main_folder = wd + "\\" + "PSE"
    # if os.path.isdir(main_folder) == False:
    #     os.mkdir(main_folder)
    # specific_folder = main_folder + "\\PSE_PARALLELfittingW" + model_id + "-" + time.strftime("m%md%dy%Y")
    # if os.path.isdir(specific_folder) == False:
    #     os.mkdir(specific_folder)
    #
    # ctb_folder = "D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\\CTB_data\\output\\"

    # Prepare simulation parameters
    simLength = 10 * 1000  # ms
    samplingFreq = 1000  # Hz
    transient = 1000  # ms

    # Parameters from Stefanovski 2019. Good working point at g=33, s=15.5 on AAL2red connectome.
    m = models.JansenRit(A=np.array([3.25]), B=np.array([22]), J=np.array([1]),
                         a=np.array([0.1]), a_1=np.array([135]), a_2=np.array([108]),
                         a_3=np.array([33.75]), a_4=np.array([33.75]), b=np.array([0.06]),
                         mu=np.array([0.1085]), nu_max=np.array([0.0025]), p_max=np.array([0]), p_min=np.array([0]),
                         r=np.array([0.56]), v0=np.array([6]))

    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)
    mon = (monitors.Raw(),)

    conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2red.zip")
    conn.weights = conn.scaled_weights(mode="tract")

    coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                       r=np.array([0.56]))
    conn.speed = np.array([s])

    # Run simulation
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
    sim.configure()
    output = sim.run(simulation_length=simLength)

    # Extract data: "output[a][b][:,0,:,0].T" where:
    # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
    raw_data = output[0][1][transient:, 0, :, 0].T
    regionLabels = conn.region_labels

    # ROIs of interest to measure alpha peak increase #i
    rois = [62, 63, 64, 65, 70, 71]  # Parietal complex (sup [63,64] & inf [65,66] parietal) + precuneus [71,72]. 0-indexing in Python.
    fft_peaks_hzAAL = FFTpeaks(raw_data, simLength - transient)[0][rois]
    fft_peaks_modulesAAL = FFTpeaks(raw_data, simLength - transient)[1][rois]

    fft_data = np.empty((1, 6), dtype=object) # columns=[param1=w, param2=rep, regLabs, fft_tot, freq_tot, param3=initPeak]
    fft_data = np.concatenate((fft_data, FFTarray(raw_data[rois], simLength, transient, regionLabels[rois], w, 0, fft_peaks_hzAAL)))[1:]

    resultsAAL = pd.DataFrame.from_dict(
        {"w": [0], "peak_hz": [np.average(fft_peaks_hzAAL)], "peak_module": [np.average(fft_peaks_modulesAAL)]})

    initialPeak = np.average(fft_peaks_hzAAL)


    ##### STIMULUS

    for r in range(n_rep):

        tic0 = time.time()
        ## Sinusoid input
        eqn_t = equations.Sinusoid()
        eqn_t.parameters['amp'] = 1  # Amplitud diferencial por áreas ajustada en stimWeights
        eqn_t.parameters['frequency'] = initialPeak  # Hz
        eqn_t.parameters['onset'] = 0  # ms
        eqn_t.parameters['offset'] = 10000  # ms

        # electric field * orthogonal to surface
        weighting = np.loadtxt(ctb_folder +'orthogonals/' + emp_subj + '-roast_OzCzModel_efnorm_mag-AAL2red.txt') * w

        stimulus = patterns.StimuliRegion(
            temporal=eqn_t,
            connectivity=conn,
            weight=weighting)

        # Configure space and time
        stimulus.configure_space()
        stimulus.configure_time(np.arange(0, simLength, 1))
        # And take a look
        # plot_pattern(stimulus)

        # Run simulation
        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon,
                                  stimulus=stimulus)
        sim.configure()
        output = sim.run(simulation_length=simLength)
        # Extract data cutting initial transient
        raw_data = output[0][1][transient:, 0, :, 0].T
        raw_time = output[0][0][transient:]
        regionLabels = conn.region_labels

        # ROIs of interest to measure alpha peak increase
        fft_peaks_hzAAL = FFTpeaks(raw_data, simLength - transient)[0][rois]
        fft_peaks_modulesAAL = FFTpeaks(raw_data, simLength - transient)[1][rois]

        fft_data = np.concatenate((fft_data, FFTarray(raw_data[rois], simLength, transient, regionLabels[rois], w, r, fft_peaks_hzAAL)))

        resultsAAL = resultsAAL.append(
            {"w": w, "peak_hz": np.average(fft_peaks_hzAAL), "peak_module": np.average(fft_peaks_modulesAAL)},
            ignore_index=True)

        # resultsMEG=resultsMEG.append({"w":w, "peak_hz":np.average(fft_peaks_hz), "peak_module":np.average(fft_peaks_modules)}, ignore_index=True)
        print("w = %0.2f - round = %i" % (w, r))
        print("LOOP ROUND REQUIRED %0.4f seconds.\n\n\n\n" % (time.time() - tic0,))

    #### Save results

    resultsAAL.to_csv('w_' + str(round(w, 3)) + "_AAL2red-ParietalComplex_alphaRise.csv")

    fft_df = pd.DataFrame(fft_data, columns=["w", "rep", "regLab", "fft_module", "freq", "initPeak"])
    fft_df.to_csv('w_' + str(round(w, 3)) + "_FFTs_AAL2red-ParietalComplex.csv")


if __name__ == "__main__":

    processed_list = Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(simulate_parallel)(i) for i in weights)

