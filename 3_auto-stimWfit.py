import os
import random
import time
import shutil

import numpy as np
import pandas as pd

from tvb.simulator.lab import *
from toolbox import FFTpeaks, FFTarray
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np
import pandas as pd
import os
import plotly.io as pio
import plotly.express as px

# from mountParallel_stimWfit import collectData, boxPlot, lines3dFFT

# Set up parallelization
# try:
#     num_cores = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
# except KeyError:
#     num_cores = multiprocessing.cpu_count() - 1

# working_points = [('NEMOS_035', 65, 11.5),
#                   ('NEMOS_049', 19, 18.5),
#                   ('NEMOS_050', 38, 22.5),
#                   ('NEMOS_058', 37, 16.5),
#                   ('NEMOS_059', 35, 12.5),
#                   ('NEMOS_064', 54, 6.5),
#                   ('NEMOS_065', 47, 22.5),
#                   ('NEMOS_071', 81, 6.5),
#                   ('NEMOS_075', 14, 5.5),
#                   ('NEMOS_077', 20, 20.5)]
#
working_points = [('NEMOS_035', 37, 22.5),
                  ('NEMOS_049', 37, 22.5),
                  ('NEMOS_050', 37, 22.5),
                  ('NEMOS_058', 37, 22.5),
                  ('NEMOS_059', 37, 22.5),
                  ('NEMOS_064', 37, 22.5),
                  ('NEMOS_065', 37, 22.5),
                  ('NEMOS_071', 37, 22.5),
                  ('NEMOS_075', 37, 22.5),
                  ('NEMOS_077', 37, 22.5)]

w_space = np.logspace(-4, 1, 1000)  ## Loop over weights to get a 14% raise in alpha peak

model_id = ".1995JansenRit"

## Folder structure - Local
wd = os.getcwd()
ctb_folder = "D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\\CTB_data\\output\\"

main_folder = wd + "\\" + "PSE"
if os.path.isdir(main_folder) == False:
    os.mkdir(main_folder)
specific_folder = main_folder + "\\PSE_autoWfit-NEMOS" + model_id + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")

if os.path.isfile(specific_folder + "\\" + "FFT_fullDF.csv"):
    shutil.rmtree(specific_folder)
if os.path.isdir(specific_folder) == False:
    os.mkdir(specific_folder)

#def simulate_parallel(inputs_):
# w = inputs_

for wp in working_points:

    resultsAAL = pd.DataFrame()

    emp_subj, g, s = wp

    # Prepare simulation parameters
    simLength = 10 * 1000  # ms
    samplingFreq = 1000  # Hz
    transient = 1000  # ms

    n_rep = 10

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

    #### First Block (1m 15sec):
    # Calculate initial peak from n reps
    tic0 = time.time()
    for r in range(n_rep):
        tic1 = time.time()

        # Run simulation
        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
        sim.configure()
        output = sim.run(simulation_length=simLength)

        # Extract data: "output[a][b][:,0,:,0].T" where:
        # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
        raw_data = output[0][1][transient:, 0, :, 0].T
        regionLabels = conn.region_labels

        # ROIs of interest to measure alpha peak increase #i
        # All occipito-parietal regins w/ 0-indexing in Python.
        # Calcarine fissure [47,48]; Cuneus [49,50]; Lingual gyrus [51,52]; Occipital cortex [53:58];
        # Parietal sup [63,64] & inf [65,66]; Supramarginal [67,68]; Angular gyryus [69,70]; Precuneus [71,72];
        rois = [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,  62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
        fftpeaks = FFTpeaks(raw_data, simLength - transient)
        fft_peaks_hzAAL = fftpeaks[0][rois]
        fft_peaks_modulesAAL = fftpeaks[1][rois]
        fft_band_modulesAAL = fftpeaks[2][rois]

        # fft_data = np.empty((1, 6),
        #                     dtype=object)  # columns=[param1=w, param2=rep, regLabs, fft_tot, freq_tot, param3=initPeak]
        # fft_data = np.concatenate(
        #     (fft_data, FFTarray(raw_data[rois], simLength, transient, regionLabels[rois], 0, 0, fft_peaks_hzAAL)))[1:]

        resultsAAL = resultsAAL.append({"w": 0, "peak_hz": np.average(fft_peaks_hzAAL),
                                        "peak_module": np.average(fft_peaks_modulesAAL),
                                        "band_module": np.average(fft_band_modulesAAL)}, ignore_index=True)

        print("LOOP ROUND REQUIRED %0.4f seconds.\n\n\n\n" % (time.time() - tic1,))
    print("WHOLE BLOCK (baseline alpha peak) REQUIRED %0.4f seconds.\n\n\n\n" % (time.time() - tic0,))


    initialPeak = np.average(resultsAAL["peak_hz"])
    baseline_modulePeak = np.average(resultsAAL["band_module"])

    ### Second Block
    ## Auto-set top w value

    secondBlock = [["top-up W", 50],
                   ["optimum-range W ", 14]]

    for j, sB in enumerate(secondBlock):
        name, target_rise = sB[0], sB[1]

        # Reset variables: peakRise, repetition, IDw_space, rise_target-rise, sim number
        rise, r, i, cost, n = 0, 0, 0, 0, 0
        tic = time.time()
        while rise < target_rise-5 or rise > target_rise+5 or r < 3:
            tic0 = time.time()
            n = n + 1
            ## Sinusoid input
            eqn_t = equations.Sinusoid()
            eqn_t.parameters['amp'] = 1  # Amplitud diferencial por áreas ajustada en stimWeights
            eqn_t.parameters['frequency'] = initialPeak  # Hz
            eqn_t.parameters['onset'] = 0  # ms
            eqn_t.parameters['offset'] = simLength  # ms

            # Define current w
            i = int(np.trunc(i + cost))
            w = w_space[i]
            # electric field * orthogonal to surface
            weighting = np.loadtxt(ctb_folder + 'orthogonals/' + emp_subj + '-roast_OzCzModel_ef_mag-AAL2red.txt') * w

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
            fftpeaks = FFTpeaks(raw_data, simLength - transient)
            fft_peaks_hzAAL = fftpeaks[0][rois]
            fft_peaks_modulesAAL = fftpeaks[1][rois]
            fft_band_modulesAAL = fftpeaks[2][rois]
            # fft_data = np.concatenate(
            #     (fft_data, FFTarray(raw_data[rois], simLength, transient, regionLabels[rois], w, r, fft_peaks_hzAAL)))

            resultsAAL = resultsAAL.append({"w": w, "peak_hz": np.average(fft_peaks_hzAAL),
                 "peak_module": np.average(fft_peaks_modulesAAL),
                 "band_module": np.average(fft_band_modulesAAL)}, ignore_index=True)

            rise = (np.average(fft_band_modulesAAL) - baseline_modulePeak) / baseline_modulePeak * 100
            if target_rise-10 < rise < target_rise+10:
                r = r+1
                cost = 0
            else:
                r = 0
                cost = target_rise - rise

            print("w = %0.4f - round = %i" % (w, r))
            print("RISE = %0.2f; sim number = %i" % (rise, n))
            print("LOOP ROUND REQUIRED %0.4f seconds.\n\n\n\n" % (time.time() - tic0,))

        secondBlock[j].append(w)
        secondBlock[j].append(i)

        print("WHOLE BLOCK (%s) REQUIRED %0.4f seconds.\n\n\n\n" % (name, time.time() - tic,))


    ### Third Block:
    # Gather a range of data
    topup_id, wrange_id = secondBlock[0][3], secondBlock[1][3]

    interval = int(np.trunc((topup_id-wrange_id)/20))
    init_id = int(np.trunc(wrange_id-10*interval))

    resultsAALb=pd.DataFrame()
    tic = time.time()
    for w in w_space[init_id:topup_id:interval]:
        for r in range(n_rep):

                tic0 = time.time()

                ## Sinusoid input
                eqn_t = equations.Sinusoid()
                eqn_t.parameters['amp'] = 1  # Amplitud diferencial por áreas ajustada en stimWeights
                eqn_t.parameters['frequency'] = initialPeak  # Hz
                eqn_t.parameters['onset'] = 0  # ms
                eqn_t.parameters['offset'] = simLength  # ms

                # electric field * orthogonal to surface
                weighting = np.loadtxt(ctb_folder + 'orthogonals/' + emp_subj + '-roast_OzCzModel_ef_mag-AAL2red.txt') * w

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
                fftpeaks = FFTpeaks(raw_data, simLength - transient)
                fft_peaks_hzAAL = fftpeaks[0][rois]
                fft_peaks_modulesAAL = fftpeaks[1][rois]
                fft_band_modulesAAL = fftpeaks[2][rois]

                # fft_data = np.concatenate(
                #     (fft_data, FFTarray(raw_data[rois], simLength, transient, regionLabels[rois], w, r, fft_peaks_hzAAL)))

                resultsAALb = resultsAAL.append({"w": w, "peak_hz": np.average(fft_peaks_hzAAL),
                                                "peak_module": np.average(fft_peaks_modulesAAL),
                                                "band_module": np.average(fft_band_modulesAAL)}, ignore_index=True)


                print("w = %0.4f - round = %i" % (w, r))
                print("LOOP ROUND REQUIRED %0.4f seconds.\n\n\n\n" % (time.time() - tic0,))
    print("WHOLE BLOCK REQUIRED %0.4f seconds.\n\n\n\n" % (time.time() - tic,))


    #### Fourth block:
    ## Save and Plot results
    resultsAALb.to_csv(specific_folder + '/w_' + emp_subj + "_AAL2red_alphaRise.csv", index=False)

    # fft_df = pd.DataFrame(fft_data, columns=["w", "rep", "regLab", "fft_module", "freq", "initPeak"])
    # fft_df.to_csv(specific_folder + '/w_' + str(round(w, 3)) + "_FFTs_AAL2red-ParietalComplex.csv", index=False)

    # if __name__ == "__main__":
    #
    #     processed_list = Parallel(n_jobs=num_cores, backend='multiprocessing')(
    #         delayed(simulate_parallel)(i) for i in weights)
    #
    #     print("Collecting data...")
    #     df_fft, resultsAALb = collectData(specific_folder)
    #     print("Box plot...")

    #     print("Lines 3d plot...")
    #     lines3dFFT(df_fft, specific_folder)

    # calculate percentages to Plot
    resultsAALb["percent"] = [
        ((resultsAALb.band_module[i] - baseline_modulePeak) / baseline_modulePeak) * 100 for i in
        range(len(resultsAALb))]

    resultsAALb_avg = resultsAALb.groupby('w').mean()
    resultsAALb_avg["sd"] = resultsAALb.groupby('w')[['w', 'peak_module']].std()

    fig = px.box(resultsAALb, x="w", y="peak_module",
                 title="Alpha peak module rise @Pareto-Occipital regions<br>(%i simulations | %s AAL2red)" % (
                 n_rep, emp_subj),
                 labels={  # replaces default labels by column name
                     "w": "Weight", "peak_module": "Alpha peak module"},
                 template="plotly")
    pio.write_html(fig, file=specific_folder + '\\' + emp_subj + "AAL_alphaRise_modules_" + str(n_rep) + "sim.html",
                   auto_open=False)

    fig = px.box(resultsAALb, x="w", y="percent",
                 title="Alpha peak module rise @Pareto-Occipital regions<br>(%i simulations | %s AAL2red)" % (
                 n_rep, emp_subj),
                 labels={  # replaces default labels by column name
                     "w": "Weight", "percent": "Percentage of alpha peak rise"},
                 template="plotly")
    pio.write_html(fig, file=specific_folder + '\\' + emp_subj + "AAL_alphaRise_percent_" + str(n_rep) + "sim.html",
                   auto_open=True)

