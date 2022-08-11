import os
import random
import time
import shutil

import numpy as np
import pandas as pd

from tvb.simulator.lab import *

import sys
sys.path.append("E:\\LCCN_Local\PycharmProjects\\")  # temporal append
from toolbox.fft import multitapper
from toolbox.fc import PLV
from toolbox.signals import epochingTool
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003_N

import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

# coge el baseline y guarda los puntos y las medias baseline

params = {"jrd": {"g": 45, "s": 3.5, "model_id": ".2003JansenRitDavid", "init_w": 0.0001, "learning_rate": 1e-3},
          "cb": {"g": 60, "s": 5.5, "model_id": ".1995JansenRit", "init_w": 0.5, "learning_rate": 1e-3},
          "jrdcb": {"g": 90, "s": 1.5, "model_id": ".2003JansenRitDavid", "init_w": 0.01, "learning_rate": 1e-3},
          "jr": {"g": 15, "s": 11.5, "model_id": ".2003JansenRit", "init_w": 0.1, "learning_rate": 1e-3}}

mode = "jr"  # "jrd"; "cb"; "jrdcb"

# working_points = [("NEMOS_0"+str(i), params[mode]["g"], params[mode]["s"]) for i in [35, 49, 50, 58, 59, 64, 65, 71, 75, 77]]
working_points = ['NEMOS_AVG', params[mode]["g"], params[mode]["s"]]

## Folder structure - Local
wd = os.getcwd()
ctb_folder = "D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\\CTB_data2\\"

main_folder = wd + "\\" + "PSE"
if os.path.isdir(main_folder) == False:
    os.mkdir(main_folder)
specific_folder = main_folder + "\\PSE_autoWfit-avgNEMOS" + mode + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")

if os.path.isfile(specific_folder + "\\" + "FFT_fullDF.csv"):
    shutil.rmtree(specific_folder)
if os.path.isdir(specific_folder) == False:
    os.mkdir(specific_folder)

tic0 = time.time()

#### First Block (1m 15sec):
## Alpha peak baseline
baselineAAL = pd.DataFrame()
print("   BASELINE PHASE _")


tic1 = time.time()
emp_subj, g, s = working_points

# Prepare simulation parameters
simLength = 24 * 1000  # ms
samplingFreq = 1000  # Hz
transient = 4000  # ms

n_rep = 10

conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL.zip")
conn.weights = conn.scaled_weights(mode="tract")

# Define regions implicated in Functional analysis: remove  Cerebelum, Thalamus, Caudate (i.e. subcorticals)
cortical_rois = ['Precentral_L', 'Precentral_R', 'Frontal_Sup_L', 'Frontal_Sup_R',
                 'Frontal_Sup_Orb_L', 'Frontal_Sup_Orb_R', 'Frontal_Mid_L', 'Frontal_Mid_R',
                 'Frontal_Mid_Orb_L', 'Frontal_Mid_Orb_R',
                 'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L',
                 'Frontal_Inf_Tri_R', 'Frontal_Inf_Orb_L', 'Frontal_Inf_Orb_R',
                 'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L',
                 'Supp_Motor_Area_R', 'Olfactory_L', 'Olfactory_R',
                 'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R',
                 'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R', 'Rectus_L', 'Rectus_R',
                 'Insula_L', 'Insula_R', 'Cingulum_Ant_L', 'Cingulum_Ant_R',
                 'Cingulum_Mid_L', 'Cingulum_Mid_R', 'Cingulum_Post_L',
                 'Cingulum_Post_R', 'Hippocampus_L', 'Hippocampus_R',
                 'ParaHippocampal_L', 'ParaHippocampal_R', 'Amygdala_L',
                 'Amygdala_R', 'Calcarine_L', 'Calcarine_R', 'Cuneus_L', 'Cuneus_R',
                 'Lingual_L', 'Lingual_R', 'Occipital_Sup_L', 'Occipital_Sup_R',
                 'Occipital_Mid_L', 'Occipital_Mid_R', 'Occipital_Inf_L',
                 'Occipital_Inf_R', 'Fusiform_L', 'Fusiform_R', 'Postcentral_L',
                 'Postcentral_R', 'Parietal_Sup_L', 'Parietal_Sup_R',
                 'Parietal_Inf_L', 'Parietal_Inf_R', 'SupraMarginal_L',
                 'SupraMarginal_R', 'Angular_L', 'Angular_R', 'Precuneus_L',
                 'Precuneus_R', 'Paracentral_Lobule_L', 'Paracentral_Lobule_R', 'Heschl_L', 'Heschl_R',
                 'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L',
                 'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R',
                 'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L', 'Temporal_Inf_R']
cingulum_rois = ['Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R',
                 'Insula_L', 'Insula_R', 'Cingulum_Ant_L', 'Cingulum_Ant_R',
                 'Cingulum_Post_L',
                 'Cingulum_Post_R', 'Hippocampus_L', 'Hippocampus_R',
                 'ParaHippocampal_L', 'ParaHippocampal_R', 'Amygdala_L',
                 'Amygdala_R', 'Parietal_Sup_L', 'Parietal_Sup_R',
                 'Parietal_Inf_L', 'Parietal_Inf_R', 'Precuneus_L',
                 'Precuneus_R', 'Thalamus_L', 'Thalamus_R']

# load text with FC rois; check if match SC
FClabs = list(np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/roi_labels_rms.txt", dtype=str))
FC_cortex_idx = [FClabs.index(roi) for roi in cortical_rois]  # find indexes in FClabs that matches cortical_rois
FC_cb_idx = [FClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cortical_rois

SClabs = list(conn.region_labels)
SC_cortex_idx = [SClabs.index(roi) for roi in cortical_rois]
SC_cb_idx = [SClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cortical_rois

# Subset for Cingulum Bundle
if "cb" in mode:
    conn.weights = conn.weights[:, SC_cb_idx][SC_cb_idx]
    conn.tract_lengths = conn.tract_lengths[:, SC_cb_idx][SC_cb_idx]
    conn.region_labels = conn.region_labels[SC_cb_idx]

if "jrd" in mode:
    p_ = 0.13 if "cb" in mode else 0
    sigma_array = np.where((conn.region_labels == 'Thalamus_R') | (conn.region_labels == 'Thalamus_L'), 0.022, 0)
    p_array = np.where((conn.region_labels == 'Thalamus_R') | (conn.region_labels == 'Thalamus_L'), 0.22 + p_, p_)

    # Parameters edited from David and Friston (2003).
    m = JansenRitDavid2003_N(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
                             tau_e1=np.array([10.8]), tau_i1=np.array([22.0]),
                             He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
                             tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),

                             w=np.array([0.8]), c=np.array([135.0]),
                             c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                             c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                             v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                             p=np.array([p_array]), sigma=np.array([sigma_array]))

    ## Remember to hold tau*H constant.
    m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
    m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])

    coup = coupling.SigmoidalJansenRitDavid(a=np.array([g]), w=m.w, e0=m.e0, v0=m.v0, r=m.r)

else:
    # Parameters from Stefanovski 2019. Good working point at g=33, s=15.5 on AAL2red connectome.
    m = models.JansenRit(A=np.array([3.25]), B=np.array([22]), J=np.array([1]),
                         a=np.array([0.1]), a_1=np.array([135]), a_2=np.array([108]),
                         a_3=np.array([33.75]), a_4=np.array([33.75]), b=np.array([0.06]),
                         mu=np.array([0.1085]), nu_max=np.array([0.0025]), p_max=np.array([0]), p_min=np.array([0]),
                         r=np.array([0.56]), v0=np.array([6]))

    coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                       r=np.array([0.56]))

# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)
mon = (monitors.Raw(),)

conn.speed = np.array([s])

# Calculate initial peak from n reps
for r in range(n_rep):
    tic2 = time.time()

    # Run simulation
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
    sim.configure()
    output = sim.run(simulation_length=simLength)

    # Extract data: "output[a][b][:,0,:,0].T" where:
    # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
    if "jrd" in mode:
        raw_data = m.w * (output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T) + \
                   (1 - m.w) * (output[0][1][transient:, 3, :, 0].T - output[0][1][transient:, 4, :, 0].T)
    else:
        raw_data = output[0][1][transient:, 0, :, 0].T
    regionLabels = conn.region_labels

    # ROIs of interest to measure alpha peak increase #i
    # All occipito-parietal regins w/ 0-indexing in Python.
    occipital_rois = ['Calcarine_L', 'Calcarine_R', 'Cuneus_L', 'Cuneus_R',
                      'Lingual_L', 'Lingual_R', 'Occipital_Sup_L', 'Occipital_Sup_R',
                      'Occipital_Mid_L', 'Occipital_Mid_R', 'Occipital_Inf_L',
                      'Occipital_Inf_R', 'Parietal_Sup_L', 'Parietal_Sup_R',
                      'Parietal_Inf_L', 'Parietal_Inf_R', 'SupraMarginal_L',
                      'SupraMarginal_R', 'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R']

    if "cb" in mode:
        occ_cb = list(set(occipital_rois).intersection(set(cingulum_rois)))
        rois = [list(conn.region_labels).index(roi) for roi in occ_cb]
    else:
        rois = [SClabs.index(roi) for roi in occipital_rois]

    fftpeaks = multitapper(raw_data, samplingFreq, regionLabels, peaks=True)
    fft_peaks_hzAAL = fftpeaks[2][rois]
    fft_peaks_modulesAAL = fftpeaks[3][rois]
    fft_band_modulesAAL = fftpeaks[4][rois]

    # fft_data = np.empty((1, 6),
    #                     dtype=object)  # columns=[param1=w, param2=rep, regLabs, fft_tot, freq_tot, param3=initPeak]
    # fft_data = np.concatenate(
    #     (fft_data, FFTarray(raw_data[rois], simLength, transient, regionLabels[rois], 0, 0, fft_peaks_hzAAL)))[1:]

    baselineAAL = baselineAAL.append({"subject": emp_subj, "w": 0, "peak_hz": np.average(fft_peaks_hzAAL),
                                      "peak_module": np.average(fft_peaks_modulesAAL),
                                      "band_module": np.average(fft_band_modulesAAL)}, ignore_index=True)

    print("LOOP ROUND %i/%i REQUIRED %0.4f seconds" % (r, n_rep, time.time() - tic2,), end="\r")
print("SUBJECT %s REQUIRED %0.4f seconds" % (emp_subj, time.time() - tic1,))
print("\n\nWHOLE BLOCK (baseline alpha peak) REQUIRED %0.4f minutes.\n\n\n" % ((time.time() - tic0) / 60,))

# Normalize spectral modules by the average of the baseline band module (through subjects and repetitions)
normalization_factor = np.average(baselineAAL["band_module"])

# Transform previously gathered baseline data
baselineAAL["peak_module"] = baselineAAL["peak_module"] / normalization_factor
baselineAAL["band_module"] = baselineAAL["band_module"] / normalization_factor

# Average peak frequency and module per subject
baselineAAL_subj = baselineAAL.groupby('subject').mean()
baselineAAL_group = baselineAAL_subj.mean()

### Second Block (40s per round)
## Optimization for top and optimal w value; sort of ML approach
secondBlock = [["top-up W", 50],  # The maximum alpha rise I want to plot
               ["optimum-range W ", 14]]  # The optimal value of alpha rise I want to plot

w = params[mode]["init_w"]
for j, sB in enumerate(secondBlock):
    name, target_rise = sB[0], sB[1]

    # Reset variables: peakRise, repetition (at least 3 rounds with alpha rise in the desired range),
    # index_Wspace, rise_target-rise, sim_group number
    r, n = 0, 0  # To count simulations
    rise, cost = 0, 0  # To perform descent
    lr = params[mode]["learning_rate"] * params[mode]["init_w"]  # learning_rate based on w magnitude

    tic = time.time()
    print("   OPTIMIZATION for " + name)
    while rise < target_rise - 10 or rise > target_rise + 10 or r < 3:
        tic0 = time.time()
        n = n + 1

        cost = 300 if cost > 300 else cost
        cost = 2 * cost if cost < 0 else cost
        # lr = lr / (n - 10) if n > 10 else lr  # Semi-adaptive learning rate _Fast at beggining; slow in the end
        # Gradient Descent update rule
        w = w - lr * cost

        w = params[mode]["init_w"] / n if w < 0 else w

        temp_secondBlockAAL = pd.DataFrame()
        for emp_subj, alpha_data in baselineAAL_subj.iterrows():

            tic1 = time.time()
            initialPeak = alpha_data["peak_hz"]

            ## Sinusoid input
            eqn_t = equations.Sinusoid()
            eqn_t.parameters['amp'] = 1  # Amplitud diferencial por áreas ajustada en stimWeights
            eqn_t.parameters['frequency'] = initialPeak  # Hz
            eqn_t.parameters['onset'] = 0  # ms
            eqn_t.parameters['offset'] = simLength  # ms

            # electric field * orthogonal to surface
            weighting = np.loadtxt(
                ctb_folder + 'CurrentPropagationModels/' + emp_subj + '-efnorm_mag-roast_OzCzModel-AAL.txt') * w
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

            # Run simulation
            sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon,
                                      stimulus=stimulus)
            sim.configure()
            fft_peaks_hzAAL, fft_peaks_modulesAAL, fft_band_modulesAAL = [], [], []

            for r in range(3):
                output = sim.run(simulation_length=simLength)
                # Extract data cutting initial transient
                if "jrd" in mode:
                    raw_data = m.w * (output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T) + \
                               (1 - m.w) * (output[0][1][transient:, 3, :, 0].T - output[0][1][transient:, 4, :, 0].T)
                else:
                    raw_data = output[0][1][transient:, 0, :, 0].T
                raw_time = output[0][0][transient:]
                regionLabels = conn.region_labels

                # ROIs of interest to measure alpha peak increase
                fftpeaks = multitapper(raw_data, samplingFreq, regionLabels, peaks=True)
                fft_peaks_hzAAL.append(fftpeaks[2][rois])
                fft_peaks_modulesAAL.append(fftpeaks[3][rois] / normalization_factor)
                fft_band_modulesAAL.append(fftpeaks[4][rois] / normalization_factor)
                # fft_data = np.concatenate(
                #     (fft_data, FFTarray(raw_data[rois], simLength, transient, regionLabels[rois], w, r, fft_peaks_hzAAL)))
            fft_peak_hzAAL = np.average(fft_peaks_hzAAL, axis=0)
            fft_peak_modulesAAL = np.average(fft_peaks_modulesAAL, axis=0)
            fft_band_modulesAAL = np.average(fft_band_modulesAAL, axis=0)

            temp_secondBlockAAL = temp_secondBlockAAL.append(
                {"subject": emp_subj, "w": w, "peak_hz": np.average(fft_peaks_hzAAL),
                 "peak_module": np.average(fft_peaks_modulesAAL),
                 "band_module": np.average(fft_band_modulesAAL)}, ignore_index=True)

        rise = (np.average(temp_secondBlockAAL["band_module"]) - baselineAAL_group["band_module"]) / baselineAAL_group[
            "band_module"] * 100
        if target_rise - 5 < rise < target_rise + 5:
            r = r + 1
            cost = 0
        else:
            r = 0
            cost = rise - target_rise

        print("w = %0.9f - round = %i  |  RISE = %0.2f; sim number = %i\nLOOP ROUND REQUIRED %0.4f min." % (
        w, r, rise, n, (time.time() - tic0) / 60), end="\r")
    secondBlock[j].append(w)  # keep weight value for next block
    print("\n\nWHOLE BLOCK (%s) REQUIRED %0.4f min.\n\n" % (name, (time.time() - tic) / 60,))

### Third Block (~3h):
## Gather an optimized range of data for plotting.
topup_w, w_approx = secondBlock[0][2], secondBlock[1][2]

interval = (topup_w - w_approx) / 15
w0 = w_approx - 15 * interval

w_space = np.arange(w0, topup_w, interval)

resultsAAL = pd.DataFrame()
saved_signals = []  # To check simulated outliers (i.e. sudden plunge of peak)
tic = time.time()
print("   GATHERING FINAL DATA")
for i, w in enumerate(w_space):
    tic1 = time.time()
    for r in range(n_rep):
        tic0 = time.time()

        for emp_subj, alpha_data in baselineAAL_subj.iterrows():

            initialPeak = alpha_data["peak_hz"]

            ## Sinusoid input
            eqn_t = equations.Sinusoid()
            eqn_t.parameters['amp'] = 1  # Amplitud diferencial por áreas ajustada en stimWeights
            eqn_t.parameters['frequency'] = 0  # Hz
            eqn_t.parameters['onset'] = 0  # ms
            eqn_t.parameters['offset'] = simLength  # ms

            # electric field * orthogonal to surface
            weighting = np.loadtxt(
                ctb_folder + 'CurrentPropagationModels/' + emp_subj + '-efnorm_mag-roast_OzCzModel-AAL.txt') * w
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

            # Run simulation
            sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon,
                                      stimulus=stimulus)
            sim.configure()
            fft_peaks_hzAAL, fft_peaks_modulesAAL, fft_band_modulesAAL = [], [], []

            output = sim.run(simulation_length=simLength)
            # Extract data cutting initial transient
            if "jrd" in mode:
                raw_data = m.w * (output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T) + \
                           (1 - m.w) * (output[0][1][transient:, 3, :, 0].T - output[0][1][transient:, 4, :, 0].T)
            else:
                raw_data = output[0][1][transient:, 0, :, 0].T

            raw_time = output[0][0][transient:]
            regionLabels = conn.region_labels

            # ROIs of interest to measure alpha peak increase
            fftpeaks = multitapper(raw_data, samplingFreq, regionLabels, peaks=True)
            fft_peaks_hzAAL.append(fftpeaks[2][rois])
            fft_peaks_modulesAAL.append(fftpeaks[3][rois] / normalization_factor)
            fft_band_modulesAAL.append(fftpeaks[4][rois] / normalization_factor)
            # # fft_data = np.concatenate(
            # #     (fft_data, FFTarray(raw_data[rois], simLength, transient, regionLabels[rois], w, r, fft_peaks_hzAAL)))
            # fft_peak_hzAAL = np.average(fft_peaks_hzAAL, axis=0)
            # fft_peak_modulesAAL = np.average(fft_peaks_modulesAAL, axis=0)
            # fft_band_modulesAAL = np.average(fft_band_modulesAAL, axis=0)

            if ((np.average(fft_band_modulesAAL) - baselineAAL_group["band_module"]) * 100 < 0) & (
                    len(saved_signals) < 10):
                saved_signals.append([raw_data, raw_time, fftpeaks])
            # fft_data = np.concatenate(
            #     (fft_data, FFTarray(raw_data[rois], simLength, transient, regionLabels[rois], w, r, fft_peaks_hzAAL)))

            resultsAAL = resultsAAL.append({"subject": emp_subj, "w": w, "peak_hz": np.average(fft_peaks_hzAAL),
                                            "peak_module": np.average(fft_peaks_modulesAAL),
                                            "band_module": np.average(fft_band_modulesAAL)}, ignore_index=True)

        print("w = %0.4f (%i/%i) - round = %i   |  LOOP ROUND REQUIRED %0.4f min." % (
        w, i, len(w_space), r, (time.time() - tic0) / 60), end="\r")
    print("w = %0.4f REQUIRED %0.4f min." % (w, (time.time() - tic1) / 60), end="\n")
print("WHOLE BLOCK REQUIRED %0.4f min.\n\n\n\n" % ((time.time() - tic) / 60,))

#### Fourth block:
## Save and Plot results
resultsAAL.to_csv(specific_folder + "/autoWgroup_AAL_alphaRise.csv", index=False)

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


# Calculate percentages of alpha rise to Plot
resultsAAL["percent"] = [
    ((resultsAAL.band_module[i] - baselineAAL_group["band_module"]) / baselineAAL_group["band_module"]) * 100 for i in
    range(len(resultsAAL))]

resultsAAL_avg = resultsAAL.groupby(['w', 'subject']).mean()
resultsAAL_avg["sd"] = resultsAAL.groupby('w')[['w', 'peak_module']].std()

fig = px.box(resultsAAL, x="w", y="peak_module",
             title="Alpha peak module rise @Pareto-Occipital regions<br>(%i simulations | AVG subject AAL)" % n_rep,
             labels={  # replaces default labels by column name
                 "w": "Weight", "peak_module": "Alpha peak module"},
             template="plotly")
pio.write_html(fig, file=specific_folder + '\\avgNemosAAL_alphaRise_modules_' + str(n_rep) + "sim.html",
               auto_open=True)

fig = px.box(resultsAAL, x="w", y="percent",
             title="Alpha peak module rise @Pareto-Occipital regions<br>(%i simulations | AVG subjects AAL)" % n_rep,
             labels={  # replaces default labels by column name
                 "w": "Weight", "percent": "Percentage of alpha peak rise"},
             template="plotly")
pio.write_html(fig, file=specific_folder + '\\avgNemosAAL_alphaRise_percent_' + str(n_rep) + "sim.html",
               auto_open=True)

# Scatter plot with mean and median
fig = px.scatter(resultsAAL, x="w", y="percent", color="subject")

w = np.asarray(resultsAAL.groupby("w").mean().reset_index()["w"])
mean = np.asarray(resultsAAL.groupby("w").mean()["percent"])
median = np.asarray(resultsAAL.groupby("w").median()["percent"])

fig.add_trace(go.Scatter(x=w, y=mean, mode="lines", name="mean", visible="legendonly"))
fig.add_trace(go.Scatter(x=w, y=median, mode="lines", name="median", visible="legendonly"))
pio.write_html(fig, file=specific_folder + '\\avgAAL_alphaRise_percentScatter_' + str(n_rep) + "sim.html",
               auto_open=True)

# # from toolbox.signals import timeseriesPlot
# # from toolbox.fft import multitapper, FFTplot
# id = 0
# raw_data = saved_signals[id][0]
# raw_time = saved_signals[id][1]
# timeseriesPlot(raw_data, raw_time, regionLabels, title= "20HzStim", mode="html")
# multitapper(raw_data, samplingFreq, regionLabels, plot=True)
