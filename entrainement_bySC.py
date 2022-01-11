import os
import time
import subprocess

import numpy as np
import scipy.signal
import pandas as pd
import scipy.stats

from tvb.simulator.lab import *
from mne import time_frequency, filter
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
import plotly.express as px

import sys
sys.path.append("D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\")  # temporal append
from toolbox.fft import multitapper, FFTpeaks
from toolbox.fc import PLV
from toolbox.signals import epochingTool, timeseriesPlot
from jansen_rit_david_mine import JansenRitDavid2003_N

modes = {"jrd": {"stimW": 0.1106, "wp": (96, 17.5), "modelid": ".2003JansenRitDavid_N", "struct": "AAL2red"},
         "cb": {"stimW": 0.11885, "wp": (99, 6.5), "modelid": ".1995JansenRit", "struct": "CB"},
         "jr": {"stimW": 0.07787, "wp": (37, 22.5), "modelid": ".2003JansenRitDavid_N", "struct": "AAL2red"}}   # "jrd"; "cb"; "jrdcb"

### MODE
mode = "jr"
# Choose a name for your simulation and define the empirical for SC
model_id = modes[mode]["modelid"]

n_rep = 3  # For the random selection of removal rois
n_simulations = 5  # For the number of simulations with same structure

target_roi = 70  # 70 Precuneus L | 34 ACC L
check_roi = 34


# Structuring directory to organize outputs
wd = os.getcwd()
main_folder = wd + "\\" + "PSE"

if not os.path.isdir(main_folder):
    os.mkdir(main_folder)

specific_folder = main_folder + "\\PSE_entrainment_bySC_" + str(target_roi) + model_id + "-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
os.mkdir(specific_folder)

ctb_folder = "D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\\CTB_data\\output\\"

simLength = 5 * 1000  # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1000  # Hz
transient = 1000  # ms to exclude from timeseries due to initial transient

working_points = [("NEMOS_0"+str(nemos_id), modes[mode]["wp"][0], modes[mode]["wp"][1], modes[mode]["stimW"]) for nemos_id in [35]]#,49,50,58,59,64,65,71,75,77]]

for emp_subj, g, s, w in working_points:

    ## STRUCTURE
    conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2red.zip")
    conn.weights = conn.scaled_weights(mode="tract")
    # conn.weights[34, :] = 0
    # conn.weights[:, 34] = 0
    # conn.weights[35, :] = 0
    # conn.weights[:, 35] = 0

    if "cb" in mode:
        CB_rois = [18, 19, 32, 33, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 62, 63, 64, 65, 70, 71, 76, 77]
        conn.weights = conn.weights[:, CB_rois][CB_rois]
        conn.tract_lengths = conn.tract_lengths[:, CB_rois][CB_rois]
        conn.region_labels = conn.region_labels[CB_rois]

    if "jrd" in mode:
        p_ = 0.1085 if "cb" in mode else 0
        sigma_array = np.where((conn.region_labels == 'Thalamus_R') | (conn.region_labels == 'Thalamus_L'), 0.022, 0)
        p_array = np.where((conn.region_labels == 'Thalamus_R') | (conn.region_labels == 'Thalamus_L'), 0.22, p_)

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

    else:
        # Parameters from Stefanovski 2019. Good working point at g=33, s=15.5 on AAL2red connectome.
        m = models.JansenRit(A=np.array([3.25]), B=np.array([22]), J=np.array([1]),
                             a=np.array([0.1]), a_1=np.array([135]), a_2=np.array([108]),
                             a_3=np.array([33.75]), a_4=np.array([33.75]), b=np.array([0.06]),
                             mu=np.array([0.1085]), nu_max=np.array([0.0025]), p_max=np.array([0]), p_min=np.array([0]),
                             r=np.array([0.56]), v0=np.array([6]))

    coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                       r=np.array([0.56]))
    conn.speed = np.array([s])

    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)
    mon = (monitors.Raw(),)

    ###### Analyze PSE-FFT results >> define what are the frequencies of interest
    stim_freqs = np.concatenate(([0], np.linspace(8, 14, 30)))
    # if "cb" in mode:
    #     rois = [4, 5, 18, 19]
    # else:
    #     rois = [34, 35, 70, 71]  # rois implicated in the effect: 35-ACCl, 36-AACr, 71-Prl, 72-Prr [note python 0-indexing]
    # ids = [1, 2, 3, 4]  # relations of interest: indices to choose from PLV's upper triangle (no diagonal)

    Results_fft = list()
    # For n of binarized weights for the roi of interest
    connected2target_rois = np.where(conn.binarized_weights[target_roi])[0]
    for n_conn_remove in range(int(len(connected2target_rois)/2)):

        print("\n\nREMOVING %i CONNECTIONS" % n_conn_remove)
        for r in range(n_rep):
            tic = time.time()
            ## Randomly choose "n_conn_remove" connections and set them to 0
            chosen_rois2remove = np.random.choice(connected2target_rois, size=n_conn_remove, replace=False)

            for roi2remove in chosen_rois2remove:
                conn.weights[roi2remove, target_roi] = 0
                conn.weights[target_roi, roi2remove] = 0
            tracts_left = sum(conn.weights[:, target_roi])

            for i_f, f in enumerate(stim_freqs):
                tic0 = time.time()

                ## Sinusoid input
                eqn_t = equations.Sinusoid()
                eqn_t.parameters['amp'] = 1
                eqn_t.parameters['frequency'] = f  # Hz
                eqn_t.parameters['onset'] = 0  # ms
                eqn_t.parameters['offset'] = simLength  # ms
                # if w != 0:
                #     eqn_t.parameters['DC'] = 0.0005 / w

                ## electric field; electrodes placed @ P3P4 to stimulate precuneus
                weighting = np.loadtxt(ctb_folder + 'CurrentPropagationModels/' + emp_subj + '-roast_P3P4Model_ef_mag-AAL2red.txt') * w
                ## Focal stimulation on ACC electric field;
                # weighting = np.loadtxt(ctb_folder + 'CurrentPropagationModels/' + emp_subj + '-ACC_target_ef_mag-AAL2red.txt') * w
                if "cb" in mode:
                    weighting = weighting[CB_rois]

                ## TEMP: test acc indirect influence
                # weighting[34] = 0
                # weighting[35] = 0
                stimulus = patterns.StimuliRegion(temporal=eqn_t, connectivity=conn, weight=weighting)

                # Configure space and time
                stimulus.configure_space()
                stimulus.configure_time(np.arange(0, simLength, 1))

                # Run simulation
                sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon,
                                          stimulus=stimulus)
                sim.configure()

                for sim_n in range(n_simulations):
                    output = sim.run(simulation_length=simLength)
                    # Extract data cutting initial transient
                    if "jrd" in mode:
                        raw_data = m.w * (output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T) + \
                                   (1 - m.w) * (output[0][1][transient:, 3, :, 0].T - output[0][1][transient:, 4, :,
                                                                                      0].T)
                    else:
                        raw_data = output[0][1][transient:, 0, :, 0].T

                    raw_data = raw_data[[target_roi, check_roi], :]
                    raw_time = output[0][0][transient:]
                    regionLabels = conn.region_labels
                    # Fourier Analysis plot
                    # FFTplot(raw_data, simLength-transient, regionLabels, main_folder, mode="html")
                    fft_peaks = FFTpeaks(raw_data, simLength-transient)[0]
                    Results_fft.append([f, r, sim_n, len(chosen_rois2remove), list(chosen_rois2remove), tracts_left, regionLabels[target_roi], fft_peaks[0], regionLabels[check_roi], fft_peaks[1]])

                print("simulating stimFreq = %0.2f  (%i/%i)  |  %i simulations  |  time %0.4f seconds" % (f, i_f, len(stim_freqs), n_simulations, time.time() - tic0,), end="\r")
            print("LOOP ROUND %i REQUIRED %0.4f minutes." % (r, (time.time() - tic)/60,))


## GATHER RESULTS
df_fft = pd.DataFrame(Results_fft, columns=["stimFreq", "rep", "sim_n", "n_rois_removed", "rois_removed", "tracts_left", "target_roi", "target_peak", "check_roi", "check_peak"])
df_fft.to_csv(specific_folder + "/" + emp_subj + "-FFT_bySC_"+regionLabels[target_roi]+"-"+regionLabels[check_roi]+".csv", index=False)

# Average data by simulations
df_fft_avg = df_fft.groupby(["stimFreq", "rep", "n_rois_removed"]).mean().reset_index()
df_baseline = df_fft_avg.loc[df_fft_avg["stimFreq"] == 0]

max_tracts_baseline = float(df_baseline["tracts_left"].loc[(df_baseline["rep"]==0)&(df_baseline["n_rois_removed"]==0)])
conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2red.zip")
tracts_normalization_factor = conn.weights.max()

## Calculate entrainment range by n_rois_removed
entrainment_range = []
for n_conn_remove in range(int(len(connected2target_rois)/2)):
    for r in range(n_rep):

        df_subset = df_fft_avg.loc[(df_fft_avg["n_rois_removed"]==n_conn_remove)&(df_fft_avg["rep"]==r)]
        removed_tracts = tracts_normalization_factor * (max_tracts_baseline - float(df_baseline["tracts_left"].loc[(df_baseline["n_rois_removed"]==n_conn_remove)&(df_baseline["rep"]==r)]))

        ## Calculate frequency distance to baseline and to stimulation frequency.
        target_2baseline = np.asarray(df_subset["target_peak"]) - float(df_baseline["target_peak"].loc[(df_baseline["n_rois_removed"]==n_conn_remove)&(df_fft_avg["rep"]==r)])
        target_2stim = np.asarray(df_subset["target_peak"]) - np.asarray(df_subset["stimFreq"])

        # Where distance to baseline is higher than to stimulus, we assume there was entrainment
        range_hz = df_subset["stimFreq"].loc[(abs(target_2baseline) - abs(target_2stim) > 0)].max() - df_subset["stimFreq"].loc[(abs(target_2baseline) - abs(target_2stim) > 0)].min()
        entrainment_range.append([n_conn_remove, r, removed_tracts, "target", regionLabels[target_roi], range_hz])

        check_2baseline = np.asarray(df_subset["check_peak"]) - float(df_baseline["check_peak"].loc[(df_baseline["n_rois_removed"]==n_conn_remove)&(df_fft_avg["rep"]==r)])
        check_2stim = np.asarray(df_subset["check_peak"]) - np.asarray(df_subset["stimFreq"])
        range_hz = df_subset["stimFreq"].loc[(abs(check_2baseline) - abs(check_2stim) > 0)].max() - df_subset["stimFreq"].loc[(abs(check_2baseline) - abs(check_2stim) > 0)].min()
        entrainment_range.append([n_conn_remove, r, removed_tracts, "check", regionLabels[check_roi], range_hz])

df_entrainment_bySC = pd.DataFrame(entrainment_range, columns=["removed_conn", "repetition", "removed_tracts", "roi_mode", "roi_name", "entrainment_range"])
df_entrainment_bySC.to_csv(specific_folder + "/" + emp_subj + "-entrainment_bySC_"+regionLabels[target_roi]+"-"+regionLabels[check_roi]+".csv", index=False)


fig = px.strip(df_entrainment_bySC, x="removed_conn", y="entrainment_range", color="roi_name",
           title="Nodes entrainment by number of %s's connections to other regions (%i repetitions | %i simulations)" % (regionLabels[target_roi], n_rep, n_simulations))
pio.write_html(fig, file=specific_folder + "/" + emp_subj + "entrainment_bySC-target" + regionLabels[target_roi] + "_conns_3rep.html",
               auto_open=True)

fig = px.strip(df_entrainment_bySC, x="removed_tracts", y="entrainment_range", color="roi_name",
           title="Nodes entrainment by number of %s's connections to other regions (%i repetitions | %i simulations)" % (regionLabels[target_roi], n_rep, n_simulations))
pio.write_html(fig, file=specific_folder + "/" + emp_subj + "entrainment_bySC-target" + regionLabels[target_roi] + "_tracts_3rep.html",
               auto_open=True)

