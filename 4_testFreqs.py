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
from toolbox.fft import multitapper
from toolbox.fc import PLV
from toolbox.signals import epochingTool
from jansen_rit_david_mine import JansenRitDavid2003_N

modes = {"jrd": {"stimW": 0.1106, "wp": (96, 17.5), "modelid": ".2003JansenRitDavid_N", "struct": "AAL2red"},
         "cb": {"stimW": 0.11885, "wp":(99, 6.5), "modelid": ".1995JansenRit", "struct": "CB"},
         "jr": {"stimW": 0.07787, "wp": (37, 22.5), "model": ".2003JansenRitDavid_N", "struct": "AAL2red"}}   # "jrd"; "cb"; "jrdcb"

### MODE
mode = "cb"
# Choose a name for your simulation and define the empirical for SC
model_id = modes[mode]["modelid"]

# Structuring directory to organize outputs
wd = os.getcwd()
main_folder = wd + "\\" + "PSE"

if not os.path.isdir(main_folder):
    os.mkdir(main_folder)

specific_folder = main_folder + "\\PSE_testFreqs_" + mode + model_id + "-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
os.mkdir(specific_folder)

ctb_folder = "D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\\CTB_data\\output\\"

simLength = 60 * 1000  # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1000  # Hz
transient = 4000  # ms to exclude from timeseries due to initial transient
n_simulations = 10


working_points = [("NEMOS_0"+str(nemos_id), modes[mode]["wp"][0], modes[mode]["wp"][1], modes[mode]["stimW"]) for nemos_id in [35,49,50,58,59,64,65,71,75,77]]

# working_points = [('NEMOS_035', 37, 22.5, 0.07787), # Already calculated
#                   ('NEMOS_049', 37, 22.5, 0.07787), ## target_ACC couldn't be computed
#                   ('NEMOS_050', 37, 22.5, 0.07787),
#                   ('NEMOS_058', 37, 22.5, 0.07787),
#                   ('NEMOS_059', 37, 22.5, 0.07787),
#                   ('NEMOS_064', 37, 22.5, 0.07787), ## target_ACC couldn't be computed
#                   ('NEMOS_065', 37, 22.5, 0.07787),
#                   ('NEMOS_071', 37, 22.5, 0.07787),
#                   ('NEMOS_075', 37, 22.5, 0.07787),
#                   ('NEMOS_077', 37, 22.5, 0.07787)]


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
    conn.speed = np.array([s])

    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)
    mon = (monitors.Raw(),)


    ###### Analyze PSE-FFT results >> define what are the frequencies of interest
    stim_freqs = np.concatenate(([0], np.linspace(8, 14, 50)))

    if "cb" in mode:
        rois = [4, 5, 18, 19]
    else:
        rois = [34, 35, 70, 71]  # rois implicated in the effect: 35-ACCl, 36-AACr, 71-Prl, 72-Prr [note python 0-indexing]
    ids = [1, 2, 3, 4]  # relations of interest: indices to choose from PLV's upper triangle (no diagonal)

    plv_targets = list()
    fft_targets = list()

    for f in stim_freqs:
        for r in range(n_simulations):
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
            # weighting = np.loadtxt(ctb_folder + 'CurrentPropagationModels/' + emp_subj + '-roast_P3P4Model_ef_mag-AAL2red.txt') * w
            ## Focal stimulation on ACC electric field;
            weighting = np.loadtxt(ctb_folder + 'CurrentPropagationModels/' + emp_subj + '-ACC_target_ef_mag-AAL2red.txt') * w
            if "cb" in mode:
                weighting = weighting[CB_rois]

            ## TEMP: test acc indirect influence
            # weighting[34] = 0
            # weighting[35] = 0

            stimulus = patterns.StimuliRegion(temporal=eqn_t, connectivity=conn, weight=weighting)

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
            print("Simulation time: %0.4f sec" % (time.time() - tic0,))
            # Extract data cutting initial transient
            if "jrd" in mode:
                raw_data = m.w * (output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T) + \
                           (1 - m.w) * (output[0][1][transient:, 3, :, 0].T - output[0][1][transient:, 4, :, 0].T)
            else:
                raw_data = output[0][1][transient:, 0, :, 0].T

            raw_data = raw_data[:, rois]
            raw_time = output[0][0][transient:]

            # Fourier Analysis plot
            # FFTplot(raw_data, simLength-transient, regionLabels, main_folder, mode="html")
            fft_peaks = multitapper(raw_data, simLength - transient)[2]

            ##########
            ### Measure functional connectivity between regions of interest : line 81 - rois
            newRow_t = [f, r]
            bands = [["3-alfa"], [(8, 12)]]
            ## [["1-delta", "2-theta", "3-alfa", "4-beta", "5-gamma"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]

            for b in range(len(bands[0])):

                newRow_t.append(bands[0][b])
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
                    efPhase.append(np.unwrap(np.angle(analyticalSignal)))
                    # efEnvelope.append(np.abs(analyticalSignal))

                # CONNECTIVITY MEASURES
                ## PLV
                plv = PLV(efPhase)
                # fname = ctb_folder+model_id+"\\"+bands[0][b]+"plv.txt"
                # np.savetxt(fname, plv)

                plv_targets.append(newRow_t + list(plv[np.triu_indices(len(rois), 1)][ids]))
                fft_targets.append(newRow_t + list(fft_peaks))

            print("simulating stimFreq = %0.2f - round = %i" % (f, r))
            print("LOOP ROUND REQUIRED %0.4f seconds.\n\n\n\n" % (time.time() - tic0,))

    ## GATHER RESULTS
    regionLabels = list(conn.region_labels)
    # Label FC relations
    rel_labels = [[conn.region_labels[roi] + '-' + conn.region_labels[roi1] for roi1 in rois] for roi in rois]
    rel_labels = np.asarray(rel_labels)[np.triu_indices(len(rois), 1)][ids]
    rlabels = [regionLabels[roi] for roi in rois]

    df_fc = pd.DataFrame(plv_targets, columns=["stimFreq", "round", "band"] + list(rel_labels))
    df_fft = pd.DataFrame(fft_targets, columns=["stimFreq", "round", "band"] + rlabels)
    # df_c = pd.DataFrame(plv_control, columns=["controlRegion", "stimFreq", "stimAmplitude", "band"] + regionLabels)

    df_fc.to_csv(specific_folder + "/" + emp_subj + "-FC_ACC&Pr.csv", index=False)
    df_fft.to_csv(specific_folder + "/" + emp_subj + "-FFT_ACC&Pr.csv", index=False)

    ## PLOTTING v2
    df_fc_avg = df_fc.groupby("stimFreq").mean()
    fft_avg = df_fft.groupby(["stimFreq"])[["stimFreq", "ACC_L", "ACC_R", "Precuneus_L", "Precuneus_R"]].mean()

    # Plot FC ACC-Pr by stim
    for i, rel in enumerate(rel_labels):

        if i == 0:
            auto_open = True
        else:
            auto_open = False

        max_fc = df_fc_avg[rel].idxmax()
        min_fc = df_fc_avg[rel].idxmin()

        fig = px.box(df_fc, x="stimFreq", y=rel,
                     title="Functional Connectivity between %s in alpha band <br>(%i simulations | %s %s | %s )" % (
                     rel, n_simulations, emp_subj, modes[mode]["struct"], mode),
                     labels={  # replaces default labels by column name
                         "stimFreq": "Stimulation Frequency", rel: "Functional Connectivity (PLV)"},
                     color_discrete_sequence=["dimgray"],
                     template="plotly")

        fig.add_vline(x=max_fc, line_width=0.75, line_dash="dot", line_color="orange")
        fig.add_vline(x=min_fc, line_width=0.75, line_dash="dot", line_color="darkblue")

        pio.write_html(fig, file=specific_folder + "/" + emp_subj + "FC_" + rel + '-w' + str(n_simulations) + "sim_v2.html",
                       auto_open=auto_open)

        # Plot FFT peak by stim
        fig_fft = go.Figure()

        fig_fft.add_trace(go.Scatter(x=fft_avg.stimFreq, y=fft_avg.ACC_L, name="ACC_L - ef_mag = " + str(round(weighting[rois[0]], 5))))
        fig_fft.add_trace(go.Scatter(x=fft_avg.stimFreq, y=fft_avg.ACC_R, name="ACC_R - ef_mag = " + str(round(weighting[rois[1]], 5))))
        fig_fft.add_trace(go.Scatter(x=fft_avg.stimFreq, y=fft_avg.Precuneus_L, name="Precuneus_L - ef_mag = " + str(round(weighting[rois[2]], 5))))
        fig_fft.add_trace(go.Scatter(x=fft_avg.stimFreq, y=fft_avg.Precuneus_R, name="Precuneus_R - ef_mag = " + str(round(weighting[rois[3]], 5))))

        fig_fft.update_layout(title=emp_subj + " || (g = " + str(g) + "; s = " + str(s) + "; w = " + str(round(w, 5)) + ")")
        fig_fft.update_xaxes(title="Stimulation Frequency")
        fig_fft.update_yaxes(title="Alpha peak frequency (Hz)")
        fig_fft.add_vline(x=max_fc, line_width=0.75, line_dash="dot", line_color="orange", name=rel)
        fig_fft.add_vline(x=min_fc, line_width=0.75, line_dash="dot", line_color="darkblue", name=rel)

        fig_fft.add_scatter(x=[9, 10, 11, 12, 13], y=[9, 10, 11, 12, 13], mode="lines", marker_color="gray", line=dict(width=0.5), name="Stimulation reference")

        pio.write_html(fig_fft, file=specific_folder + "/" + emp_subj + "FFT_" + rel + '-w' + str(n_simulations) + "sim_v2.html", auto_open=auto_open)



    ## PLOTTING v1
    # Plot FC ACC-Pr by stim - v1
    # for rel in rel_labels:
    #     fig = px.box(df_fc, x="stimFreq", y=rel,
    #                  title="Functional Connectivity between %s in alpha band <br>(%i simulations | %s AAL2red)" % (rel, n_simulations, emp_subj),
    #                  labels={  # replaces default labels by column name
    #                      "stimFreq": "Stimulation Frequency", rel:"Functional Connectivity (PLV)"},
    #                  color_discrete_sequence=["dimgray"],
    #                  template="plotly")
    #     pio.write_html(fig, file=specific_folder + "/" + emp_subj + "FC_" + rel + '-w' + str(n_simulations) + "sim.html",
    #                    auto_open=True)
    #
    #
    # # Plot FFT peak by stim
    # fft_avg = df_fft.groupby(["stimFreq"])[["stimFreq", "ACC_L", "ACC_R", "Precuneus_L", "Precuneus_R"]].mean()
    #
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=fft_avg.stimFreq, y=fft_avg.ACC_L, name="ACC_L - ef_mag = " + str(weighting[34])))
    # fig.add_trace(go.Scatter(x=fft_avg.stimFreq, y=fft_avg.ACC_R, name="ACC_R - ef_mag = " + str(weighting[35])))
    # fig.add_trace(go.Scatter(x=fft_avg.stimFreq, y=fft_avg.Precuneus_L, name="Precuneus_L - ef_mag = " + str(weighting[70])))
    # fig.add_trace(go.Scatter(x=fft_avg.stimFreq, y=fft_avg.Precuneus_R, name="Precuneus_R - ef_mag = " + str(weighting[71])))
    #
    # fig.update_layout(title=emp_subj + " || (g = " + str(g) + "; s = " + str(s) + "; w = " + str(round(w, 5)) + ")")
    # fig.update_xaxes(title="Stimulation Frequency")
    # fig.update_yaxes(title="Alpha peak frequency (Hz)")
    # pio.write_html(fig, file=specific_folder + "/" + emp_subj + "FFTby_stimFreq-10sim.html", auto_open=True)





