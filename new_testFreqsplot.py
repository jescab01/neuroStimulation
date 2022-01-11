import numpy as np
import pandas as pd

from tvb.simulator.lab import *
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
import plotly.express as px

specific_folder = 'D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\neuroStimulation\PSE\PSE_testFreqs_wX2.1995JansenRit_NEMOS-m07d19y2021-t10h.38m.33s'
ctb_folder = "D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\\CTB_data\\output\\"

working_points = [('NEMOS_035', 37, 22.5, 0.04317),
                  ('NEMOS_049', 37, 22.5, 0.053),
                  ('NEMOS_050', 37, 22.5, 0.0423),
                  ('NEMOS_058', 37, 22.5, 0.04563),
                  ('NEMOS_059', 37, 22.5, 0.02985),
                  ('NEMOS_064', 37, 22.5, 0.04085),
                  ('NEMOS_065', 37, 22.5, 0.04398),
                  ('NEMOS_071', 37, 22.5, 0.04735),
                  ('NEMOS_075', 37, 22.5, 0.04563),
                  ('NEMOS_077', 37, 22.5, 0.03865)]

for wp in working_points:

    emp_subj, g, s, w = wp

    n_simulations = 10

    rois = [34, 35, 70, 71]  # rois implicated in the effect: 35-ACCl, 36-AACr, 71-Prl, 72-Prr [note python 0-indexing]
    ids = [1, 2, 3, 4]  # relations of interest: indices to choose from PLV's upper triangle (no diagonal)

    conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2red.zip")
    weighting = np.loadtxt(ctb_folder + 'orthogonals/' + emp_subj + '-roast_P3P4Model_ef_mag-AAL2red.txt') * w *2

    ## load data
    # Label FC relations
    rel_labels = [[conn.region_labels[roi] + '-' + conn.region_labels[roi1] for roi1 in rois] for roi in rois]
    rel_labels = np.asarray(rel_labels)[np.triu_indices(len(rois), 1)][ids]
    rlabels = [conn.region_labels[roi] for roi in rois]

    df_fc = pd.read_csv(specific_folder + '\\' + emp_subj + "-FC_ACC&Pr.csv")
    df_fft = pd.read_csv(specific_folder + '\\' + emp_subj + "-FFT_ACC&Pr.csv")

    df_fc_avg = df_fc.groupby("stimFreq").mean()
    fft_avg = df_fft.groupby(["stimFreq"])[["stimFreq", "ACC_L", "ACC_R", "Precuneus_L", "Precuneus_R"]].mean()

    ## PLOTTING
    # Plot FC ACC-Pr by stim
    for i, rel in enumerate(rel_labels):

        if i == 0:
            auto_open = True
        else:
            auto_open = False

        max_fc = df_fc_avg[rel].idxmax()
        min_fc = df_fc_avg[rel].idxmin()

        fig = px.box(df_fc, x="stimFreq", y=rel,
                     title="Functional Connectivity between %s in alpha band <br>(%i simulations | %s AAL2red)" % (
                     rel, n_simulations, emp_subj),
                     labels={  # replaces default labels by column name
                         "stimFreq": "Stimulation Frequency", rel: "Functional Connectivity (PLV)"},
                     color_discrete_sequence=["dimgray"],
                     template="plotly")

        fig.add_vline(x=max_fc, line_width=0.75, line_dash="dot", line_color="orange")
        fig.add_vline(x=min_fc, line_width=0.75, line_dash="dot", line_color="darkblue")

        pio.write_html(fig, file=specific_folder + "/v2_" + emp_subj + "FC_" + rel + '-w' + str(n_simulations) + "sim.html",
                       auto_open=auto_open)

        # Plot FFT peak by stim
        fig_fft = go.Figure()

        fig_fft.add_trace(go.Scatter(x=fft_avg.stimFreq, y=fft_avg.ACC_L, name="ACC_L - ef_mag = " + str(round(weighting[34], 5))))
        fig_fft.add_trace(go.Scatter(x=fft_avg.stimFreq, y=fft_avg.ACC_R, name="ACC_R - ef_mag = " + str(round(weighting[35], 5))))
        fig_fft.add_trace(go.Scatter(x=fft_avg.stimFreq, y=fft_avg.Precuneus_L, name="Precuneus_L - ef_mag = " + str(round(weighting[70], 5))))
        fig_fft.add_trace(go.Scatter(x=fft_avg.stimFreq, y=fft_avg.Precuneus_R, name="Precuneus_R - ef_mag = " + str(round(weighting[71], 5))))

        fig_fft.update_layout(title=emp_subj + " || (g = " + str(g) + "; s = " + str(s) + "; w = " + str(round(w, 5)) + ")")
        fig_fft.update_xaxes(title="Stimulation Frequency")
        fig_fft.update_yaxes(title="Alpha peak frequency (Hz)")
        fig_fft.add_vline(x=max_fc, line_width=0.75, line_dash="dot", line_color="orange", name=rel)
        fig_fft.add_vline(x=min_fc, line_width=0.75, line_dash="dot", line_color="darkblue", name=rel)

        fig_fft.add_scatter(x=[9, 10, 11, 12, 13], y=[9, 10, 11, 12, 13], mode="lines", marker_color="gray", line=dict(width=0.5), name="Stimulation reference")

        pio.write_html(fig_fft, file=specific_folder + "/v2_" + emp_subj + "FFT_" + rel + '-w' + str(n_simulations) + "sim.html", auto_open=auto_open)




