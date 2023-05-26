
"""
Plotting the baseline behaviour of our models by simulation mode
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import scipy.signal

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px

fig_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\\Figures\\"
nmm_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\\output_NMM\\"
spk_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\\output_SPK\\"

cingulum_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                 'Insula_L', 'Insula_R',
                 'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                 'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                 'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R',
                 'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                 'Parietal_Inf_R', 'Precuneus_L', 'Precuneus_R',
                 'Thalamus_L', 'Thalamus_R']

# rois = [ 'Precuneus_L', 'Precuneus_R', 'Parietal_Sup_L', 'Parietal_Sup_R',
#        'Frontal_Mid_2_L', 'Frontal_Mid_2_R', 'Cingulate_Ant_L', 'Cingulate_Ant_R']

# Selected rois to be plotted. id_roi1 as the stimulated
id_roi1, id_roi2 = 19, 7



####          A.   Neural Mass Models       #####
# 0. Load data
simtag = "PSEmpi_nmm_stimAllConds-m04d07y2023-t10h.27m.37s"

# df_nmm = pd.read_pickle(nmm_folder + simtag + "\\nmm_results.pkl")
# df_nmm = df_nmm.astype({"mode": str, "trial": int, "node": str, "weight": float,
#                               "fex": float, "fpeak": float, "amplitude_fex": float, "amplitude_fpeak": float})
#
# df_nmm_w0 = df_nmm.loc[df_nmm["weight"] == 0]  # baseline dataframe
# df_nmm_w0["plv_mean"] = [np.average(row["plv"][row["plv"] != 1]) for i, row in df_nmm_w0.iterrows()]

# Save the subset to further loading
# df_nmm_w0.to_pickle(nmm_folder + simtag + "\\nmm_results_baselineExtract.pkl")

df_nmm_w0 = pd.read_pickle(nmm_folder + simtag + "\\nmm_results_baselineExtract.pkl")


fig = make_subplots(rows=3, cols=11, column_widths=[0.025, 0.025, 0.05, 0.025, 0.30,   0.1,   0.025, 0.025, 0.05, 0.025, 0.30], horizontal_spacing=0,
                    column_titles=["", "", "", "                     Neural Mass Models", "", "", "", "", "", "                     Spiking Neural Networks"])
                    #vertical_spacing=0.2, shared_xaxes=True)
cmap = px.colors.qualitative.Pastel
width = 0.1

# 1. Single node
df_sub = df_nmm_w0.loc[(df_nmm_w0["mode"] == "isolatedStim_oneNode_sigma0.11") & (df_nmm_w0["node"] == cingulum_rois[id_roi1])]
name = "Node"
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["fpeak"], name=cingulum_rois[id_roi1],
                        legendgroup=cingulum_rois[id_roi1], line_color=cmap[id_roi1%len(cmap)], width=width), row=1, col=2)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["amplitude_fpeak"], name=cingulum_rois[id_roi1],
                        legendgroup=cingulum_rois[id_roi1], showlegend=False, line_color=cmap[id_roi1%len(cmap)], width=width, scalemode="width"), row=2, col=2)

## dummy plot for sizing
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["fpeak"], name=cingulum_rois[id_roi1],
                        legendgroup=cingulum_rois[id_roi1], line_color=cmap[id_roi1%len(cmap)], width=width, visible=False), row=1, col=1)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["amplitude_fpeak"], name=cingulum_rois[id_roi1],
                        legendgroup=cingulum_rois[id_roi1], showlegend=False, line_color=cmap[id_roi1%len(cmap)], width=width, visible=False, scalemode="width"), row=2, col=1)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["fpeak"], name=cingulum_rois[id_roi1],
                        legendgroup=cingulum_rois[id_roi1], line_color=cmap[id_roi1%len(cmap)], width=width, visible=False), row=1, col=3)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["amplitude_fpeak"], name=cingulum_rois[id_roi1],
                        legendgroup=cingulum_rois[id_roi1], showlegend=False, line_color=cmap[id_roi1%len(cmap)], width=width, visible="legendonly", scalemode="width"), row=2, col=3)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["amplitude_fpeak"], name=cingulum_rois[id_roi1],
                        legendgroup=cingulum_rois[id_roi1], showlegend=False, line_color=cmap[id_roi1%len(cmap)], width=width, visible="legendonly", scalemode="width"), row=3, col=3)

# 2. Couple nodes
name = "Couple"
df_sub = df_nmm_w0.loc[(df_nmm_w0["mode"] == "isolatedStim_twoNodes_sigma0.11") & (df_nmm_w0["node"] == cingulum_rois[id_roi1])]
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["fpeak"], name=cingulum_rois[id_roi1],
                        legendgroup=cingulum_rois[id_roi1], showlegend=False, line_color=cmap[id_roi1%len(cmap)],
                        width=width, scalemode="width", side='negative'), row=1, col=4)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["amplitude_fpeak"], name=cingulum_rois[id_roi1],
                        legendgroup=cingulum_rois[id_roi1], showlegend=False, line_color=cmap[id_roi1%len(cmap)],
                        width=width, scalemode="width", side='negative'), row=2, col=4)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["plv_mean"], name=cingulum_rois[id_roi1],
                        legendgroup=cingulum_rois[id_roi1], showlegend=False, line_color=cmap[id_roi1%len(cmap)],
                        width=width, scalemode="width", side='negative'), row=3, col=4)

df_sub = df_nmm_w0.loc[(df_nmm_w0["mode"] == "isolatedStim_twoNodes_sigma0.11") & (df_nmm_w0["node"] == cingulum_rois[id_roi2])]
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["fpeak"], name=cingulum_rois[id_roi2],
                        legendgroup=cingulum_rois[id_roi2], showlegend=True, line_color=cmap[id_roi2%len(cmap)],
                        width=width, scalemode="width", side='positive'), row=1, col=4)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["amplitude_fpeak"], name=cingulum_rois[id_roi2],
                        legendgroup=cingulum_rois[id_roi2], showlegend=False, line_color=cmap[id_roi2%len(cmap)],
                        width=width, scalemode="width", side='positive'), row=2, col=4)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["plv_mean"], name=cingulum_rois[id_roi2],
                        legendgroup=cingulum_rois[id_roi2], showlegend=False, line_color=cmap[id_roi2%len(cmap)],
                        width=width, scalemode="width", side='positive'), row=3, col=4)

# 3. Cingulumn bundle
fname = "PSEmpi_stimWfit_cb_indiv-m03d28y2023-t16h.49m.39s"
folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\output_NMM\\"

# cargar los datos
stimWfit_nmm = pd.read_csv(folder + fname + "\\stimWmpi_results.csv")

name = "Cingulum Bundle"
for i, roi in enumerate(cingulum_rois):
    df_sub = df_nmm_w0.loc[(df_nmm_w0["mode"].isin(["isolatedStim_cb_sigma0.11", "isolatedStim_cb_antiphase_sigma0.11"])) & (df_nmm_w0["node"] == roi)]
    sl = True if roi not in [cingulum_rois[id_roi1], cingulum_rois[id_roi2]] else False
    fig.add_trace(go.Violin(x=[name] * len(df_sub), y=df_sub["fpeak"], name=roi, legendgroup=roi,
                            showlegend=sl, offsetgroup=roi,  line_color=cmap[i % len(cmap)]), row=1, col=5)
    fig.add_trace(go.Violin(x=[name] * len(df_sub), y=df_sub["amplitude_fpeak"], name=roi, legendgroup=roi,
                            showlegend=False, offsetgroup=roi, line_color=cmap[i % len(cmap)]), row=2, col=5)
    fig.add_trace(go.Violin(x=[name] * len(df_sub), y=df_sub["plv_mean"], name=roi, legendgroup=roi,
                            showlegend=False, offsetgroup=roi,  line_color=cmap[i % len(cmap)]), row=3, col=5)



####          B.    Spiking  model            ####
# 1. Single node
spk1_df = pd.read_csv(spk_folder + "one_node_tacs.txt", delimiter="\t", index_col=0)

# Focus on mode "LFP0" and under w==2; and Average out trials
df_sub = spk1_df.loc[(spk1_df["simulation"] == "lfp") & (spk1_df["weight"] == 0)]
name = "Node"
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["fpeak"], legendgroup=cingulum_rois[id_roi1],
                        showlegend=False, line_color=cmap[id_roi1%len(cmap)], width=0.3), row=1, col=8)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["amplitude_fpeak"], legendgroup=cingulum_rois[id_roi1],
                        showlegend=False, line_color=cmap[id_roi1%len(cmap)], width=width), row=2, col=8)

## Dummy plots for sizing : visible = False
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["fpeak"], legendgroup=cingulum_rois[id_roi1],
                        showlegend=False, line_color=cmap[id_roi1%len(cmap)], width=0.3, visible=False), row=1, col=7)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["amplitude_fpeak"], legendgroup=cingulum_rois[id_roi1],
                        showlegend=False, line_color=cmap[id_roi1%len(cmap)], width=width, visible=False), row=2, col=7)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["fpeak"], legendgroup=cingulum_rois[id_roi1],
                        showlegend=False, line_color=cmap[id_roi1%len(cmap)], width=0.3, visible=False), row=1, col=9)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["amplitude_fpeak"], legendgroup=cingulum_rois[id_roi1],
                        showlegend=False, line_color=cmap[id_roi1%len(cmap)], width=width, visible=False), row=2, col=9)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["amplitude_fpeak"], legendgroup=cingulum_rois[id_roi1],
                        showlegend=False, line_color=cmap[id_roi1%len(cmap)], width=width, visible=False), row=3, col=9)

# 2. Couple nodes
spk2_df = pd.read_csv(spk_folder + "two_nodes_tacs.txt", delimiter="\t", index_col=0)
df_sub = spk2_df.loc[(spk2_df["simulation"] == "lfp") & (spk2_df["weight"] == 0)]

spk_df_plv = pd.read_csv(spk_folder + "two_nodes_plv_tacs.txt", delimiter="\t", index_col=0)
df_sub_plv = spk_df_plv.loc[(spk_df_plv["simulation"] == "lfp") & (spk_df_plv["weight"] == 0)]

name = "Couple"
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["fpeak"], legendgroup=cingulum_rois[id_roi1],
                        showlegend=False, line_color=cmap[id_roi1%len(cmap)], width=width, scalemode="width", side='negative'), row=1, col=10)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["amplitude_fpeak"], legendgroup=cingulum_rois[id_roi1],
                        showlegend=False, line_color=cmap[id_roi1%len(cmap)], width=width, scalemode="width", side='negative'), row=2, col=10)
fig.add_trace(go.Violin(x=[name]*len(df_sub_plv), y=df_sub_plv["plv"], legendgroup=cingulum_rois[id_roi1],
                        showlegend=False, line_color=cmap[id_roi1%len(cmap)], width=width, scalemode="width", side='negative'), row=3, col=10)

fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["fpeak"], legendgroup=cingulum_rois[id_roi2],
                        showlegend=False, line_color=cmap[id_roi2%len(cmap)], width=width, scalemode="width", side='positive'), row=1, col=10)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["amplitude_fpeak"],  legendgroup=cingulum_rois[id_roi2],
                        showlegend=False, line_color=cmap[id_roi2%len(cmap)], width=width, scalemode="width", side='positive'), row=2, col=10)
fig.add_trace(go.Violin(x=[name]*len(df_sub_plv), y=df_sub_plv["plv"], legendgroup=cingulum_rois[id_roi2],
                        showlegend=False, line_color=cmap[id_roi2%len(cmap)], width=width, scalemode="width", side='positive'), row=3, col=10)

# 3. Cingulum bundle
stimWfit = pd.read_csv(spk_folder + "stimulation_OzCz_nodes.txt", delimiter="\t", index_col=0)
name = "Cingulum Bundle"
for i, roi in enumerate(cingulum_rois):
    df_sub = stimWfit.loc[(stimWfit["w"] != 0) & (stimWfit["subject"] == "NEMOS_035") & (stimWfit["node"] == roi)]

    fig.add_trace(go.Violin(x=[name] * len(df_sub), y=df_sub["fpeak"], legendgroup=roi, showlegend=False,
                            offsetgroup=roi,  line_color=cmap[i % len(cmap)]), row=1, col=11)
    fig.add_trace(go.Violin(x=[name] * len(df_sub), y=df_sub["amp_fpeak"], legendgroup=roi, showlegend=False,
                            offsetgroup=roi,  line_color=cmap[i % len(cmap)]), row=2, col=11)
    fig.add_trace(go.Violin(x=[name] * len(df_sub), y=df_sub["plv_mean"], name=roi, legendgroup=roi, showlegend=False,
                            offsetgroup=roi, line_color=cmap[i % len(cmap)]), row=3, col=11)

freq_range_nmm, freq_range_spk = [8, 12], [8, 12]
amp_range_nmm, amp_range_spk = [18, 110], [-5, 35]
plv_range_nmm, plv_range_spk = [0, 1], [0, 1]

fig.update_layout(violinmode="group", template="plotly_white", height=700, width=900,
                  legend=dict(orientation="h", y=-0.2),
                  yaxis1=dict(showgrid=True, title="Frequency (Hz)", range=freq_range_nmm, title_standoff=0),
                  yaxis2=dict(range=freq_range_nmm, showticklabels=False),
                  yaxis3=dict(range=freq_range_nmm, showticklabels=False),
                  yaxis4=dict(range=freq_range_nmm, showticklabels=False),
                  yaxis5=dict(range=freq_range_nmm, showticklabels=False),

                  yaxis7=dict(showgrid=True, title="Frequency (Hz)", range=freq_range_spk, title_standoff=0),
                  yaxis8=dict(range=freq_range_spk, showticklabels=False),
                  yaxis9=dict(range=freq_range_spk, showticklabels=False),
                  yaxis10=dict(range=freq_range_spk, showticklabels=False),
                  yaxis11=dict(range=freq_range_spk, showticklabels=False),

                  yaxis12=dict(showgrid=True, title="Power (dB)", range=amp_range_nmm, title_standoff=0),
                  yaxis13=dict(range=amp_range_nmm, showticklabels=False),
                  yaxis14=dict(range=amp_range_nmm, showticklabels=False),
                  yaxis15=dict(range=amp_range_nmm, showticklabels=False),
                  yaxis16=dict(range=amp_range_nmm, showticklabels=False),

                  yaxis18=dict(showgrid=True, title="Power (dB)", range=amp_range_spk, title_standoff=0),
                  yaxis19=dict(range=amp_range_spk, showticklabels=False),
                  yaxis20=dict(range=amp_range_spk, showticklabels=False),
                  yaxis21=dict(range=amp_range_spk, showticklabels=False),
                  yaxis22=dict(range=amp_range_spk, showticklabels=False),

                  yaxis25=dict(showgrid=True, title="Mean PLV", range=plv_range_nmm, title_standoff=0),
                  yaxis26=dict(range=plv_range_nmm, showticklabels=False),
                  yaxis27=dict(range=plv_range_nmm, showticklabels=False),

                  yaxis31=dict(showgrid=True, title="Mean PLV", range=plv_range_spk, title_standoff=0),
                  yaxis32=dict(range=plv_range_spk, showticklabels=False),
                  yaxis33=dict(range=plv_range_spk, showticklabels=False),)

pio.write_html(fig, fig_folder + "\\R1_baselines.html", auto_open=True)
pio.write_image(fig, fig_folder + "\\R1_baselines.svg")








