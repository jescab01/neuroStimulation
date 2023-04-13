
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



####          A.   Neural Mass Models       #####
# 0. Load data
simtag = "PSEmpi_nmm_stimAllConds-m04d03y2023-t19h.45m.21s"

df_arnold = pd.read_pickle(nmm_folder + simtag + "\\nmm_results.pkl")
df_arnold = df_arnold.astype({"mode": str, "trial": int, "node": str, "weight": float,
                              "fex": float, "fpeak": float, "amplitude_fex": float, "amplitude_fpeak": float})
df_arnold_w0 = df_arnold.loc[df_arnold["weight"] == 0]  # baseline dataframe
df_arnold_w0["plv_mean"] = [np.average(row["plv"][row["plv"] != 1]) for i, row in df_arnold_w0.iterrows()]

# TODO change all the roi namings, you are not using Precuneus R anymore for the couple simukations




fig = make_subplots(rows=3, cols=2, column_titles=["Neural Mass Models", "Spiking Neural Networks"],
                    vertical_spacing=0.2, shared_xaxes=True)
cmap = px.colors.qualitative.Plotly
width = .2
xticks = [0, 1, 5]

# 1. Single node
name="Node"
df_sub = df_arnold_w0.loc[(df_arnold_w0["mode"] == "isolatedStim_oneNode_sigma0.11") & (df_arnold_w0["node"] == "stim_Precuneus_L")]
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["fpeak"], name="Precuneus_L", legendgroup="Precuneus_L", line_color=cmap[0], width=width), row=1, col=1)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["amplitude_fpeak"], name="Precuneus_L", legendgroup="Precuneus_L", showlegend=False, line_color=cmap[0], width=width), row=2, col=1)

# 2. Couple nodes
name = "Couple"
df_sub = df_arnold_w0.loc[(df_arnold_w0["mode"] == "isolatedStim_twoNodes_sigma0.11") & (df_arnold_w0["node"] == "stim_Precuneus_L")]
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["fpeak"], name="Precuneus_L", legendgroup="Precuneus_L", showlegend=False, line_color=cmap[0], width=width, side='negative'), row=1, col=1)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["amplitude_fpeak"], name="Precuneus_L", legendgroup="Precuneus_L", showlegend=False, line_color=cmap[0], width=width, side='negative'), row=2, col=1)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["plv_mean"], name="Precuneus_L", legendgroup="Precuneus_L", showlegend=False, line_color=cmap[0], width=width, side='negative'), row=3, col=1)

df_sub = df_arnold_w0.loc[(df_arnold_w0["mode"] == "isolatedStim_twoNodes_sigma0.11") & (df_arnold_w0["node"] == "Precuneus_R")]
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["fpeak"], name="Precuneus_R", legendgroup="Precuneus_R", showlegend=True, line_color=cmap[1], width=width, side='positive'), row=1, col=1)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["amplitude_fpeak"], name="Precuneus_R", legendgroup="Precuneus_R", showlegend=False, line_color=cmap[1], width=width, side='positive'), row=2, col=1)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["plv_mean"], name="Precuneus_R", legendgroup="Precuneus_R", showlegend=False, line_color=cmap[1], width=width, side='positive'), row=3, col=1)

# 3. Cingulumn bundle
for i, roi in enumerate(cingulum_rois):
    df_sub = df_arnold_w0.loc[(df_arnold_w0["mode"] == "isolatedStim_cb_sigma0.11") & (df_arnold_w0["node"] == roi)]
    sl = True if roi not in ["Precuneus_L", "Precuneus_R"] else False
    fig.add_trace(go.Violin(x=[name+"a"] * len(df_sub), y=df_sub["fpeak"], name=roi, legendgroup=roi, showlegend=sl, offsetgroup=roi, line_color=cmap[i % len(cmap)]), row=1, col=1)
    fig.add_trace(go.Violin(x=[name+"b"] * len(df_sub), y=df_sub["amplitude_fpeak"],  showlegend=False, offsetgroup=roi, line_color=cmap[i % len(cmap)]), row=2, col=1)
    fig.add_trace(go.Violin(x=[name+"c"] * len(df_sub), y=df_sub["plv_mean"], showlegend=False, offsetgroup=roi, line_color=cmap[i % len(cmap)]), row=3, col=1)


####          B.    Spiking  model            ####
# 1. Single node
spk_df = pd.read_csv(spk_folder + "one_node_tacs.txt", delimiter="\t", index_col=0)

# Focus on mode "LFP0" and under w==2; and Average out trials
df_sub = spk_df.loc[(spk_df["simulation"] == "lfp") & (spk_df["mode"] == 0) & (spk_df["weight"] == 0)]

fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["fpeak"], legendgroup="Precuneus_L", showlegend=False, line_color=cmap[0], width=width), row=1, col=2)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["amplitude_fpeak"], legendgroup="Precuneus_L", showlegend=False, line_color=cmap[0], width=width), row=2, col=2)

# 2. Couple nodes
spk_df = pd.read_csv(spk_folder + "two_nodes_tacs.txt", delimiter="\t", index_col=0)
df_sub = spk_df.loc[(spk_df["simulation"] == "lfp") & (spk_df["weight"] == 0)]

spk_df_plv = pd.read_csv(spk_folder + "two_nodes_plv_tacs.txt", delimiter="\t", index_col=0)
df_sub_plv = spk_df_plv.loc[(spk_df_plv["simulation"] == "lfp") & (spk_df_plv["weight"] == 0)]


fig.add_trace(go.Violin(x=[xticks[1]]*len(df_sub), y=df_sub["fpeak"], legendgroup="Precuneus_L", showlegend=False, line_color=cmap[0], width=width, side='negative'), row=1, col=2)
fig.add_trace(go.Violin(x=[xticks[1]]*len(df_sub), y=df_sub["amplitude_fpeak"], legendgroup="Precuneus_L", showlegend=False, line_color=cmap[0], width=width, side='negative'), row=2, col=2)
fig.add_trace(go.Violin(x=[xticks[1]]*len(df_sub_plv), y=df_sub_plv["plv"], legendgroup="Precuneus_L", showlegend=False, line_color=cmap[0], width=width, side='negative'), row=3, col=2)

fig.add_trace(go.Violin(x=[xticks[1]]*len(df_sub), y=df_sub["fpeak"], legendgroup="Precuneus_R", showlegend=False, line_color=cmap[1], width=width, side='positive'), row=1, col=2)
fig.add_trace(go.Violin(x=[xticks[1]]*len(df_sub), y=df_sub["amplitude_fpeak"],  legendgroup="Precuneus_R", showlegend=False, line_color=cmap[1], width=width, side='positive'), row=2, col=2)
fig.add_trace(go.Violin(x=[xticks[1]]*len(df_sub_plv), y=df_sub_plv["plv"], legendgroup="Precuneus_R", showlegend=False, line_color=cmap[1], width=width, side='positive'), row=3, col=2)

# 3. Cingulum bundle
stimWfit = pd.read_csv(spk_folder + "stimulation_OzCz_nodes.txt", delimiter="\t", index_col=0)

for i, roi in enumerate(cingulum_rois):
    df_sub = stimWfit.loc[(stimWfit["w"] == 0) & (stimWfit["subject"] == "NEMOS_035") & (stimWfit["node"] == i)]

    fig.add_trace(go.Violin(x=[name] * len(df_sub), y=df_sub["fpeak"], name=roi, legendgroup=roi, showlegend=False, offsetgroup=roi+"spkfpeak", line_color=cmap[i % len(cmap)]), row=1, col=2)
    fig.add_trace(go.Violin(x=[name] * len(df_sub), y=df_sub["amp_fpeak"], name=roi, legendgroup=roi, showlegend=False, offsetgroup=roi+"spkfamp", line_color=cmap[i % len(cmap)]), row=2, col=2)
    # fig.add_trace(go.Violin(x=[name] * len(df_sub), y=df_sub["plv_mean"], name=roi, legendgroup=roi, showlegend=False, offsetgroup=roi+"spkplv", line_color=cmap[i % len(cmap)]), row=3, col=1)

fig.update_layout(violinmode="group", template="plotly_white", #height=500, width=800,
                  legend=dict(orientation="h", y=-0.2), yaxis1=dict(title="Frequency (Hz)"),
                  yaxis3=dict(title="Power (dB)"), yaxis5=dict(title="PLV"),
                  xaxis1=dict(tickmode="array", tickvals=xticks, ticktext=["Node", "Couple", "Cingulum Bundle"]),
                  xaxis2=dict(tickmode="array", tickvals=xticks, ticktext=["Node", "Couple", "Cingulum Bundle"]),
                  xaxis3=dict(tickmode="array", tickvals=xticks, ticktext=["Node", "Couple", "Cingulum Bundle"]),
                  xaxis4=dict(tickmode="array", tickvals=xticks, ticktext=["Node", "Couple", "Cingulum Bundle"]),
                  xaxis5=dict(tickmode="array", tickvals=xticks, ticktext=["Node", "Couple", "Cingulum Bundle"]),
                  xaxis6=dict(tickmode="array", tickvals=xticks, ticktext=["Node", "Couple", "Cingulum Bundle"]),
                  )


pio.write_html(fig, fig_folder + "\\R1_baselines.html", auto_open=True)



pio.write_image(fig, fig_folder + "\\R1_baselines.svg")








