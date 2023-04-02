
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


##########          Neural Mass Models          ###########

## Load data

main_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\\R1_IsolatedStim\\"

simtag = "PSEmpi_baselines_stimAllConds-m04d02y2023-t00h.49m.15s"
df_base = pd.read_pickle(main_folder + "R1c_baselines\\" + simtag + "\\nmm_results.pkl")

df_base["plv_mean"] = [np.average(row["plv"][row["plv"]!=1]) for i, row in df_base.iterrows()]

# TODO select a set of rois that are progressively disconnected (sc1>sc2>sc3...)
rois = [ 'Precuneus_L', 'Precuneus_R','Parietal_Sup_L','Parietal_Sup_R',
       'Frontal_Mid_2_L', 'Frontal_Mid_2_R', 'Cingulate_Ant_L', 'Cingulate_Ant_R']

fig = make_subplots(rows=3, cols=2, column_titles=["Neural Mass Models", "Spiking Neural Networks"], vertical_spacing=0.2)
cmap = px.colors.qualitative.Plotly
width=0.2

# 1. Single node
df_sub = df_base.loc[(df_base["mode"] == "isolatedStim_oneNode_sigma0.11") & (df_base["node"] == "Precuneus_L")]
name = "Node"
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["fpeak"], name="Precuneus_L", legendgroup="Precuneus_L", line_color=cmap[0], width=width), row=1, col=1)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["amplitude_fpeak"], name="Precuneus_L", legendgroup="Precuneus_L", showlegend=False, line_color=cmap[0], width=width), row=2, col=1)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["plv_mean"], name="Precuneus_L", legendgroup="Precuneus_L", showlegend=False, line_color=cmap[0], width=width), row=3, col=1)

# 2. Couple nodes
name="Couple"
df_sub = df_base.loc[(df_base["mode"] == "isolatedStim_twoNodes_sigma0.11") & (df_base["node"] == "Precuneus_L")]
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["fpeak"], name="Precuneus_L", legendgroup="Precuneus_L", showlegend=False, line_color=cmap[0], width=width, side='negative'), row=1, col=1)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["amplitude_fpeak"], name="Precuneus_L", legendgroup="Precuneus_L", showlegend=False, line_color=cmap[0], width=width, side='negative'), row=2, col=1)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["plv_mean"], name="Precuneus_L", legendgroup="Precuneus_L", showlegend=False, line_color=cmap[0], width=width, side='negative'), row=3, col=1)
df_sub = df_base.loc[(df_base["mode"] == "isolatedStim_twoNodes_sigma0.11") & (df_base["node"] == "Precuneus_R")]
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["fpeak"], name="Precuneus_R", legendgroup="Precuneus_R", showlegend=True, line_color=cmap[1], width=width, side='positive'), row=1, col=1)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["amplitude_fpeak"], name="Precuneus_R", legendgroup="Precuneus_R", showlegend=False, line_color=cmap[1], width=width, side='positive'), row=2, col=1)
fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["plv_mean"], name="Precuneus_R", legendgroup="Precuneus_R", showlegend=False, line_color=cmap[1], width=width, side='positive'), row=3, col=1)

# 3. Cingulumn bundle
name = "Cingulum Bundle"
for i, roi in enumerate(rois):
    df_sub = df_base.loc[(df_base["mode"] == "isolatedStim_cb_sigma0.11") & (df_base["node"] == roi)]
    if roi in ["Precuneus_L", "Precuneus_R"]:
        fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["fpeak"], name=roi, legendgroup=roi, showlegend=False, line_color=cmap[i%len(cmap)]), row=1, col=1)
        fig.add_trace(go.Violin(x=[name] * len(df_sub), y=df_sub["amplitude_fpeak"], name=roi, legendgroup=roi, showlegend=False, line_color=cmap[i%len(cmap)]), row=2, col=1)
        fig.add_trace(go.Violin(x=[name] * len(df_sub), y=df_sub["plv_mean"], name=roi, legendgroup=roi, showlegend=False, line_color=cmap[i%len(cmap)]), row=3, col=1)
    else:
        fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["fpeak"], name=roi, legendgroup=roi, showlegend=True, line_color=cmap[i%len(cmap)]), row=1, col=1)
        fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["amplitude_fpeak"], name=roi, legendgroup=roi, showlegend=False, line_color=cmap[i%len(cmap)]), row=2, col=1)
        fig.add_trace(go.Violin(x=[name]*len(df_sub), y=df_sub["plv_mean"], name=roi, legendgroup=roi, showlegend=False, line_color=cmap[i%len(cmap)]), row=3, col=1)


fig.update_layout(violinmode="group", template="plotly_white", legend=dict(orientation="h", y=-0.2),yaxis1=dict(title="Frequency (Hz)"),
                  yaxis3=dict(title="Power (dB)"), yaxis5=dict(title="PLV"))


###  TODO spiking ###


pio.write_html(fig, main_folder + "\\R1c_baselines.html", auto_open=True)
pio.write_image(fig, main_folder + "\\R1c_baselines.svg")








