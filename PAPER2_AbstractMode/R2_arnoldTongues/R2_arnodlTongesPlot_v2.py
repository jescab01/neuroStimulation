
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pickle
import pandas as pd
import numpy as np
import scipy.signal

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio


fig_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\Figures\\"
nmm_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\output_NMM\\"
spk_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\output_SPK\\"


# Load structural data to work with regions connectivity
cingulum_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                 'Insula_L', 'Insula_R',
                 'Cingulate_Ant_L', 'Cingulate_Ant_R',
                 'Cingulate_Post_L', 'Cingulate_Post_R', #7
                 'Hippocampus_L', 'Hippocampus_R',
                 'ParaHippocampal_L', 'ParaHippocampal_R',
                 'Amygdala_L', 'Amygdala_R', #13
                 'Parietal_Sup_L', 'Parietal_Sup_R',
                 'Parietal_Inf_L', 'Parietal_Inf_R',
                 'Precuneus_L', 'Precuneus_R',
                 'Thalamus_L', 'Thalamus_R']


####           NMM          ######

# 0a. Load data
simtag = "PSEmpi_nmm_stimAllConds-m04d07y2023-t10h.27m.37s"
df_arnold = pd.read_pickle(nmm_folder + simtag + "\\nmm_results.pkl")

df_arnold = df_arnold.astype({"mode": str, "trial": int, "node": str, "weight": float,
                              "fex": float, "fpeak": float, "amplitude_fex": float, "amplitude_fpeak": float})

rois = ["Precuneus_R", "Cingulate_Post_R", "Frontal_Mid_2_R"]
# rois = ["Precuneus_L", "Precuneus_R", "Frontal_Mid_2_R"]

rois_ids = [cingulum_rois.index(roi) for roi in rois]

df_arnold["plv_Roi1Roi2"], df_arnold["plv_Roi1Roi3"], df_arnold["plv_Roi2Roi3"] = np.nan, np.nan, np.nan
plv_Roi1Roi2, plv_Roi1Roi3, plv_Roi2Roi3 = [], [], []
for i, row in df_arnold.loc[df_arnold["node"].isin(rois)].iterrows():

    if len(row["plv"]) == 2:
        plv_Roi1Roi2.append(row["plv"][row["plv"] != 1][0])
        plv_Roi1Roi3.append(np.nan)
        plv_Roi2Roi3.append(np.nan)

    elif (len(row["plv"]) > 3) & (row["node"] == rois[0]):
        plv_Roi1Roi2.append(row["plv"][rois_ids[1]])
        plv_Roi1Roi3.append(row["plv"][rois_ids[2]])
        plv_Roi2Roi3.append(np.nan)

    elif (len(row["plv"]) > 3) & (row["node"] == rois[1]):
        plv_Roi1Roi2.append(row["plv"][rois_ids[0]])
        plv_Roi1Roi3.append(np.nan)
        plv_Roi2Roi3.append(row["plv"][rois_ids[2]])

    elif (len(row["plv"]) > 3) & (row["node"] == rois[2]):
        plv_Roi1Roi2.append(np.nan)
        plv_Roi1Roi3.append(row["plv"][rois_ids[0]])
        plv_Roi2Roi3.append(row["plv"][rois_ids[1]])

df_arnold["plv_Roi1Roi2"].loc[df_arnold["node"].isin(rois)] = plv_Roi1Roi2
df_arnold["plv_Roi1Roi3"].loc[df_arnold["node"].isin(rois)] = plv_Roi1Roi3
df_arnold["plv_Roi2Roi3"].loc[df_arnold["node"].isin(rois)] = plv_Roi2Roi3



df_arnold_avg = df_arnold.loc[df_arnold["weight"] <= 100].groupby(["mode", "weight", "fex", "node"]).mean().reset_index()  # average out repetitions

measures = ["fpeak", "amplitude_fpeak", "plv_Roi1Roi2", "plv_Roi1Roi3", "plv_Roi2Roi3"]

# # 0b. Pre-allocate relative changes and compute them
# if "reldiv" in measures[0]:
#     # With division
#     df_arnold_avg["amplitude_fex_reldiv"], df_arnold_avg["amplitude_fpeak_reldiv"], df_arnold_avg["fpeak_reldiv"] = 0, 0, 0
#
#     for mode in list(set(df_arnold_avg["mode"].values)):
#         for node in list(set(df_arnold_avg["node"].values)):
#             df_arnold_avg["fpeak_reldiv"].loc[(df_arnold_avg["mode"] == mode) & (df_arnold_avg["node"] == node)] = \
#                 df_arnold_avg["fpeak"].loc[(df_arnold_avg["mode"] == mode) & (df_arnold_avg["node"] == node)] / df_arnold_avg["fpeak"].loc[(df_arnold_avg["weight"] == 0) & (df_arnold_avg["mode"] == mode) & (df_arnold_avg["node"] == node)].mean()
#
#             df_arnold_avg["amplitude_fex_reldiv"].loc[(df_arnold_avg["mode"] == mode) & (df_arnold_avg["node"] == node)] = \
#                 df_arnold_avg["amplitude_fex"].loc[(df_arnold_avg["mode"] == mode) & (df_arnold_avg["node"] == node)] / df_arnold_avg["amplitude_fex"].loc[(df_arnold_avg["weight"] == 0) & (df_arnold_avg["mode"] == mode) & (df_arnold_avg["node"] == node)].mean()
#
#             df_arnold_avg["amplitude_fpeak_reldiv"].loc[(df_arnold_avg["mode"] == mode) & (df_arnold_avg["node"] == node)] = \
#                 df_arnold_avg["amplitude_fpeak"].loc[(df_arnold_avg["mode"] == mode) & (df_arnold_avg["node"] == node)] / df_arnold_avg["amplitude_fpeak"].loc[(df_arnold_avg["weight"] == 0) & (df_arnold_avg["mode"] == mode) & (df_arnold_avg["node"] == node)].mean()
# # With resta
# elif "relrest" in measures[0]:
#     df_arnold_avg["amplitude_fex_relrest"], df_arnold_avg["amplitude_fpeak_relrest"], df_arnold_avg["fpeak_relrest"] = 0, 0, 0
#     for mode in list(set(df_arnold_avg["mode"].values)):
#         for node in list(set(df_arnold_avg["node"].values)):
#             df_arnold_avg["fpeak_relrest"].loc[(df_arnold_avg["mode"] == mode) & (df_arnold_avg["node"] == node)] = \
#                 df_arnold_avg["fpeak"].loc[(df_arnold_avg["mode"] == mode) & (df_arnold_avg["node"] == node)] - \
#                 df_arnold_avg["fpeak"].loc[(df_arnold_avg["weight"] == 0) & (df_arnold_avg["mode"] == mode) & (
#                             df_arnold_avg["node"] == node)].mean()
#
#             df_arnold_avg["amplitude_fex_relrest"].loc[(df_arnold_avg["mode"] == mode) & (df_arnold_avg["node"] == node)] = \
#                 df_arnold_avg["amplitude_fex"].loc[(df_arnold_avg["mode"] == mode) & (df_arnold_avg["node"] == node)] - \
#                 df_arnold_avg["amplitude_fex"].loc[(df_arnold_avg["weight"] == 0) & (df_arnold_avg["mode"] == mode) & (
#                             df_arnold_avg["node"] == node)].mean()
#
#             df_arnold_avg["amplitude_fpeak_relrest"].loc[
#                 (df_arnold_avg["mode"] == mode) & (df_arnold_avg["node"] == node)] = \
#                 df_arnold_avg["amplitude_fpeak"].loc[
#                     (df_arnold_avg["mode"] == mode) & (df_arnold_avg["node"] == node)] - \
#                 df_arnold_avg["amplitude_fpeak"].loc[
#                     (df_arnold_avg["weight"] == 0) & (df_arnold_avg["mode"] == mode) & (
#                                 df_arnold_avg["node"] == node)].mean()

freq_min, freq_max, freq_colorscale = df_arnold_avg[measures[0]].min(), df_arnold_avg[measures[0]].max(), px.colors.diverging.balance
pow_min, pow_max, pow_colorscale = df_arnold_avg[measures[1]].min(), df_arnold_avg[measures[1]].max(), px.colors.sequential.Sunsetdark
plv_min, plv_max, plv_colorscale = 0, 1, px.colors.sequential.Darkmint

colormode = "general"  # general (all plots same colorscale); rowspecific (each row its own colorscale); individual.


###       PLOTTING       #####
sp_titles = ["             "+rois[0]+"<br>Frequency", "Power"] + [""] * 9 +\
          ["             "+rois[1]+"<br>Frequency", "Power", "FC"] + [""] * 9 +\
          ["            "+rois[2]+"<br>Frequency", "Power", "             FC"]
fig = make_subplots(rows=3, cols=9, shared_yaxes=True, subplot_titles=sp_titles, vertical_spacing=0.05,
                    x_title="Stimulation Frequency (Hz)", y_title="Stimulation Weight")

## 1. arnold tongue for the single node stimulation
df_sub = df_arnold_avg.loc[(df_arnold_avg["mode"] == "isolatedStim_oneNode_sigma0.11") & (df_arnold_avg["node"] == rois[0])]

freq_colorbar = dict(title="Hz", thickness=10, len=0.22, y=0.98, x=0.27)
pow_colorbar = dict(title="dB", thickness=10, len=0.22, y=0.98, x=0.33)
plv_colorbar = dict(title="PLV", thickness=10, len=0.22, y=0.98, x=0.39)
sl = True

if colormode == "rowspecific":
    freq_min, freq_max = df_sub[measures[0]].min(), df_sub[measures[0]].max()
    pow_min, pow_max = df_sub[measures[1]].min(), df_sub[measures[1]].max()
    plv_min, plv_max = df_sub[measures[2]].min(), df_sub[measures[2]].max()

fig.add_trace(go.Heatmap(z=df_sub[measures[0]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=freq_colorscale, zmin=freq_min, zmax=freq_max,
                         colorbar=freq_colorbar), row=1, col=1)

fig.add_trace(go.Heatmap(z=df_sub[measures[1]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=pow_colorscale, zmin=pow_min, zmax=pow_max,
                         colorbar=pow_colorbar), row=1, col=2)
# # PLV
# fig.add_trace(go.Heatmap(z=df_sub[measures[2]], x=df_sub.fex, y=df_sub.weight,
#                          showscale=sl, colorscale=plv_colorscale, zmin=plv_min, zmax=plv_max,
#                          colorbar=plv_colorbar), row=1, col=3)

## 2. arnold tongue for the Two nodes stimulation
# 2.1 Precuneus L (stimulated)
df_sub = df_arnold_avg.loc[(df_arnold_avg["mode"] == "isolatedStim_twoNodes_sigma0.11")]
if colormode == "rowspecific":
    freq_min, freq_max = df_sub[measures[0]].min(), df_sub[measures[0]].max()
    pow_min, pow_max = df_sub[measures[1]].min(), df_sub[measures[1]].max()
    plv_min, plv_max = df_sub[measures[2]].min(), df_sub[measures[2]].max()
    freq_colorbar = dict(title="Hz", thickness=6, len=0.23, y=0.63, x=0.56)
    pow_colorbar = dict(title="dB", thickness=6, len=0.23, y=0.63, x=0.60)
    plv_colorbar = dict(title="PLV", thickness=6, len=0.23, y=0.63, x=0.64)
    sl = True
else:
    False

df_sub = df_arnold_avg.loc[(df_arnold_avg["mode"] == "isolatedStim_twoNodes_sigma0.11") & (df_arnold_avg["node"] == rois[0])]
fig.add_trace(go.Heatmap(z=df_sub[measures[0]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=freq_colorscale, zmin=freq_min, zmax=freq_max,
                         colorbar=freq_colorbar), row=2, col=1)
fig.add_trace(go.Heatmap(z=df_sub[measures[1]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=pow_colorscale, zmin=pow_min, zmax=pow_max,
                         colorbar=pow_colorbar), row=2, col=2)

# 2.2 Precuneus R
df_sub = df_arnold_avg.loc[(df_arnold_avg["mode"] == "isolatedStim_twoNodes_sigma0.11") & (df_arnold_avg["node"] == rois[1])]
sl = False
fig.add_trace(go.Heatmap(z=df_sub[measures[0]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=freq_colorscale, zmin=freq_min, zmax=freq_max,
                         colorbar=freq_colorbar), row=2, col=3)
fig.add_trace(go.Heatmap(z=df_sub[measures[1]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=pow_colorscale, zmin=pow_min, zmax=pow_max,
                         colorbar=pow_colorbar), row=2, col=4)
# 2.3 plv
fig.add_trace(go.Heatmap(z=df_sub[measures[2]], x=df_sub.fex, y=df_sub.weight,
                         showscale=True, colorscale=plv_colorscale, zmin=plv_min, zmax=plv_max,
                         colorbar=plv_colorbar), row=2, col=5)

## 3. arnold tongue for the Cingulum bundle
# 3.1 Precuneus L (stimulated)

df_sub = df_arnold_avg.loc[(df_arnold_avg["mode"] == "isolatedStim_cb_sigma0.11")]

if colormode == "rowspecific":
    freq_min, freq_max = df_sub[measures[0]].min(), df_sub[measures[0]].max()
    pow_min, pow_max = df_sub[measures[1]].min(), df_sub[measures[1]].max()
    plv_min, plv_max = 0, 1
    freq_colorbar = dict(title="Hz", thickness=6, len=0.47, y=0.22, x=1.01)
    pow_colorbar = dict(title="dB", thickness=6, len=0.47, y=0.22, x=1.05)
    plv_colorbar = dict(title="PLV", thickness=6, len=0.47, y=0.22, x=1.09)
    sl = True

df_sub = df_arnold_avg.loc[(df_arnold_avg["mode"] == "isolatedStim_cb_sigma0.11") & (df_arnold_avg["node"] == rois[0])]

fig.add_trace(go.Heatmap(z=df_sub[measures[0]], x=df_sub.fex, y=df_sub.weight,
                         showscale=True, colorscale=freq_colorscale, zmin=freq_min, zmax=freq_max,
                         colorbar=freq_colorbar), row=3, col=1)
fig.add_trace(go.Heatmap(z=df_sub[measures[1]], x=df_sub.fex, y=df_sub.weight,
                         showscale=True, colorscale=pow_colorscale, zmin=pow_min, zmax=pow_max,
                         colorbar=pow_colorbar), row=3, col=2)
# 3.2 Precuneus R
df_sub = df_arnold_avg.loc[(df_arnold_avg["mode"] == "isolatedStim_cb_sigma0.11") & (df_arnold_avg["node"] == rois[1])]

sl = False


fig.add_trace(go.Heatmap(z=df_sub[measures[0]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=freq_colorscale, zmin=freq_min, zmax=freq_max,
                         colorbar=freq_colorbar), row=3, col=3)
fig.add_trace(go.Heatmap(z=df_sub[measures[1]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=pow_colorscale, zmin=pow_min, zmax=pow_max,
                         colorbar=pow_colorbar), row=3, col=4)
# PLV
fig.add_trace(go.Heatmap(z=df_sub[measures[2]], x=df_sub.fex, y=df_sub.weight,
                         showscale=True, colorscale=plv_colorscale, zmin=plv_min, zmax=plv_max,
                         colorbar=plv_colorbar), row=3, col=5)

# 3.3 Frontal Mid L
df_sub = df_arnold_avg.loc[(df_arnold_avg["mode"] == "isolatedStim_cb_sigma0.11") & (df_arnold_avg["node"] ==  rois[2])]

sl = False

fig.add_trace(go.Heatmap(z=df_sub[measures[0]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=freq_colorscale, zmin=freq_min, zmax=freq_max,
                         colorbar=freq_colorbar), row=3, col=6)
fig.add_trace(go.Heatmap(z=df_sub[measures[1]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=pow_colorscale, zmin=pow_min, zmax=pow_max,
                         colorbar=pow_colorbar), row=3, col=7)
# PLV
fig.add_trace(go.Heatmap(z=df_sub[measures[3]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=plv_colorscale, zmin=plv_min, zmax=plv_max,
                         colorbar=plv_colorbar), row=3, col=8)
fig.add_trace(go.Heatmap(z=df_sub[measures[4]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=plv_colorscale, zmin=plv_min, zmax=plv_max,
                         colorbar=plv_colorbar), row=3, col=9)

# # 4. Arnold tongue for the Cingulum bundle with antiphase stimulation
# # 4.1 Precuneus L (Stimulated)
# df_sub = df_arnold_avg.loc[(df_arnold_avg["mode"] == "isolatedStim_cb_antiphase_sigma0.11") & (df_arnold_avg["node"] == rois[0])]
#
# fig.add_trace(go.Heatmap(z=df_sub[measures[0]], x=df_sub.fex, y=df_sub.weight,
#                          showscale=sl, colorscale=freq_colorscale, zmin=freq_min, zmax=freq_max,
#                          colorbar=freq_colorbar), row=4, col=1)
# fig.add_trace(go.Heatmap(z=df_sub[measures[1]], x=df_sub.fex, y=df_sub.weight,
#                          showscale=sl, colorscale=pow_colorscale, zmin=pow_min, zmax=pow_max,
#                          colorbar=pow_colorbar), row=4, col=2)
#
# # 4.2 Precuneus R (antiphase stimulation)
# df_sub = df_arnold_avg.loc[(df_arnold_avg["mode"] == "isolatedStim_cb_antiphase_sigma0.11") & (df_arnold_avg["node"] == rois[1])]
#
# sl = False
# fig.add_trace(go.Heatmap(z=df_sub[measures[0]], x=df_sub.fex, y=df_sub.weight,
#                          showscale=sl, colorscale=freq_colorscale, zmin=freq_min, zmax=freq_max,
#                          colorbar=freq_colorbar), row=4, col=3)
# fig.add_trace(go.Heatmap(z=df_sub[measures[1]], x=df_sub.fex, y=df_sub.weight,
#                          showscale=sl, colorscale=pow_colorscale, zmin=pow_min, zmax=pow_max,
#                          colorbar=pow_colorbar), row=4, col=4)
# # PLV
# fig.add_trace(go.Heatmap(z=df_sub[measures[2]], x=df_sub.fex, y=df_sub.weight,
#                          showscale=sl, colorscale=plv_colorscale, zmin=plv_min, zmax=plv_max,
#                          colorbar=plv_colorbar), row=4, col=5)
#
# # 4.3 Frontal Mid L
# df_sub = df_arnold_avg.loc[(df_arnold_avg["mode"] == "isolatedStim_cb_antiphase_sigma0.11") & (df_arnold_avg["node"] ==  rois[2])]
#
# sl = False
#
# fig.add_trace(go.Heatmap(z=df_sub[measures[0]], x=df_sub.fex, y=df_sub.weight,
#                          showscale=sl, colorscale=freq_colorscale, zmin=freq_min, zmax=freq_max,
#                          colorbar=freq_colorbar), row=4, col=6)
# fig.add_trace(go.Heatmap(z=df_sub[measures[1]], x=df_sub.fex, y=df_sub.weight,
#                          showscale=sl, colorscale=pow_colorscale, zmin=pow_min, zmax=pow_max,
#                          colorbar=pow_colorbar), row=4, col=7)
# # PLV
# fig.add_trace(go.Heatmap(z=df_sub[measures[3]], x=df_sub.fex, y=df_sub.weight,
#                          showscale=sl, colorscale=plv_colorscale, zmin=plv_min, zmax=plv_max,
#                          colorbar=plv_colorbar), row=4, col=8)
# fig.add_trace(go.Heatmap(z=df_sub[measures[4]], x=df_sub.fex, y=df_sub.weight,
#                          showscale=sl, colorscale=plv_colorscale, zmin=plv_min, zmax=plv_max,
#                          colorbar=plv_colorbar), row=4, col=9)

fig.update_layout(height=500, width=900)
pio.write_html(fig, fig_folder + "R2_NMM_arnoldTongues.html", auto_open=True)
pio.write_image(fig, fig_folder + "R2_NMM_arnoldTongues.svg")








####             SPIKING           ####
spk1_df = pd.read_csv(spk_folder + "one_node_tacs.txt", delimiter="\t", index_col=0)
spk1_df["mode"] = "lfp_single"
spk1_df_avg = spk1_df.loc[spk1_df["simulation"]=="lfp"].groupby(["mode", "node", "weight", "fex"]).mean().reset_index().iloc[:, [0,1,2,3,5,6,7]]
colnames = spk1_df_avg.columns

spk2_df = pd.read_csv(spk_folder + "two_nodes_tacs.txt", delimiter="\t", index_col=0)
spk2_df["mode"] = "lfp_couple"
spk2_df_avg = spk2_df.loc[spk2_df["simulation"]=="lfp"].groupby(["mode", "node", "weight", "fex"]).mean().reset_index().iloc[:, [0,1,2,3,5,6,7]]


spk_cb_df = pd.read_csv(spk_folder + "stimulation_OzCz_precuneus_nodes.txt", delimiter="\t", index_col=0)
spk_cb_df["mode"] = "lfp_cb"
spk_cb_df_avg = spk_cb_df.groupby(["mode", "node", "w", "fex"]).mean().reset_index().iloc[:, [0,1,2,3,5,6,7]]
spk_cb_df_avg.columns = colnames

spk_cb_anti_df = pd.read_csv(spk_folder + "stimulation_OzCz_precuneus_antiphase_nodes.txt", delimiter="\t", index_col=0)
spk_cb_anti_df["mode"] = "lfp_cb_anti"
spk_cb_anti_df_avg = spk_cb_anti_df.groupby(["mode", "node", "w", "fex"]).mean().reset_index().iloc[:, [0,1,2,3,5,6,7]]
spk_cb_anti_df_avg.columns = colnames

spk_df_avg = pd.concat([spk1_df_avg, spk2_df_avg, spk_cb_df_avg, spk_cb_anti_df_avg])


# PLV dfs
spk_cbplv_df_avg = spk_cb_df.groupby(["mode", "node", "w", "fex"]).mean().reset_index().iloc[:, [0,1,2,3,5,6,7,10,11]]
spk_cbplv_anti_df_avg = spk_cb_anti_df.groupby(["mode", "node", "w", "fex"]).mean().reset_index().iloc[:, [0,1,2,3,5,6,7,10,11]]

spk2plv_df = pd.read_csv(spk_folder + "two_nodes_plv_tacs.txt", delimiter="\t", index_col=0)
spk2plv_df_avg = spk2plv_df.loc[spk2plv_df["simulation"]=="lfp"].groupby(["weight", "fex"]).mean().reset_index()



freq_min, freq_max, freq_colorscale = spk_df_avg[measures[0]].min(), spk_df_avg[measures[0]].max(), px.colors.diverging.balance
pow_min, pow_max, pow_colorscale = spk_df_avg[measures[1]].min(), spk_df_avg[measures[1]].max(), px.colors.sequential.Sunsetdark
plv_min, plv_max, plv_colorscale = 0, 1, px.colors.sequential.Darkmint


sp_titles = ["             "+rois[0]+"<br>Frequency", "Power"] + [""] * 9 +\
          ["             "+rois[1]+"<br>Frequency", "Power", "FC"] + [""] * 9 +\
          ["            "+rois[2]+"<br>Frequency", "Power", "             FC"]
fig = make_subplots(rows=4, cols=9, shared_yaxes=True, subplot_titles=sp_titles, vertical_spacing=0.05,
                    x_title="Stimulation Frequency (Hz)", y_title="Stimulation Weight")

measures = ["fpeak", "amplitude_fpeak", "plv"]

## 1. ONE NODE

df_sub = spk_df_avg.loc[(spk_df_avg["mode"] == "lfp_single")]

freq_colorbar = dict(title="Hz", thickness=10, len=0.22, y=0.98, x=0.27)
pow_colorbar = dict(title="dB", thickness=10, len=0.22, y=0.98, x=0.33)
plv_colorbar = dict(title="PLV", thickness=10, len=0.22, y=0.98, x=0.39)
sl = True

fig.add_trace(go.Heatmap(z=df_sub[measures[0]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=freq_colorscale, zmin=freq_min, zmax=freq_max,
                         colorbar=freq_colorbar), row=1, col=1)

fig.add_trace(go.Heatmap(z=df_sub[measures[1]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=pow_colorscale, zmin=pow_min, zmax=pow_max,
                         colorbar=pow_colorbar), row=1, col=2)

## 2. arnold tongue for the TWO NODES stimulation

# Average out trials
df_sub = spk_df_avg.loc[(spk_df_avg["mode"] == "lfp_couple") & (spk_df_avg["node"] == rois[0])]

fig.add_trace(go.Heatmap(z=df_sub[measures[0]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=freq_colorscale, zmin=freq_min, zmax=freq_max,
                         colorbar=freq_colorbar), row=2, col=1)
fig.add_trace(go.Heatmap(z=df_sub[measures[1]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=pow_colorscale, zmin=pow_min, zmax=pow_max,
                         colorbar=pow_colorbar), row=2, col=2)

# 2.2 Precuneus R
df_sub = spk_df_avg.loc[(spk_df_avg["mode"] == "lfp_couple") & (spk_df_avg["node"] == rois[1])]
sl = False
fig.add_trace(go.Heatmap(z=df_sub[measures[0]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=freq_colorscale, zmin=freq_min, zmax=freq_max,
                         colorbar=freq_colorbar), row=2, col=3)
fig.add_trace(go.Heatmap(z=df_sub[measures[1]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=pow_colorscale, zmin=pow_min, zmax=pow_max,
                         colorbar=pow_colorbar), row=2, col=4)
# 2.3 plv
fig.add_trace(go.Heatmap(z=spk2plv_df_avg[measures[2]], x=spk2plv_df_avg.fex, y=spk2plv_df_avg.weight,
                         showscale=True, colorscale=plv_colorscale, zmin=plv_min, zmax=plv_max,
                         colorbar=plv_colorbar), row=2, col=5)


## 3. arnold tongue for the Cingulum bundle
# 3.1 Precuneus L (stimulated)

df_sub = spk_df_avg.loc[(spk_df_avg["mode"] == "lfp_cb")]

if colormode == "rowspecific":
    freq_min, freq_max = df_sub[measures[0]].min(), df_sub[measures[0]].max()
    pow_min, pow_max = df_sub[measures[1]].min(), df_sub[measures[1]].max()
    plv_min, plv_max = 0, 1
    freq_colorbar = dict(title="Hz", thickness=6, len=0.47, y=0.22, x=1.01)
    pow_colorbar = dict(title="dB", thickness=6, len=0.47, y=0.22, x=1.05)
    plv_colorbar = dict(title="PLV", thickness=6, len=0.47, y=0.22, x=1.09)
    sl = True

df_sub = spk_df_avg.loc[(spk_df_avg["mode"] == "lfp_cb") & (spk_df_avg["node"] == rois[0])]

fig.add_trace(go.Heatmap(z=df_sub[measures[0]], x=df_sub.fex, y=df_sub.weight,
                         showscale=True, colorscale=freq_colorscale, zmin=freq_min, zmax=freq_max,
                         colorbar=freq_colorbar), row=3, col=1)
fig.add_trace(go.Heatmap(z=df_sub[measures[1]], x=df_sub.fex, y=df_sub.weight,
                         showscale=True, colorscale=pow_colorscale, zmin=pow_min, zmax=pow_max,
                         colorbar=pow_colorbar), row=3, col=2)
# 3.2 Precuneus R
df_sub = spk_df_avg.loc[(spk_df_avg["mode"] == "lfp_cb") & (spk_df_avg["node"] == rois[1])]

sl = False


fig.add_trace(go.Heatmap(z=df_sub[measures[0]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=freq_colorscale, zmin=freq_min, zmax=freq_max,
                         colorbar=freq_colorbar), row=3, col=3)
fig.add_trace(go.Heatmap(z=df_sub[measures[1]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=pow_colorscale, zmin=pow_min, zmax=pow_max,
                         colorbar=pow_colorbar), row=3, col=4)
# # PLV
# df_sub = spk_cbplv_df_avg.loc[spk_cbplv_df_avg[""]]
# fig.add_trace(go.Heatmap(z=df_sub[measures[2]], x=df_sub.fex, y=df_sub.weight,
#                          showscale=True, colorscale=plv_colorscale, zmin=plv_min, zmax=plv_max,
#                          colorbar=plv_colorbar), row=3, col=5)

# 3.3 Frontal Mid L
df_sub = spk_df_avg.loc[(spk_df_avg["mode"] == "lfp_cb") & (spk_df_avg["node"] == rois[2])]

sl = False

fig.add_trace(go.Heatmap(z=df_sub[measures[0]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=freq_colorscale, zmin=freq_min, zmax=freq_max,
                         colorbar=freq_colorbar), row=3, col=6)
fig.add_trace(go.Heatmap(z=df_sub[measures[1]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=pow_colorscale, zmin=pow_min, zmax=pow_max,
                         colorbar=pow_colorbar), row=3, col=7)
# # PLV
# fig.add_trace(go.Heatmap(z=df_sub[measures[3]], x=df_sub.fex, y=df_sub.weight,
#                          showscale=sl, colorscale=plv_colorscale, zmin=plv_min, zmax=plv_max,
#                          colorbar=plv_colorbar), row=3, col=8)
# fig.add_trace(go.Heatmap(z=df_sub[measures[4]], x=df_sub.fex, y=df_sub.weight,
#                          showscale=sl, colorscale=plv_colorscale, zmin=plv_min, zmax=plv_max,
#                          colorbar=plv_colorbar), row=3, col=9)

# 4. Arnold tongue for the Cingulum bundle with antiphase stimulation
# 4.1 Precuneus L (Stimulated)
df_sub = spk_df_avg.loc[(spk_df_avg["mode"] == "lfp_cb_anti") & (spk_df_avg["node"] == rois[0])]

fig.add_trace(go.Heatmap(z=df_sub[measures[0]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=freq_colorscale, zmin=freq_min, zmax=freq_max,
                         colorbar=freq_colorbar), row=4, col=1)
fig.add_trace(go.Heatmap(z=df_sub[measures[1]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=pow_colorscale, zmin=pow_min, zmax=pow_max,
                         colorbar=pow_colorbar), row=4, col=2)

# 4.2 Precuneus R (antiphase stimulation)
df_sub = spk_df_avg.loc[(spk_df_avg["mode"] == "lfp_cb_anti") & (spk_df_avg["node"] == rois[1])]

sl = False
fig.add_trace(go.Heatmap(z=df_sub[measures[0]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=freq_colorscale, zmin=freq_min, zmax=freq_max,
                         colorbar=freq_colorbar), row=4, col=3)
fig.add_trace(go.Heatmap(z=df_sub[measures[1]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=pow_colorscale, zmin=pow_min, zmax=pow_max,
                         colorbar=pow_colorbar), row=4, col=4)
# # PLV
# fig.add_trace(go.Heatmap(z=df_sub[measures[2]], x=df_sub.fex, y=df_sub.weight,
#                          showscale=sl, colorscale=plv_colorscale, zmin=plv_min, zmax=plv_max,
#                          colorbar=plv_colorbar), row=4, col=5)

# 4.3 Frontal Mid L
df_sub = spk_df_avg.loc[(spk_df_avg["mode"] == "lfp_cb_anti") & (spk_df_avg["node"] == rois[2])]

sl = False

fig.add_trace(go.Heatmap(z=df_sub[measures[0]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=freq_colorscale, zmin=freq_min, zmax=freq_max,
                         colorbar=freq_colorbar), row=4, col=6)
fig.add_trace(go.Heatmap(z=df_sub[measures[1]], x=df_sub.fex, y=df_sub.weight,
                         showscale=sl, colorscale=pow_colorscale, zmin=pow_min, zmax=pow_max,
                         colorbar=pow_colorbar), row=4, col=7)
# # PLV
# fig.add_trace(go.Heatmap(z=df_sub[measures[3]], x=df_sub.fex, y=df_sub.weight,
#                          showscale=sl, colorscale=plv_colorscale, zmin=plv_min, zmax=plv_max,
#                          colorbar=plv_colorbar), row=4, col=8)
# fig.add_trace(go.Heatmap(z=df_sub[measures[4]], x=df_sub.fex, y=df_sub.weight,
#                          showscale=sl, colorscale=plv_colorscale, zmin=plv_min, zmax=plv_max,
#                          colorbar=plv_colorbar), row=4, col=9)

fig.update_layout(template="plotly_white", height=600, width=900)
pio.write_html(fig, fig_folder + "R2_SPK_arnoldTongues.html", auto_open=True)
pio.write_image(fig, fig_folder + "R2_SPK_arnoldTongues.svg")


