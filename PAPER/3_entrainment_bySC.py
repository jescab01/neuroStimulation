
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
import glob

figures_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER\FIGURES\\"

######### Plot entraiment by SC
import statsmodels.api as sm
from tvb.simulator.lab import *

# Load data
fname = "PSEmpi_entrainment_bySC_indWPpass-m06d11y2022-t21h.59m.16s"
results = pd.read_csv("E:\\LCCN_Local\PycharmProjects\\neuroStimulation\\3entrainment_bySC\PSE\\" + fname + "\entrainment_bySC_results.csv")
df_ent_bySC = pd.read_csv("E:\\LCCN_Local\PycharmProjects\\neuroStimulation\\3entrainment_bySC\PSE\\"+ fname + "\entrainment_bySC_10subjs.csv")

ctb_folder = "E:\LCCN_Local\PycharmProjects\\CTB_data2\\"
conn = connectivity.Connectivity.from_file(ctb_folder + "NEMOS_AVG_AAL2_pass.zip")
regionLabels = conn.region_labels

# color palette
cmap = px.colors.qualitative.Plotly

## Functions

def entrainment_bySC_plot(results, df_ent_bySC, disconn_metric="tracts_left", lowess_frac=0.1):

    fig = go.Figure()

    for i, target in enumerate(sorted(set(results.target.values))):

        # subset data
        df_2plot = df_ent_bySC.loc[(df_ent_bySC["target"]==target) & (df_ent_bySC["roi"]==target)]
        df_2plot = df_2plot.sort_values(by=[disconn_metric])

        # original scatter points
        fig.add_trace(go.Scatter(x=df_2plot[disconn_metric], y=df_2plot.ent_range, opacity=0.2,
                                 mode="markers", marker_symbol="circle-open", marker_color=cmap[i],
                                 name=regionLabels[target], legendgroup=regionLabels[target], showlegend=True))

        # smoothed line
        smoothed_line = sm.nonparametric.lowess(exog=df_2plot[disconn_metric], endog=df_2plot.ent_range, frac=lowess_frac)
        # savgol_filter(df_2plot.ent_range, window_length=25, polyorder=2)

        fig.add_trace(go.Scatter(x=smoothed_line[:, 0], y=smoothed_line[:, 1], marker_color=cmap[i],
                                 line=dict(width=4), name=regionLabels[target], legendgroup=regionLabels[target],
                                 showlegend=True))

    fig.update_xaxes(title="Number of streamlines")
    fig.update_yaxes(title="Entrainment range (Hz)")
    fig.update_layout(template="plotly_white")
    pio.write_html(fig, file=figures_folder + '/ent_bySC.html', auto_open=True)
    pio.write_image(fig, file=figures_folder + '/ent_bySC.png')


entrainment_bySC_plot(results, df_ent_bySC, disconn_metric="tracts_left", lowess_frac=0.15)



def entrainment_bySC_x4plot(results, df_ent_bySC, disconn_metric="tracts_left", lowess_frac=0.1):
    fig = make_subplots(rows=2, cols=2, subplot_titles=("disconnecting ACC_L","disconnecting Precuneus_L",
                                                        "disconnecting ACC_R", "disconnecting Precuneus_R"),
                        specs=[[{}, {}], [{}, {}]], shared_yaxes=True,
                        x_title=disconn_metric)

    for i, target in enumerate(sorted(set(results.target.values))):
        for ii, roi in enumerate(sorted(set(results.target.values))):
            sl = True if i<1 else False
            # subset data
            df_2plot = df_ent_bySC.loc[(df_ent_bySC["target"]==target) & (df_ent_bySC["roi"]==roi)]
            df_2plot = df_2plot.sort_values(by=[disconn_metric])

            # original scatter points
            fig.add_trace(go.Scatter(x=df_2plot[disconn_metric], y=df_2plot.ent_range,
                                     mode="markers", marker_symbol="circle-open", marker_color=cmap[ii], opacity=0.3,
                                     name=regionLabels[roi], legendgroup=regionLabels[roi], showlegend=sl), row=i%2+1, col=i//2+1)

            # smoothed line
            smoothed_line = sm.nonparametric.lowess(exog=df_2plot[disconn_metric], endog=df_2plot.ent_range,
                                                             frac=lowess_frac)

            # smoothed_y = savgol_filter(df_2plot.ent_range, window_length=25, polyorder=4)

            fig.add_trace(go.Scatter(x=smoothed_line[:, 0], y=smoothed_line[:,1], marker_color=cmap[ii],
                                     line=dict(width=4), name=regionLabels[roi], legendgroup=regionLabels[roi],
                                     showlegend=sl), row=i%2+1, col=i//2+1)

    pio.write_html(fig, file=figures_folder + '/ent_bySCx4.html', auto_open=True)

entrainment_bySC_x4plot(results, df_ent_bySC, disconn_metric="tracts_left", lowess_frac=0.3)



## PLOTS per SUBJECT
## X1 Plot
for subj in set(results.subject.values):
    subset_df_ent = df_ent_bySC.loc[df_ent_bySC["subj"]==subj]
    entrainment_bySC_plot(results, subset_df_ent, disconn_metric="tracts_left", lowess_frac=0.35)
## X4 plot
for subj in set(results.subject.values):
    subset_df_ent = df_ent_bySC.loc[df_ent_bySC["subject"]==subj]
    entrainment_bySC_x4plot(results, subset_df_ent, disconn_metric="tracts_left")

