
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
import glob
from tvb.datatypes import connectivity
from scipy.signal import savgol_filter
import statsmodels.api as sm

fname = "PSEmpi_entrainment_bySC_indWPpass-m06d11y2022-t21h.59m.16s"
specific_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\\3entrainment_bySC\PSE\\" + fname
ctb_folder = "E:\LCCN_Local\PycharmProjects\\CTB_data2\\"


# Cargar los datos
results = pd.read_csv(glob.glob(specific_folder + "\\*results.csv")[0])

# max_tracts_baseline = float(df_baseline["tracts_left"].loc[(df_baseline["rep"]==0) & (df_baseline["n_removed"]==0)])


## Calculate entrainment range by n_removed  ## LASTs for 30 min: not efficient.
entrainment_range = []
for subj in set(results.subject.values):
    print(subj)

    conn = connectivity.Connectivity.from_file(ctb_folder + subj + "_AAL2_pass.zip")
    tracts_normalization_factor = conn.weights.max()

    regionLabels = conn.region_labels
    df_baseline = results.loc[(results["subject"] == subj) & (results["n_remove"] == 0) & (results["stim_params"] == 0)]

    for target in set(results.target.values):
        for n_remove in set(results.n_remove.values):
            for r in set(results.rep.values):

                df_temp = results.loc[(results["subject"] == subj) & (results["target"] == target) &
                                      (results["n_remove"] == n_remove) & (results["rep"] == r)]

                removed_tracts = (df_temp.removed_tracts.values * tracts_normalization_factor).mean().round()
                tracts_left = (df_temp.tracts_left.values * tracts_normalization_factor).mean().round()

                percent_remTracts = removed_tracts / (removed_tracts + tracts_left)
                percent_tractsLeft = tracts_left / (removed_tracts + tracts_left)

                for roi in set(results.target.values):

                    ## Calculate frequency distance to baseline and to stimulation frequency.
                    roi_2baseline = np.asarray(df_temp[regionLabels[roi]]) - df_baseline[regionLabels[target]].values.mean()
                    roi_2stim = np.asarray(df_temp[regionLabels[roi]]) - np.asarray(df_temp["stim_params"])

                    # Where distance to baseline is higher than distance to stimulus, we assume there was entrainment
                    range_hz = df_temp["stim_params"].loc[(abs(roi_2baseline) - abs(roi_2stim) > 0)].max() - \
                               df_temp["stim_params"].loc[(abs(roi_2baseline) - abs(roi_2stim) > 0)].min()

                    entrainment_range.append([subj, target, n_remove, tracts_left, percent_tractsLeft, removed_tracts, percent_remTracts, r, roi, range_hz])



df_ent_bySC = pd.DataFrame(entrainment_range, columns=["subj", "target", "removed_conn", "tracts_left", "percent_tractsLeft",
                                                               "removed_tracts", "percent_remTracts", "rep", "roi", "ent_range"])
df_ent_bySC = df_ent_bySC.dropna()
df_ent_bySC.to_csv(specific_folder + "/entrainment_bySC_10subjs.csv", index=False)

# color palette
cmap = px.colors.qualitative.Plotly


## Functions
# disconn_metric = removed_conn, tracts_left, percent_tractsLeft, removed_tracts, percent_remTracts

def entrainment_bySC_plot(results, df_ent_bySC, disconn_metric="tracts_left", lowess_frac=0.1):

    fig = go.Figure()

    for i, target in enumerate(sorted(set(results.target.values))):

        # subset data
        df_2plot = df_ent_bySC.loc[(df_ent_bySC["subj"]==subj) & (df_ent_bySC["target"]==target) & (df_ent_bySC["roi"]==target)]
        df_2plot = df_2plot.sort_values(by=[disconn_metric])

        # original scatter points
        fig.add_trace(go.Scatter(x=df_2plot[disconn_metric], y=df_2plot.ent_range, opacity=0.3,
                                 mode="markers", marker_symbol="circle-open", marker_color=cmap[i],
                                 name=regionLabels[target], legendgroup=regionLabels[target], showlegend=True))

        # smoothed line
        smoothed_line = sm.nonparametric.lowess(exog=df_2plot[disconn_metric], endog=df_2plot.ent_range, frac=lowess_frac)
        # savgol_filter(df_2plot.ent_range, window_length=25, polyorder=2)

        fig.add_trace(go.Scatter(x=smoothed_line[:, 0], y=smoothed_line[:, 1], marker_color=cmap[i],
                                 line=dict(width=4), name=regionLabels[target], legendgroup=regionLabels[target],
                                 showlegend=True))

    fig.show(renderer="browser")

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

    fig.show(renderer="browser")


## AVERAGE PLOTS
## X4 plot
entrainment_bySC_x4plot(results, df_ent_bySC, disconn_metric="tracts_left", lowess_frac=0.3)
## X1 Plot
entrainment_bySC_plot(results, df_ent_bySC, disconn_metric="tracts_left", lowess_frac=0.15)


## PLOTS per SUBJECT
## X4 plot
for subj in set(results.subject.values):
    subset_df_ent = df_ent_bySC.loc[df_ent_bySC["subject"]==subj]
    entrainment_bySC_x4plot(results, subset_df_ent, disconn_metric="tracts_left")
## X1 Plot
for subj in set(results.subject.values):
    subset_df_ent = df_ent_bySC.loc[df_ent_bySC["subj"]==subj]
    entrainment_bySC_plot(results, subset_df_ent, disconn_metric="tracts_left", lowess_frac=0.35)




# # Check what happens in fft plots
# res_sub = results.loc[(results["subject"]==subj)&(results["target"]==34)]
#
# fig.add_trace(go.Scatter(x=res_sub.relFreq, y=res_sub.fft, line=dict(width=4),name=roi), row=2, col=1)



