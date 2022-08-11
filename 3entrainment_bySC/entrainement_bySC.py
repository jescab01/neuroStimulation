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


## GATHER RESULTS
df_fft = pd.DataFrame(Results_fft, columns=["stimFreq", "rep", "sim_n", "n_rois_removed", "rois_removed", "tracts_left", "target_roi", "target_peak", "check_roi", "check_peak"])
df_fft.to_csv(specific_folder + "/" + emp_subj + "-FFT_bySC_"+regionLabels[target_roi]+"-"+regionLabels[check_roi]+".csv", index=False)

# Average data by simulations
df_fft_avg = df_fft.groupby(["stimFreq", "rep", "n_rois_removed"]).mean().reset_index()
df_baseline = df_fft_avg.loc[df_fft_avg["stimFreq"] == 0]

max_tracts_baseline = float(df_baseline["tracts_left"].loc[(df_baseline["rep"]==0) & (df_baseline["n_rois_removed"]==0)])
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
pio.write_html(fig, file=specific_folder + "/" + emp_subj + "3entrainment_bySC-target" + regionLabels[target_roi] + "_conns_3rep.html",
               auto_open=True)

fig = px.strip(df_entrainment_bySC, x="removed_tracts", y="entrainment_range", color="roi_name",
           title="Nodes entrainment by number of %s's connections to other regions (%i repetitions | %i simulations)" % (regionLabels[target_roi], n_rep, n_simulations))
pio.write_html(fig, file=specific_folder + "/" + emp_subj + "3entrainment_bySC-target" + regionLabels[target_roi] + "_tracts_3rep.html",
               auto_open=True)

