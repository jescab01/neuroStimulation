
import pickle

import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
import glob
import pingouin as pg
from tvb.simulator.lab import connectivity


fname = "PSEmpi_variabilities_cb_indiv-m03d30y2023-t12h.26m.36s"
folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\R2_CingulumBundle\R2c_variabilities\\"

# cargar los datos
stimWfit = pd.read_pickle(glob.glob(folder + fname + "\\*_results.pkl")[0])

n_trials = stimWfit["trial"].max() + 1

# Calculate percentage
baseline = stimWfit.iloc[:, [0, 5, 6, 8]].loc[stimWfit["w"] == 0].groupby(["subject"]).mean().reset_index()

## Substitute internal coding NEMOS by subject
# for i, subj in enumerate(sorted(set(stimWfit.subject))):
#     new_name = "Subject " + str(i+1).zfill(2)
#     stimWfit["subject"].loc[stimWfit["subject"] == subj] = new_name



# CREATE a ROIS long dataframe
# for changes occured during the fitted stimulation
ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_dataOLD2\\"
cingulum_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                 'Insula_L', 'Insula_R',
                 'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                 'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                 'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R',
                 'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                 'Parietal_Inf_R', 'Precuneus_L', 'Precuneus_R',
                 'Thalamus_L', 'Thalamus_R']

empCluster_rois = ['Precentral_L', 'Frontal_Sup_2_L', 'Frontal_Sup_2_R', 'Frontal_Mid_2_L',
                   'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L', 'Frontal_Inf_Tri_R',
                   'Frontal_Inf_Orb_2_L', 'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Frontal_Sup_Medial_L',
                   'Frontal_Sup_Medial_R', 'Rectus_L', 'OFCmed_L', 'Insula_L', 'Insula_R', 'Cingulate_Ant_L',
                   'Cingulate_Ant_R',
                   'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L', 'ParaHippocampal_R',
                   'Amygdala_L', 'Calcarine_L', 'Calcarine_R', 'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R',
                   'Occipital_Sup_R', 'Occipital_Mid_L', 'Occipital_Mid_R', 'Occipital_Inf_L',
                   'Occipital_Inf_R', 'Fusiform_L', 'Fusiform_R', 'Postcentral_L', 'Parietal_Sup_R',
                   'Parietal_Inf_R', 'Angular_R', 'Precuneus_R', 'Temporal_Sup_L',
                   'Temporal_Sup_R', 'Temporal_Pole_Sup_L', 'Temporal_Pole_Sup_R', 'Temporal_Mid_L',
                   'Temporal_Mid_R', 'Temporal_Pole_Mid_L', 'Temporal_Inf_L', 'Temporal_Inf_R']

occ_cb = list(set(empCluster_rois).intersection(set(cingulum_rois)))

stimWfit_longdf = []
for i, emp_subj in enumerate(sorted(set(stimWfit.subject))):

    # Load data
    conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2_pass.zip")
    conn.weights = conn.scaled_weights(mode="tract")

    # load text with FC rois
    FClabs = list(np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/roi_labels_rms.txt", dtype=str))
    # Subset for Cingulum Bundle
    FC_cb_idx = [FClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cortical_rois

    SClabs = list(conn.region_labels)
    SC_cb_idx = [SClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cortical_rois
    conn.weights = conn.weights[:, SC_cb_idx][SC_cb_idx]
    conn.region_labels = conn.region_labels[SC_cb_idx]

    # Load electric field magnitude values
    weighting = np.loadtxt(
        ctb_folder + 'CurrentPropagationModels/' + emp_subj + '-efnorm_mag-roast_OzCzModel-AAL2.txt',
        delimiter=",") * stimWfit.w.mean()

    weighting = weighting[SC_cb_idx]

    rois_cluster = [list(conn.region_labels).index(roi) for roi in occ_cb]

    for ii, row in stimWfit.loc[(stimWfit["subject"] == emp_subj)].iterrows():

        # Add cluster values
        sc = conn.weights[:, rois_cluster][rois_cluster]
        stimWfit_longdf.append(["cluster", row["subject"], row["trial"], row["w"], row["fpeak"], row["amp_fpeak"],
                                row["plv"], np.average(weighting[rois_cluster]), np.average(sc[np.triu_indices(len(sc), 1)])])

        # Add rois values
        fpeak_rois = row["fpeak_rois"]
        amp_fpeak_rois = row["amp_fpeak_rois"]
        plv_rois = row["plv_rois"]

        for r, roi in enumerate(conn.region_labels):
            plv_roi = plv_rois[r, :]
            sc = conn.weights[r, :]
            stimWfit_longdf.append([roi, row["subject"], row["trial"], row["w"], fpeak_rois[r], amp_fpeak_rois[r],
                                    np.average(plv_roi[plv_roi != 1]), weighting[r], np.average(sc[sc != 1])])

        # Add avg values
        stimWfit_longdf.append(["avg", row["subject"], row["trial"], row["w"], np.average(fpeak_rois),
                                np.average(amp_fpeak_rois), np.average(plv_rois[np.triu_indices(len(plv_rois), 1)]),
                                np.average(weighting), np.average(conn.weights[np.triu_indices(len(conn.weights), 1)])])

stimWfit_longdf = pd.DataFrame(stimWfit_longdf, columns=["location", "subject", "trial", "w", "fpeak", "amp_fpeak", "plv", "ef_mag", "sc"])


stimWfit_longdf["dPow_percent"] = [(row["amp_fpeak"] - baseline.loc[baseline["subject"] == row["subject"]].amp_fpeak.values[0])
                                   / baseline.loc[baseline["subject"] == row["subject"]].amp_fpeak.values[0] * 100
                                   for i, row in stimWfit_longdf.iterrows()]
stimWfit_longdf["dFreq"] = [(row["fpeak"] - baseline.loc[baseline["subject"] == row["subject"]].fpeak.values[0])
                            for i, row in stimWfit_longdf.iterrows()]
stimWfit_longdf["dFC"] = [(row["plv"] - baseline.loc[baseline["subject"] == row["subject"]].plv.values[0])
                          for i, row in stimWfit_longdf.iterrows()]

stimWfit_longdf["efmag_sc"] = [(abs(row["ef_mag"]) * row["sc"]) for i, row in stimWfit_longdf.iterrows()]


## 1. What has happened with the fitted stimulation?
# Plot all trials 0 vs 0.6 in terms of dPow, dFreq, dFC
fig = px.strip(stimWfit_longdf.loc[stimWfit_longdf["location"] == "cluster"], x="w", y="dPow_percent", color="subject")
fig.show("browser")
fig = px.strip(stimWfit_longdf.loc[stimWfit_longdf["location"] == "cluster"], x="w", y="dFreq", color="subject")
fig.show("browser")
fig = px.strip(stimWfit_longdf.loc[stimWfit_longdf["location"] == "cluster"], x="w", y="dFC", color="subject")
fig.show("browser")

# 1b. Changes in intrasubject variability?
"""
It is slightly different (p<0.05), but the initial one is already big lets focus first in intersubject
"""
intraVar_base = [np.std(stimWfit["percent"].loc[(stimWfit["subject"] == subj) & (stimWfit["w"] == 0)].values) for subj in sorted(set(stimWfit.subject))]
intraVar_wfit = [np.std(stimWfit["percent"].loc[(stimWfit["subject"] == subj) & (stimWfit["w"] == 0.6)].values) for subj in sorted(set(stimWfit.subject))]

ttest = pg.wilcoxon(intraVar_wfit, intraVar_base, alternative="greater")



##

fig = px.strip(stimWfit_longdf.loc[(stimWfit_longdf["location"] != "avg") & (stimWfit_longdf["location"] != "cluster")],
               x="efmag_sc", y="dPow_percent", color="location")
fig.show("browser")


fig = px.strip(stimWfit_longdf.loc[(stimWfit_longdf["location"] != "avg") & (stimWfit_longdf["location"] != "cluster") & (stimWfit_longdf["subject"] == "NEMOS_035") & (stimWfit_longdf["w"] == 0)],
               x="efmag_sc", y="fpeak", color="location")

fig.add_hline(y=stimWfit_longdf["fpeak"].loc[(stimWfit_longdf["location"] == "cluster") & (stimWfit_longdf["subject"] == "NEMOS_035") & (stimWfit_longdf["w"] == 0)].mean())

fig.show("browser")

## 2. INTER-SUBJECT variability: is it different at w==0 than at the fit (w=0.6)
"""

"""

stimWfit_sub = stimWfit.loc[(stimWfit["w"] == 0.6)].groupby(["subject", "w"]).mean().reset_index()

# Pre-calculate the correlations: df with (subj, roi, percent)



# per subject, add trace of ef_mags (color - roi, size - degree, x - ef_mag)
# legend roi; hovertext the roi and the correlation through subjects

fig = go.Figure()
for i, emp_subj in enumerate(sorted(set(stimWfit_sub.subject))):

    sl = True if i == 0 else False

    # Load data for plotting
    conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2_pass.zip")
    conn.weights = conn.scaled_weights(mode="tract")



    # load text with FC rois
    FClabs = list(np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/roi_labels_rms.txt", dtype=str))
    # Subset for Cingulum Bundle
    FC_cb_idx = [FClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cortical_rois


    SClabs = list(conn.region_labels)
    SC_cb_idx = [SClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cortical_rois
    conn.weights = conn.weights[:, SC_cb_idx][SC_cb_idx]
    conn.tract_lengths = conn.tract_lengths[:, SC_cb_idx][SC_cb_idx]
    conn.region_labels = conn.region_labels[SC_cb_idx]

    # Load electric field magnitude values
    weighting = np.loadtxt(
        ctb_folder + 'CurrentPropagationModels/' + emp_subj + '-efnorm_mag-roast_OzCzModel-AAL2.txt',
        delimiter=",") * stimWfit_sub.w.mean()

    weighting = weighting[SC_cb_idx]

    ## PREPARE TRACE
    ## Define size per degree
    degree, size_range = np.sum(conn.weights, axis=1), [8, 25]
    size = ((degree - np.min(degree)) * (size_range[1] - size_range[0]) / (np.max(degree) - np.min(degree))) + size_range[0]

    ## Define color per roi
    cmap = px.colors.qualitative.Light24
    # cmap = px.colors.sequential.Turbo
    color = np.arange(len(size))

    percent = [stimWfit_sub["percent"].loc[stimWfit_sub["subject"] == emp_subj].values[0]]*len(weighting)
    for ii, roi in enumerate(conn.region_labels):
        fig.add_trace(go.Scatter(x=weighting[ii], y=percent[ii], mode="markers", showlegend=sl, name=roi, legendgroup=roi,
                                 marker=dict(size=size[ii], color=color[ii]), opacity=0.8))

fig.update_layout(template="plotly_white")
fig.show("browser")