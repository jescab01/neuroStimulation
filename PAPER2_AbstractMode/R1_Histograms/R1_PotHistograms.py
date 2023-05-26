

import numpy as np
import scipy.io
import scipy.stats
from tvb.simulator.lab import connectivity

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


emp_subj = "NEMOS_035"

fname = emp_subj + "-ROIvals_orth-roast_OzCzModel.mat"
main_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\R1_Histograms\\"

fig_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\Figures\\"

## 0a. Load histograms
histograms = scipy.io.loadmat(main_folder + fname)


## 0b. Load Connectivity to get regions order
ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_dataOLD2\\"

conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2_pass.zip")
conn.weights = conn.scaled_weights(mode="tract")


# Define regions implicated in Functional analysis: remove subcorticals
cingulum_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                 'Insula_L', 'Insula_R',
                 'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                 'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                 'ParaHippocampal_R',
                 'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                 'Parietal_Inf_R', 'Precuneus_L', 'Precuneus_R']

# load text with FC rois; check if match SC
SClabs = list(conn.region_labels)
SC_cb_idx = [SClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cortical_rois

subjects = ["NEMOS_0" + str(id) for id in [35, 49, 50, 58, 59, 64, 65, 71, 75, 77]]


# 1. CB plots with accum. all subjects
nrows, ncols = 3, 6
fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=cingulum_rois, y_title="Probability density",
                    x_title="Orthogonal component of the electric field<br>to white matter surface")

for i, roi in enumerate(SC_cb_idx):

    ef_orthvals, ef_orthmean = [], []
    for emp_subj in subjects:
        fname = emp_subj + "-ROIvals_orth-roast_OzCzModel.mat"

        histograms = scipy.io.loadmat(main_folder + fname)

        ef_orthvals.append(np.squeeze(histograms["ROIvals"][0, roi][0]))  # Last index for 0-array, 1-mean
        ef_orthmean.append(np.average(np.squeeze(histograms["ROIvals"][0, roi][0])))  # Last index for 0-array, 1-mean

    ef_orthvals = np.hstack(ef_orthvals)
    ef_orthmean = np.average(ef_orthmean)

    fig.add_trace((go.Histogram(x=ef_orthvals, name=cingulum_rois[i], histnorm="probability", opacity=0.6, showlegend=False,
                        marker_color="steelblue")), row=i//ncols+1, col=i%ncols+1)

    fig.add_vline(x=ef_orthmean, line=dict(color="indianred", dash="dash", width=2), opacity=0.5, row=i//ncols+1, col=i%ncols+1)

for i in np.arange(0, ncols * nrows):
    fig["layout"]["xaxis" + str(i+1)]["showgrid"]=True
    fig["layout"]["xaxis" + str(i+1)]["range"] = [-0.35, 0.35]


fig.update_layout(template="plotly_white", width=1200, height=700)

pio.write_html(fig, fig_folder + "R1_Histograms_CB.html", auto_open=True)
pio.write_image(fig, fig_folder + "R1_Histograms_CB.svg")



# 2. AAL plots - accum. all subjects

cortical_rois = ['Precentral_L', 'Precentral_R', 'Frontal_Sup_2_L',
       'Frontal_Sup_2_R', 'Frontal_Mid_2_L', 'Frontal_Mid_2_R',
       'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L',
       'Frontal_Inf_Tri_R', 'Frontal_Inf_Orb_2_L', 'Frontal_Inf_Orb_2_R',
       'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L',
       'Supp_Motor_Area_R', 'Olfactory_L', 'Olfactory_R',
       'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R',
       'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R', 'Rectus_L', 'Rectus_R',
       'OFCmed_L', 'OFCmed_R', 'OFCant_L', 'OFCant_R', 'OFCpost_L',
       'OFCpost_R', 'OFClat_L', 'OFClat_R', 'Insula_L', 'Insula_R',
       'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Mid_L',
       'Cingulate_Mid_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
       'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
       'ParaHippocampal_R', 'Calcarine_L',
       'Calcarine_R', 'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R',
       'Occipital_Sup_L', 'Occipital_Sup_R', 'Occipital_Mid_L',
       'Occipital_Mid_R', 'Occipital_Inf_L', 'Occipital_Inf_R',
       'Fusiform_L', 'Fusiform_R', 'Postcentral_L', 'Postcentral_R',
       'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
       'Parietal_Inf_R', 'SupraMarginal_L', 'SupraMarginal_R',
       'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R',
       'Paracentral_Lobule_L', 'Paracentral_Lobule_R',
       'Heschl_L', 'Heschl_R',
       'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L',
       'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R',
       'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L',
       'Temporal_Inf_R']

SC_cortex_idx = [SClabs.index(roi) for roi in cortical_rois]  # find indexes in FClabs that matches cortical_rois



nrows, ncols = 14, 6
fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=cortical_rois, y_title="Probability density",
                    x_title="Orthogonal component of the electric field<br>to white matter surface")

for i, roi in enumerate(SC_cortex_idx):

    ef_orthvals, ef_orthmean = [], []
    for emp_subj in subjects:
        fname = emp_subj + "-ROIvals_orth-roast_OzCzModel.mat"

        histograms = scipy.io.loadmat(main_folder + fname)

        ef_orthvals.append(np.squeeze(histograms["ROIvals"][0, roi][0]))  # Last index for 0-array, 1-mean
        ef_orthmean.append(np.average(np.squeeze(histograms["ROIvals"][0, roi][0])))  # Last index for 0-array, 1-mean

    ef_orthvals = np.hstack(ef_orthvals)
    ef_orthmean = np.average(ef_orthmean)

    fig.add_trace((go.Histogram(x=ef_orthvals, name=conn.region_labels[roi], histnorm="probability", opacity=0.6, showlegend=False,
                        marker_color="steelblue")), row=i//ncols+1, col=i%ncols+1)

    fig.add_vline(x=ef_orthmean, line=dict(color="indianred", dash="dash", width=2), opacity=0.5, row=i//ncols+1, col=i%ncols+1)

for i in np.arange(0, ncols * nrows):
    fig["layout"]["xaxis" + str(i+1)]["showgrid"]=True
    fig["layout"]["xaxis" + str(i+1)]["range"] = [-0.35, 0.35]


fig.update_layout(template="plotly_white", width=1200, height=1500)

pio.write_html(fig, fig_folder + "Rsm1_Histograms_AAL.html", auto_open=True)
pio.write_image(fig, fig_folder + "Rsm1_Histograms_AAL.svg")

