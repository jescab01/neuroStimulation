

import numpy as np
import scipy.io
import scipy.stats
from tvb.simulator.lab import connectivity

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


emp_subj = "NEMOS_035"

fname = emp_subj + "-ROIvals_orth-roast_OzCzModel.mat"
main_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\R0_Histograms\\"

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


# 1. Load region data and plot :: all subjects v2
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

pio.write_html(fig, fig_folder + "R0_Histograms_CB_v2.html", auto_open=True)
pio.write_image(fig, fig_folder + "R0_Histograms_CB_v2.svg")





# 1. Load region data and plot :: all subjects
nrows, ncols = 3, 6
fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=cingulum_rois, y_title="Probability density", shared_xaxes=True,
                    x_title="Orthogonal component of the electric field<br>to white matter surface", specs=[[{"secondary_y": True}] * ncols]*nrows)

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

    # fig.add_vline(x=0, line=dict(color="lightgray"), row=i//ncols + 1, col=i % ncols + 1)
    #
    # fig.add_vline(x=ef_orthmean, line=dict(color="indianred", dash="dash"), row=i//ncols+1, col=i%ncols+1)

    kde = scipy.stats.gaussian_kde(ef_orthvals)
    xseq = np.linspace(np.min(ef_orthvals), np.max(ef_orthvals), 1001)

    # fig.add_trace(go.Scatter(x=xseq, y=kde(xseq), mode="lines", line=dict(color="black"), showlegend=False), secondary_y=True, row=i//ncols+1, col=i%ncols+1)

for i in np.arange(1, ncols * nrows * 2, 2):
    fig["layout"]["yaxis" + str(i+1)]["visible"] = False

fig.update_layout(template="plotly_white", width=1200, height=700)

pio.write_html(fig, fig_folder + "Histograms_CB.html", auto_open=True)
pio.write_image(fig, fig_folder + "Histograms_CB.svg")




# 2. Load region data and plot :: NEMOS 035
nrows, ncols = 3, 6
fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=cingulum_rois, y_title="Probability density",
                    x_title="Orthogonal component of the electric field<br>to white matter surface", specs=[[{"secondary_y": True}] * ncols]*nrows)

for i, roi in enumerate(SC_cb_idx):

    ef_orthvals = np.squeeze(histograms["ROIvals"][0, roi][0])  # Last index for 0-array, 1-mean
    ef_orthmean = np.average(ef_orthvals)  # Last index for 0-array, 1-mean

    fig.add_trace((go.Histogram(x=ef_orthvals, name=cingulum_rois[i], histnorm="probability", opacity=0.6, showlegend=False,
                        marker_color="steelblue")), row=i//ncols+1, col=i%ncols+1)

    fig.add_vline(x=0, line=dict(color="lightgray"), row=i//ncols + 1, col=i % ncols + 1)

    fig.add_vline(x=ef_orthmean, line=dict(color="indianred", dash="dash"), row=i//ncols+1, col=i%ncols+1)

    kde = scipy.stats.gaussian_kde(ef_orthvals)
    xseq = np.linspace(np.min(ef_orthvals), np.max(ef_orthvals), 1001)

    fig.add_trace(go.Scatter(x=xseq, y=kde(xseq), mode="lines", line=dict(color="black"), showlegend=False), secondary_y=True, row=i//ncols+1, col=i%ncols+1)

for i in np.arange(1, ncols * nrows * 2, 2):
    fig["layout"]["yaxis" + str(i+1)]["visible"] = False

fig.update_layout(template="plotly_white", width=1200, height=700)

pio.write_html(fig, fig_folder + "Histograms_CB_N35.html", auto_open=True)
pio.write_image(fig, fig_folder + "Histograms_CB_N35.svg")





## WITH SNS

import seaborn as sns
from matplotlib import pyplot as plt



fig, axes = plt.subplots(nrows, ncols, sharex=True)
for i, roi in enumerate(SC_cb_idx):

    ef_orthvals = np.squeeze(histograms["ROIvals"][0, roi][0])  # Last index for 0-array, 1-mean
    ef_orthmean = np.average(ef_orthvals)  # Last index for 0-array, 1-mean

    sns.distplot(ef_orthvals, kde=True, ax=axes[i//ncols, i%ncols])

    if i%ncols != 0:
        axes[i // ncols, i % ncols].set(ylabel=None)

    # if i//ncols == 3:
    #     axes[i // ncols, i % ncols].set(xlabel="EF magnitude<br>orthogonal component to WM surface")

fig.text(0.5, 0.04, 'EF magnitude orthogonal component to white matter surface', ha='center')
fig.tight_layout()

