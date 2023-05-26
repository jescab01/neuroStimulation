
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
import glob

fig_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\Figures\\"


#####     NEURAL MASS MODELS

## 0a. Load NMM data
fname = "PSEmpi_stimWfit_cb_indiv-m04d11y2023-t21h.58m.02s"
folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\output_NMM\\"

# cargar los datos
stimWfit_nmm = pd.read_csv(glob.glob(folder + fname + "\\*results.csv")[0])
n_trials = stimWfit_nmm["trial"].max() + 1

# Calculate percentage
baseline = stimWfit_nmm.loc[stimWfit_nmm["w"] == 0].groupby("subject").mean().reset_index()

stimWfit_nmm["percent"] = [(row["amp_fpeak"] - baseline.loc[baseline["subject"] == row["subject"]].amp_fpeak.values[0]) / baseline.loc[baseline["subject"] == row["subject"]].amp_fpeak.values[0] * 100 for i, row in stimWfit_nmm.iterrows()]
stimWfit_nmm.columns = ['subject', 'mode', 'coup', 'trial', 'w', 'fpeak', 'amp_fpeak', 'amp_fbaseline', 'percent']

# Just show half the calibration constants to make a clearer picture
include_w = sorted(set(stimWfit_nmm.w))[::4]
stimWfit_nmm_sub = stimWfit_nmm[stimWfit_nmm["w"].isin(include_w)]


## Substitute internal coding NEMOS by subject
for i, subj in enumerate(sorted(set(stimWfit_nmm_sub.subject))):
    new_name = "Subject " + str(i+1).zfill(2)
    stimWfit_nmm_sub["subject"].loc[stimWfit_nmm_sub["subject"] == subj] = new_name


### A. Scatter plot with Mean line for percentage
sim = "NMM"
fig = px.strip(stimWfit_nmm_sub, x="w", y="percent", color="subject", labels={"subject": ""}, log_x=False)

w = np.asarray(stimWfit_nmm_sub.groupby("w").mean().reset_index()["w"])
mean = np.asarray(stimWfit_nmm_sub.groupby("w").mean()["percent"])
median = np.asarray(stimWfit_nmm_sub.groupby("w").median()["percent"])

fig.add_trace(go.Scatter(x=w, y=mean, mode="lines", name="mean", line=dict(color='darkslategray', width=5)))
fig.update_layout(height=470, width=900)
fig.update_xaxes(title="Calibration constant (K) <br>i.e. Stimulation Weight")

fig.update_yaxes(title="Alpha band power change (%)", tickvals=[-10, 0, 8.02, 10, 20, 30])


pio.write_image(fig, file=fig_folder + '\\R3_'+sim+'_allNemosAAL_scatterpercent_alphaRise_' + str(n_trials) + "sim.svg")
# pio.write_image(fig, file=fig_folder + '\\R3_'+sim+'_allNemosAAL_scatterpercent_alphaRise_' + str(n_trials) + "sim.png")

fig.add_trace(go.Scatter(x=w, y=median, mode="lines", name="median", line=dict(color='slategray', width=4), visible="legendonly"))
pio.write_html(fig, file=fig_folder + '\\R3_'+sim+'_allNemosAAL_scatterpercent_alphaRise_' + str(n_trials) + "sim.html", auto_open=True)

## A2. plot for absolute power change bands || Is it needed? Just if some subject behaves weirdly.
fig = px.strip(stimWfit_nmm_sub, x="w", y="amp_fpeak", color="subject", log_x=False,
             title="Alpha band power rise @Pareto-Occipital regions<br>(%i simulations | 10 subjects AAL)" % n_trials,
             labels={  # replaces default labels by column name
                 "w": "Calibration constant (K) <br>i.e. Stimulation Weight", "bModule": "Alpha band power (mV*Hz)"})

pio.write_html(fig, file=fig_folder + '\\R3_'+sim+'_allNemosAAL_scatterModules_alphaRise_' + str(n_trials) + "sim.html",
               auto_open=True)





######       SPIKING

## 0b. Load SPK data
sim_tag = "stimWfit\\"
fname = "stimulation_OzCz_densities_nodes.txt"
folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\\output_SPK\\"

# cargar los datos
stimWfit_spk_pre = pd.read_csv(folder + sim_tag + fname, delimiter="\t", index_col=0)

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
stimWfit_spk = stimWfit_spk_pre.loc[stimWfit_spk_pre["node"].isin(empCluster_rois)].copy()

stimWfit_spk = stimWfit_spk.groupby(["subject", "trial", "w"]).mean().reset_index()

n_trials = stimWfit_spk["trial"].max() + 1


# Calculate percentage
baseline_spk = stimWfit_spk.loc[stimWfit_spk["w"] == 0].groupby("subject").mean().reset_index()

stimWfit_spk["percent"] = [(row["amp_fpeak"] - baseline_spk.loc[baseline_spk["subject"] == row["subject"]].amp_fpeak.values[0]) / baseline_spk.loc[baseline_spk["subject"] == row["subject"]].amp_fpeak.values[0] * 100 for i, row in stimWfit_spk.iterrows()]
# stimWfit_spk["percent"] = [((row["amp_fpeak"] / baseline_spk.loc[baseline_spk["subject"] == row["subject"]].amp_fpeak.values[0]) - 1) * 100 for i, row in stimWfit_spk.iterrows()]

stimWfit_spk_avg = stimWfit_spk.groupby(["subject", "w"]).mean().reset_index()

# Just show half the calibration constants to make a clearer picture
include_w = [sorted(set(stimWfit_spk_avg.w))[0]] + sorted(set(stimWfit_spk_avg.w))[3:-4:1]
stimWfit_spk_sub = stimWfit_spk_avg[stimWfit_spk_avg["w"].isin(include_w)]


## Substitute internal coding NEMOS by subject
for i, subj in enumerate(sorted(set(stimWfit_spk_sub.subject))):
    new_name = "Subject " + str(i+1).zfill(2)
    stimWfit_spk_sub["subject"].loc[stimWfit_spk_sub["subject"] == subj] = new_name


### A. Scatter plot with Mean line for percentage
sim="SPK"
fig = px.line(stimWfit_spk_sub, x="w", y="percent", color="subject", labels={"subject": ""}, log_x=False, markers=True)

w = np.asarray(stimWfit_spk_sub.groupby("w").mean().reset_index()["w"])
mean = np.asarray(stimWfit_spk_sub.groupby("w").mean()["percent"])
median = np.asarray(stimWfit_spk_sub.groupby("w").median()["percent"])

fig.add_trace(go.Scatter(x=w, y=mean, mode="lines", name="mean", line=dict(color='darkslategray', width=5)))
fig.add_shape(x0=33, x1=37, y0=-35, y1=65, fillcolor="lightgray", opacity=0.3, line=dict(width=1))

fig.update_layout(height=470, width=700, template="plotly_white")
fig.update_xaxes(title="Calibration constant (K) <br>i.e. Stimulation intensity")
fig.update_yaxes(title="Alpha band power change (%)", tickvals=[-50, -25, 0, 8.02, 25, 50, 75, 100, 125, 150, 175])#,tickvals=[-50, 0, 8.02, 50, 100, 150])

pio.write_image(fig, file=fig_folder + '\\R3_'+sim+'_allNemosAAL_scatterpercent_alphaRise_' + str(n_trials) + "sim.svg")
pio.write_image(fig, file=fig_folder + '\\R3_'+sim+'_allNemosAAL_scatterpercent_alphaRise_' + str(n_trials) + "sim.png")

fig.add_trace(go.Scatter(x=w, y=median, mode="lines", name="median", line=dict(color='slategray', width=4), visible="legendonly"))
pio.write_html(fig, file=fig_folder + '\\R3_'+sim+'_allNemosAAL_scatterpercent_alphaRise_' + str(n_trials) + "sim.html", auto_open=True)




## A2. plot for absolute power change bands || Is it needed? Just if some subject behaves weirdly.
fig = px.strip(stimWfit_spk_sub, x="w", y="amp_fpeak", color="subject", log_x=False,
             title="Alpha band power rise @Pareto-Occipital regions<br>(%i simulations | 10 subjects AAL)" % n_trials,
             labels={  # replaces default labels by column name
                 "w": "Calibration constant (K) <br>i.e. Stimulation intensity", "bModule": "Alpha band power (mV*Hz)"})

pio.write_html(fig, file=fig_folder + '\\R3_'+sim+'_allNemosAAL_scatterModules_alphaRise_' + str(n_trials) + "sim.html",
               auto_open=True)



## A2. plot for frequency change || Is it needed? Just if some subject behaves weirdly.
fig = px.strip(stimWfit_spk_sub, x="w", y="fpeak", color="subject", log_x=False,
             title="Alpha band power rise @Pareto-Occipital regions<br>(%i simulations | 10 subjects AAL)" % n_trials,
             labels={  # replaces default labels by column name
                 "w": "Calibration constant (K) <br>i.e. Stimulation intensity", "bModule": "Alpha band power (mV*Hz)"})

pio.write_html(fig, file=fig_folder + '\\R3_'+sim+'_allNemosAAL_scatterFreqs_alphaRise_' + str(n_trials) + "sim.html",
               auto_open=True)


## Discarding intrasubject variability: is it different at w==0 than at the fit (w=0.6)
# intraVar_base = [stimWfit_spk["percent"].loc[(stimWfit_spk["subject"] == subj) & (stimWfit_spk["w"] == 0)].values for subj in sorted(set(stimWfit_spk.subject))]
# intraVar_wfit = [stimWfit_spk["percent"].loc[(stimWfit_spk["subject"] == subj) & (stimWfit_spk["w"] > 34) & (stimWfit_spk["w"] < 36)].values for subj in sorted(set(stimWfit_spk.subject))]
#
# import pingouin as pg
#
# results = pd.DataFrame()
# for i in range(len(intraVar_base)):
#
#     temp = pd.DataFrame(pg.wilcoxon(intraVar_wfit[i], intraVar_base[i]))
#     results = results.append(temp)
#     print(pg.wilcoxon(intraVar_wfit[i], intraVar_base[i]))
#
# pg.multicomp(results["p-val"])
