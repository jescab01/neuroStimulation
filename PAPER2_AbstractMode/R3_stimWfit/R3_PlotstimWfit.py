
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
import glob

fig_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\Figures\\"

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

stimWfit_nmm_sub["sim"] = "NMM"

## 0b. Load SPK data

fname = "stimulation_OzCz_cluster.txt"
folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\\output_SPK\\"

# cargar los datos
stimWfit_spk = pd.read_csv(folder + fname, delimiter="\t", index_col=0)

n_trials = stimWfit_spk["trial"].max() + 1

# Calculate percentage
baseline_spk = stimWfit_spk.loc[stimWfit_spk["w"] == 0].groupby("subject").mean().reset_index()

stimWfit_spk["percent"] = [(row["amp_fpeak"] - baseline_spk.loc[baseline_spk["subject"] == row["subject"]].amp_fpeak.values[0]) / baseline_spk.loc[baseline_spk["subject"] == row["subject"]].amp_fpeak.values[0] * 100 for i, row in stimWfit_spk.iterrows()]

# Just show half the calibration constants to make a clearer picture
include_w = [sorted(set(stimWfit_spk.w))[0]] + sorted(set(stimWfit_spk.w))[3::2]
stimWfit_spk_sub = stimWfit_spk[stimWfit_spk["w"].isin(include_w)]

stimWfit_spk_sub["sim"] = "SPK"


## 0c. MERGE datasets
stimWfit = pd.concat([stimWfit_nmm_sub.iloc[:, [0,3,4,5,6,7,8,9]], stimWfit_spk_sub.iloc[:, [0,2,3,5,9,11,-2,-1]]])



## Substitute internal coding NEMOS by subject
for i, subj in enumerate(sorted(set(stimWfit.subject))):
    new_name = "Subject " + str(i+1).zfill(2)
    stimWfit["subject"].loc[stimWfit["subject"] == subj] = new_name



### A. Scatter plot with Mean line for percentage
for sim in ["NMM", "SPK"]:
    fig = px.strip(stimWfit.loc[stimWfit["sim"]==sim], x="w", y="percent", color="subject", labels={"subject": ""}, log_x=False)

    w = np.asarray(stimWfit.loc[stimWfit["sim"]==sim].groupby("w").mean().reset_index()["w"])
    mean = np.asarray(stimWfit.loc[stimWfit["sim"]==sim].groupby("w").mean()["percent"])
    median = np.asarray(stimWfit.loc[stimWfit["sim"]==sim].groupby("w").median()["percent"])

    fig.add_trace(go.Scatter(x=w, y=mean, mode="lines", name="mean", line=dict(color='darkslategray', width=5)))
    fig.update_layout(height=470, width=900)
    fig.update_xaxes(title="Calibration constant (K) <br>i.e. Stimulation Weight")
    if sim == "NMM":
        fig.update_yaxes(title="Alpha band power change (%)", tickvals=[-10, 0, 8.02, 10, 20, 30])
    elif sim == "SPK":
        fig.update_yaxes(title="Alpha band power change (%)", tickvals=[-50, 0, 8.02, 50, 100, 150])#, range=[-55, 100])#,tickvals=[-50, 0, 8.02, 50, 100, 150])

    pio.write_image(fig, file=fig_folder + '\\R3_'+sim+'_allNemosAAL_scatterpercent_alphaRise_' + str(n_trials) + "sim.svg")
    pio.write_image(fig, file=fig_folder + '\\R3_'+sim+'_allNemosAAL_scatterpercent_alphaRise_' + str(n_trials) + "sim.png")

    fig.add_trace(go.Scatter(x=w, y=median, mode="lines", name="median", line=dict(color='slategray', width=4), visible="legendonly"))
    pio.write_html(fig, file=fig_folder + '\\R3_'+sim+'_allNemosAAL_scatterpercent_alphaRise_' + str(n_trials) + "sim.html", auto_open=True)

    ## A2. plot for absolute power change bands || Is it needed? Just if some subject behaves weirdly.
    fig = px.strip(stimWfit.loc[stimWfit["sim"] == sim], x="w", y="amp_fpeak", color="subject", log_x=False,
                 title="Alpha band power rise @Pareto-Occipital regions<br>(%i simulations | 10 subjects AAL)" % n_trials,
                 labels={  # replaces default labels by column name
                     "w": "Calibration constant (K) <br>i.e. Stimulation Weight", "bModule": "Alpha band power (mV*Hz)"})

    pio.write_html(fig, file=fig_folder + '\\R3_'+sim+'_allNemosAAL_scatterModules_alphaRise_' + str(n_trials) + "sim.html",
                   auto_open=True)




# ## Discarding intrasubject variability: is it different at w==0 than at the fit (w=0.6)
# intraVar_base = [np.std(stimWfit["percent"].loc[(stimWfit["subject"] == subj) & (stimWfit["w"] == 0)].values) for subj in sorted(set(stimWfit.subject))]
# intraVar_wfit = [np.std(stimWfit["percent"].loc[(stimWfit["subject"] == subj) & (stimWfit["w"] > 0.59) & (stimWfit["w"] < 0.61)].values) for subj in sorted(set(stimWfit.subject))]
#
# import pingouin as pg
#
# ttest = pg.wilcoxon(intraVar_wfit, intraVar_base, alternative="greater")
