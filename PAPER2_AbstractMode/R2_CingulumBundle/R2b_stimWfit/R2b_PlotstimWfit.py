
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
import glob

fname = "PSEmpi_stimWfit_cb_indiv-m03d28y2023-t16h.49m.39s"
folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\R2_CingulumBundle\R2b_stimWfit\\"

# cargar los datos
stimWfit = pd.read_csv(glob.glob(folder + fname + "\\*results.csv")[0])
n_trials = stimWfit["trial"].max() + 1

# Calculate percentage
baseline = stimWfit.loc[stimWfit["w"] == 0].groupby("subject").mean().reset_index()

stimWfit["percent"] = [(row["amp_fpeak"] - baseline.loc[baseline["subject"] == row["subject"]].amp_fpeak.values[0]) / baseline.loc[baseline["subject"] == row["subject"]].amp_fpeak.values[0] * 100 for i, row in stimWfit.iterrows()]

# Just show half the calibration constants to make a clearer picture
include_w = sorted(set(stimWfit.w))[::4]
stimWfit_sub = stimWfit[stimWfit["w"].isin(include_w)]


## Substitute internal coding NEMOS by subject
for i, subj in enumerate(sorted(set(stimWfit_sub.subject))):
    new_name = "Subject " + str(i+1).zfill(2)
    stimWfit_sub["subject"].loc[stimWfit_sub["subject"] == subj] = new_name


### A. Scatter plot with Mean line for percentage
fig = px.strip(stimWfit_sub, x="w", y="percent", color="subject", labels={"subject": ""}, log_x=False)

w = np.asarray(stimWfit_sub.groupby("w").mean().reset_index()["w"])
mean = np.asarray(stimWfit_sub.groupby("w").mean()["percent"])
median = np.asarray(stimWfit_sub.groupby("w").median()["percent"])

fig.add_trace(go.Scatter(x=w, y=mean, mode="lines", name="mean", line=dict(color='darkslategray', width=5)))
fig.update_layout(height=470, width=900)
fig.update_xaxes(title="Calibration constant (K) <br>i.e. Stimulation Weight")
fig.update_yaxes(title="Alpha band power change (%)", tickvals=[-10, 0, 8.02, 10, 20, 30])

pio.write_image(fig, file=folder + fname + '\\R2a_allNemosAAL_4scatterpercent_alphaRise_' + str(n_trials) + "sim.svg")
pio.write_image(fig, file=folder + fname + '\\R2a_allNemosAAL_4scatterpercent_alphaRise_' + str(n_trials) + "sim.png")

fig.add_trace(go.Scatter(x=w, y=median, mode="lines", name="median", line=dict(color='slategray', width=4), visible="legendonly"))
pio.write_html(fig, file=folder + fname + '\\R2a_allNemosAAL_4scatterpercent_alphaRise_' + str(n_trials) + "sim.html",
               auto_open=True)

## A2. plot for absolute power change bands || Is it needed? Just if some subject behaves weirdly.
fig = px.strip(stimWfit_sub, x="w", y="amp_fpeak", color="subject", log_x=False,
             title="Alpha band power rise @Pareto-Occipital regions<br>(%i simulations | 10 subjects AAL)" % n_trials,
             labels={  # replaces default labels by column name
                 "w": "Calibration constant (K) <br>i.e. Stimulation Weight", "bModule": "Alpha band power (mV*Hz)"})

pio.write_html(fig, file=folder + fname + '\\R2b_allNemosAAL_1scatterModules_alphaRise_' + str(n_trials) + "sim.html",
               auto_open=True)


## Discarding intrasubject variability: is it different at w==0 than at the fit (w=0.6)
intraVar_base = [np.std(stimWfit["percent"].loc[(stimWfit["subject"] == subj) & (stimWfit["w"] == 0)].values) for subj in sorted(set(stimWfit.subject))]
intraVar_wfit = [np.std(stimWfit["percent"].loc[(stimWfit["subject"] == subj) & (stimWfit["w"] > 0.59) & (stimWfit["w"] < 0.61)].values) for subj in sorted(set(stimWfit.subject))]

import pingouin as pg

ttest = pg.wilcoxon(intraVar_wfit, intraVar_base, alternative="greater")
