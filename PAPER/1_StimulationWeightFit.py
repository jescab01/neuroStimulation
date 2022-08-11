
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
import glob

figures_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER\FIGURES\\"



## PART A:
####### Stimulation Weight Fit
fname = "PSEmpi_stimWfit_indWP-C3Ndata-pass-m06d11y2022-t11h.15m.31s - w0.29"
folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\\1_stimWeight_Fitting\PSE\\" + fname

# Load data
resultsAAL = pd.read_csv(glob.glob(folder + "\\*results.csv")[0])
n_rep = resultsAAL["rep"].max() + 1

# Calculate percentage
baseline = resultsAAL.loc[resultsAAL["w"] == 0].groupby("Subject").mean()

resultsAAL["percent"] = \
    [(row["bModule"] - baseline.loc[row["Subject"]].bModule) / baseline.loc[row["Subject"]].bModule * 100
     for i, row in resultsAAL.iterrows()]

# Just show half the calibration constants to make a clearer picture
include_w = np.arange(0, 0.41, 0.04)
resultsAAL_sub = resultsAAL[resultsAAL["w"].isin(include_w)]

## Substitute internal coding NEMOS by subject
for i, subj in enumerate(sorted(set(resultsAAL_sub.Subject))):
    new_name = "Subject " + str(i+1).zfill(2)
    resultsAAL_sub["Subject"].loc[resultsAAL_sub["Subject"] == subj] = new_name


### A. Scatter plot with Mean line for percentage
fig = px.strip(resultsAAL_sub, x="w", y="percent", color="Subject", labels={"Subject":""})

w = np.asarray(resultsAAL_sub.groupby("w").mean().reset_index()["w"])
mean = np.asarray(resultsAAL_sub.groupby("w").mean()["percent"])
median = np.asarray(resultsAAL_sub.groupby("w").median()["percent"])

fig.add_trace(go.Scatter(x=w, y=mean, mode="lines", name="mean", line=dict(color='darkslategray', width=5)))
fig.update_layout(height=470, width=900)
fig.update_xaxes(title="Calibration constant (K) <br>i.e. Stimulation Weight")
fig.update_yaxes(title="Alpha band power change (%)", tickvals=[-40, -20, 0, 8.02, 20, 40, 60, 80, 100])

pio.write_image(fig, file=figures_folder + '\\1_allNemosAAL_4scatterpercent_alphaRise_' + str(n_rep) + "sim.svg")
pio.write_image(fig, file=figures_folder + '\\1_allNemosAAL_4scatterpercent_alphaRise_' + str(n_rep) + "sim.png")

fig.add_trace(go.Scatter(x=w, y=median, mode="lines", name="median", line=dict(color='slategray', width=4), visible="legendonly"))
pio.write_html(fig, file=figures_folder + '\\1_allNemosAAL_4scatterpercent_alphaRise_' + str(n_rep) + "sim.html",
               auto_open=True)

## A2. plot for absolute power change bands || Is it needed? Just if some subject behaves weirdly.
fig = px.strip(resultsAAL_sub, x="w", y="bModule", color="Subject",
             title="Alpha band power rise @Pareto-Occipital regions<br>(%i simulations | 10 subjects AAL)" % n_rep,
             labels={  # replaces default labels by column name
                 "w": "Calibration constant (K) <br>i.e. Stimulation Weight", "bModule": "Alpha band power (mV*Hz)"})

pio.write_html(fig, file=figures_folder + '\\1_allNemosAAL_1scatterModules_alphaRise_' + str(n_rep) + "sim.html",
               auto_open=False)
