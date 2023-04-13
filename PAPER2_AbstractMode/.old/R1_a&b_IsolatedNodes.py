


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px

main_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\\R1_IsolatedStim\\"

## Load NMM
nmm_tag = "R1a_arnoldTongues\\PSEmpi_nmmx2plv_stim-m03d13y2023-t19h.37m.55s"
nmm_df = pd.read_csv(main_folder + nmm_tag + "\\nmm_results.csv")



#####################################
##   PLOT1: SINGLE NODE - Absolute changes
fig = make_subplots(rows=3, cols=6, column_titles=nmm_modes_exp1+spk_modes_exp1,
                    shared_xaxes=True, shared_yaxes=False, horizontal_spacing=0.04,
                    x_title="Stimulation Frequency (Hz)", y_title="Stimulation Weight",
                    row_titles=["Node's Freq. peak", "Node's Pow @peak", "Node's Pow @fex"])

# Plot NMM
for j, mode in enumerate(nmm_modes_exp1):

    sl = True if j == 0 else False

    nmm_df_avg_cond = nmm_df_avg.loc[(nmm_df_avg["mode"] == mode) & (nmm_df_avg["weight"] != 0) & (nmm_df_avg["node"] == "stim")]

    fig.add_trace(go.Heatmap(z=nmm_df_avg_cond.fpeak, x=nmm_df_avg_cond.fex, y=nmm_df_avg_cond.weight,
                             colorscale='Turbo', showscale=sl, zmin=nmm_df_avg["fpeak"].min(), zmax=nmm_df_avg["fpeak"].max(),
                             colorbar=dict(title="Hz", thickness=4, len=0.3, y=0.85, x=1)), row=1, col=1+j)

    fig.add_trace(go.Heatmap(z=nmm_df_avg_cond.amplitude_fpeak, x=nmm_df_avg_cond.fex, y=nmm_df_avg_cond.weight,
                             showscale=sl, zmin=nmm_df_avg["amplitude_fpeak"].min(), zmax=nmm_df_avg["amplitude_fpeak"].max(),
                             colorbar=dict(title="dB", thickness=4, len=0.3, y=0.5, x=0.47)),
                  row=2, col=1+j)

    fig.add_trace(go.Heatmap(z=nmm_df_avg_cond.amplitude_fex, x=nmm_df_avg_cond.fex, y=nmm_df_avg_cond.weight,
                             showscale=sl, zmin=nmm_df_avg["amplitude_fex"].min(), zmax=nmm_df_avg["amplitude_fex"].max(),
                             colorbar=dict(title="dB", thickness=4, len=0.3, y=0.15, x=0.47)),
                  row=3, col=1+j)

# Plot spiking
for j, mode in enumerate(spk_modes_exp1):

    pos = [0.64, 0.81, 1]
    spk_df_avg_cond = spk_df_avg.loc[(spk_df_avg["mode"].str.contains(mode)) & (spk_df_avg["weight"] != 0) & (spk_df_avg["weight"] <= 2)]
    spk_df_avg_cond = spk_df_avg_cond.groupby(["node", "weight", "fex"]).mean().reset_index()

    fig.add_trace(go.Heatmap(z=spk_df_avg_cond.fpeak, x=spk_df_avg_cond.fex, y=spk_df_avg_cond.weight,
                             colorscale='Turbo', showscale=sl, zmin=nmm_df_avg["fpeak"].min(), zmax=nmm_df_avg["fpeak"].max(),
                             colorbar=dict(title="Hz", thickness=4, len=0.3, y=0.85, x=1)), row=1, col=4+j)

    fig.add_trace(go.Heatmap(z=spk_df_avg_cond.amplitude_fpeak, x=spk_df_avg_cond.fex, y=spk_df_avg_cond.weight,
                             showscale=True, # zmin=spk_df["amplitude_fpeak"].min(), zmax=spk_df["amplitude_fpeak"].max(),
                             colorbar=dict(title="dB", thickness=4, len=0.3, y=0.5, x=pos[j])),
                  row=2, col=4+j)

    fig.add_trace(go.Heatmap(z=spk_df_avg_cond.amplitude_fex, x=spk_df_avg_cond.fex, y=spk_df_avg_cond.weight,
                             showscale=True, # zmin=spk_df["amplitude_fex"].min(), zmax=spk_df["amplitude_fex"].max(),
                             colorbar=dict(title="dB", thickness=4, len=0.3, y=0.15, x=pos[j])),
                  row=3, col=4+j)

fig.write_html("PAPER2_AbstractMode/figures/paperExpress_EXP1_viz.html", auto_open=True)


##   PLOT1rel: SINGLE NODE - Relative changes
fig = make_subplots(rows=3, cols=6, column_titles=nmm_modes_exp1+spk_modes_exp1,
                    shared_xaxes=True, shared_yaxes=False, horizontal_spacing=0.04,
                    x_title="Stimulation Frequency (Hz)", y_title="Stimulation Weight",
                    row_titles=["Node's Freq. peak", "inc. Node's Pow @peak", "inc. Node's Pow @fex"])

# Plot NMM
for j, mode in enumerate(nmm_modes_exp1):

    sl = True if j == 0 else False

    nmm_df_avg_cond = nmm_df_avg.loc[(nmm_df_avg["mode"] == mode) & (nmm_df_avg["weight"] != 0) & (nmm_df_avg["node"] == "stim")]

    fig.add_trace(go.Heatmap(z=nmm_df_avg_cond.fpeak, x=nmm_df_avg_cond.fex, y=nmm_df_avg_cond.weight,
                             colorscale='Turbo', showscale=sl, zmin=nmm_df_avg["fpeak"].min(), zmax=nmm_df_avg["fpeak"].max(),
                             colorbar=dict(title="Hz", thickness=4, len=0.3, y=0.85, x=1)), row=1, col=1+j)

    fig.add_trace(go.Heatmap(z=nmm_df_avg_cond.amplitude_fpeak_rel, x=nmm_df_avg_cond.fex, y=nmm_df_avg_cond.weight,
                             showscale=sl, zmin=nmm_df_avg["amplitude_fpeak_rel"].min(), zmax=nmm_df_avg["amplitude_fpeak_rel"].max(),
                             colorbar=dict(title="dB", thickness=4, len=0.3, y=0.5, x=0.47)),
                  row=2, col=1+j)

    fig.add_trace(go.Heatmap(z=nmm_df_avg_cond.amplitude_fex_rel, x=nmm_df_avg_cond.fex, y=nmm_df_avg_cond.weight,
                             showscale=sl, zmin=nmm_df_avg["amplitude_fex_rel"].min(), zmax=nmm_df_avg["amplitude_fex_rel"].max(),
                             colorbar=dict(title="dB", thickness=4, len=0.3, y=0.15, x=0.47)),
                  row=3, col=1+j)

# Plot spiking
for j, mode in enumerate(spk_modes_exp1):

    pos = [0.64, 0.81, 1]
    spk_df_avg_cond = spk_df_avg.loc[(spk_df_avg["mode"].str.contains(mode)) & (spk_df_avg["weight"] != 0) & (spk_df_avg["weight"] <= 2)]
    spk_df_avg_cond = spk_df_avg_cond.groupby(["node", "weight", "fex"]).mean().reset_index()

    fig.add_trace(go.Heatmap(z=spk_df_avg_cond.fpeak, x=spk_df_avg_cond.fex, y=spk_df_avg_cond.weight,
                             colorscale='Turbo', showscale=sl, zmin=nmm_df_avg["fpeak"].min(), zmax=nmm_df_avg["fpeak"].max(),
                             colorbar=dict(title="Hz", thickness=4, len=0.3, y=0.85, x=1)), row=1, col=4+j)

    fig.add_trace(go.Heatmap(z=spk_df_avg_cond.amplitude_fpeak_rel, x=spk_df_avg_cond.fex, y=spk_df_avg_cond.weight,
                             showscale=True, # zmin=spk_df["amplitude_fpeak"].min(), zmax=spk_df["amplitude_fpeak"].max(),
                             colorbar=dict(title="dB", thickness=4, len=0.3, y=0.5, x=pos[j])),
                  row=2, col=4+j)

    fig.add_trace(go.Heatmap(z=spk_df_avg_cond.amplitude_fex_rel, x=spk_df_avg_cond.fex, y=spk_df_avg_cond.weight,
                             showscale=True, # zmin=spk_df["amplitude_fex"].min(), zmax=spk_df["amplitude_fex"].max(),
                             colorbar=dict(title="dB", thickness=4, len=0.3, y=0.15, x=pos[j])),
                  row=3, col=4+j)

fig.write_html("PAPER2_AbstractMode/figures/paperExpress_EXP1rel_viz.html", auto_open=True)




###############################################
##   PLOT2: COUPLE NODES - Relative changes

# 2.1 Comparing mode 'n2_g0sigma0.22' to 'xxx'
nmm_modes_exp2 = ['n2_g0sigma0',
                 'n2_g0sigma0.11',
                 'n2_g0sigma0.22',
                 'n2_g10000sigma0',
                 'n2_g10000sigma0.11',
                 'n2_g10000sigma0.22',
                 'n2_g20000sigma0',
                 'n2_g20000sigma0.11',
                 'n2_g20000sigma0.22',
                 'n2_g30000sigma0',
                 'n2_g30000sigma0.11',
                 'n2_g30000sigma0.22',
                 'n2_g3000sigma0',
                 'n2_g3000sigma0.11',
                 'n2_g3000sigma0.22']

mode_nmm = 'n2_g10000sigma0.11'

fig = make_subplots(rows=4, cols=4, column_titles=["NMM - stim", "NMM - n1", "SPK - stim", "SPK - n1"],
                    shared_xaxes=True, shared_yaxes=False, specs=[[{},{},{},{}],[{},{},{},{}],[{},{},{},{}],
                                                                  [{"colspan":2},{},{"colspan":2},{}]],
                    x_title="Stimulation Frequency (Hz)", y_title="Stimulation Weight",
                    row_titles=["Node's Freq. peak", "inc. Node's Pow @peak", "inc. Node's Pow @fex", "PLV"])

y_pos = [0.94, 0.63, 0.37, 0.125]
# Plot NMM
for j, node in enumerate(["stim", "1"]):

    sl = True if j == 0 else False

    nmm_df_avg_cond = nmm_df_avg.loc[(nmm_df_avg["mode"] == mode_nmm) & (nmm_df_avg["weight"] != 0) & (nmm_df_avg["node"] == node)]

    fig.add_trace(go.Heatmap(z=nmm_df_avg_cond.fpeak, x=nmm_df_avg_cond.fex, y=nmm_df_avg_cond.weight,
                             colorscale='Turbo', showscale=sl, zmin=nmm_df_avg["fpeak"].min(), zmax=nmm_df_avg["fpeak"].max(),
                             colorbar=dict(title="Hz", thickness=4, len=0.25, y=y_pos[0], x=1)), row=1, col=1+j)

    fig.add_trace(go.Heatmap(z=nmm_df_avg_cond.amplitude_fpeak_rel, x=nmm_df_avg_cond.fex, y=nmm_df_avg_cond.weight,
                             showscale=sl, zmin=nmm_df_avg["amplitude_fpeak_rel"].min(), zmax=nmm_df_avg["amplitude_fpeak_rel"].max(),
                             colorbar=dict(title="dB", thickness=4, len=0.25, y=y_pos[1], x=0.47)),
                  row=2, col=1+j)

    fig.add_trace(go.Heatmap(z=nmm_df_avg_cond.amplitude_fex_rel, x=nmm_df_avg_cond.fex, y=nmm_df_avg_cond.weight,
                             showscale=sl, zmin=nmm_df_avg["amplitude_fex_rel"].min(), zmax=nmm_df_avg["amplitude_fex_rel"].max(),
                             colorbar=dict(title="dB", thickness=4, len=0.25, y=y_pos[2], x=0.47)),
                  row=3, col=1+j)

## ADD PLV values
fig.add_trace(go.Heatmap(z=nmm_df_avg_cond.plv, x=nmm_df_avg_cond.fex, y=nmm_df_avg_cond.weight, colorscale="Viridis", showscale=False,
                         colorbar=dict(title="plv", len=0.25, y=0.11, x=1, thickness=4), zmin=0, zmax=1), row=4, col=1)

# Plot spiking
spk_df = pd.read_csv(main_folder + "PSE/Jaime/two_nodes_tacs.txt", delimiter="\t", index_col=0)
# Reshape mode
spk_df["mode"] = [row.simulation + str(0) for i, row in spk_df.iterrows()]
# Average out trials
spk_df_avg = spk_df.groupby(["mode", "node", "weight", "fex"]).mean().reset_index()

mode_spk='lfp0'

for j, node in enumerate([0, 1]):

    x_pos = [0.72, 1]
    spk_df_avg_cond = spk_df_avg.loc[(spk_df_avg["mode"].str.contains(mode_spk)) &
                                     (spk_df_avg["weight"] != 0) & (spk_df_avg["node"] == node)]
    spk_df_avg_cond = spk_df_avg_cond.groupby(["node", "weight", "fex"]).mean().reset_index()

    fig.add_trace(go.Heatmap(z=spk_df_avg_cond.fpeak, x=spk_df_avg_cond.fex, y=spk_df_avg_cond.weight,
                             colorscale='Turbo', showscale=sl, zmin=nmm_df_avg["fpeak"].min(), zmax=nmm_df_avg["fpeak"].max(),
                             colorbar=dict(title="Hz", thickness=4, len=0.25, y=y_pos[0], x=1)), row=1, col=3+j)

    fig.add_trace(go.Heatmap(z=spk_df_avg_cond.amplitude_fpeak_rel, x=spk_df_avg_cond.fex, y=spk_df_avg_cond.weight,
                             showscale=True, # zmin=spk_df["amplitude_fpeak"].min(), zmax=spk_df["amplitude_fpeak"].max(),
                             colorbar=dict(title="dB", thickness=4, len=0.25, y=y_pos[1], x=x_pos[j])),
                  row=2, col=3+j)

    fig.add_trace(go.Heatmap(z=spk_df_avg_cond.amplitude_fex_rel, x=spk_df_avg_cond.fex, y=spk_df_avg_cond.weight,
                             showscale=True, # zmin=spk_df["amplitude_fex"].min(), zmax=spk_df["amplitude_fex"].max(),
                             colorbar=dict(title="dB", thickness=4, len=0.25, y=y_pos[2], x=x_pos[j])),
                  row=3, col=3+j)

# Spiking PLV
spk_df = pd.read_csv(main_folder + "PSE/Jaime/two_nodes_plv_tacs.txt", delimiter="\t", index_col=0)
# Reshape mode
spk_df["mode"] = [row.simulation + str(0) for i, row in spk_df.iterrows()]
# Average out trials
spk_df_avg = spk_df.groupby(["mode", "weight", "fex"]).mean().reset_index()

spk_df_avg = spk_df_avg.loc[spk_df_avg["weight"] <= 2]
fig.add_trace(go.Heatmap(z=spk_df_avg.plv, x=spk_df_avg.fex, y=spk_df_avg.weight, colorscale="Viridis",
                         colorbar=dict(title="plv", len=0.25, y=0.11, x=1, thickness=4), zmin=0, zmax=1), row=4, col=3)


fig.update_layout(title=mode_nmm + " - " + mode_spk)
fig.write_html("PAPER2_AbstractMode/figures/paperExpress_EXP2rel_CoulpedNodes_viz.html", auto_open=True)



##     Process and plot the baseline dots      ##

simtag = "R1b_phasediffs\\PSEmpi_phasediff_stimAllConds-m03d31y2023-t16h.29m.19s"
df = pd.read_pickle(main_folder + simtag + "\\nmm_results.pkl")


## plot baseline values of fft, pow and plv

df_sub = df.loc[df["cond"] == "baseline"]

fig = px.histogram(df_sub, x="fpeak")
fig.show("browser")

fig = px.histogram(df_sub, x="amplitude_fpeak")
fig.show("browser")

import numpy as np
a = np.array(df.iloc[0, 9])  # epochs x roi x time

# plot
fig = px.line(x=np.arange(len(b)), y=np.abs(b-c))
fig.show("browser")

b = np.hstack(a[:,0,:])
c = np.hstack(a[:,1,:])

x=np.arange(0,10,0.001)
fig = px.line(x=x, y=np.sin(2*np.pi*x*10))
fig.show("browser")
import scipy.signal
analyticalSignal = scipy.signal.hilbert(np.sin(2*np.pi*x*10))
# Get instantaneous phase and amplitude envelope by channel
np.unwrap(np.angle(analyticalSignal))








"""
### This is going to be a full plot for the isolated stimulation
# The plot will be made independently for nmm y spk, 
# and finally merged with illustrations for the stimulation
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import scipy.signal

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


##########          Neural Mass Models          ###########

## 1. Load data

main_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\\R1_IsolatedStim\\"

# 1.1 Phase differences experiments
simtag = "PSEmpi_phasediff_stimAllConds-m04d01y2023-t22h.50m.48s"
df_phases = pd.read_pickle(main_folder + "R1b_phasediffs\\" + simtag + "\\nmm_results.pkl")

df_ph_single = df_phases.loc[(df_phases["mode"] == "isolatedStim_oneNode_sigma0.11") & (df_phases["node"] == "stim_Precuneus_L")]

# add several metrics: is fex==tofpeak?; number of 2pi dephases; initial phase difference signal-stim.
df_ph_single["fex==fpeak"] = [True if row["fex"] == row["fpeak"] else False for i, row in df_ph_single.iterrows()]

init_diff, n_2pi = [], []
for i, row in df_ph_single.iterrows():

    freq = row["fex"]
    t = 12 if row["cond"] == "baseline" else 34
    timepoints = np.linspace(0, t, len(row["fPhase"]))

    sin = np.sin(2 * np.pi * freq * timepoints)

    analyticalSignal = scipy.signal.hilbert(sin)
    sin_phase = np.unwrap(np.angle(analyticalSignal))

    diff = row["fPhase"] - sin_phase

    # save initial difference value
    init_diff.append(np.average(diff[10:20]))

    # calculate 2pi dephases
    pi2 = diff + np.pi // (2 * np.pi)
    pi2_changes = np.array([val - pi2[i] for i, val in enumerate(pi2[1:])])

    n_2pi.append(sum(pi2_changes)/t)

df_ph_single["init_ph_diff"] = init_diff
df_ph_single["n_2pi_per_s"] = n_2pi

# 1.2 Arnold tongue experiments
simtag = "PSEmpi_nmm_stimAllConds-m03d31y2023-t16h.16m.11s"
df_arnold = pd.read_pickle(main_folder + "R1a_arnoldTongues\\" + simtag + "\\nmm_results.pkl")


df_arnold_avg = df_arnold[["mode", "node", "weight", "fex", "fpeak", "amplitude_fpeak", "amplitude_fex"]].groupby(["mode", "node", "weight", "fex"]).mean().reset_index()  # average out repetitions

# 1.2a Pre-allocate relative changes and compute them
df_arnold_avg["amplitude_fex_rel"], df_arnold_avg["amplitude_fpeak_rel"] = 0, 0
for mode in list(set(df_arnold_avg["mode"].values)):
    df_arnold_avg["amplitude_fex_rel"].loc[df_arnold_avg["mode"] == mode] = df_arnold_avg["amplitude_fex"].loc[df_arnold_avg["mode"]==mode] / df_arnold_avg["amplitude_fex"].loc[(df_arnold_avg["weight"] == 0) & (df_arnold_avg["mode"]==mode)].mean()
    df_arnold_avg["amplitude_fpeak_rel"].loc[df_arnold_avg["mode"] == mode] = df_arnold_avg["amplitude_fpeak"].loc[df_arnold_avg["mode"]==mode] / df_arnold_avg["amplitude_fpeak"].loc[(df_arnold_avg["weight"] == 0) & (df_arnold_avg["mode"]==mode)].mean()




## 3. PLOTTING
n_cols = 12
fig = make_subplots(rows=4, cols=n_cols)

df_base = df_ph_single.loc[df_ph_single["cond"] == "baseline"]

# 3.1 Single node simulations
# 3.1a violin plots for baseline (wo/ stimulation) conditions on single node [frequency, power, plv]
df_sub = df_base.loc[(df_base["node"] == "stim_Precuneus_L") & (df_base["mode"] == "isolatedStim_oneNode_sigma0.11")]  # selects only single node simulation
fig.add_trace(go.Violin(x=["Frequency"]*len(df_sub["fpeak"].values), y=df_sub["fpeak"].values), row=1, col=n_cols)
fig.add_trace(go.Violin(x=["Power"]*len(df_sub["amplitude_fpeak"].values), y=df_sub["amplitude_fpeak"].values), row=1, col=n_cols-1)

plvs = [row["plv"][1] for i, row in df_sub.iterrows()]
fig.add_trace(go.Violin(x=["PLV"]*len(plvs), y=plvs), row=1, col=n_cols-2)


# 3.1b arnold tongue for the single node stimulation
df_sub = df_arnold.loc[(df_arnold["mode"] == "isolatedStim_oneNode_sigma0.11") & (df_arnold["node"] == "stim_Precuneus_L")]

df_sub = df_sub.groupby(["weight", "fex"]).mean().reset_index()

fig.add_trace(go.Heatmap(z=df_sub.fpeak, x=df_sub.fex, y=df_sub.weight,
                         colorscale='Turbo', showscale=False,
                         colorbar=dict(title="Hz", thickness=4, len=0.3, y=0.85, x=1)), row=1, col=n_cols-3)

fig.add_trace(go.Heatmap(z=df_sub.amplitude_fpeak, x=df_sub.fex, y=df_sub.weight,
                         showscale=False,
                         colorbar=dict(title="dB", thickness=4, len=0.3, y=0.5, x=0.47)),
              row=1, col=n_cols-4)


# 3.1c NOT::Plot time_to_entrainment (if entrainment) by init_phase_diff (after transient) per frequency.
# It's 4s of transient; 12s of baseline; 34s of stim.
df_sub = df_ph_single.loc[df_ph_single["cond"] == "baseline"]
fig.add_trace(go.Violin(x=df_sub.fex, y=df_sub.n_2pi_per_s), row=1, col=n_cols-5)  # baseline

df_sub = df_ph_single.loc[df_ph_single["cond"] == "stimulation"]
fig.add_trace(go.Violin(x=df_sub.fex, y=df_sub.n_2pi_per_s), row=1, col=n_cols-5)  # stimulation


# ## UNDERSTANDING PHASE DIFFERENCES
# fig_aux = px.scatter(df_ph_single, x="init_ph_diff", y="n_2pi_per_s", color="fex")
# fig_aux.show("browser")

# fig_aux = px.strip(df_ph_single, x="fex", y="n_2pi_per_s", color="fex==fpeak")
# fig_aux.show("browser")

# fig_aux = px.strip(df_ph_single, x="fex", y="n_2pi_per_s", color="cond")
# fig_aux.show("browser")

# # PLOT the phases and difference to understand whats happening
# fig_aux = go.Figure()
# df_sub = df_ph_single.iloc[5, :]

# freq = df_sub["fex"]
# t = 12 if df_sub["cond"] == "baseline" else 34
# timepoints = np.linspace(0, t, len(df_sub["fPhase"]))

# # Phase of the node
# sig_phase = np.arctan2(np.sin(df_sub["fPhase"]), np.cos(df_sub["fPhase"]))
# fig_aux.add_trace(go.Scatter(x=timepoints, y=sig_phase, name="phase_signal"))

# # Phase of the stimulation
# sin_1 = np.sin(2 * np.pi * freq * timepoints)
# analyticalSignal = scipy.signal.hilbert(sin_1)
# sin_phase = np.angle(analyticalSignal)
# fig_aux.add_trace(go.Scatter(x=timepoints, y=sin_phase, name="phase_stim"))
#
# # Warped difference of phases
# fig_aux.add_trace(go.Scatter(x=timepoints, y=sig_phase-sin_phase, name="diff"))
#
# # unwraped difference of phases
# sin_phase = np.unwrap(np.angle(analyticalSignal))
# fig_aux.add_trace(go.Scatter(x=timepoints, y=df_sub["fPhase"]-sin_phase, name="diff_unwarped"))
# fig_aux.update_layout(xaxis=dict(title="time (seconds)"), yaxis=dict(title="radians"))
#
# fig_aux.show("browser")



# 3.2 Coupled node simulations
# 3.2a violin plots for baseline (wo/ stimulation) conditions on single node [frequency, power, plv]
df_base = df_phases.loc[(df_phases["node"] == "stim_Precuneus_L") & (df_phases["mode"] == "isolatedStim_twoNodes_sigma0.11") & (df_phases["cond"] == "baseline")]
fig.add_trace(go.Violin(x=["Frequency"]*len(df_base["fpeak"].values), y=df_base["fpeak"].values), row=2, col=n_cols)
fig.add_trace(go.Violin(x=["Power"]*len(df_base["amplitude_fpeak"].values), y=df_base["amplitude_fpeak"].values), row=2, col=n_cols-1)

df_base = df_phases.loc[(df_phases["node"] == "Precuneus_R") & (df_phases["mode"] == "isolatedStim_twoNodes_sigma0.11") & (df_phases["cond"] == "baseline")]
fig.add_trace(go.Violin(x=["Frequency"]*len(df_base["fpeak"].values), y=df_base["fpeak"].values), row=2, col=n_cols)
fig.add_trace(go.Violin(x=["Power"]*len(df_base["amplitude_fpeak"].values), y=df_base["amplitude_fpeak"].values), row=2, col=n_cols-1)

df_base = df_phases.loc[(df_phases["node"] == "stim_Precuneus_L") & (df_phases["mode"] == "isolatedStim_twoNodes_sigma0.11") & (df_phases["cond"] == "baseline")]
plvs = [row["plv"][1] for i, row in df_base.iterrows()]
fig.add_trace(go.Violin(x=["PLV"]*len(plvs), y=plvs), row=2, col=n_cols-2)


# 3.2b arnold tongue for the single node stimulation
df_sub = df_arnold.loc[(df_arnold["mode"] == "isolatedStim_twoNodes_sigma0.11") & (df_arnold["node"] == "stim_Precuneus_L")]
df_sub = df_sub.groupby(["weight", "fex"]).mean().reset_index()
fig.add_trace(go.Heatmap(z=df_sub.fpeak, x=df_sub.fex, y=df_sub.weight,
                         colorscale='Turbo', showscale=False,
                         colorbar=dict(title="Hz", thickness=4, len=0.3, y=0.85, x=1)), row=2, col=n_cols-3)

fig.add_trace(go.Heatmap(z=df_sub.amplitude_fpeak, x=df_sub.fex, y=df_sub.weight,
                         showscale=False,
                         colorbar=dict(title="dB", thickness=4, len=0.3, y=0.5, x=0.47)),
              row=2, col=n_cols-4)


df_sub = df_arnold.loc[(df_arnold["mode"] == "isolatedStim_twoNodes_sigma0.11") & (df_arnold["node"] == "Precuneus_R")]
df_sub = df_sub.groupby(["weight", "fex"]).mean().reset_index()
fig.add_trace(go.Heatmap(z=df_sub.fpeak, x=df_sub.fex, y=df_sub.weight,
                         colorscale='Turbo', showscale=False,
                         colorbar=dict(title="Hz", thickness=4, len=0.3, y=0.85, x=1)), row=2, col=n_cols-5)

fig.add_trace(go.Heatmap(z=df_sub.amplitude_fpeak, x=df_sub.fex, y=df_sub.weight,
                         showscale=False,
                         colorbar=dict(title="dB", thickness=4, len=0.3, y=0.5, x=0.47)),
              row=2, col=n_cols-6)


plvs = [row["plv"][0] for i, row in df_sub.iterrows()]

fig.add_trace(go.Heatmap(z=plvs, x=df_sub.fex, y=df_sub.weight,
                         showscale=False, colorscale="RdBu",
                         colorbar=dict(title="dB", thickness=4, len=0.3, y=0.5, x=0.47)),
              row=2, col=n_cols-7)




# 3.2 Cingulum bundle simulations :: stim Precuneus_L
# 3.2a violin plots for baseline (wo/ stimulation) conditions on single node [frequency, power, plv]
df_base = df_phases.loc[(df_phases["node"] == "stim_Precuneus_L") & (df_phases["mode"] == "isolatedStim_twoNodes_sigma0.11") & (df_phases["cond"] == "baseline")]
fig.add_trace(go.Violin(x=["Frequency"]*len(df_base["fpeak"].values), y=df_base["fpeak"].values), row=2, col=n_cols)
fig.add_trace(go.Violin(x=["Power"]*len(df_base["amplitude_fpeak"].values), y=df_base["amplitude_fpeak"].values), row=2, col=n_cols-1)

df_base = df_phases.loc[(df_phases["node"] == "Precuneus_R") & (df_phases["mode"] == "isolatedStim_twoNodes_sigma0.11") & (df_phases["cond"] == "baseline")]
fig.add_trace(go.Violin(x=["Frequency"]*len(df_base["fpeak"].values), y=df_base["fpeak"].values), row=2, col=n_cols)
fig.add_trace(go.Violin(x=["Power"]*len(df_base["amplitude_fpeak"].values), y=df_base["amplitude_fpeak"].values), row=2, col=n_cols-1)

df_base = df_phases.loc[(df_phases["node"] == "stim_Precuneus_L") & (df_phases["mode"] == "isolatedStim_twoNodes_sigma0.11") & (df_phases["cond"] == "baseline")]
plvs = [row["plv"][1] for i, row in df_base.iterrows()]
fig.add_trace(go.Violin(x=["PLV"]*len(plvs), y=plvs), row=2, col=n_cols-2)


# 3.2b arnold tongue for the single node stimulation
df_sub = df_arnold.loc[(df_arnold["mode"] == "isolatedStim_twoNodes_sigma0.11") & (df_arnold["node"] == "stim_Precuneus_L")]
df_sub = df_sub.groupby(["weight", "fex"]).mean().reset_index()
fig.add_trace(go.Heatmap(z=df_sub.fpeak, x=df_sub.fex, y=df_sub.weight,
                         colorscale='Turbo', showscale=False,
                         colorbar=dict(title="Hz", thickness=4, len=0.3, y=0.85, x=1)), row=2, col=n_cols-3)

fig.add_trace(go.Heatmap(z=df_sub.amplitude_fpeak, x=df_sub.fex, y=df_sub.weight,
                         showscale=False,
                         colorbar=dict(title="dB", thickness=4, len=0.3, y=0.5, x=0.47)),
              row=2, col=n_cols-4)


df_sub = df_arnold.loc[(df_arnold["mode"] == "isolatedStim_twoNodes_sigma0.11") & (df_arnold["node"] == "Precuneus_R")]
df_sub = df_sub.groupby(["weight", "fex"]).mean().reset_index()
fig.add_trace(go.Heatmap(z=df_sub.fpeak, x=df_sub.fex, y=df_sub.weight,
                         colorscale='Turbo', showscale=False,
                         colorbar=dict(title="Hz", thickness=4, len=0.3, y=0.85, x=1)), row=2, col=n_cols-5)

fig.add_trace(go.Heatmap(z=df_sub.amplitude_fpeak, x=df_sub.fex, y=df_sub.weight,
                         showscale=False,
                         colorbar=dict(title="dB", thickness=4, len=0.3, y=0.5, x=0.47)),
              row=2, col=n_cols-6)


plvs = [row["plv"][0] for i, row in df_sub.iterrows()]

fig.add_trace(go.Heatmap(z=plvs, x=df_sub.fex, y=df_sub.weight,
                         showscale=False, colorscale="RdBu",
                         colorbar=dict(title="dB", thickness=4, len=0.3, y=0.5, x=0.47)),
              row=2, col=n_cols-7)



fig.update_layout(violinmode="group")
fig.show("browser")



##########          Spiking Neural Network          ###########

## 1. Load data
spk_df = pd.read_csv(main_folder + "PSE/Jaime/one_node_tacs.txt", delimiter="\t", index_col=0)
# Reshape mode
spk_df["mode"] = [row.simulation + str(row["mode"]) for i, row in spk_df.iterrows()]
# Average out trials
spk_df_avg = spk_df.groupby(["mode", "node", "weight", "fex"]).mean().reset_index()

spk_modes_exp1 = ['lfp0', 'lfp1', 'lfp2']
