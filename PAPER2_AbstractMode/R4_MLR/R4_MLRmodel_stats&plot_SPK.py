
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import pingouin as pg
from tvb.simulator.lab import connectivity
import scipy
import scipy.io

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_dataOLD2\\"
fig_folder = "E:\\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\Figures\\"
hist_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\R1_Histograms\\"

# Define regions implicated in Functional analysis: remove  Cerebelum, Thalamus, Caudate (i.e. subcorticals)
cingulum_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                 'Insula_L', 'Insula_R',
                 'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                 'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                 'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R',
                 'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                 'Parietal_Inf_R', 'Precuneus_L', 'Precuneus_R',
                 'Thalamus_L', 'Thalamus_R']



#### B. SPIKING NEURAL NETWORK

spk_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\\output_SPK\\"
sim_tag = "MLR\\"
results = pd.read_csv(spk_folder + sim_tag + "stimulation_OzCz_densities_nodes.txt", delimiter="\t", index_col=0)

## 1. Prepare the dataframe:
# 1.1 Separate in baseline and stimulation dataframes
base = results.loc[results["stage"] == "baseline"]
stim = results.loc[results["stage"] == "stimulation"]

del results

# 1.2 Create df with deltas of amp_fbase pre vs. stim (y), fpeak pre vs. stim, and fpeak-initialPeak in pre
df = base.copy()

## Dependent variables
df["delta_ampfex"] = (stim.amp_fex.values - base.amp_fex.values) / base.amp_fex.values
df["delta_ampfpeak"] = (stim.amp_fpeak.values - base.amp_fpeak.values) / base.amp_fpeak.values
df["delta_fpeak"] = (stim.fpeak.values - base.fpeak.values)  # **2?

## Independent variables
df["delta_initPeak"] = (base.fpeak.values - stim.fex.values)  # **2?
df["abs_delta_initPeak"] = abs(base.fpeak.values - stim.fex.values)  # **2?


# 1.3 add SC and ef_mags per roi and subject
w, working_points = 35, ["NEMOS_035", "NEMOS_049", "NEMOS_050", "NEMOS_058", "NEMOS_059",
                         "NEMOS_064", "NEMOS_065", "NEMOS_071",  "NEMOS_075",  "NEMOS_077"]


df["ef_mag"], df["sc"], df["abs_ef_mag"], df["log_sc"] = np.nan, np.nan, np.nan, np.nan
df["EFdist_modes"], df["EFdist_mode1"], df["EFdist_mode2"], df["EFdist_mode3"] = np.nan, np.nan, np.nan, np.nan
df["EFdist_skew"], df["abs_EFdist_skew"], df["EFdist_kurtosis"], df["EFdist_avg"], df["abs_EFdist_avg"] = np.nan, np.nan, np.nan, np.nan, np.nan
df["abs_ef_mag*sc"], df["abs_ef_mag*sc"] = np.nan, np.nan

for emp_subj in working_points:

    conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2_pass.zip")
    conn.weights = conn.scaled_weights(mode="tract")

    # load text with FC rois; check if match SC
    SClabs = list(conn.region_labels)
    SC_cb_idx = [SClabs.index(roi) for roi in
                 cingulum_rois]  # find indexes in FClabs that matches cortical_rois
    conn.weights = conn.weights[:, SC_cb_idx]

    weighting = np.loadtxt(
        ctb_folder + 'CurrentPropagationModels/' + emp_subj + '-efnorm_mag-roast_OzCzModel-AAL2.txt',
        delimiter=",")

    histograms = scipy.io.loadmat(hist_folder + emp_subj + "-ROIvals_orth-roast_OzCzModel.mat")

    for i, roi in enumerate(conn.region_labels):
        if roi in cingulum_rois:
            ef_orthvals = np.squeeze(histograms["ROIvals"][0, i][0])

            # Find modes
            kde = scipy.stats.gaussian_kde(ef_orthvals)
            xseq = np.linspace(np.min(ef_orthvals), np.max(ef_orthvals), 1001)
            modes_id = scipy.signal.find_peaks(kde(xseq))[0]
            modes = [xseq[id] for id in modes_id]

            df.loc[(df["subject"] == emp_subj) & (df["node"] == roi), "EFdist_mode1"] = modes[0]
            if len(modes) > 1:
                df.loc[(df["subject"] == emp_subj) & (df["node"] == roi), "EFdist_mode2"] = modes[1]
            if len(modes) > 2:
                df.loc[(df["subject"] == emp_subj) & (df["node"] == roi), "EFdist_mode3"] = modes[2]

            df.loc[(df["subject"] == emp_subj) & (df["node"] == roi), "EFdist_modes"] = len(modes)
            df.loc[(df["subject"] == emp_subj) & (df["node"] == roi), "EFdist_skew"] = scipy.stats.skew(ef_orthvals)
            df.loc[(df["subject"] == emp_subj) & (df["node"] == roi), "abs_EFdist_skew"] = abs(scipy.stats.skew(ef_orthvals))

            df.loc[(df["subject"] == emp_subj) & (df["node"] == roi), "EFdist_kurtosis"] = scipy.stats.kurtosis(ef_orthvals)

            df.loc[(df["subject"] == emp_subj) & (df["node"] == roi), "EFdist_avg"] = np.average(ef_orthvals)
            df.loc[(df["subject"] == emp_subj) & (df["node"] == roi), "abs_EFdist_avg"] = np.average(abs(ef_orthvals))

            # subset del dataframe para el sujeto y la roi, y añades el sc medio y el ef_mag
            df.loc[(df["subject"] == emp_subj) & (df["node"] == roi), "sc"] = np.average(conn.weights[i, conn.weights[i, :] != 0])
            df.loc[(df["subject"] == emp_subj) & (df["node"] == roi), "ef_mag"] = weighting[i]
            df.loc[(df["subject"] == emp_subj) & (df["node"] == roi), "log_sc"] = np.log2(np.average(conn.weights[i, conn.weights[i, :] != 0]))
            df.loc[(df["subject"] == emp_subj) & (df["node"] == roi), "abs_ef_mag"] = weighting[i]**2
            df.loc[(df["subject"] == emp_subj) & (df["node"] == roi), "abs_ef_mag*fc"] = abs(weighting[i]) * df.loc[(df["subject"] == emp_subj) & (df["node"] == roi), "plv_mean"]
            df.loc[(df["subject"] == emp_subj) & (df["node"] == roi), "abs_ef_mag*sc"] = abs(weighting[i]) * np.log(np.average(conn.weights[i, conn.weights[i, :] != 0]))



df_avg = df.groupby(["subject", "node"]).mean().reset_index()

# 1.4 Normalize metrics involved in the analysis
variables = ["delta_ampfex", "delta_ampfpeak", "delta_fpeak", "amp_fex", "delta_initPeak", "abs_delta_initPeak", "ef_mag", "abs_ef_mag",
             "sc", "log_sc", "plv_mean", "abs_ef_mag*fc", "abs_ef_mag*sc",
             "EFdist_modes", "EFdist_skew", "abs_EFdist_skew", "EFdist_kurtosis", "EFdist_avg", "abs_EFdist_avg"]
for var in variables:
    df_avg["norm_" + var] = (df_avg[var] - df_avg[var].mean()) / df_avg[var].std()



## 2.1 Explore scatters
import plotly.express as px
import plotly.io as pio

cmap = px.colors.qualitative.Plotly
fig = make_subplots(rows=5, cols=2, column_titles=["delta_ampfex", "delta_fpeak"], vertical_spacing=0.1)

fig.add_trace(go.Scatter(x=df_avg.norm_sc, y=df_avg.norm_delta_ampfex, name="norm_sc", legendgroup="norm_sc", marker=dict(color=cmap[0])), row=1, col=1)
fig.add_trace(go.Scatter(x=df_avg.norm_sc, y=df_avg.norm_delta_fpeak, name="norm_sc", legendgroup="norm_sc", showlegend=False, marker=dict(color=cmap[0])), row=1, col=2)

fig.add_trace(go.Scatter(x=df_avg.norm_log_sc, y=df_avg.norm_delta_ampfex, name="norm_log_sc", legendgroup="norm_log_sc", marker=dict(color=cmap[1])), row=1, col=1)
fig.add_trace(go.Scatter(x=df_avg.norm_log_sc, y=df_avg.norm_delta_fpeak, name="norm_log_sc", legendgroup="norm_log_sc", showlegend=False, marker=dict(color=cmap[1])), row=1, col=2)

fig.add_trace(go.Scatter(x=df_avg.norm_ef_mag, y=df_avg.norm_delta_ampfex, name="norm_ef_mag", legendgroup="norm_ef_mag", marker=dict(color=cmap[2])), row=2, col=1)
fig.add_trace(go.Scatter(x=df_avg.norm_ef_mag, y=df_avg.norm_delta_fpeak, name="norm_ef_mag", legendgroup="norm_ef_mag", showlegend=False, marker=dict(color=cmap[2])), row=2, col=2)

fig.add_trace(go.Scatter(x=df_avg.norm_abs_ef_mag, y=df_avg.norm_delta_ampfex, name="norm_abs_ef_mag", legendgroup="norm_abs_ef_mag", marker=dict(color=cmap[3])), row=2, col=1)
fig.add_trace(go.Scatter(x=df_avg.norm_abs_ef_mag, y=df_avg.norm_delta_fpeak, name="norm_abs_ef_mag", legendgroup="norm_abs_ef_mag", showlegend=False, marker=dict(color=cmap[3])), row=2, col=2)

fig.add_trace(go.Scatter(x=df_avg.norm_plv_mean, y=df_avg.norm_delta_ampfex, name="norm_plv_mean", legendgroup="norm_plv_mean", marker=dict(color=cmap[4])), row=3, col=1)
fig.add_trace(go.Scatter(x=df_avg.norm_plv_mean, y=df_avg.norm_delta_fpeak, name="norm_plv_mean", legendgroup="norm_plv_mean", showlegend=False, marker=dict(color=cmap[4])), row=3, col=2)

fig.add_trace(go.Scatter(x=df_avg.norm_delta_initPeak, y=df_avg.norm_delta_ampfex, name="norm_dInitPeak", legendgroup="norm_dInitPeak", marker=dict(color=cmap[5])), row=4, col=1)
fig.add_trace(go.Scatter(x=df_avg.norm_delta_initPeak, y=df_avg.norm_delta_fpeak, name="norm_dInitPeak", legendgroup="norm_dInitPeak", showlegend=False, marker=dict(color=cmap[5])), row=4, col=2)

fig.add_trace(go.Scatter(x=df_avg.norm_abs_delta_initPeak, y=df_avg.norm_delta_ampfex, name="norm_abs_dInitPeak", legendgroup="norm_abs_dInitPeak", marker=dict(color=cmap[7])), row=4, col=1)
fig.add_trace(go.Scatter(x=df_avg.norm_abs_delta_initPeak, y=df_avg.norm_delta_fpeak, name="norm_abs_dInitPeak", legendgroup="norm_abs_dInitPeak", showlegend=False, marker=dict(color=cmap[7])), row=4, col=2)

fig.add_trace(go.Scatter(x=df_avg.norm_sc, y=df_avg.norm_plv_mean, marker=dict(color=cmap[6])), row=5, col=2)

fig.update_traces(mode="markers")
fig.update_layout(xaxis1=dict(title="Mean SC"), yaxis1=dict(title="\u0394 power"),
                  xaxis2=dict(title="Mean SC"), yaxis2=dict(title="\u0394 fpeak"),
                  xaxis3=dict(title="|Orth. EF magnitude|"), yaxis3=dict(title="\u0394 power"),
                  xaxis4=dict(title="|Orth. EF magnitude|"), yaxis4=dict(title="\u0394 fpeak"),
                  xaxis5=dict(title="Mean PLV (pre-stimulation)"), yaxis5=dict(title="\u0394 power"),
                  xaxis6=dict(title="Mean PLV (pre-stimulation)"), yaxis6=dict(title="\u0394 fpeak"),
                  xaxis7=dict(title="\u0394 Initial peak (to stim)"), yaxis7=dict(title="\u0394 power"),
                  xaxis8=dict(title="\u0394 Initial peak (to stim)"), yaxis8=dict(title="\u0394 fpeak"))

pio.write_image(fig, fig_folder + "abs.svg")

fig.show("browser")



## 3. STATS
import statsmodels.api as sm
import seaborn as sns
import scipy.stats as stats
from statsmodels.robust.scale import mad

# import pingouin as pg
# ind_vars = ["sc", "log_sc", "ef_mag", "abs_ef_mag", "plv_mean", "delta_initPeak", "abs_delta_initPeak", "EFdist_modes", "EFdist_skew", "EFdist_kurtosis", "EFdist_avg", "abs_EFdist_avg"]
# mlr = pg.linear_regression(df_avg[ind_vars], df_avg["delta_ampfex"], relimp=True)

ind_vars = ["norm_abs_ef_mag", "norm_log_sc", "norm_plv_mean",  "norm_abs_delta_initPeak",
             "norm_EFdist_modes", "norm_EFdist_skew", "norm_EFdist_kurtosis"]


model = sm.OLS(df_avg["norm_delta_ampfex"], sm.add_constant(df_avg[ind_vars])).fit()
model.summary()


#### 4. ROBUST LINEAR REGRESSION - This one ##
model_robust = sm.RLM(df_avg["norm_delta_ampfex"], sm.add_constant(df_avg[ind_vars])).fit()
model_robust.summary()
sm.robust.scale.mad(model_robust.resid)


#### 3.1 MULTIPLE LINEAR REGRESSION MODEL
# Compute it and check assumptions [NORMALITY NOT MET].
# max log likelihood = ...
model = sm.OLS(df_avg["norm_delta_ampfex"], sm.add_constant(df_avg[ind_vars])).fit()
model.summary()

## ASSUMPTIONS:
## 3.1.a NORMALITY of the residuals [not met]
# - approach 1
sm.qqplot(df_avg[["norm_delta_ampfex"]].values[:, 0], line="45")
# - approach 2
residuals = model.resid
statistic, p_val = stats.normaltest(residuals)
# Print results
print('Normality test statistic: ', statistic)
print('P-value: ', p_val)
if p_val < 0.05:
    print('Residuals are not normally distributed')
else:
    print('Residuals are normally distributed')

fig = px.histogram(residuals)
fig.show("browser")


# 3.1.b LINEARITY (linear relation between variables) [NOT MET]
pred = model.predict()

fig = make_subplots(rows=1, cols=len(ind_vars), shared_yaxes=True, y_title="Residuals")
for i, var in enumerate(ind_vars):
    fig.add_trace(go.Scatter(x=df_avg[var], y=residuals, name=var, mode="markers"), row=1, col=i+1)
    fig["layout"]["xaxis"+str(i+1)]["title"] = var
fig.show("browser")

# 3.1.c Multicolinearity (correlation between variables) # VIF
corr = df_avg[["delta_fbase"] + ind_vars].corr()



## 4 PLOT for paper
## Size by node strength
weights = df_avg.sc.values
weights_norm = weights/max(weights)
range_size = [5, 30]
size = weights_norm * (range_size[1]-range_size[0]) + (range_size[0] - np.min(weights_norm))


vars = [("abs_ef_mag", "Norm. Electric field mean\u00B2 (mV/mm)<br>Distribution mean", [1, 2, 2]),

        ("EFdist_modes", "Distribution modes", [2,1, 4]),
        ("EFdist_skew", "Distribution skewness",[2,2, 5]),
        ("EFdist_kurtosis", "Distribution Kurtosis", [2,3, 6]),

        ("log_sc", "log. Node strength",[3,1, 7]),
        ("plv_mean", "PLV mean", [3,2, 8]),
        ("delta_initPeak", "\u0394 Frequency (Hz)",[3,3, 9])]

fig = make_subplots(rows=3, cols=3, shared_yaxes=True, vertical_spacing=0.15, y_title="\u0394 Alpha power (dB)")

for j, (var, title, [row, col, ax]) in enumerate(vars):

    hovertext2d = [row.subject + " - " + row.node + "<br>" +
                   str(round(row.delta_ampfex, 3)) + " dB - \u0394 power<br>" +

                   str(round(row[var], 3)) + " " + title + "<br>" +
                   str(round(row.ef_mag, 5)) + " mV/mm - orth. Electric field input^2<br>" +
                   str(round(row.sc, 3)) + " - Normalized node strength"
                   for i, row in df_avg.iterrows()]

    colorbar = dict(thickness=10, title="    Orth.<br>Electric field<br> (mV/mm)", len=0.3, x=0.7, y=0.85) if j==0 else None
    fig.add_scatter(x=df_avg[var].values, y=df_avg.delta_ampfex.values, mode="markers", hovertext=hovertext2d, showlegend=False,
                             marker=dict(size=size,
                                         color=df_avg.ef_mag.values, cmin=-max(df_avg.ef_mag.values),
                                         cmax=max(df_avg.ef_mag.values), colorscale="RdBu", reversescale=True,
                                         line=dict(color="grey", width=1),
                                         colorbar=colorbar), row=row, col=col)

    fig["layout"]["xaxis" + str(ax)]["title"] = title

fig.update_layout(yaxis=dict(title="\u0394 Power norm."), xaxis2=dict(title_standoff=0),
                  template="plotly_white", height=700, width=900)

pio.write_html(fig, file=fig_folder + 'R4_MLRmodel.html', auto_open=True)
pio.write_image(fig, file=fig_folder + 'R4_MLRmodel.svg')










## UNDERSTANDING LOWERING of the POWER
df_avg_under = df_avg.loc[df_avg["delta_ampfex"]<0]

# Initial variables included in the stepwise MLR
ind_vars = ["norm_abs_ef_mag", "norm_log_sc", "norm_plv_mean",  "norm_delta_initPeak",
             "norm_EFdist_modes", "norm_EFdist_skew", "norm_EFdist_kurtosis"]

# Final variables
ind_vars = ["norm_log_sc", "norm_plv_mean"]

#### 4. ROBUST LINEAR REGRESSION - This one ##
model_robust = sm.RLM(df_avg_under["norm_delta_ampfex"], sm.add_constant(df_avg_under[ind_vars])).fit()
model_robust.summary()


sm.robust.scale.mad(model_robust.resid)


weights = df_avg.sc.values
weights_norm = weights/max(weights)
range_size = [5, 30]
size = weights_norm * (range_size[1]-range_size[0]) + (range_size[0] - np.min(weights_norm))


vars = [("abs_ef_mag", "Norm. Electric field mean\u00B2 (mV/mm)<br>Distribution mean", [1, 2, 2]),

        ("EFdist_modes", "Distribution modes", [2,1, 4]),
        ("EFdist_skew", "Distribution skewness",[2,2, 5]),
        ("EFdist_kurtosis", "Distribution Kurtosis", [2,3, 6]),

        ("log_sc", "log. Node strength",[3,1, 7]),
        ("plv_mean", "PLV mean", [3,2, 8]),
        ("delta_initPeak", "\u0394 Frequency (Hz)",[3,3, 9])]

fig = make_subplots(rows=3, cols=3, shared_yaxes=True, vertical_spacing=0.15, y_title="\u0394 Alpha power (dB)")


for j, (var, title, [row, col, ax]) in enumerate(vars):

    hovertext2d = [row.subject + " - " + row.node + "<br>" +
                   str(round(row.delta_ampfex, 3)) + " dB - \u0394 power<br>" +

                   str(round(row[var], 3)) + " " + title + "<br>" +
                   str(round(row.ef_mag, 5)) + " mV/mm - orth. Electric field input^2<br>" +
                   str(round(row.sc, 3)) + " - Normalized node strength"
                   for i, row in df_avg_under.iterrows()]

    colorbar = dict(thickness=10, title="    Orth.<br>Electric field<br> (mV/mm)", len=0.3, x=0.7, y=0.85) if j==0 else None
    fig.add_scatter(x=df_avg_under[var].values, y=df_avg_under.delta_ampfex.values, mode="markers", hovertext=hovertext2d, showlegend=False,
                             marker=dict(size=size,
                                         color=df_avg_under.ef_mag.values, cmin=-max(df_avg_under.ef_mag.values),
                                         cmax=max(df_avg_under.ef_mag.values), colorscale="RdBu", reversescale=True,
                                         line=dict(color="grey", width=1),
                                         colorbar=colorbar), row=row, col=col)

    fig["layout"]["xaxis" + str(ax)]["title"] = title

fig.update_layout(yaxis=dict(title="\u0394 Power norm."), xaxis2=dict(title_standoff=0),
                  template="plotly_white", height=700, width=900)

pio.write_html(fig, file=fig_folder + 'R4_MLRmodel_under.html', auto_open=True)
pio.write_image(fig, file=fig_folder + 'R4_MLRmodel_under.svg')

