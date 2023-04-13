
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import pingouin as pg
from tvb.simulator.lab import connectivity

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_dataOLD2\\"
fig_folder = "E:\\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\Figures\\"
nmm_folder = "E:\\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\output_NMM\\"


# Define regions implicated in Functional analysis: remove  Cerebelum, Thalamus, Caudate (i.e. subcorticals)
cingulum_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                 'Insula_L', 'Insula_R',
                 'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                 'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                 'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R',
                 'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                 'Parietal_Inf_R', 'Precuneus_L', 'Precuneus_R',
                 'Thalamus_L', 'Thalamus_R']

## 0. Load data
sim_tag = "PSEmpi_mlr_cb_indiv-m04d11y2023-t17h.29m.20s"

results = pd.read_pickle(nmm_folder + sim_tag + "\\stimWmpi_pd_results.pkl")
results = results.astype({"mode": str, "trial": int, "roi": str, "w": float, "plv_mean":float,
                              "fex": float, "fpeak": float, "amp_fbase": float, "amp_fpeak": float})


## 1. Prepare the dataframe:
# 1.1 Separate in baseline and stimulation dataframes
base = results.loc[results["stage"] == "baseline"]
stim = results.loc[results["stage"] == "stimulation"]


# 1.2 Create df with deltas of amp_fbase pre vs. stim (y), fpeak pre vs. stim, and fpeak-initialPeak in pre
df = base.copy()

df["delta_fbase"] = (stim.amp_fbase.values - base.amp_fbase.values) / base.amp_fbase.values
df["delta_fpeak"] = (stim.fpeak.values - base.fpeak.values)  # **2?

df["delta_initPeak"] = (stim.fex.values - base.fpeak.values)  # **2?
df["abs_delta_initPeak"] = abs(stim.fex.values - base.fpeak.values)  # **2?

# 1.3 add SC and ef_mags per roi and subject
w, working_points = 0.6, [("NEMOS_035", 27),  # JR pass @ 27/03/2023
                  ("NEMOS_049", 28),
                  ("NEMOS_050", 42),
                  ("NEMOS_058", 50),
                  ("NEMOS_059", 36),
                  ("NEMOS_064", 39),
                  ("NEMOS_065", 37),
                  ("NEMOS_071", 36),
                  ("NEMOS_075", 48),
                  ("NEMOS_077", 38)]
df["ef_mag"], df["sc"], df["abs_ef_mag"], df["log_sc"] = np.nan, np.nan, np.nan, np.nan

for emp_subj, _ in working_points:

    conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2_pass.zip")
    conn.weights = conn.scaled_weights(mode="tract")

    # load text with FC rois; check if match SC
    SClabs = list(conn.region_labels)

    SC_cb_idx = [SClabs.index(roi) for roi in
                 cingulum_rois]  # find indexes in FClabs that matches cortical_rois
    conn.weights = conn.weights[:, SC_cb_idx][SC_cb_idx]
    conn.tract_lengths = conn.tract_lengths[:, SC_cb_idx][SC_cb_idx]
    conn.region_labels = conn.region_labels[SC_cb_idx]

    weighting = np.loadtxt(
        ctb_folder + 'CurrentPropagationModels/' + emp_subj + '-efnorm_mag-roast_OzCzModel-AAL2.txt',
        delimiter=",") * w

    weighting = weighting[SC_cb_idx]

    for i, roi in enumerate(conn.region_labels):

        # subset del dataframe para el sujeto y la roi, y añades el sc medio y el ef_mag
        df.loc[(df["subject"] == emp_subj) & (df["roi"] == roi), "sc"] = np.average(conn.weights[i, conn.weights[i, :] != 0])
        df.loc[(df["subject"] == emp_subj) & (df["roi"] == roi), "ef_mag"] = weighting[i]
        df.loc[(df["subject"] == emp_subj) & (df["roi"] == roi), "log_sc"] = np.log2(np.average(conn.weights[i, conn.weights[i, :] != 0]))
        df.loc[(df["subject"] == emp_subj) & (df["roi"] == roi), "abs_ef_mag"] = weighting[i]**2


# df = df.groupby(["subject", "roi"]).mean().reset_index()

# 1.4 Normalize metrics involved in the analysis
variables = ["delta_fbase", "delta_fpeak", "delta_initPeak", "abs_delta_initPeak", "ef_mag", "abs_ef_mag", "sc", "log_sc", "plv_mean"]
for var in variables:
    df["norm_" + var] = (df[var] - df[var].mean()) / df[var].std()




##  2. PLOT
# 2.1 explore scatters
import plotly.express as px
cmap = px.colors.qualitative.Plotly
fig = make_subplots(rows=5, cols=2, column_titles=["delta_fbase", "delta_fpeak"], vertical_spacing=0.1)

fig.add_trace(go.Scatter(x=df.norm_sc, y=df.norm_delta_fbase, name="norm_sc", legendgroup="norm_sc", marker=dict(color=cmap[0])), row=1, col=1)
fig.add_trace(go.Scatter(x=df.norm_sc, y=df.norm_delta_fpeak, name="norm_sc", legendgroup="norm_sc", showlegend=False, marker=dict(color=cmap[0])), row=1, col=2)

fig.add_trace(go.Scatter(x=df.norm_log_sc, y=df.norm_delta_fbase, name="norm_log_sc", legendgroup="norm_log_sc", marker=dict(color=cmap[1])), row=1, col=1)
fig.add_trace(go.Scatter(x=df.norm_log_sc, y=df.norm_delta_fpeak, name="norm_log_sc", legendgroup="norm_log_sc", showlegend=False, marker=dict(color=cmap[1])), row=1, col=2)

fig.add_trace(go.Scatter(x=df.norm_ef_mag, y=df.norm_delta_fbase, name="norm_ef_mag", legendgroup="norm_ef_mag", marker=dict(color=cmap[2])), row=2, col=1)
fig.add_trace(go.Scatter(x=df.norm_ef_mag, y=df.norm_delta_fpeak, name="norm_ef_mag", legendgroup="norm_ef_mag", showlegend=False, marker=dict(color=cmap[2])), row=2, col=2)

fig.add_trace(go.Scatter(x=df.norm_abs_ef_mag, y=df.norm_delta_fbase, name="norm_abs_ef_mag", legendgroup="norm_abs_ef_mag", marker=dict(color=cmap[3])), row=2, col=1)
fig.add_trace(go.Scatter(x=df.norm_abs_ef_mag, y=df.norm_delta_fpeak, name="norm_abs_ef_mag", legendgroup="norm_abs_ef_mag", showlegend=False, marker=dict(color=cmap[3])), row=2, col=2)

fig.add_trace(go.Scatter(x=df.norm_plv_mean, y=df.norm_delta_fbase, name="norm_plv_mean", legendgroup="norm_plv_mean", marker=dict(color=cmap[4])), row=3, col=1)
fig.add_trace(go.Scatter(x=df.norm_plv_mean, y=df.norm_delta_fpeak, name="norm_plv_mean", legendgroup="norm_plv_mean", showlegend=False, marker=dict(color=cmap[4])), row=3, col=2)

fig.add_trace(go.Scatter(x=df.norm_delta_initPeak, y=df.norm_delta_fbase, name="norm_dInitPeak", legendgroup="norm_dInitPeak", marker=dict(color=cmap[5])), row=4, col=1)
fig.add_trace(go.Scatter(x=df.norm_delta_initPeak, y=df.norm_delta_fpeak, name="norm_dInitPeak", legendgroup="norm_dInitPeak", showlegend=False, marker=dict(color=cmap[5])), row=4, col=2)

fig.add_trace(go.Scatter(x=df.norm_abs_delta_initPeak, y=df.norm_delta_fbase, name="norm_abs_dInitPeak", legendgroup="norm_abs_dInitPeak", marker=dict(color=cmap[7])), row=4, col=1)
fig.add_trace(go.Scatter(x=df.norm_abs_delta_initPeak, y=df.norm_delta_fpeak, name="norm_abs_dInitPeak", legendgroup="norm_abs_dInitPeak", showlegend=False, marker=dict(color=cmap[7])), row=4, col=2)

fig.add_trace(go.Scatter(x=df.norm_sc, y=df.norm_plv_mean, marker=dict(color=cmap[6])), row=5, col=2)

fig.update_traces(mode="markers")
fig.update_layout(xaxis1=dict(title="Mean SC"), yaxis1=dict(title="\u0394 power"),
                  xaxis2=dict(title="Mean SC"), yaxis2=dict(title="\u0394 fpeak"),
                  xaxis3=dict(title="|Orth. EF magnitude|"), yaxis3=dict(title="\u0394 power"),
                  xaxis4=dict(title="|Orth. EF magnitude|"), yaxis4=dict(title="\u0394 fpeak"),
                  xaxis5=dict(title="Mean PLV (pre-stimulation)"), yaxis5=dict(title="\u0394 power"),
                  xaxis6=dict(title="Mean PLV (pre-stimulation)"), yaxis6=dict(title="\u0394 fpeak"),
                  xaxis7=dict(title="\u0394 Initial peak (to stim)"), yaxis7=dict(title="\u0394 power"),
                  xaxis8=dict(title="\u0394 Initial peak (to stim)"), yaxis8=dict(title="\u0394 fpeak"))
fig.show("browser")


## 2.2 PLOT for paper
## Size by node strength
weights = df.sc.values
weights_norm = weights/max(weights)
range_size = [5, 40]
size = weights_norm * (range_size[1]-range_size[0]) + (range_size[0] - np.min(weights_norm))

hovertext2d = [row.subject + " - " + row.roi + "<br>" +
               str(round(row.ef_mag, 5)) + " mV/mm - Electric field input<br>"+
               str(round(row.delta_fbase, 3)) + " dB - \u0394 power<br>" +
               str(round(row.sc, 3)) + " - Normalized node strength"
               for i, row in df.iterrows()]

fig = go.Figure(go.Scatter(x=df.ef_mag_abs.values, y=df.delta_fbase.values, mode="markers", hovertext=hovertext2d,
                         marker=dict(size=size,
                                     color=df.ef_mag.values, cmin=-max(df.ef_mag_abs.values),
                                     cmax=max(df.ef_mag_abs.values), colorscale="RdBu", reversescale=True,
                                     line=dict(color="grey", width=1),
                                     colorbar=dict(thickness=10, title=" Electric field<br> (mV/mm)"))))

fig.update_layout(xaxis=dict(title="|Electric field input| (mV/mm)"), yaxis=dict(title="\u0394 Power norm."),
                  template="plotly_white", height=500, width=700)

pio.write_html(fig, file=fig_folder + 'R4_MLRmodel.html', auto_open=True)
pio.write_image(fig, file=fig_folder + 'R4_MLRmodel.svg')



## 3. STATS
import statsmodels.api as sm
import seaborn as sns
import scipy.stats as stats

ind_vars = ["norm_log_sc", "norm_abs_ef_mag", "norm_plv_mean", "norm_delta_initPeak"]

#### 4. ROBUST LINEAR REGRESSION - This one ##
model_robust = sm.RLM(df["norm_delta_fbase"], sm.add_constant(df[ind_vars])).fit()
model_robust.summary()

#### 3.1 MULTIPLE LINEAR REGRESSION MODEL
# Compute it and check assumptions [NORMALITY NOT MET].
# max log likelihood = ...
model = sm.OLS(df["norm_delta_fbase"], sm.add_constant(df[ind_vars])).fit()
model.summary()

## ASSUMPTIONS:
## 3.1.a NORMALITY of the residuals [not met]
# - approach 1
sm.qqplot(df[["delta_fbase"]].values[:, 0], line="45")
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
    fig.add_trace(go.Scatter(x=df[var], y=residuals, name=var, mode="markers"), row=1, col=i+1)
    fig["layout"]["xaxis"+str(i+1)]["title"] = var
fig.show("browser")

# 3.1.c Multicolinearity (correlation between variables) # VIF
corr = df[["delta_fbase"] + ind_vars].corr()





