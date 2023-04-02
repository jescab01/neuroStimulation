
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.genmod.families
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import glob

figures_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER\FIGURES\\"

##############         WHOLE NETWORK IMPACT
import statsmodels.api as sm
from tvb.simulator.lab import *
import pickle
from scipy import stats

ctb_folder = "E:\LCCN_Local\PycharmProjects\\CTB_dataOLD2\\"
conn = connectivity.Connectivity.from_file(ctb_folder + "NEMOS_AVG_AAL2_pass.zip")

## Load simulated data - baseline vs stim simulations
# Data gathered just for cortical rois
fname = "wholeNetwork_impactm06d15y2022-t18h.16m.22s.pkl"
file_name = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\\4wholeNetwork_impact\\" + fname
open_file = open(file_name, "rb")
wn_impact_results = pickle.load(open_file)
open_file.close()

fftpeaks_baseline_avg = np.average(np.asarray([subject[0] for subject in wn_impact_results]), axis=0)
plv_baseline_avg = np.average(np.asarray([subject[1] for subject in wn_impact_results]), axis=0)
fftpeaks_stimulated_avg = np.average(np.asarray([subject[2] for subject in wn_impact_results]), axis=0)
plv_stimulated_avg = np.average(np.asarray([subject[3] for subject in wn_impact_results]), axis=0)

ef_mag_avg = np.average(np.asarray([subject[4] for subject in wn_impact_results]), axis=0)

## Simple measure of FC polarization: mean+std of both conditions
import pingouin as pg
plv_baseline_std = np.asarray([np.std(subject[1]) for subject in wn_impact_results])
plv_stimulated_std = np.asarray([np.std(subject[3]) for subject in wn_impact_results])

plv_baseline_m = np.asarray([np.average(subject[1]) for subject in wn_impact_results])
plv_stimulated_m = np.asarray([np.average(subject[3]) for subject in wn_impact_results])

ttest_std = pg.ttest(plv_stimulated_std, plv_baseline_std, paired=True, alternative="greater")
# Greater because we assume that it is the stimulated network the one getting extreme values.
# Result for std: [T=1.88, dof=9, p<0.05, cohens d=0.8, power=0.79]
ttest_m = pg.ttest(plv_stimulated_m, plv_baseline_m, paired=True, alternative="less")
# Result for mean: [T=-5.17, dof=9, p<0.001, cohens d=1.14, power=0.95]

## Plot distributions
plv_baseline_vals = np.average(np.asarray([subject[1][np.triu_indices(len(subject[1]), 1)] for subject in wn_impact_results]), axis=0)
plv_stimulated_vals = np.average(np.asarray([subject[3][np.triu_indices(len(subject[3]), 1)] for subject in wn_impact_results]), axis=0)

ttest = pg.ttest(plv_stimulated_vals, plv_baseline_vals, paired=True, alternative="less")

# fig = ff.create_distplot([plv_baseline_vals, plv_stimulated_vals], ["baseline", "stimulated"], bin_size=0.025)
# fig.update_layout(title="Phase locking values distribution for the average of 10 simulated virtual brains")
# fig.show("browser")

deltas = np.array(plv_stimulated_vals - plv_baseline_vals)

fig = ff.create_distplot([deltas], ["deltas fc"], bin_size=0.005)
fig.update_layout(title="delta of Phase locking values for the average of 10 simulated virtual brains")
fig.show("browser")

# percentages of reduce, higher and stable
len(deltas[deltas < -0.1])/len(deltas)  # 47%
len(deltas[(deltas > -0.1) & (deltas < 0.1)])/len(deltas)  # 52%
len(deltas[deltas > 0.1])/len(deltas)  # <1% 0.46%
len(deltas[deltas > 0.1])

# Preparation of Just cortical labels and indexing
# Define regions implicated in Functional analysis: remove  Cerebelum, Thalamus, Caudate (i.e. subcorticals)
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
                 'Paracentral_Lobule_L', 'Paracentral_Lobule_R', 'Heschl_L', 'Heschl_R',
                 'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L',
                 'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R',
                 'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L',
                 'Temporal_Inf_R']
# load text with FC rois; check if match SC
SClabs = list(conn.region_labels)
SC_cortex_idx = [SClabs.index(roi) for roi in cortical_rois]

regionLabels = conn.region_labels[SC_cortex_idx]
acc_index = [list(regionLabels).index(roi) for roi in regionLabels if "Cingulate_Ant" in roi]



delta_fc_avg = np.average(plv_stimulated_avg[acc_index, :] - plv_baseline_avg[acc_index, :], axis=0)
delta_fft_avg = (fftpeaks_stimulated_avg - fftpeaks_baseline_avg)



#############         PLOT FC base + FC stimulated + AVERAGE delta fft and delta fc to acc
fig = make_subplots(rows=2, cols=2, specs=[[{}, {}], [{"colspan": 2, "secondary_y": True}, {}]], row_heights=[0.7, 0.3], shared_yaxes=True,
                    vertical_spacing=0.35, column_titles=["Baseline", "During stimulation"])

fig.add_trace(go.Heatmap(z=plv_baseline_avg, x=regionLabels, y=regionLabels, zmax=1, zmin=0,
                         colorbar=dict(thickness=10, y=0.77, len=0.45), colorscale="Viridis"), row=1, col=1)
fig.add_trace(go.Heatmap(z=plv_stimulated_avg, x=regionLabels, y=regionLabels, zmax=1, zmin=0, colorscale="Viridis", showscale=False), row=1, col=2)


order = np.argsort(delta_fc_avg)

fig.add_trace(go.Scatter(x=regionLabels[order], y=delta_fc_avg[order], name="\u0394 FC ACC-others", line=dict(width=3)), row=2, col=1)

fig.add_trace(go.Scatter(x=regionLabels[order], y=delta_fft_avg[order], name="\u0394 FFT peak", line=dict(width=3)), secondary_y=True, row=2, col=1)

# fig.add_trace(go.Scatter(x=regionLabels[order], y=ef_mag_avg[order], name="Electric field input", line=dict(width=3)), secondary_y=True, row=2, col=1)

fig.update_layout(legend=dict(orientation="h", y=0.25, x=0.05), template="plotly_white",
                  yaxis3=dict(title="\u0394 FC", zerolinewidth=3), yaxis4=dict(title="\u0394 FFT-peak", zerolinewidth=3),
                  xaxis=dict(tickangle=45), xaxis2=dict(tickangle=45), xaxis3=dict(tickangle=45), height=750, width=950)

pio.write_html(fig, file=figures_folder + '/wholeNetwork_impact.html', auto_open=True)
pio.write_image(fig, file=figures_folder + '/wholeNetwork_impact.svg')



##############         PLOT correlation between delta-FFTpeak and avg_ef_mag

##### Correlation
stats.pearsonr(delta_fft_avg, abs(ef_mag_avg[SC_cortex_idx]))
## Regression line
X = sm.add_constant(abs(ef_mag_avg[SC_cortex_idx]))
model = sm.OLS(delta_fft_avg, X)
y_linear_regression = model.fit().fittedvalues

## Prepare brain volume for plotting
# Idea from Matteo Mancini: https://neurosnippets.com/posts/interactive-network/
def obj_data_to_mesh3d(odata):
    # odata is the string read from an obj file
    vertices = []
    faces = []
    lines = odata.splitlines()

    for line in lines:
        slist = line.split()
        if slist:
            if slist[0] == 'v':
                vertex = np.array(slist[1:], dtype=float)
                vertices.append(vertex)
            elif slist[0] == 'f':
                face = []
                for k in range(1, len(slist)):
                    face.append([int(s) for s in slist[k].replace('//', '/').split('/')])
                if len(face) > 3:  # triangulate the n-polyonal face, n>3
                    faces.extend(
                        [[face[0][0] - 1, face[k][0] - 1, face[k + 1][0] - 1] for k in range(1, len(face) - 1)])
                else:
                    faces.append([face[j][0] - 1 for j in range(len(face))])
            else:
                pass

    return np.array(vertices), np.array(faces)

obj_file="E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER\lh.pial_simplified.obj"
with open(obj_file, "r") as f:
    obj_data = f.read()
[vertices, faces] = obj_data_to_mesh3d(obj_data)

vert_x, vert_y, vert_z = vertices[:, :3].T
face_i, face_j, face_k = faces.T


## Color array for pairs of nodes
import plotly.express as px
cmap = px.colors.qualitative.Plotly
# ## Colors w/ Opcity
# ## Opacity array by ef_mag
# range_op = [0, 1]
# ef_mag_avg_normalized = abs(ef_mag_avg)/max(abs(ef_mag_avg))
# opacity = ef_mag_avg_normalized * (range_op[1]-range_op[0]) + (range_op[0] - np.min(ef_mag_avg_normalized))
#
# colors_op = [px.colors.label_rgb(px.colors.hex_to_rgb(cmap[(i//2)-((i//2)//len(cmap))*len(cmap)]) + (op, )) for i, op in enumerate(opacity[:94])] +\
#             [(0, 0, 0, op) for op in opacity[94:112]] + \
#             [(128, 128, 128, op) for op in opacity[112:]]  ## (128,128,128) Gray ## (0,0,0) Black


## Size by node strength
weights = np.sum(conn.weights, axis=1)
weights_norm = weights/max(weights)
range_size = [7, 70]
size = weights_norm * (range_size[1]-range_size[0]) + (range_size[0] - np.min(weights_norm))

# Size of black markers in 3D brain: for deltaFFT
range_size = [0, 5]
delta_fft_norm = (delta_fft_avg - np.min(delta_fft_avg))/(max(delta_fft_avg) - np.min(delta_fft_avg))
size_deltaFFT = delta_fft_norm * (range_size[1]-range_size[0]) + (range_size[0] - np.min(delta_fft_norm))
size_deltaFFT = list(size_deltaFFT) + [0] * (120 - len(size_deltaFFT))
delta_fft_hover = list(delta_fft_avg) + ["-"] * (120 - len(size_deltaFFT))

hovertext2d=[roi + "<br>" + str(round(ef_mag_avg[i], 5)) + " mV/mm - Electric field input<br>"
           + str(round(delta_fft_hover[i], 3)) + " Hz - delta FFT peak<br>"
           + str(round(weights_norm[i], 3)) + " - Normalized node strength"
           for i, roi in enumerate(regionLabels)]

## PLOT
fig = go.Figure(go.Scatter(x=abs(ef_mag_avg[SC_cortex_idx]), y=delta_fft_avg, mode="markers", hovertext=hovertext2d,
                         marker=dict(size=size[SC_cortex_idx], color=ef_mag_avg[SC_cortex_idx], cmin=-max(abs(ef_mag_avg)), reversescale=True,
                                     cmax=max(abs(ef_mag_avg)), colorscale="RdBu", line=dict(color="grey", width=1),
                                     colorbar=dict(thickness=10, title=" Electric field<br> (mV/mm)"))))

    # Add regression line
# fig.add_trace(go.Scatter(x=abs(ef_mag_avg[SC_cortex_idx]), y=y_linear_regression, marker_color="gray", line=dict(width=2),
#                          showlegend=True), row=1, col=1)

fig.update_layout(xaxis=dict(title="|Electric field input| (mV/mm)"), yaxis=dict(title="\u0394 FFT peak"),
                  template="plotly_white", height=500, width=700)

pio.write_html(fig, file=figures_folder + '/GlobalImpact_2Dscatter.html', auto_open=True)
pio.write_image(fig, file=figures_folder + '/GlobalImpact_2Dscatter.svg')



    # Add scatter3D
hovertext3d=[roi + "<br>" + str(round(ef_mag_avg[i], 5)) + " mV/mm - Electric field input<br>"
           + str(round(delta_fft_hover[cortical_rois.index(roi)], 3)) + " Hz - delta FFT peak<br>"
           + str(round(weights_norm[i], 3)) + " - Normalized node strength"
           if roi in cortical_rois else roi + "<br>" + str(round(ef_mag_avg[i], 5)) + " mV/mm - Electric field input<br>"
           + "Not measured delta FFT peak in subcortical areas<br>"
           + str(round(weights_norm[i], 3)) + " - Normalized node strength"
           for i, roi in enumerate(conn.region_labels)]

fig = go.Figure(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hovertext=hovertext3d, mode="markers",
                           marker=dict(size=size, color=ef_mag_avg, cmin=-max(abs(ef_mag_avg)), cmax=max(abs(ef_mag_avg)),
                                       opacity=1, line=dict(color="grey", width=10), colorscale="RdBu", reversescale=True,
                                       colorbar=dict(thickness=10, title=" Electric field<br> (mV/mm)")),
                           showlegend=False))

    # Add additional scatter with entrainment measure
fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hovertext=hovertext3d, mode="markers",
                           marker=dict(size=size_deltaFFT, color="black", opacity=1, line=dict(color="grey", width=10),
                                       colorscale="RdBu", reversescale=True),
                           showlegend=False))

fig.add_trace(go.Mesh3d(x=vert_x, y=vert_y, z=vert_z, i=face_i, j=face_j, k=face_k,
                        color='silver', opacity=0.25, showscale=False, hoverinfo='none'))

# camera = dict(
#     up=dict(x=0, y=0, z=1),
#     center=dict(x=0, y=0, z=0),
#     eye=dict(x=-1.5, y=1.25, z=0.5))

fig.update_layout(xaxis=dict(title="|Electric field input| (mV/mm)"), yaxis=dict(title="delta FFT peak"),
                  template="plotly_white")

pio.write_html(fig, file=figures_folder + '/GlobalImpact_3Dbrain.html', auto_open=True)
pio.write_image(fig, file=figures_folder + '/GlobalImpact_3Dbrain.svg', auto_open=True)





#### Model the influence of degree and ef_mag on delta FFT: with Linear or Logistic REGRESSIONs?
# The distribution of deltaFFT is not really continuous. In each subject there are mainly two options
# it was entrained, or it was not - that's categorical.
#
# I've tried with binary logistic (maxLL=-250) and ordered logistic regressions (maxLL=-450).
# I dont think we should use a linear regression (maxLL=-900) because we wouldn't meet assumptions due to that
# data distribution.
#
# We could use the Robust MR, but I dont think it would be correct anyway. I think Robust MR
# allows to face problems with outliers, but not the problem of the distribution in my data.


# PREPARE DATA

ctb_folder = "E:\LCCN_Local\PycharmProjects\\CTB_dataOLD2\\"
subjects = ["NEMOS_0" + str(idx) for idx in [35, 49, 50, 58, 59, 64, 65, 71, 75, 77]]

# Unpack wn_impact_results
fftpeaks_baseline_all = np.asarray([subject[0] for subject in wn_impact_results])
plv_baseline_all = np.asarray([subject[1] for subject in wn_impact_results])
fftpeaks_stimulated_all = np.asarray([subject[2] for subject in wn_impact_results])
plv_stimulated_all = np.asarray([subject[3] for subject in wn_impact_results])
delta_fc_all = np.average(plv_stimulated_all[:, acc_index, :] - plv_baseline_all[:, acc_index, :], axis=1)
delta_fft_all = fftpeaks_stimulated_all - fftpeaks_baseline_all
ef_mag_all = np.asarray([subject[4] for subject in wn_impact_results])

## Concatenate subjects data and standardize measures
conn = connectivity.Connectivity.from_file(ctb_folder + "NEMOS_035_AAL2_pass.zip")
weights_ = list(np.sum(conn.weights, axis=1)[SC_cortex_idx])
for i, emp_subj in enumerate(subjects[1:]):
    conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2_pass.zip")
    weights_ = weights_ + list(np.sum(conn.weights, axis=1)[SC_cortex_idx])

weights = stats.zscore(np.log(weights_))  # logaritmic relationship weight-entrainment
delta_fc = stats.zscore(np.arctanh(delta_fc_all.flatten()))  # Fisher's transformation of correlation values
ef_mag = stats.zscore(np.abs(ef_mag_all)[:, SC_cortex_idx].flatten())  # abs-both phases generates same effect on deltaFFT

delta_fft = stats.zscore(delta_fft_all.flatten())
df_z = pd.DataFrame(np.asarray([delta_fc, delta_fft, weights, ef_mag]).T,
                    columns=["deltaFC", "deltaFFT", "log_weights", "abs_ef_mag"])
df_z["deltaFFT_binary"] = df_z["deltaFFT"].values > 0.1
df_z["deltaFFT_ordered"] = ["0-zero" if value <= 0.1 else
                    "1-mid" if value <= 0.9 else "2-high"
                    for value in df_z["deltaFFT"].values]

from pandas.api.types import CategoricalDtype
categories_type = CategoricalDtype(categories=["0-zero", "1-mid", "2-high"], ordered=True)
df_z["deltaFFT_ordered"] = df_z["deltaFFT_ordered"].astype(categories_type)


## MODELS
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor

### 1. MULTINOMIAL LOGISTIC MODEL: max Log Likelihood = -218
model_logit = sm.MNLogit(df_z["deltaFFT_binary"], sm.add_constant(df_z[["log_weights", "abs_ef_mag"]])).fit()
model_logit.summary()

# ASSUMPTIONS - https://towardsdatascience.com/assumptions-of-logistic-regression-clearly-explained-44d85a22b290
# 1.1 ASSUMPTIONS: Linearity (not met)
df_z["weights*LogWeights"] = np.log(weights_) * np.log(np.log(weights_))
df_z["efmag*Logefmag"] = np.abs(ef_mag_all)[:, SC_cortex_idx].flatten() * np.log(np.abs(ef_mag_all)[:, SC_cortex_idx].flatten())
model_logit = sm.MNLogit(df_z["deltaFFT_binary"], sm.add_constant(df_z[["log_weights", "abs_ef_mag", "weights*LogWeights", "efmag*Logefmag"]])).fit()
model_logit.summary()

# 1.2 ASSUMPTIONS: Not strong outliers
logit = sm.GLM(df_z["deltaFFT_binary"], sm.add_constant(df_z[["log_weights", "abs_ef_mag"]]), family=sm.genmod.families.Binomial()).fit()
influence = logit.get_influence()  # Get influence measures
summ_df = influence.summary_frame()  # Obtain summary df of influence measure
diagnosis_df = summ_df[['cooks_d']]  # Filter summary df to Cook's distance values only
cook_threshold = 4 / len(X)  # Set Cook's distance threshold
diagnosis_df['std_resid'] = stats.zscore(logit.resid_pearson) # Append absolute standardized residual values
diagnosis_df['std_resid'] = diagnosis_df['std_resid'].apply(lambda x: np.abs(x))
# Find observations which are BOTH outlier (std dev > 3) and highly influential
extreme = diagnosis_df[(diagnosis_df['cooks_d'] > cook_threshold) & (diagnosis_df['std_resid'] > 3)]
extreme.sort_values("cooks_d", ascending=False).head()  # Show top 5 highly influential outlier observations

# 1.4 ASSUMPTIONS: Multicollinearity
vif_df = pd.DataFrame()
vif_df["Features"] = df_z[["log_weights", "abs_ef_mag"]].columns
vif_df["VIF Factor"] = [variance_inflation_factor(df_z[["log_weights", "abs_ef_mag"]].values, i) for i in range(df_z[["log_weights", "abs_ef_mag"]].shape[1])]
print(vif_df.sort_values("VIF Factor").round(2))  # If VIF < 5, its ok.

df_z[["deltaFFT", "log_weights", "abs_ef_mag"]].corr()

# 1.5 ASSUMPTIONS: Independent observations
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 5))  # Setup plot
ax = fig.add_subplot(111, title="Residual Series Plot", xlabel="Index Number", ylabel="Deviance Residuals")
ax.plot(df_z.index.tolist(), stats.zscore(logit.resid_deviance)) # Generate residual series plot using standardized deviance residuals
plt.axhline(y=0, ls="--", color='red')  # Draw horizontal line at y=0


#### 2. ORDERED LOGISTIC REGRESSION: max Log likelihood = -555
model_ordinal = OrderedModel(df_z["deltaFFT_ordered"], df_z[["log_weights", "abs_ef_mag"]], distr="probit").fit(method="bfgs")
model_ordinal.summary()

predicted = model_ordinal.model.predict(model_ordinal.params)
df_z["pred_Ordinal"] = predicted.argmax(axis=1)




# 3. LINEAR REGRESSION MODEL: max log likelihood = -908
model = sm.OLS(df_z["deltaFFT"], sm.add_constant(df_z[["log_weights", "abs_ef_mag"]])).fit()
model.summary()

# 3.1 ASSUMPTIONS: Linearity (linear relation between variables)
# scatter plot
import seaborn as sns
sns.set(style="ticks", color_codes=True, font_scale=2)
g=sns.pairplot(df_z[["deltaFFT", "log_weights", "abs_ef_mag"]], height=3, diag_kind="hist", kind="reg")
g.fig.suptitle("Scatter plot", y=1.08)
# 3.2 ASSUMPTIONS: Normality (variables follow a normal distribution)
sm.qqplot(df_z[["deltaFFT"]].values[:, 0], line="45")
# 3.3 ASSUMPTIONS: Multicolinearity (correlation between variables) # VIF
df_z[["deltaFFT", "log_weights", "abs_ef_mag"]].corr()


# 4. ROBUST LINEAR REGRESSION - This one ##
model_robust = sm.RLM(df_z["deltaFFT"], sm.add_constant(df_z[["log_weights", "abs_ef_mag"]])).fit()
model_robust.summary()













# ## Utiliza los DATOS TIPIFICADOS: para poder comparar la importancia de las variables en la ecuación.
# # En regresión simple el coeficiente de regresión tipificado es el coeficiente de correlación de Pearson;
# # en regresión múltiple no lo es.
#
#
#
# ## Explore data distribution
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# fig, ax = plt.subplots(1, 3)
#
# sns.regplot(x=df["weights"], y=df["deltaFFT"], lowess=True, ax=ax[0], line_kws={'color': 'red'})
# ax[0].set_title('weight - deltaFFT', fontsize=16)
# ax[0].set(xlabel='Degree', ylabel='deltaFFT')
#
# sns.regplot(x=df["log_weight"], y=df["deltaFFT"], lowess=True, ax=ax[1], line_kws={'color': 'red'})
# ax[1].set_title('log(weight) - deltaFFT', fontsize=16)
# ax[1].set(xlabel='log(Degree)', ylabel='deltaFFT')
#
# sns.regplot(x=df["ef_mag"], y=df["deltaFFT"], lowess=True, ax=ax[2], line_kws={'color': 'red'})
# ax[2].set_title('ef_mag - deltaFFT', fontsize=16)
# ax[2].set(xlabel='ef_mag', ylabel='deltaFFT')
#
#
#
# # DIAGNOSTICS
# from PAPER.LinearRegressionDiagnostic import Linear_Reg_Diagnostic
# cls = Linear_Reg_Diagnostic(model)
# fig, ax = cls()
#
#
# ## Check Assumptions
# import seaborn as sns
# import matplotlib.pyplot as plt
# # Linearity of the model :: NOT MET
# ### Diagramas de dispersion parcial: Residuos de pronosticar Y con todas las variables independientes excepto Xj; por los
# ### residuos de pronosticar Xj a partir del resto de variables  independientes.
#
# #### Working on Degree
# X = df_z[["ef_mag"]]
# X = sm.add_constant(X)
# y = df_z["deltaFFT"]
# model1 = sm.OLS(y, X).fit()
#
# X = df_z[["ef_mag"]]
# X = sm.add_constant(X)
# y = df_z["weights"]
# model2 = sm.OLS(y, X).fit()
#
# #   colinealidad
# # homocedasticidad
# #      ...
#
#
#
#
#
#
#
#
# ##############  - delta FC by delta FFT, weight and ef_mag   ##########################################
# # LINEAR REGRESSION MODEL
# X = df_z[["deltaFFT", "weights", "ef_mag"]]
# X = sm.add_constant(X)
# y = df_z["deltaFC"]
#
# model = sm.OLS(y, X).fit()
# model.summary()
#
# ## Check Assumptions
#
# # Linearity of the model :: NOT MET
# fitted_values = model.predict()
# residuals = model.resid
#
# fig, ax = plt.subplots(1, 2)
#
# sns.regplot(x=fitted_values, y=y, lowess=True, ax=ax[0], line_kws={'color': 'red'})
# ax[0].set_title('Observed vs. Predicted Values', fontsize=16)
# ax[0].set(xlabel='Predicted', ylabel='Observed')
#
# sns.regplot(x=fitted_values, y=residuals, lowess=True, ax=ax[1], line_kws={'color': 'red'})
# ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)
# ax[1].set(xlabel='Predicted', ylabel='Residuals')
#
# #   colinealidad
# # homocedasticidad
# #      ...
#
# ## ROBUST LINEAR REGRESSION
# model_robust = sm.RLM(y, X).fit()
# model_robust.summary()
#
#
#
#
#
#
#
