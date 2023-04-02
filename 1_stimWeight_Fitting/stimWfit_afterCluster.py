
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
import glob

fname = "PSEmpi_stimWfit_prebif-m11d08y2022-t16h.21m.58s"
specific_folder = "E:\\LCCN_Local\PycharmProjects\\neuroStimulation\\1_stimWeight_Fitting\PSE\\" + fname

# cargar los datos
resultsAAL = pd.read_csv(glob.glob(specific_folder + "\\*results.csv")[0])
n_rep = resultsAAL["rep"].max() + 1

# Calculate percentage
baseline = resultsAAL.loc[resultsAAL["w"] == 0].groupby("Subject").mean()

resultsAAL["percent"] = [(row["bModule"] - baseline.loc[row["Subject"]].bModule) / baseline.loc[row["Subject"]].bModule * 100 for i, row in resultsAAL.iterrows()]

# plotear
# resultsAAL_avg = resultsAAL.groupby(['w', 'Subject']).mean()
# resultsAAL_avg["sd"] = resultsAAL.groupby('w')[['module']].std()

fig = px.scatter(resultsAAL, x="w", y="bModule", color="Subject",
             title="Alpha peak module rise @Pareto-Occipital regions<br>(%i simulations | 10 subjects AAL)" % n_rep,
             labels={  # replaces default labels by column name
                 "w": "Weight", "band_module": "Alpha peak module"},
             template="plotly", log_x=True)
pio.write_html(fig, file=specific_folder + '\\allNemosAAL_1scatterModules_alphaRise_' + str(n_rep) + "sim.html",
               auto_open=True)

fig = px.box(resultsAAL, x="w", y="percent",
             title="Alpha peak module rise @Pareto-Occipital regions<br>(%i simulations | 10 subjects AAL)" % n_rep,
             labels={  # replaces default labels by column name
                 "w": "Weight", "percent": "Percentage of alpha rise"},
             template="plotly", log_x=True)
pio.write_html(fig, file=specific_folder + '\\allNemosAAL_2boxpercent_alphaRise_' + str(n_rep) + "sim.html",
               auto_open=True)

fig = px.box(resultsAAL, x="w", y="percent", color="Subject",
             title="Alpha peak module rise @Pareto-Occipital regions<br>(%i simulations | 10 subjects AAL)" % n_rep,
             labels={  # replaces default labels by column name
                 "w": "Weight", "percent": "Percentage of alpha rise"},
             template="plotly", log_x=True)
pio.write_html(fig, file=specific_folder + '\\allNemosAAL_3disgregboxpercent_alphaRise_' + str(n_rep) + "sim.html",
               auto_open=True)

# Scatter plot with mean and median
fig = px.scatter(resultsAAL, x="w", y="percent", color="Subject", log_x=True)

w = np.asarray(resultsAAL.groupby("w").mean().reset_index()["w"])
mean = np.asarray(resultsAAL.groupby("w").mean()["percent"])
median = np.asarray(resultsAAL.groupby("w").median()["percent"])

fig.add_trace(go.Scatter(x=w, y=mean, mode="lines", name="mean", visible="legendonly"))
fig.add_trace(go.Scatter(x=w, y=median, mode="lines", name="median", visible="legendonly"))
pio.write_html(fig, file=specific_folder + '\\allNemosAAL_4scatterpercent_alphaRise_' + str(n_rep) + "sim.html",
               auto_open=True)



# a = resultsAAL[(resultsAAL["w"]==0)|(resultsAAL["w"]==1e-8)]
# fig=px.scatter(resultsAAL, x="w", y="percent", log_x=True)
# fig.show("browser")
# deprecated
# def boxPlot_stimWfit(df_ar, emp_subj, specific_folder,  n_simulations):
#
#     # calculate percentages
#     df_ar_avg = df_ar.groupby('w').mean()
#
#     df_ar["percent"] = [
#         ((df_ar.peak_module[i] - df_ar_avg.peak_module[0]) / df_ar_avg.peak_module[0]) * 100 for i in
#         range(len(df_ar))]
#
#     df_ar_avg = df_ar.groupby('w').mean()
#     df_ar_avg["sd"] = df_ar.groupby('w')[['w', 'peak_module']].std()
#
#
#     fig = px.box(df_ar, x="w", y="peak_module",
#                  title="Alpha peak module rise @ParietalComplex<br>(%i simulations | %s AAL2red)" % (n_simulations, emp_subj),
#                  labels={  # replaces default labels by column name
#                      "w": "Weight", "peak_module": "Alpha peak module"},
#                  template="plotly")
#     pio.write_html(fig, file=specific_folder + '\\' + emp_subj + "AAL_alphaRise_modules_" + str(n_simulations) + "sim.html",
#                    auto_open=False)
#
#     fig = px.box(df_ar, x="w", y="percent",
#                  title="Alpha peak module rise @ParietalComplex<br>(%i simulations | %s AAL2red)" % (n_simulations, emp_subj),
#                  labels={  # replaces default labels by column name
#                      "w": "Weight", "percent": "Percentage of alpha peak rise"},
#                  template="plotly")
#     pio.write_html(fig, file=specific_folder + '\\' + emp_subj + "AAL_alphaRise_percent_" + str(n_simulations) + "sim.html",
#                    auto_open=True)
#
#
# def lines3dFFT_stimWfit(df_fft, specific_folder, show_rois=False):
#
#     rois_ = list(set(df_fft.regLab))
#     rois_.sort()
#     rois=rois_[0:len(rois_):2]+rois_[1:len(rois_):2]
#     weights = np.sort(np.array(list(set(df_fft.w))))
#     reps = list(set(df_fft.rep))
#     initPeak=np.average(df_fft.initPeak)
#
#     fig_global = make_subplots(rows=2, cols=5, vertical_spacing=0.1, horizontal_spacing=0.001,
#                                subplot_titles=(
#                                "Alpha rise <br>@Parietal complex + precuneus", rois[0],
#                                rois[1], rois[2], rois[3], rois[4], rois[5]),
#                                specs=[[{"rowspan": 2, "colspan": 2, 'type': 'surface'}, None, {'type': 'surface'},
#                                        {'type': 'surface'},{'type': 'surface'}],
#                                       [None, None, {'type': 'surface'}, {'type': 'surface'},{'type': 'surface'}]])
#
#     pos = [(1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]
#
#     # Plots per ROI
#     for i, roi in enumerate(rois[0:5]):
#         dft_roi=df_fft.loc[df_fft["regLab"] == roi]
#
#         weights_subset = np.sort(np.random.choice(weights[1:], 4, replace=False))
#         fig = make_subplots(rows=2, cols=4, vertical_spacing=0.1, horizontal_spacing=0.001,
#                             subplot_titles=(roi + "<br> 40 sim averaged", "w = "+str(weights_subset[0]),
#                                             "w = "+str(weights_subset[1]),"w = "+str(weights_subset[2]),"w = "+str(weights_subset[3])),
#                             specs=[[{"rowspan": 2, "colspan": 2, 'type': 'surface'}, None, {'type': 'surface'}, {'type': 'surface'}],
#                                    [None, None, {'type': 'surface'}, {'type': 'surface'}]])
#
#         (row, col)=pos[i]
#         dft_roi_repavg = dft_roi.groupby(['w', 'freq'])[['w', 'freq', 'rep', 'fft_module', 'initPeak']].mean()
#         for w_ in np.sort(np.array(list(set(dft_roi_repavg.w)))):
#             dft_roi_repavg_w = dft_roi_repavg.loc[dft_roi_repavg["w"] == w_]
#
#             # GLOBAL
#             fig_global.add_trace(go.Scatter3d(x=dft_roi_repavg_w.w, y=dft_roi_repavg_w.freq, z=dft_roi_repavg_w.fft_module,
#                                               legendgroup="w = " + str(w_), showlegend=False, mode="lines", line=dict(width=2.5, color="gray")), row=row, col=col)
#             fig_global.add_trace(go.Scatter3d(x=np.array([max(dft_roi_repavg_w.w)]),
#                                        y=np.array([initPeak]),
#                                        z=np.array([max(dft_roi_repavg_w.fft_module)]), legendgroup="w = " + str(w_), showlegend=False,
#                                        marker=dict(symbol="cross", size=5, color="black", opacity=0.5)), row=row, col=col)
#             # ROIS
#             fig.add_trace(go.Scatter3d(x=dft_roi_repavg_w.w, y=dft_roi_repavg_w.freq, z=dft_roi_repavg_w.fft_module, legendgroup="w = " + str(w_), name="w = " + str(round(w_, 4)), mode="lines", line=dict(width=4)), row=1, col=1)
#             fig.add_trace(go.Scatter3d(x=np.array([max(dft_roi_repavg_w.w)]),
#                                        y=np.array([initPeak]),
#                                        z=np.array([max(dft_roi_repavg_w.fft_module)]), legendgroup="w = " + str(w_), showlegend=False,
#                                        marker=dict(symbol="cross", size=5, color="black", opacity=0.5)), row=1, col=1)
#
#         del dft_roi_repavg_w, dft_roi_repavg
#
#         for j, w_ in enumerate(weights_subset[:2]):
#             dft_roi_w = dft_roi.loc[dft_roi["w"] == w_]
#             for r_ in reps:
#                 dft_roi_w_r = dft_roi_w.loc[dft_roi["rep"] == r_]
#                 if j==0:
#                     fig.add_trace(go.Scatter3d(x=dft_roi_w_r.rep, y=dft_roi_w_r.freq, z=dft_roi_w_r.fft_module, legendgroup="rep = " + str(r_), name="rep = " + str(r_), mode="lines", line=dict(width=2.5, color="gray")), row=1, col=3+j)
#                 else:
#                     fig.add_trace(go.Scatter3d(x=dft_roi_w_r.rep, y=dft_roi_w_r.freq, z=dft_roi_w_r.fft_module, legendgroup="rep = " + str(r_), showlegend=False, mode="lines", line=dict(width=2.5, color="gray")), row=1, col=3+j)
#                 fig.add_trace(go.Scatter3d(x=np.array([max(dft_roi_w_r.rep)]),
#                                            y=np.array([initPeak]),
#                                            z=np.array([max(dft_roi_w_r.fft_module)]), legendgroup="rep = " + str(r_), showlegend=False, marker=dict(symbol="cross", size=5, color="black", opacity=0.5)), row=1, col=3+j)
#
#         for j, w_ in enumerate(weights_subset[2:]):
#             dft_roi_w = dft_roi.loc[dft_roi["w"] == w_]
#             for r_ in reps:
#                 dft_roi_w_r = dft_roi_w.loc[dft_roi["rep"] == r_]
#                 fig.add_trace(go.Scatter3d(x=dft_roi_w_r.rep, y=dft_roi_w_r.freq, z=dft_roi_w_r.fft_module, legendgroup="rep = " + str(r_), showlegend=False, name="rep = " + str(r_), mode="lines", line=dict(width=2, color="gray")), row=2, col=3+j)
#                 fig.add_trace(go.Scatter3d(x=np.array([max(dft_roi_w_r.rep)]),
#                                        y=np.array([initPeak]),
#                                        z=np.array([max(dft_roi_w_r.fft_module)]), legendgroup="rep = " + str(r_), showlegend=False, marker=dict(symbol='cross', size=5, color="black", opacity=0.5)), row=2, col=3+j)
#
#         del dft_roi_w, dft_roi_w_r, dft_roi
#
#         fig.update_layout(legend=dict(y=1.2, x=-0.1),
#             scene1=dict(xaxis_title='Stim Weight', yaxis_title='Frequency (Hz)', zaxis_title='Module'),
#             scene2=dict(xaxis_title='Repetition', yaxis_title='Frequency (Hz)', zaxis_title='Module'),
#             scene3=dict(xaxis_title='Repetition', yaxis_title='Frequency (Hz)', zaxis_title='Module'),
#             scene4=dict(xaxis_title='Repetition', yaxis_title='Frequency (Hz)', zaxis_title='Module'),
#             scene5=dict(xaxis_title='Repetition', yaxis_title='Frequency (Hz)', zaxis_title='Module'))
#
#         pio.write_html(fig, file=specific_folder + "/lines3dFFT-%s.html" % roi, auto_open=show_rois)
#
#
#     # Global plot
#     dft_roiavg = df_fft.groupby(['w', 'freq'])[['w', 'freq', 'rep', 'fft_module', 'initPeak']].mean()
#     for iii, w_ in enumerate(np.sort(np.array(list(set(dft_roiavg.w))))):
#
#         dft_roiavg_w = dft_roiavg.loc[dft_roiavg["w"]==w_]
#
#         fig_global.add_trace(
#             go.Scatter3d(x=dft_roiavg_w.w, y=dft_roiavg_w.freq, z=dft_roiavg_w.fft_module, legendgroup="w = " + str(w_), name="w = " + str(round(w_, 4)), mode="lines", line=dict(width=4)), row=1, col=1)
#
#         fig_global.add_trace(go.Scatter3d(x=np.array([max(dft_roiavg_w.w)]),
#                                           y=np.array([initPeak]),
#                                           z=np.array([max(dft_roiavg_w.fft_module)]),
#                                           legendgroup="w = " + str(w_), showlegend=False,
#                                           marker=dict(symbol="cross", size=5, color="black", opacity=0.5)), row=1, col=1)
#     del dft_roiavg_w, dft_roiavg
#     fig_global.update_layout(legend=dict(y=1.2, x=-0.1),
#         scene1=dict(xaxis_title='Stim Weight', yaxis_title='Frequency (Hz)', zaxis_title='Module'))
#
#     pio.write_html(fig_global, file=specific_folder + "/global_lines3dFFT-ParietalComplex.html", auto_open=True)