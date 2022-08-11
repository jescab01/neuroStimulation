
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
import glob

fname = "PSEmpi_testFrequenciesWmean_indWPpass-m06d11y2022-t17h.11m.18s"

specific_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\\2testFrequencies\PSE\\" + fname
ctb_folder = "E:\LCCN_Local\PycharmProjects\\CTB_data2\\"

# Cargar los datos
results = pd.read_csv(glob.glob(specific_folder + "\\*results.csv")[0])
results.columns = "stimulation_site", "stimulus_type", "stim_params", "mode", "subject", "g", "speed", "stimW", "rep", "band", 'plv0', 'plv1', 'plv2', 'plv3', 'fft0', 'fft1', 'fft2', 'fft3', 'IAF', 'pre_Prec_peak', 'pre_ACC_peak'

# From wide to long in connections
results = pd.wide_to_long(results, stubnames='plv',
                          i=["stimulation_site", "stimulus_type", "stim_params", "mode", "subject",
                             "g", "speed", "stimW", "rep", "band", 'fft0', 'fft1', 'fft2', 'fft3', 'IAF', 'pre_Prec_peak', 'pre_ACC_peak'],
                          j='connection').reset_index()
# Back to rel labels
rel_labels = ['Cingulate_Ant_L-Precuneus_L', 'Cingulate_Ant_L-Precuneus_R', 'Cingulate_Ant_R-Precuneus_L',
              'Cingulate_Ant_R-Precuneus_R']
results["connection"] = rel_labels * int(len(results) / len(rel_labels))

results = pd.wide_to_long(results, stubnames='fft',
                          i=["stimulation_site", "stimulus_type", "stim_params", "mode", "subject",
                             "g", "speed", "stimW", "rep", 'connection', 'plv', 'IAF', 'pre_Prec_peak', 'pre_ACC_peak'], j='roi').reset_index()

# back to roi labels
roi_labels = ['Cingulate_Ant_L', 'Cingulate_Ant_R', 'Precuneus_L', 'Precuneus_R']
results["roi"] = roi_labels * int(len(results) / len(roi_labels))

# attributes
n_simulations, w, g, s, struct = results["rep"].max() + 1, \
                                 results["stimW"][0], results["g"][0], results["speed"][0], "AAL2"
# rois of interest
rois = [34, 35, 70, 71]  # rois implicated in the effect: 35-ACCl, 36-AACr, 71-Prl, 72-Prr [note python 0-indexing]

## TODO Simple plot of IAFs per subject - Variability should be low, or inexistent.




## PLOTTING v4 # Trying to fit all information in a single plot per subject
for mode in (set(results["mode"])):

    for ii, emp_subj in enumerate(set(results["subject"])):

        auto_open = True if ii == 0 else False

        subset = results.loc[(results["subject"] == emp_subj) & (results["mode"]==mode)]

        # prepara el subset: w=0 (baseline) ; "noise"
        subset["stim_params"].loc[subset["stimulus_type"] == "baseline"] = -9
        subset["stim_params"].loc[(subset["stimulus_type"] == "noise") & (subset["stim_params"] == 0)] = -8

        # Average for FFT plots
        subset_avg = subset.groupby(["stimulation_site", "stim_params", "roi"]).mean().reset_index()

        freq_max_fc = subset_avg["stim_params"][subset_avg["plv"].idxmax()]
        freq_min_fc = subset_avg["stim_params"][subset_avg["plv"].idxmin()]

        IAF = subset_avg.IAF.mean().round(2)

        ## Passive ROI peak; ppp = Pre Precuneus Peak.
        ppp = subset_avg.pre_Prec_peak.mean().round(2)

        # color palette
        cmap = px.colors.qualitative.Plotly

        # gomagerit.cesvima.um
        fig = make_subplots(rows=2, cols=3, column_titles=("P3P4 Model", "F3F4 Model", "targetACC Model"),
                            specs=[[{}, {}, {}], [{}, {}, {}]], shared_yaxes=True, shared_xaxes=True,
                            x_title="Stimulation Frequency relative to passive ROI oscillation frequency (Hz)")

        for i, stim_site in enumerate(["roast_P3P4Model", "roast_F3F4Model", "roast_ACCtarget"]):

            sl = True if i == 0 else False

            for c, coi in enumerate(sorted(set(subset["connection"]))):
                df_sub = subset.loc[(subset["stimulation_site"] == stim_site) & (subset["connection"] == coi)]

                fig.add_trace(go.Box(x=df_sub.stim_params, y=df_sub.plv, marker_color=cmap[c], name=coi,
                                     legendgroup=coi, showlegend=sl), row=1, col=i + 1)

            # Vertical lines @ min FC frequencies
            # fig.add_shape(go.layout.Shape(type="line", xref="x" + str(i+1), yref="y" + str(i+1),
            #                               x0=freq_min_fc, y0=0, x1=freq_min_fc, y1=1,
            #                               line={"dash": "dash", "color": "blue", "width": 1}), row=1, col=i+1)
            #
            # fig.add_shape(go.layout.Shape(type="line", xref="x" + str(i+1), yref="y" + str(i+1),
            #                               x0=freq_min_fc, y0=5, x1=freq_min_fc, y1=15,
            #                               line={"dash": "dash", "color": "blue", "width": 1}), row=2, col=i+1)

            for c, roi in enumerate(sorted(list(set(subset["roi"])))):
                df_sub = subset_avg.loc[(subset_avg["stimulation_site"] == stim_site) & (subset_avg["roi"] == roi)]

                fig.add_trace(go.Scatter(x=df_sub.stim_params, y=df_sub.fft, marker_color=cmap[c + 4], line=dict(width=4),
                                         name=roi, legendgroup=roi, showlegend=sl), row=2, col=i + 1)

            fig.add_trace(
                go.Scatter(x=[-5, -2.5, 0, 2.5, 5], y=[ppp - 5, ppp - 2.5, ppp, ppp + 2.5, ppp + 5], mode="lines",
                           marker_color="gray", line=dict(width=0.5, dash="dash"), name="Stimulation reference",
                           showlegend=sl), row=2, col=i + 1)

            # Horizontal line in freq plot to show IAF
            # fig.add_trace(
            #     go.Scatter(x=[-5, 5], y=[IAF, IAF], mode="lines", marker_color="red",
            #                line=dict(width=0.5, dash="dash"), name="IAF", showlegend=sl), row=2,
            #     col=i + 1)


        fig.update_layout(boxmode='group', title=mode + emp_subj + 'FC&FFT_w',
                          yaxis_title="Functional Connectivity (PLV)", yaxis4_title="FFT peak (Hz)",
                          xaxis=dict(showticklabels=True, tickmode='array',
                                     tickvals=[-9, -8, -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6],
                                     ticktext=["Baseline", "tRNS", -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6], tickangle=45),
                          xaxis2=dict(showticklabels=True, tickmode='array',
                                      tickvals=[-9, -8, -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6],
                                      ticktext=["Baseline", "tRNS", -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6], tickangle=45),
                          xaxis3=dict(showticklabels=True, tickmode='array',
                                      tickvals=[-9, -8, -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6],
                                      ticktext=["Baseline", "tRNS", -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6], tickangle=45),
                          xaxis4=dict(showticklabels=True, tickmode='array',
                                      tickvals=[-9, -8, -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6],
                                      ticktext=["Baseline", "tRNS", -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6], tickangle=45),
                          xaxis5=dict(showticklabels=True, tickmode='array',
                                      tickvals=[-9, -8, -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6],
                                      ticktext=["Baseline", "tRNS", -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6], tickangle=45),
                          xaxis6=dict(showticklabels=True, tickmode='array',
                                      tickvals=[-9, -8, -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6],
                                      ticktext=["Baseline", "tRNS", -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6], tickangle=45))

        pio.write_html(fig, file=specific_folder + "/" + mode + emp_subj + 'FC&FFT_w' + str(n_simulations) + "sim_v4.html",
                       auto_open=auto_open)


    ## Plot an average result
    subset = results.copy()

    subset = results.loc[results["mode"] == mode]

    # prepara el subset: w=0 (baseline) to w=6; "noise" param= 0  to param=7
    subset["stim_params"].loc[subset["stimulus_type"] == "baseline"] = -9
    subset["stim_params"].loc[(subset["stimulus_type"] == "noise") & (subset["stim_params"] == 0)] = -8


    # Average for FFT plots
    subset_avg = subset.groupby(["stimulation_site", "stim_params", "roi"]).mean().reset_index()

    freq_max_fc = subset_avg["stim_params"][subset_avg["plv"].idxmax()]
    freq_min_fc = subset_avg["stim_params"][subset_avg["plv"].idxmin()]

    # color palette
    cmap = px.colors.qualitative.Plotly

    # gomagerit.cesvima.um
    fig = make_subplots(rows=2, cols=3, column_titles=("P3P4 Model", "F3F4 Model", "targetACC Model"),
                        specs=[[{}, {}, {}], [{}, {}, {}]], shared_yaxes=True, shared_xaxes=True,
                        x_title="Stimulation Frequency relative to passive ROI frequency peak (Hz)")

    for i, stim_site in enumerate(["roast_P3P4Model", "roast_F3F4Model", "roast_ACCtarget"]):

        sl = True if i == 0 else False

        for c, coi in enumerate(sorted(set(subset["connection"]))):
            df_sub = subset.loc[(subset["stimulation_site"] == stim_site) & (subset["connection"] == coi)]

            fig.add_trace(go.Box(x=df_sub.stim_params, y=df_sub.plv, marker_color=cmap[c], name=coi,
                                 legendgroup=coi, showlegend=sl), row=1, col=i + 1)

        for c, roi in enumerate(sorted(list(set(subset["roi"])))):
            df_sub = subset_avg.loc[(subset_avg["stimulation_site"] == stim_site) & (subset_avg["roi"] == roi)]

            fig.add_trace(go.Scatter(x=df_sub.stim_params, y=df_sub.fft, marker_color=cmap[c + 4], line=dict(width=4),
                                     name=roi, legendgroup=roi, showlegend=sl), row=2, col=i + 1)

        fig.add_trace(
            go.Scatter(x=[-5, -2.5, 0, 2.5, 5], y=[ppp - 5, ppp - 2.5, ppp, ppp + 2.5, ppp + 5], mode="lines",
                       marker_color="gray",
                       line=dict(width=0.5, dash="dash"), name="Stimulation reference", showlegend=sl), row=2,
            col=i + 1)

    fig.update_layout(boxmode='group', title= mode + 'AVG_FC&FFT_w',
                      yaxis_title="Functional Connectivity (PLV)", yaxis4_title="FFT peak (Hz)",
                      xaxis=dict(showticklabels=True, tickmode='array',
                                 tickvals=[-9, -8, -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6],
                                 ticktext=["Baseline", "tRNS", -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6], tickangle=45),
                      xaxis2=dict(showticklabels=True, tickmode='array',
                                  tickvals=[-9, -8, -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6],
                                  ticktext=["Baseline", "tRNS", -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6], tickangle=45),
                      xaxis3=dict(showticklabels=True, tickmode='array',
                                  tickvals=[-9, -8, -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6],
                                  ticktext=["Baseline", "tRNS", -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6], tickangle=45),
                      xaxis4=dict(showticklabels=True, tickmode='array',
                                  tickvals=[-9, -8, -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6],
                                  ticktext=["Baseline", "tRNS", -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6], tickangle=45),
                      xaxis5=dict(showticklabels=True, tickmode='array',
                                  tickvals=[-9, -8, -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6],
                                  ticktext=["Baseline", "tRNS", -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6], tickangle=45),
                      xaxis6=dict(showticklabels=True, tickmode='array',
                                  tickvals=[-9, -8, -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6],
                                  ticktext=["Baseline", "tRNS", -6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6], tickangle=45))

    pio.write_html(fig, file=specific_folder + '/' + mode + 'AVG_FC&FFT_w' + str(n_simulations) + "sim_v4.html",
                   auto_open=True)




### PLOTTING v3 # one plot per stimulation site with grouped boxplots + FFT aside

# subset = results.loc[
#     (results["stimulus_type"] == "sinusoid") & (results["subject"] == emp_subj) & (results["stim_params"] != 0)]
#
# for stim_site in set(results["stimulation_site"]):
#     # Iterate over stimulation_sites models: ACC vs P3P4; and subjects
#     for i, emp_subj in enumerate(set(results["subject"])):
#         ## define what you are ging to process
#         for stim_type in set(results["stimulus_type"]):
#
#             weighting = np.loadtxt(
#                 glob.glob(
#                     ctb_folder + 'CurrentPropagationModels/' + emp_subj + '-efnorm_mag-' + stim_site + '*-AAL2.txt')[0],
#                 delimiter=",") * w
#
#             subset_w0 = results.loc[(results["stimulus_type"] == "sinusoid") & (results["subject"] == emp_subj) & (
#                     results["stim_params"] == 0)]
#
#             if i == 0:
#                 auto_open = False
#             else:
#                 auto_open = False
#
#             if stim_type == "sinusoid":
#
#                 ## Subset y separa el baseline
#                 subset = results.loc[
#                     (results["stimulation_site"] == stim_site) & (results["stimulus_type"] == stim_type) & (
#                             results["subject"] == emp_subj) & (results["stim_params"] != 0)]
#
#                 subset_avg = subset.groupby(["stim_params", "connection", "roi"]).mean().reset_index()
#
#                 freq_max_fc = subset_avg["stim_params"][subset_avg["plv"].idxmax()]
#                 freq_min_fc = subset_avg["stim_params"][subset_avg["plv"].idxmin()]
#                 baseline_fc = subset_w0["plv"].mean()
#
#                 # Plot FC ACC-Pr by stim
#                 fig = px.box(subset, x="stim_params", y="plv", color="connection",
#                              title="Functional Connectivity between ROIs in alpha band - stimulation: %s | %s <br>(%i simulations | %s %s | %s )" % (
#                                  stim_site, stim_type, n_simulations, emp_subj, struct, results["mode"][0]),
#                              labels={  # replaces default labels by column name
#                                  "stim_params": "Stimulation Frequency", "plv": "Functional Connectivity (PLV)"},
#                              # color_discrete_sequence=["dimgray"],
#                              template="plotly")
#
#                 fig.add_vline(x=freq_max_fc, line_width=0.75, line_dash="dot", line_color="orange",
#                               layer='below')  # , annotation_text="Freq max PLV", annotation_position="top left")
#                 fig.add_vline(x=freq_min_fc, line_width=0.75, line_dash="dot", line_color="darkblue",
#                               layer='below')  # , annotation_text="Freq min PLV", annotation_position="top left")
#                 fig.add_hline(y=baseline_fc, line_width=0.75, line_dash="dash", line_color="darkgray", layer='below',
#                               annotation_text="Baseline PLV (no stimulation)", annotation_position="top left")
#
#                 pio.write_html(fig,
#                                file=specific_folder + "/" + emp_subj + "FC_" + stim_site + '_' + stim_type + '-w' + str(
#                                    n_simulations) + "sim_v3.html",
#                                auto_open=auto_open)
#
#                 fig_fft = px.line(x=subset_avg.stim_params, y=subset_avg.fft, color=subset_avg.roi,
#                                   title="Frequency Peaks in ROIs along stimulation : " + stim_site + '_' + stim_type + " - " + emp_subj + " || (g = " + str(
#                                       g) + "; s = " + str(s) + "; w = " + str(round(w, 5)) + ")",
#                                   labels={"x": "Stimulation Frequency", "y": "Frequency peak (Hz)", "color": ""})
#
#                 fig_fft.add_vline(x=freq_max_fc, line_width=0.75, line_dash="dot", line_color="orange")
#                 fig_fft.add_vline(x=freq_min_fc, line_width=0.75, line_dash="dot", line_color="darkblue")
#                 fig_fft.add_scatter(x=[9, 10, 11, 12, 13], y=[9, 10, 11, 12, 13], mode="lines", marker_color="gray",
#                                     line=dict(width=0.5), name="Stimulation reference")
#
#                 pio.write_html(fig_fft,
#                                file=specific_folder + "/" + emp_subj + "FFT_" + stim_site + '_' + stim_type + '-w' + str(
#                                    n_simulations) + "sim_v3.html",
#                                auto_open=auto_open)
#
#             elif stim_type == "noise":
#
#                 subset = results.loc[
#                     (results["stimulation_site"] == stim_site) & (results["stimulus_type"] == stim_type) & (
#                             results["subject"] == emp_subj)]
#
#                 subset_avg = subset.groupby(["stim_params", "connection", "roi"]).mean().reset_index()
#
#                 freq_max_fc = subset_avg["stim_params"][subset_avg["plv"].idxmax()]
#                 freq_min_fc = subset_avg["stim_params"][subset_avg["plv"].idxmin()]
#                 baseline_fc = subset_w0["plv"].mean()
#
#                 # Plot FC ACC-Pr by stim
#                 fig = px.box(subset, x="stim_params", y="plv", color="connection",
#                              title="Functional Connectivity between ROIs in alpha band - stimulation: %s | %s <br>(%i simulations | %s %s | %s )" % (
#                                  stim_site, stim_type, n_simulations, emp_subj, struct, results["mode"][0]),
#                              labels={  # replaces default labels by column name
#                                  "stim_params": "Gaussian Mean (DC)", "plv": "Functional Connectivity (PLV)"},
#                              # color_discrete_sequence=["dimgray"],
#                              template="plotly")
#
#                 fig.add_vline(x=freq_max_fc, line_width=0.75, line_dash="dot", line_color="orange",
#                               layer='below')  # , annotation_text="Freq max PLV", annotation_position="top left")
#                 fig.add_vline(x=freq_min_fc, line_width=0.75, line_dash="dot", line_color="darkblue",
#                               layer='below')  # , annotation_text="Freq min PLV", annotation_position="top left")
#                 fig.add_hline(y=baseline_fc, line_width=0.75, line_dash="dash", line_color="darkgray", layer='below',
#                               annotation_text="Baseline PLV (no stimulation)", annotation_position="top left")
#
#                 pio.write_html(fig,
#                                file=specific_folder + "/" + emp_subj + "FC_" + stim_site + '_' + stim_type + '-w' + str(
#                                    n_simulations) + "sim_v3.html",
#                                auto_open=auto_open)


# ## PLOTTING v2 # one plot per connection
# # Iterate over stimulation models: ACC vs P3P4; and subjects
# for stimulation in set(results["stimulation"]):
#
#     for emp_subj in set(results["subject"]):
#
#         weighting = np.loadtxt(
#             glob.glob(ctb_folder + 'CurrentPropagationModels/' + emp_subj + '-efnorm_mag-' + stimulation + '*-AAL2.txt')[0],
#             delimiter=",") * w
#
#         subset = results.loc[(results["stimulation"] == stimulation) & (results["subject"] == emp_subj)]
#
#         subset_avg = subset.groupby("stim_params").mean().reset_index()
#         # fft_avg = df_fft.groupby(["stim_params"])[["stim_params", "Cingulate_Ant_L", "Cingulate_Ant_R", "Precuneus_L", "Precuneus_R"]].mean()
#
#         # Plot FC ACC-Pr by stim
#         rel_labels = ['Cingulate_Ant_L-Precuneus_L', 'Cingulate_Ant_L-Precuneus_R',
#                       'Cingulate_Ant_R-Precuneus_L', 'Cingulate_Ant_R-Precuneus_R']
#
#         for i, rel in enumerate(rel_labels):
#
#             if i == 0:
#                 auto_open = False
#             else:
#                 auto_open = False
#
#             max_fc = subset_avg["stim_params"][subset_avg[rel].idxmax()]
#             min_fc = subset_avg["stim_params"][subset_avg[rel].idxmin()]
#
#             fig = px.box(subset, x="stim_params", y=rel,
#                          title="Functional Connectivity between %s in alpha band <br>(%i simulations | %s %s | %s )" % (
#                              rel, n_simulations, emp_subj, struct, mode),
#                          labels={  # replaces default labels by column name
#                              "stim_params": "Stimulation Frequency", rel: "Functional Connectivity (PLV)"},
#                          color_discrete_sequence=["dimgray"],
#                          template="plotly")
#
#             fig.add_vline(x=max_fc, line_width=0.75, line_dash="dot", line_color="orange")
#             fig.add_vline(x=min_fc, line_width=0.75, line_dash="dot", line_color="darkblue")
#
#             pio.write_html(fig, file=specific_folder + "/" + emp_subj + "FC_" + rel + '-w' + str(n_simulations) + "sim_v2.html",
#                            auto_open=auto_open)
#
#             # Plot FFT peak by stim
#             fig_fft = go.Figure()
#
#             fig_fft.add_trace(go.Scatter(x=subset_avg.stim_params, y=subset_avg.Cingulate_Ant_L,
#                                          name="ACC_L - ef_mag = " + str(round(weighting[rois[0]], 5))))
#             fig_fft.add_trace(go.Scatter(x=subset_avg.stim_params, y=subset_avg.Cingulate_Ant_R,
#                                          name="ACC_R - ef_mag = " + str(round(weighting[rois[1]], 5))))
#             fig_fft.add_trace(go.Scatter(x=subset_avg.stim_params, y=subset_avg.Precuneus_L,
#                                          name="Precuneus_L - ef_mag = " + str(round(weighting[rois[2]], 5))))
#             fig_fft.add_trace(go.Scatter(x=subset_avg.stim_params, y=subset_avg.Precuneus_R,
#                                          name="Precuneus_R - ef_mag = " + str(round(weighting[rois[3]], 5))))
#
#             fig_fft.update_layout(title=emp_subj + " || (g = " + str(g) + "; s = " + str(s) + "; w = " + str(round(w, 5)) + ")")
#             fig_fft.update_xaxes(title="Stimulation Frequency")
#             fig_fft.update_yaxes(title="Alpha peak frequency (Hz)")
#             fig_fft.add_vline(x=max_fc, line_width=0.75, line_dash="dot", line_color="orange", name=rel)
#             fig_fft.add_vline(x=min_fc, line_width=0.75, line_dash="dot", line_color="darkblue", name=rel)
#
#             fig_fft.add_scatter(x=[9, 10, 11, 12, 13], y=[9, 10, 11, 12, 13], mode="lines", marker_color="gray",
#                                 line=dict(width=0.5), name="Stimulation reference")
#
#             pio.write_html(fig_fft,
#                            file=specific_folder + "/" + emp_subj + "FFT_" + rel + '-w' + str(n_simulations) + "sim_v2.html",
#                            auto_open=auto_open)
#
