
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
import glob

figures_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER\FIGURES\\"

### PART B:
## Plot 2testFrequencies with statistics
fname = "PSEmpi_testFrequenciesWmean_indWPpass-m06d11y2022-t17h.11m.18s"
specific_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\\2testFrequencies\PSE\\" + fname
# ctb_folder = "E:\LCCN_Local\PycharmProjects\\CTB_data2\\"

# Cargar los datos -- IAF (last column) refers here to occipito-parietal rois freq peak
results = pd.read_csv(glob.glob(specific_folder + "\\*results.csv")[0])
results.columns = "stimulation_site", "stimulus_type", "stim_params", "mode", "subject", "g", "speed", "stimW", "rep", \
                  "band", 'plv0', 'plv1', 'plv2', 'plv3', 'fft0', 'fft1', 'fft2', 'fft3', 'IAF', "pre_prec_peak", "pre_acc_peak"

# Extract baseline dataframe
baseline = results.loc[(results["stimulus_type"] == "baseline")]
baseline = baseline.groupby(["subject"]).mean().reset_index()

# Extract results dataframe to compose relative to baseline results
rel_results = results.loc[(results["stimulus_type"] != "baseline")].copy()

# Average out repetitions
rel_results = rel_results.groupby(["stimulation_site", "stimulus_type", "stim_params", "mode", "subject", "band"]).mean().reset_index()
rel_results.loc[rel_results["stimulus_type"] == "noise", "stim_params"] = -8  # set tRNS to be shown before sinusoid

# Calculate deltaPLVs (plvs - baseline plvs)
for i, row in baseline.iterrows():

    subset = rel_results[["plv0", "plv1", "plv2", "plv3"]].loc[rel_results["subject"] == row["subject"]]

    subset = subset - row[["plv0", "plv1", "plv2", "plv3"]]
    subset = subset.astype(float)

    rel_results.loc[rel_results["subject"] == row["subject"], ["plv0", "plv1", "plv2", "plv3"]] = subset


# Rename columns (PLVs vs deltaPLV)
rel_results.columns = "stimulation_site", "stimulus_type", "stim_params", "mode", "subject", "g", "speed", "stimW", "rep", \
                  "band", 'd_plv0', 'd_plv1', 'd_plv2', 'd_plv3', 'fft0', 'fft1', 'fft2', 'fft3', 'IAF', "pre_prec_peak", "pre_acc_peak"


# From wide to long in connections
rel_results_w = pd.wide_to_long(rel_results, stubnames='d_plv',
                          i=["stimulation_site", "stimulus_type", "stim_params", "mode", "subject",
                             "g", "speed", "stimW", "rep", "band", 'fft0', 'fft1', 'fft2', 'fft3'],
                          j='connection').reset_index()

# Back to rel labels
conn_labels = ['Cingulate_Ant_L-Precuneus_L', 'Cingulate_Ant_L-Precuneus_R', 'Cingulate_Ant_R-Precuneus_L',
              'Cingulate_Ant_R-Precuneus_R']
rel_results_w["connection"] = conn_labels * int(len(rel_results_w) / len(conn_labels))

rel_results_w = pd.wide_to_long(rel_results_w, stubnames='fft',
                          i=["stimulation_site", "stimulus_type", "stim_params", "mode", "subject",
                             "g", "speed", "stimW", "rep", 'connection', 'd_plv'], j='roi').reset_index()

# back to roi labels
roi_labels = ['ACC_L', 'ACC_R', 'Precuneus_L', 'Precuneus_R']
rel_results_w["roi"] = roi_labels * int(len(rel_results_w) / len(roi_labels))

# attributes
n_simulations = rel_results_w["rep"].max() + 1

# rois of interest
rois = [34, 35, 70, 71]  # rois implicated in the effect: 35-ACCl, 36-AACr, 71-Prl, 72-Prr [note python 0-indexing]


## FUNCTIONS
def plot_results(data, signif=None, title="final_plot"):

    cmap = px.colors.qualitative.Plotly  # color palette

    fig = make_subplots(rows=3, cols=1, specs=[[{}], [{}], [{}]], shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.5, 0.25, 0.25],
                        x_title="Stimulation Frequency<br>relative to passive ROI (Hz)")

    ## Functional connectivity
    ## P3P4 Model
    df_sub = data.loc[(data["mode"]=="jr") & (data["stimulation_site"] == "roast_P3P4Model")].\
        groupby(["stimulation_site", "stim_params", "mode", "subject"]).mean().reset_index()
    # Boxpoints: suspectedoutliers, all, outliers, False
    fig.add_trace(go.Box(x=df_sub.stim_params, y=df_sub.d_plv, marker_color="steelblue", name="P3P4 Protocol",
                         showlegend=True, boxpoints="suspectedoutliers"), row=1, col=1)

    ## F3F4 Model
    df_sub = data.loc[(data["mode"]=="jr") & (data["stimulation_site"] == "roast_F3F4Model")].\
        groupby(["stimulation_site", "stim_params", "mode", "subject"]).mean().reset_index()
    # Boxpoints: suspectedoutliers, all, outliers, False
    fig.add_trace(go.Box(x=df_sub.stim_params, y=df_sub.d_plv, marker_color="indianred", name="F3F4 Protocol",
                         showlegend=True, boxpoints="suspectedoutliers"), row=1, col=1)

    # White label space
    fig.add_trace(go.Scatter(x=[0], y=[0], opacity=0,marker_color="white", name="    "), row=1, col=1)

    ## FFT plots
    # P3P4 model
    data_avg = data.loc[(data["mode"]=="jr") & (data["stimulation_site"] == "roast_P3P4Model")].\
        groupby(["stimulation_site", "stim_params", "mode", "roi"]).mean().reset_index()
    ppp = data_avg["pre_prec_peak"].mean()
    for c, roi in enumerate(sorted(list(set(data["roi"])))):
        df_sub = data_avg.loc[(data_avg["roi"] == roi)]
        fig.add_trace(go.Scatter(x=df_sub.stim_params, y=df_sub.fft, marker_color=cmap[c + 4], line=dict(width=4),
                                 name=roi, legendgroup=roi, showlegend=True), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=[-2, -1, 0, 1, 2], y=[ppp - 2, ppp - 1, ppp, ppp + 1, ppp + 2], mode="lines", marker_color="gray",
                   line=dict(width=0.5, dash="dash"), name="Reference", showlegend=False), row=2, col=1)
    # F3F4 model
    data_avg = data.loc[(data["mode"] == "jr") & (data["stimulation_site"] == "roast_F3F4Model")].\
        groupby(["stimulation_site", "stim_params", "mode", "roi"]).mean().reset_index()
    ppp = data_avg["pre_prec_peak"].mean()
    for c, roi in enumerate(sorted(list(set(data["roi"])))):
        df_sub = data_avg.loc[(data_avg["roi"] == roi)]
        fig.add_trace(go.Scatter(x=df_sub.stim_params, y=df_sub.fft, marker_color=cmap[c + 4], line=dict(width=4),
                                 name=roi, legendgroup=roi, showlegend=False), row=3, col=1)
    fig.add_trace(
        go.Scatter(x=[-2, -1, 0, 1, 2], y=[ppp - 2, ppp - 1, ppp, ppp + 1, ppp + 2], mode="lines", marker_color="gray",
                   line=dict(width=0.5, dash="dash"), name="Reference", showlegend=False), row=3, col=1)

    # Add significance level
    if signif is not None:

        signif_data = signif.loc[(signif["stim_site"] == "mean") & (signif["sig"] != "ns")]

        [fig.add_annotation(dict(x=row.stim_params, y=row.maxVal + 0.15, xref="x", yref="y", text=row.sig, textangle=90,
                                 showarrow=False)) for index, row in signif_data.iterrows() if len(signif_data) > 0]

    fig.update_layout(boxmode='group',
                      xaxis=dict(showgrid=True, zerolinewidth=3, showticklabels=True, tickmode='array',
                                 tickvals=[-8, -6, -4, -2, 0, 2, 4, 6], ticktext=["tRNS", -6, -4, -2, 0, 2, 4, 6]),
                      xaxis2=dict(zerolinewidth=3, showticklabels=True, tickmode='array',
                                 tickvals=[-8, -6, -4, -2, 0, 2, 4, 6], ticktext=["tRNS", -6, -4, -2, 0, 2, 4, 6]),
                      xaxis3=dict(zerolinewidth=3, showticklabels=True, tickmode='array',
                                  tickvals=[-8, -6, -4, -2, 0, 2, 4, 6], ticktext=["tRNS", -6, -4, -2, 0, 2, 4, 6]),
                      yaxis=dict(title="\u0394 PLV"),
                      yaxis2=dict(title="FFT peaks (Hz)<br><b>P3P4 protocol</b>", title_font=dict(color="steelblue")),
                      yaxis3=dict(title="FFT peaks (Hz)<br><b>F3F4 protocol</b>", title_font=dict(color="indianred")),
                      legend=dict(orientation="h", y=1.05, x=0.01), width=900, height=800)

    pio.write_html(fig, file=figures_folder + '/dFC_&_FFTs-' + title + "_" + str(n_simulations) + "sim.html", auto_open=True)
    pio.write_image(fig, figures_folder + '/dFC_&_FFTs-' + title + "_" + str(n_simulations) + "sim.svg")
    pio.write_image(fig, figures_folder + '/dFC_&_FFTs-' + title + "_" + str(n_simulations) + "sim.png")


# stat_test: ttest xor non-parametric
def pairwise_comparisons(data_avg, stat_test=None):

    # pairwise comparisons vs 0: where do we actually reduce FC?
    pwc = pd.DataFrame()
    assumptions = pd.DataFrame()

    # initFreq = 1  # remove from this analysis tRNS

    # for stim_site in set(data_avg.stimulation_site):

    df = data_avg.loc[(data_avg["stimulation_site"] == "roast_P3P4Model") | (data_avg["stimulation_site"] == "roast_F3F4Model")]

    for f in list(sorted(set(df.stim_params))):

        test_group = df.loc[df["stim_params"] == f]

        ## CHECK PARAMETRIC ASSUMPTIONS
        # Samples' Normality
        n_test = stats.shapiro(test_group.d_plv.values)

        assumptions = assumptions.append(
            [["mean", f, n_test.statistic, n_test.pvalue]])

        if ((n_test.pvalue > 0.05) or (stat_test == "ttest")) and (stat_test != "non-parametric"):
            test = pg.ttest(x=test_group.d_plv.values, y=0, alternative="less")
        elif (n_test.pvalue > 0.05) or (stat_test == "non-parametric"):
            test = pg.mwu(x=test_group.d_plv.values, y=0, alternative="less")

        test["assump_met"] = "yes" if (n_test.pvalue > 0.05) else "no"
        test["stim_site"] = "mean"
        test["stim_params"] = f
        test["maxVal"] = max(test_group.d_plv.values)
        test["testMean"] = np.average(test_group.d_plv.values)

        pwc = pwc.append(test)

    # Pairwise comparisons between protocols: what protocol produces better results?
    for f in list(sorted(set(df.stim_params))):

        p3p4 = data_avg.loc[(data_avg["stimulation_site"] == "roast_P3P4Model") & (data_avg["stim_params"] == f)]
        f3f4 = data_avg.loc[(data_avg["stimulation_site"] == "roast_F3F4Model") & (data_avg["stim_params"] == f)]

        ## CHECK PARAMETRIC ASSUMPTIONS
        # Samples' Normality
        n_p3p4 = stats.shapiro(p3p4.d_plv.values)
        n_f3f4 = stats.shapiro(f3f4.d_plv.values)

        assumptions = assumptions.append(
            [[f, n_p3p4.statistic, n_p3p4.pvalue, n_f3f4.statistic, n_f3f4.pvalue]])

        if stat_test == "ttest":
            test = pg.ttest(x=p3p4.d_plv.values, y=f3f4.d_plv.values, paired=True, alternative="two-sided")

        else:
            test = pg.wilcoxon(x=p3p4.d_plv.values, y=f3f4.d_plv.values, alternative="two-sided")

        test["assump_met"] = "yes" if (n_p3p4.pvalue > 0.05) & (n_f3f4.pvalue > 0.05) else "no"
        test["stim_site"] = "f3f4_vs_p3p4"
        test["maxVal"] = max([max(p3p4.d_plv.values), max(f3f4.d_plv.values)])
        test["stim_params"] = f
        test["p3p4_mean"] = np.average(p3p4.d_plv.values)
        test["f3f4_mean"] = np.average(f3f4.d_plv.values)

        pwc = pwc.append(test)

    # Correct with bonferroni
    pwc["p.corr"] = pg.multicomp(pwc["p-val"].values, method="bonf")[1]

    # Add asterisks
    pwc["sig"] = ["****" if row["p.corr"] <= 0.0001 else
                  "***" if row["p.corr"] <= 0.001 else
                  "**" if row["p.corr"] <= 0.01 else
                  "*" if row["p.corr"] <= 0.05 else "ns"
                  for index, row in pwc.iterrows()]

    return pwc



data = rel_results_w.copy()
stim_params = list(sorted(set(data["stim_params"].values)))
#### SELECT A SUBSET to PLOT (e.g. [::2])
data = data.loc[data["stim_params"].isin(stim_params[::2])]



## STATS
import pingouin as pg
import scipy.stats as stats

data_avg = data.loc[(data["mode"] == "jr") & (data["stimulation_site"] != "roast_ACCtarget")].\
    groupby(["stimulation_site", "stim_params", "subject"]).mean().reset_index()

# Anova two-way to check for the influence of protocol and stim freq
rmAnova_twoway = pg.rm_anova(data_avg, dv="d_plv", within=["stimulation_site", "stim_params"], subject="subject")

# plot to explore the influence of stim site - thus plotting stim site Anova effect that is like a ttest.
pg.plot_paired(data_avg, dv="d_plv", within="stimulation_site", subject="subject")

# Pairwise comparisons both "vs baseline" and "between protocols"
signif = pairwise_comparisons(data_avg, stat_test="ttest")


## FINAL PLOT
plot_results(data, signif=signif)














#############          OLDER STUFF        ##############

# # mode :: "P3P4_vs_ACCtarget" or "baseline"
# def pairwise_comparisons(data, mode, stat_test="auto"):
#     """
#     It runs pairwise comparisons after checking for parametric assumptions.
#     It will decide based on assumption whether to use ttest or wilcoxon test.
#
#     :param data:
#     :param mode:
#     :return:
#     """
#
#     pwc = pd.DataFrame()
#     assumptions = pd.DataFrame()
#
#     if mode == "baseline":
#
#         initFreq = 1
#
#         for stim_site in set(data.stimulation_site):
#
#             df = data.loc[data["stimulation_site"] == stim_site]
#
#             for f in set(data.stim_params)[initFreq:]:
#
#                 df_temp = df.groupby(["subject", "stim_params"]).mean().reset_index()
#
#                 # avg per subject df(b)
#                 baseline = df_temp.loc[(df_temp["stim_params"] == 0) & (df_temp["relFreq"] == base_f)]
#
#                 # average per subject df(f)
#                 test_group = df_temp.loc[df_temp["relFreq"] == f]
#
#                 ## CHECK PARAMETRIC ASSUMPTIONS
#                 # Samples' Normality
#                 n_base = stats.shapiro(baseline.plv.values)
#                 n_test = stats.shapiro(test_group.plv.values)
#
#                 assumptions = assumptions.append([[stim_site, f, n_base.statistic, n_base.pvalue, n_test.statistic, n_test.pvalue]])
#
#                 if ((n_base.pvalue > 0.05) & (n_test.pvalue > 0.05)) or (stat_test == "ttest"):
#                     test = pg.ttest(x=test_group.plv.values, y=baseline.plv.values, paired=True, alternative="less")
#                 else:
#                     test = pg.wilcoxon(x=test_group.plv.values, y=baseline.plv.values, alternative="less")
#
#                 test["assump_met"] = "yes" if (n_base.pvalue > 0.05) & (n_test.pvalue > 0.05) else "no"
#                 test["stim_site"] = stim_site
#                 test["relFreq"] = f
#                 test["maxVal"] = max(test_group.plv.values)
#                 test["baselineMean"] = np.average(baseline.plv.values)
#                 test["testMean"] = np.average(test_group.plv.values)
#
#                 pwc = pwc.append(test)
#
#     elif mode == "P3P4_vs_ACCtarget":
#
#         df_p3p4 = data.loc[data["stimulation_site"] == "roast_P3P4Model"].groupby(["subject", "relFreq"]).mean().reset_index()
#
#         df_acc = data.loc[data["stimulation_site"] == "roast_ACCtarget"].groupby(["subject", "relFreq"]).mean().reset_index()
#
#         for f in relFreqs_sub:
#
#             baseline = df_p3p4.loc[df_p3p4["relFreq"] == f]
#
#             # average per subject df(f)
#             test_group = df_acc.loc[df_acc["relFreq"] == f]
#
#             ## CHECK PARAMETRIC ASSUMPTIONS
#             # Samples' Normality
#             n_base = stats.shapiro(baseline.plv.values)
#             n_test = stats.shapiro(test_group.plv.values)
#
#             assumptions = assumptions.append(
#                 [[f, n_base.statistic, n_base.pvalue, n_test.statistic, n_test.pvalue]])
#
#             if stat_test == "t-test":
#                 test = pg.ttest(x=test_group.plv.values, y=baseline.plv.values, paired=True, alternative="less")
#             else:
#                 test = pg.wilcoxon(x=test_group.plv.values, y=baseline.plv.values, alternative="less")
#
#             test["assump_met"] = "yes" if (n_base.pvalue > 0.05) & (n_test.pvalue > 0.05) else "no"
#             test["mode"] = mode
#             test["relFreq"] = f
#             test["maxVal"] = max(baseline.plv.values)
#             test["baselineMean"] = np.average(baseline.plv.values)
#             test["testMean"] = np.average(test_group.plv.values)
#
#             pwc = pwc.append(test)
#
#     # Correct with bonferroni
#     pwc["p.corr"] = pg.multicomp(pwc["p-val"].values, method="bonf")[1]
#
#     # Add asterisks
#     pwc["sig"] = ["****" if row["p.corr"] <= 0.0001 else
#                     "***" if row["p.corr"] <= 0.001 else
#                     "**" if row["p.corr"] <= 0.01 else
#                     "*" if row["p.corr"] <= 0.05 else "ns"
#                     for index, row in pwc.iterrows()]
#
#     return pwc, assumptions










###### SEPARATE plots


data = rel_results_w.copy()
# Assign color to subject to override with scatterplot
cmap = px.colors.qualitative.Plotly  # color palette
data["color"] = 0  # Pre-allocation
for i, subj in enumerate(set(data["subject"])):
    data.loc[data["subject"]==subj, "color"] = cmap[i]

mode = "jr"
connMode = "all"
title = "test"
for mode in ["jr", "jr_abstract"]:

    fig = make_subplots(rows=2, cols=3, specs=[[{},{},{}], [{},{},{}]],
                        column_titles=["roast_P3P4Model", "roast_F3F4Model", "roast_ACCtarget"],shared_yaxes=True,
                        shared_xaxes=True, vertical_spacing=0.15, x_title="Stimulation Frequency<br>relative to passive roi (Hz)")

    for i, stim_site in enumerate(["roast_P3P4Model", "roast_F3F4Model", "roast_ACCtarget"]):

        if "avg" in connMode:
            df_sub = data.loc[(data["mode"]==mode) & (data["stimulation_site"] == stim_site)].groupby(["stimulation_site", "stim_params", "mode", "subject", "color"]).mean().reset_index()

            # fig = px.scatter(df_sub, x="stim_params", y="d_plv", color="subject")
            # fig.show(renderer="browser")

            # Boxpoints: suspectedoutliers, all, outliers, False
            fig.add_trace(go.Box(x=df_sub.stim_params, y=df_sub.d_plv, marker_color="steelblue", name="ACC - Precuneus",
                                 showlegend=False, boxpoints=False), row=1, col=1+i)

            fig.add_trace(go.Scatter(x=df_sub.stim_params, y=df_sub.d_plv, mode="markers",
                                     marker=dict(color=df_sub.color,colorscale=px.colors.qualitative.Plotly, opacity=0.6, size=3),
                                     showlegend=False), row=1, col=1+i)

        if "all" in connMode:
            for c, coi in enumerate(sorted(set(data["connection"]))):
                df_sub = data.loc[(data["stimulation_site"] == stim_site) & (data["connection"] == coi)]

                fig.add_trace(go.Box(x=df_sub.relFreq, y=df_sub.plv, marker_color=cmap[c], name=coi), row=1, col=1+i)

        ## Add significance level
        # if ttest is not None:
        #     signif_data = ttest.loc[(ttest["stim_site"] == stim_site) & (ttest["sig"] != "ns")]
        #
        #     fig["layout"].update(annotations=[dict(x=row.relFreq, y=row.maxVal + 0.15,
        #                                            xref="x", yref="y",
        #                                            text=row.sig, textangle=90, showarrow=False)
        #                                       for index, row in signif_data.iterrows() if len(signif_data) > 0])


        data_avg = data.loc[(data["mode"]==mode) & (data["stimulation_site"] == stim_site)].groupby(["stimulation_site", "stim_params", "mode", "roi"]).mean().reset_index()
        ppp = data_avg["pre_prec_peak"].mean()
        for c, roi in enumerate(sorted(list(set(data["roi"])))):

            df_sub = data_avg.loc[(data_avg["roi"] == roi)]

            fig.add_trace(go.Scatter(x=df_sub.stim_params, y=df_sub.fft, marker_color=cmap[c + 4], line=dict(width=4),
                                     name=roi, showlegend=False), row=2, col=1+i)

        fig.add_trace(
            go.Scatter(x=[-2, -1, 0, 1, 2], y=[ppp - 2, ppp - 1, ppp, ppp + 1, ppp + 2], mode="lines", marker_color="gray",
                       line=dict(width=0.5, dash="dash"), name="Reference", showlegend=False), row=2, col=1+i)

        fig.update_layout(boxmode='group',
                          yaxis_title="Functional Connectivity (PLV)", yaxis4_title="FFT peak (Hz)",
                          # xaxis1=dict(showgrid=True, zerolinewidth=3, showticklabels=True, tickmode='array', tickangle=270,
                          #             tickvals=[base_f, tRNS_f, -2.45, -1.47, -0.49, "ppp", 0.49, 1.47, 2.45],
                          #             ticktext=["Baseline", "tRNS", -2.45, -1.47, -0.49, "ppp", 0.49, 1.47, 2.45]),
                          # xaxis2=dict(showgrid=True, zerolinewidth=3, showticklabels=True, tickmode='array', tickangle=270,
                          #             tickvals=[base_f, tRNS_f, -2.45, -1.47, -0.49, 0, 0.49, 1.47, 2.45],
                          #             ticktext=["Baseline", "tRNS", -2.45, -1.47, -0.49, "IAF", 0.49, 1.47, 2.45],
                          )

        if "avg" in connMode:
            fig.update_layout(legend=dict(orientation="h", y=0.48, x=0.35), title=mode + " - desynch averaged in 4 conn")

    pio.write_html(fig, file=figures_folder + '/FC&FFT_' + mode + str(n_simulations) + "sim.html",
                   auto_open=True)











def plot_results_x4(data, IAF, connMode="avg", ttest=None, title="AVG"):

    data_avg = data.groupby(["stimulation_site", "relFreq", "roi"]).mean().reset_index()

    ## Common processes
    cmap = px.colors.qualitative.Plotly  # color palette
    fig = make_subplots(rows=2, cols=2, column_titles=("P3P4 Model", "targetACC Model"),
                        specs=[[{}, {}], [{}, {}]], shared_yaxes=True, shared_xaxes=True,
                        x_title="Stimulation Frequency (Hz)")

    for i, stim_site in enumerate(["roast_P3P4Model", "roast_ACCtarget"]):

        sl = True if i == 0 else False

        if "avg" in connMode:
            df_sub = data.loc[data["stimulation_site"] == stim_site]
            df_sub = df_sub.groupby(["relFreq", "subject"]).mean().reset_index()
            # Boxpoints: suspectedoutliers, all, outliers, False
            fig.add_trace(go.Box(x=df_sub.relFreq, y=df_sub.plv, marker_color="steelblue", name="ACC - Precuneus",
                                 legendgroup="ACC - Precuneus", showlegend=sl, boxpoints='suspectedoutliers'), row=1, col=1 + i)

        if "all" in connMode:
            for c, coi in enumerate(sorted(set(data["connection"]))):
                df_sub = data.loc[(data["stimulation_site"] == stim_site) & (data["connection"] == coi)]

                fig.add_trace(go.Box(x=df_sub.relFreq, y=df_sub.plv, marker_color=cmap[c], showlegend=sl, name=coi,
                                     legendgroup="ACC - Precuneus"), row=1, col=1 + i)

        ## Add significance level
        if ttest is not None:

            signif_data = ttest.loc[(ttest["stim_site"] == stim_site) & (ttest["sig"] != "ns")]

            fig["layout"].update(annotations=[dict(x=row.relFreq, y=row.maxVal + 0.15,
                                                   xref="x"+str(i+1), yref="y"+str(i+1),
                                                   text=row.sig, textangle=90, showarrow=False)
                                              for index, row in signif_data.iterrows() if len(signif_data) > 0])

        for c, roi in enumerate(sorted(list(set(data["roi"])))):

            df_sub = data_avg.loc[
                (data_avg["stimulation_site"] == stim_site) & (data_avg["roi"] == roi)]

            fig.add_trace(go.Scatter(x=df_sub.relFreq, y=df_sub.fft, marker_color=cmap[c + 4], line=dict(width=4),
                                     name=roi, showlegend=sl, legendgroup="rois"), row=2, col=1 + i)

        fig.add_trace(
            go.Scatter(x=[-2, -1, 0, 1, 2], y=[IAF - 2, IAF - 1, IAF, IAF + 1, IAF + 2], mode="lines",
                       marker_color="gray",
                       line=dict(width=0.5, dash="dash"), name="Stimulation reference", showlegend=sl,
                       legendgroup="rois"), row=2, col=1 + i)

    fig.update_layout(boxmode='group',
                      yaxis_title="Functional Connectivity (PLV)", yaxis2_title="FFT peak (Hz)",
                      xaxis1=dict(showgrid=True, zerolinewidth=3, showticklabels=True, tickmode='array', tickangle=270,
                                  tickvals=[base_f, tRNS_f, -2.45, -1.47, -0.49, 0, 0.49, 1.47, 2.45],
                                  ticktext=["Baseline", "tRNS", -2.45, -1.47, -0.49, "IAF", 0.49, 1.47, 2.45]),
                      xaxis2=dict(showgrid=True, zerolinewidth=3, showticklabels=True, tickmode='array', tickangle=270,
                                  tickvals=[base_f, tRNS_f, -2.45, -1.47, -0.49, 0, 0.49, 1.47, 2.45],
                                  ticktext=["Baseline", "tRNS", -2.45, -1.47, -0.49, "IAF", 0.49, 1.47, 2.45], title="Stimulation Frequency (Hz)"),
                      legend_tracegroupgap=320)

    pio.write_html(fig, file=figures_folder + '/FC&FFT_' + title + str(n_simulations) + "sim_v4.html",
                   auto_open=True)


### C. Statistics.
# Starting with the global average df: results_2plot_avg
import pingouin as pg
from scipy import stats


## C1. P3P4 protocol - Pairwise Comparisons. vs baseline .
### about ASSUMPTIONS # We cannot assume normality because the sample is too small (n=10).
# Additionally, shapiro tests show significance difference from normal
pwc_vsBase, assumptions = pairwise_comparisons(results_2plot, mode="baseline", stat_test="ttest")
assumptions.columns = ["stimulation_site", "relFreq", "stat_base", "p_base", "stat_test", "p_test"]

# PLOTTING w/ STATS (vsBaseline)
plot_results_x2(results_2plot, IAF, "roast_P3P4Model", connMode="avg", ttest=pwc_vsBase, title="P3P4_vsBase_sig")
plot_results_x2(results_2plot, IAF, "roast_P3P4Model", connMode="avg+all", ttest=pwc_vsBase, title="P3P4_vsBase_sig_all")

## General Statistical Measure (ANOVA/Friedman's) for the effect of stimulating with different frquencies.
df_temp = results_2plot.loc[(results_2plot["stim_params"] != 0) & (results_2plot["stimulus_type"] == "sinusoid")]
pg.friedman(df_temp, dv="plv", within="relFreq", subject="subject", method="f")




## C2. targetACC protocol - Pairwise Comparisons vs P3P4 model
pwc_interProtocols, assumptions = pairwise_comparisons(results_2plot, mode="P3P4_vs_ACCtarget", stat_test="auto")
assumptions.columns = ["relFreq", "stat_base", "p_base", "stat_test", "p_test"]

## General Statistical Measure (ANOVA/Friedman's)
df_temp = results_2plot.loc[(results_2plot["stim_params"] != 0) & (results_2plot["stimulus_type"] == "sinusoid")]
pg.friedman(results_2plot, dv="plv", within="stimulation_site", subject="subject", method="f")

def plot_results_x2_doublebox(data, IAF, ttest=None, title="AVG"):

    data_avg = data.groupby(["stimulation_site", "relFreq", "roi", "subject"]).mean().reset_index()

    df_p3p4 = data_avg.loc[data_avg["stimulation_site"] == "roast_P3P4Model"]
    df_acc = data_avg.loc[data_avg["stimulation_site"] == "roast_ACCtarget"]

    ## Common processes
    cmap = px.colors.qualitative.Plotly  # color palette
    fig = make_subplots(rows=2, cols=1, specs=[[{}], [{}]], shared_yaxes=True, vertical_spacing=0.20)

    # Boxpoints: suspectedoutliers, all, outliers, False
    fig.add_trace(go.Box(x=df_p3p4.relFreq, y=df_p3p4.plv, marker_color="steelblue", name="P3-P4 protocol",
                           boxpoints='suspectedoutliers'), row=1, col=1)

    fig.add_trace(go.Box(x=df_acc.relFreq, y=df_acc.plv, marker_color="indianred", name="ACC-target protocol",
                           boxpoints='suspectedoutliers'), row=1, col=1)

    fig.add_trace(go.Scatter(x=[1], y=[2], marker_color="white", name=" "), row=1, col=1)
    fig.add_trace(go.Scatter(x=[1], y=[2], marker_color="white", name=" "), row=1, col=1)


    ## Add significance level
    if ttest is not None:

        signif_data = ttest.loc[ttest["sig"] != "ns"]

        fig["layout"].update(annotations=[dict(x=row.relFreq, y=row.maxVal + 0.15,
                                               xref="x", yref="y",
                                               text=row.sig, textangle=90, showarrow=False)
                                          for index, row in signif_data.iterrows() if len(signif_data) > 0])

    for c, roi in enumerate(sorted(list(set(data["roi"])))):

        data_avg = data_avg.groupby(["stimulation_site", "roi", "relFreq"]).mean().reset_index()

        df_sub = data_avg.loc[
            (data_avg["stimulation_site"] == "roast_ACCtarget") & (data_avg["roi"] == roi)]

        fig.add_trace(go.Scatter(x=df_sub.relFreq, y=df_sub.fft, marker_color=cmap[c + 4], line=dict(width=4),
                                 name=roi), row=2, col=1)

    fig.add_trace(
        go.Scatter(x=[-2, -1, 0, 1, 2], y=[IAF - 2, IAF - 1, IAF, IAF + 1, IAF + 2], mode="lines",
                   marker_color="gray",
                   line=dict(width=0.5, dash="dash"), name="Reference", showlegend=False), row=2, col=1)

    fig.update_layout(boxmode='group',
                      yaxis_title="Functional Connectivity (PLV)", yaxis2_title="FFT peak (Hz)",
                      xaxis1=dict(showgrid=True, zerolinewidth=3, showticklabels=True, tickmode='array', tickangle=270,
                                  tickvals=[base_f, tRNS_f, -2.45, -1.47, -0.49, 0, 0.49, 1.47, 2.45],
                                  ticktext=["Baseline", "tRNS", -2.45, -1.47, -0.49, "IAF", 0.49, 1.47, 2.45]),
                      xaxis2=dict(showgrid=True, zerolinewidth=3, showticklabels=True, tickmode='array', tickangle=270,
                                  tickvals=[base_f, tRNS_f, -2.45, -1.47, -0.49, 0, 0.49, 1.47, 2.45],
                                  ticktext=["Baseline", "tRNS", -2.45, -1.47, -0.49, "IAF", 0.49, 1.47, 2.45], title="Relative Stimulation Frequency (Hz-IAF)"),
                      yaxis=dict(range=[0, 1.1]), height=700)

    fig.update_layout(legend=dict(orientation="h", y=0.51, x=0.05))

    pio.write_html(fig, file=figures_folder + '/FC&FFT_' + title + str(n_simulations) + "sim.html",
                   auto_open=True)

plot_results_x2_doublebox(results_2plot, IAF, title="ACCtarget_vsProtocol")


# ## T-Tests vs Global mean
# ttest_vsGlob = pairwise_comparisons(results_2plot, mode="global_mean")
# ## AVERAGE w/ STATS (vsGlob) - PLOTTING v4 ### Plot an average result
# # Average for FFT plots
# plot_results_x2(results_2plot, IAF, "roast_ACCtarget", connMode="avg", ttest=ttest_vsGlob, title="ACCtarget_vsGlob_sig")




## C3. Others
# plot_results_x4(results_2plot, IAF, connMode="all+avg", ttest=ttest_vsBase, title="AVG")
# SUBJECT - PLOTTING v4 # Trying to fit all information in a single plot per
for ii, emp_subj in enumerate(set(results_2plot["subject"])):

    axis_config = dict(showgrid=True, zerolinewidth=2, showticklabels=True, tickmode='array', tickangle=270,
                       tickvals=[base_f, tRNS_f, -2.45, -1.47, -0.49, 0, 0.49, 1.47, 2.45],
                       ticktext=["Baseline", "tRNS", -2.45, -1.47, -0.49, "IAF", 0.49, 1.47, 2.45])

    auto_open = True if ii == 0 else False

    subset = results_2plot.loc[results_2plot["subject"] == emp_subj]

    # subset = subset.loc[subset["stim_params"] > 1]

    # Average for FFT plots
    subset_avg = subset.groupby(["stimulation_site", "relFreq", "roi"]).mean().reset_index()

    # freq_max_fc = subset_avg["relFreq"][subset_avg["plv"].idxmax()]
    # freq_min_fc = subset_avg["relFreq"][subset_avg["plv"].idxmin()]

    # color palette
    cmap = px.colors.qualitative.Plotly

    # gomagerit.cesvima.um
    fig = make_subplots(rows=2, cols=2, column_titles=("P3P4 Model", "targetACC Model"),
                        specs=[[{}, {}], [{}, {}]], shared_yaxes=True, shared_xaxes=True,
                        x_title="Stimulation Frequency (Hz)")

    for i, stim_site in enumerate(["roast_P3P4Model", "roast_ACCtarget"]):

        IAF = bSubj["closest_stimFreq"].loc[(bSubj["stimulation_site"]==stim_site)&(bSubj["subject"]==emp_subj)].values[0]


        sl = True if i == 0 else False

        for c, coi in enumerate(sorted(set(subset["connection"]))):
            df_sub = subset.loc[(subset["stimulation_site"] == stim_site) & (subset["connection"] == coi)]

            fig.add_trace(go.Box(x=df_sub.relFreq, y=df_sub.plv, marker_color=cmap[c], name=coi,
                                 legendgroup=coi, showlegend=sl), row=1, col=i + 1)

        for c, roi in enumerate(sorted(list(set(subset["roi"])))):
            df_sub = subset_avg.loc[(subset_avg["stimulation_site"] == stim_site) & (subset_avg["roi"] == roi)]

            fig.add_trace(go.Scatter(x=df_sub.relFreq, y=df_sub.fft, marker_color=cmap[c + 4], line=dict(width=4),
                                     name=roi, legendgroup=roi, showlegend=sl), row=2, col=i + 1)

        fig.add_trace(go.Scatter(x=[-2, -1, 0, 1, 2], y=[IAF-2, IAF-1, IAF, IAF+1, IAF+2], mode="lines", marker_color="gray",
                                 line=dict(width=0.5, dash="dash"), name="Stimulation reference", showlegend=sl), row=2,
                      col=i + 1)

    fig.update_layout(boxmode='group',
                      yaxis_title="Functional Connectivity (PLV)", yaxis3_title="FFT peak (Hz)",
                      xaxis=axis_config, xaxis2=axis_config, xaxis3=axis_config, xaxis4=axis_config)

    pio.write_html(fig, file=figures_folder + "/" + emp_subj + 'FC&FFT_w' + str(n_simulations) + "sim_v4.html",
                   auto_open=auto_open)

# CONNECTION - PLOTTING v4 ### Plot an average result BY CONNECTION
# Average for FFT plots
plot_results_x4(results_2plot, IAF, connMode="all+avg", title="AVG")

# AVERAGE - PLOTTING v4 ### Plot an average result
# Average for FFT plots
plot_results_x4(results_2plot, IAF, connMode="all+avg", title="AVG")



## DEPRECATED @ 05/05/2022: used when we collected data in a fixed range of frequencies
# ## Prepare dataframe to PLOTTING.
# stimFreqs = sorted(list(set(results.loc[results["stimulus_type"] == "sinusoid"].stim_params.values)))
#
# # Define precuneus natural frequency per subject (an average between L and R)
# baseline_Prec_peak = results.loc[(results["stim_params"] == 0) & (results["stimulus_type"] == "sinusoid") & (results["roi"].isin(["Precuneus_L", "Precuneus_R"]))]
# baseline_Prec_peak = baseline_Prec_peak.drop(columns=["mode", "stimW", "stim_params", "g", "speed", "connection", "plv", "band"])
# bSubj = baseline_Prec_peak.groupby(["stimulation_site", "subject"]).mean().reset_index().drop(columns="rep")
#
# # Make the mean of Precuneus natural freqs (for all subjects) and get the reference to set the mean IAF
# bProtocol = baseline_Prec_peak.groupby(["stimulation_site"]).mean().reset_index().drop(columns="rep")
#
# del baseline_Prec_peak
#
# # Transform all stim freq range in sth relative to IAF
# # choose the closest stimFreqs value to the one in each baseline protocol
# bSubj["closest_stimFreq"] = \
#     [stimFreqs[abs((row.fft - np.array(stimFreqs))).argmin()] for index, row in bSubj.iterrows()]
#
# bProtocol["closest_stimFreq"] = \
#     [stimFreqs[abs((row.fft - np.array(stimFreqs))).argmin()] for index, row in bProtocol.iterrows()]
#
# IAF = stimFreqs[abs(np.average(bProtocol["closest_stimFreq"].values) - np.array(stimFreqs)).argmin()]
#
# # work on Results: Tarda un rato
# # results_2plot = results.copy()
# results["relFreq"] = 0
#
# for index, row in bSubj.iterrows():
#     print(row)
#     for freq in stimFreqs:
#         results["relFreq"].loc[(results["stimulation_site"] == row.stimulation_site) &
#                                (results["subject"] == row.subject) &
#                                (results["stim_params"] == freq) &
#                                (results["stimulus_type"] == "sinusoid")] = round(freq - row.closest_stimFreq, 2)
#
# # prepara el subset: w=0 (baseline) to w=6; "noise" param= 0  to param=7
# base_f = -3.5
# tRNS_f = -3
# results["relFreq"].loc[(results["stimulus_type"] == "sinusoid") & (results["stim_params"] == 0)] = base_f
# results["relFreq"].loc[(results["stimulus_type"] == "noise") & (results["stim_params"] == 0)] = tRNS_f
#
# relFreqs = sorted(list(set(results.relFreq)))
#
# # Selecciona solo aquellas relFreqs compartidas por todos los sujetos y reduce el número de ellos by spacing
# spacing = 2
# min_f = max(results.loc[~results["relFreq"].isin([base_f, tRNS_f])].groupby(["subject"]).min().relFreq.values)
# max_f = min(results.loc[~results["relFreq"].isin([base_f, tRNS_f])].groupby(["subject"]).max().relFreq.values)
#
# relFreqs_sub = [base_f, tRNS_f] + relFreqs[relFreqs.index(min_f):relFreqs.index(max_f):spacing]
#
# results_2plot = results.loc[results["relFreq"].isin(relFreqs_sub)]
