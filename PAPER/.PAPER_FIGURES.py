
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
import glob

figures_folder = "PAPER\\"



## PART A:
####### Stimulation Weight Fit
fname = "PSEmpi_stimWfit_indWP3-m02d03y2022-t19h.43m.56s - w0.25"
folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\\1stimWeight_Fitting\PSE\\" + fname

# Load data
resultsAAL = pd.read_csv(glob.glob(folder + "\\*results.csv")[0])
n_rep = resultsAAL["rep"].max() + 1

# Calculate percentage
baseline = resultsAAL.loc[resultsAAL["w"] == 0].groupby("Subject").mean()

resultsAAL["percent"] = \
    [(row["bModule"] - baseline.loc[row["Subject"]].bModule) / baseline.loc[row["Subject"]].bModule * 100
     for i, row in resultsAAL.iterrows()]

# Just show half the calibration constants to make a clearer picture
include_w = np.arange(0, 0.5, 0.02)
resultsAAL_sub = resultsAAL

### A. Scatter plot with Mean line for percentage
fig = px.strip(resultsAAL_sub, x="w", y="percent", color="Subject")

w = np.asarray(resultsAAL_sub.groupby("w").mean().reset_index()["w"])
mean = np.asarray(resultsAAL_sub.groupby("w").mean()["percent"])
median = np.asarray(resultsAAL_sub.groupby("w").median()["percent"])

fig.add_trace(go.Scatter(x=w, y=mean, mode="lines", name="mean", line=dict(color='darkslategray', width=5)))
fig.add_trace(go.Scatter(x=w, y=median, mode="lines", name="median", line=dict(color='slategray', width=4), visible="legendonly"))

fig.update_xaxes(title="Calibration constant (K) <br>i.e. Stimulation Weight")
fig.update_yaxes(title="Alpha band power change (%)", tickvals=[-40, -20, 0, 14, 20, 40, 60, 80, 100])

pio.write_html(fig, file=figures_folder + '\\allNemosAAL_4scatterpercent_alphaRise_' + str(n_rep) + "sim.html",
               auto_open=True)

## A2. plot for absolute power change bands || Is it needed? Just if some subject behaves weirdly.
fig = px.strip(resultsAAL_sub, x="w", y="bModule", color="Subject",
             title="Alpha band power rise @Pareto-Occipital regions<br>(%i simulations | 10 subjects AAL)" % n_rep,
             labels={  # replaces default labels by column name
                 "w": "Calibration constant (K) <br>i.e. Stimulation Weight", "bModule": "Alpha band power (mV*Hz)"})

pio.write_html(fig, file=figures_folder + '\\allNemosAAL_1scatterModules_alphaRise_' + str(n_rep) + "sim.html",
               auto_open=True)




### PART B:
## Plot 2testFrequencies with statistics
fname = "PSEmpi_testFrequenciesWmean_indWP3-m02d04y2022-t00h.43m.45s"
specific_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\\2testFrequencies\PSE\\" + fname
# ctb_folder = "E:\LCCN_Local\PycharmProjects\\CTB_data2\\"

# Cargar los datos
results = pd.read_csv(glob.glob(specific_folder + "\\*results.csv")[0])
results.columns = "stimulation_site", "stimulus_type", "stim_params", "mode", "subject", "g", "speed", "stimW", "rep", "band", 'plv0', 'plv1', 'plv2', 'plv3', 'fft0', 'fft1', 'fft2', 'fft3'

del specific_folder, fname


# From wide to long in connections
results = pd.wide_to_long(results, stubnames='plv',
                          i=["stimulation_site", "stimulus_type", "stim_params", "mode", "subject",
                             "g", "speed", "stimW", "rep", "band", 'fft0', 'fft1', 'fft2', 'fft3'],
                          j='connection').reset_index()
# Back to rel labels
rel_labels = ['Cingulate_Ant_L-Precuneus_L', 'Cingulate_Ant_L-Precuneus_R', 'Cingulate_Ant_R-Precuneus_L',
              'Cingulate_Ant_R-Precuneus_R']
results["connection"] = rel_labels * int(len(results) / len(rel_labels))

results = pd.wide_to_long(results, stubnames='fft',
                          i=["stimulation_site", "stimulus_type", "stim_params", "mode", "subject",
                             "g", "speed", "stimW", "rep", 'connection', 'plv'], j='roi').reset_index()

# back to roi labels
roi_labels = ['ACC_L', 'ACC_R', 'Precuneus_L', 'Precuneus_R']
results["roi"] = roi_labels * int(len(results) / len(roi_labels))

# attributes
n_simulations = results["rep"].max() + 1
# w, g, s, struct = results["stimW"][0], results["g"][0], results["speed"][0], "AAL2"

# rois of interest
rois = [34, 35, 70, 71]  # rois implicated in the effect: 35-ACCl, 36-AACr, 71-Prl, 72-Prr [note python 0-indexing]


## Prepare dataframe to PLOTTING.
stimFreqs = sorted(list(set(results.loc[results["stimulus_type"] == "sinusoid"].stim_params.values)))

# Define precuneus natural frequency per subject (an average between L and R)
baseline_Prec_peak = results.loc[(results["stim_params"] == 0) & (results["stimulus_type"] == "sinusoid") & (results["roi"].isin(["Precuneus_L", "Precuneus_R"]))]
baseline_Prec_peak = baseline_Prec_peak.drop(columns=["mode", "stimW", "stim_params", "g", "speed", "connection", "plv", "band"])
bSubj = baseline_Prec_peak.groupby(["stimulation_site", "subject"]).mean().reset_index().drop(columns="rep")

# Make the mean of Precuneus natural freqs (for all subjects) and get the reference to set the mean IAF
bProtocol = baseline_Prec_peak.groupby(["stimulation_site"]).mean().reset_index().drop(columns="rep")

del baseline_Prec_peak

# Transform all stim freq range in sth relative to IAF
# choose the closest stimFreqs value to the one in each baseline protocol
bSubj["closest_stimFreq"] = \
    [stimFreqs[abs((row.fft - np.array(stimFreqs))).argmin()] for index, row in bSubj.iterrows()]

bProtocol["closest_stimFreq"] = \
    [stimFreqs[abs((row.fft - np.array(stimFreqs))).argmin()] for index, row in bProtocol.iterrows()]

IAF = stimFreqs[abs(np.average(bProtocol["closest_stimFreq"].values) - np.array(stimFreqs)).argmin()]

# work on Results: Tarda un rato
# results_2plot = results.copy()
results["relFreq"] = 0

for index, row in bSubj.iterrows():
    print(row)
    for freq in stimFreqs:
        results["relFreq"].loc[(results["stimulation_site"] == row.stimulation_site) &
                               (results["subject"] == row.subject) &
                               (results["stim_params"] == freq) &
                               (results["stimulus_type"] == "sinusoid")] = round(freq - row.closest_stimFreq, 2)

# prepara el subset: w=0 (baseline) to w=6; "noise" param= 0  to param=7
base_f = -3.5
tRNS_f = -3
results["relFreq"].loc[(results["stimulus_type"] == "sinusoid") & (results["stim_params"] == 0)] = base_f
results["relFreq"].loc[(results["stimulus_type"] == "noise") & (results["stim_params"] == 0)] = tRNS_f

relFreqs = sorted(list(set(results.relFreq)))

# Selecciona solo aquellas relFreqs compartidas por todos los sujetos y reduce el número de ellos by spacing
spacing = 2
min_f = max(results.loc[~results["relFreq"].isin([base_f, tRNS_f])].groupby(["subject"]).min().relFreq.values)
max_f = min(results.loc[~results["relFreq"].isin([base_f, tRNS_f])].groupby(["subject"]).max().relFreq.values)

relFreqs_sub = [base_f, tRNS_f] + relFreqs[relFreqs.index(min_f):relFreqs.index(max_f):spacing]

results_2plot = results.loc[results["relFreq"].isin(relFreqs_sub)]


#### FUNCTIONS

def plot_results_x2(data, IAF, stim_site, connMode="avg", ttest=None, title="AVG"):

    data_avg = data.groupby(["stimulation_site", "relFreq", "roi"]).mean().reset_index()

    ## Common processes
    cmap = px.colors.qualitative.Plotly  # color palette
    fig = make_subplots(rows=2, cols=1, specs=[[{}], [{}]], shared_yaxes=True, vertical_spacing=0.15)

    if "avg" in connMode:
        df_sub = data.loc[data["stimulation_site"] == stim_site]
        df_sub = df_sub.groupby(["relFreq", "subject"]).mean().reset_index()
        # Boxpoints: suspectedoutliers, all, outliers, False
        fig.add_trace(go.Box(x=df_sub.relFreq, y=df_sub.plv, marker_color="steelblue", name="ACC - Precuneus",
                             showlegend=False,  boxpoints='suspectedoutliers'), row=1, col=1)

    if "all" in connMode:
        for c, coi in enumerate(sorted(set(data["connection"]))):
            df_sub = data.loc[(data["stimulation_site"] == stim_site) & (data["connection"] == coi)]

            fig.add_trace(go.Box(x=df_sub.relFreq, y=df_sub.plv, marker_color=cmap[c], name=coi), row=1, col=1)

    ## Add significance level
    if ttest is not None:

        signif_data = ttest.loc[(ttest["stim_site"] == stim_site) & (ttest["sig"] != "ns")]

        fig["layout"].update(annotations=[dict(x=row.relFreq, y=row.maxVal + 0.15,
                                               xref="x", yref="y",
                                               text=row.sig, textangle=90, showarrow=False)
                                          for index, row in signif_data.iterrows() if len(signif_data) > 0])

    for c, roi in enumerate(sorted(list(set(data["roi"])))):

        df_sub = data_avg.loc[
            (data_avg["stimulation_site"] == stim_site) & (data_avg["roi"] == roi)]

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

    if "avg" in connMode:
        fig.update_layout(legend=dict(orientation="h", y=0.48, x=0.35))

    pio.write_html(fig, file=figures_folder + '/FC&FFT_' + title + str(n_simulations) + "sim.html",
                   auto_open=True)

# mode :: "P3P4_vs_ACCtarget" or "baseline"
def pairwise_comparisons(data, mode, stat_test="auto"):
    """
    It runs pairwise comparisons after checking for parametric assumptions.
    It will decide based on assumption whether to use ttest or wilcoxon test.

    :param data:
    :param mode:
    :return:
    """

    pwc = pd.DataFrame()
    assumptions = pd.DataFrame()

    if mode == "baseline":

        initFreq = 1

        for stim_site in set(results_2plot.stimulation_site):

            df = data.loc[data["stimulation_site"] == stim_site]

            for f in relFreqs_sub[initFreq:]:

                df_temp = df.groupby(["subject", "relFreq"]).mean().reset_index()

                # if mode == "global_mean":
                #     # avg per subject df(b) through all relFReqs
                #     baseline = df_temp.groupby("subject").mean().reset_index()

                # elif mode == "baseline":
                    # avg per subject df(b)
                baseline = df_temp.loc[(df_temp["stim_params"] == 0) & (df_temp["relFreq"] == base_f)]

                # average per subject df(f)
                test_group = df_temp.loc[df_temp["relFreq"] == f]

                ## CHECK PARAMETRIC ASSUMPTIONS
                # Samples' Normality
                n_base = stats.shapiro(baseline.plv.values)
                n_test = stats.shapiro(test_group.plv.values)

                assumptions = assumptions.append([[stim_site, f, n_base.statistic, n_base.pvalue, n_test.statistic, n_test.pvalue]])

                if ((n_base.pvalue > 0.05) & (n_test.pvalue > 0.05)) or (stat_test == "ttest"):
                    test = pg.ttest(x=test_group.plv.values, y=baseline.plv.values, paired=True, alternative="less")
                else:
                    test = pg.wilcoxon(x=test_group.plv.values, y=baseline.plv.values, alternative="less")

                test["assump_met"] = "yes" if (n_base.pvalue > 0.05) & (n_test.pvalue > 0.05) else "no"
                test["stim_site"] = stim_site
                test["relFreq"] = f
                test["maxVal"] = max(test_group.plv.values)
                test["baselineMean"] = np.average(baseline.plv.values)
                test["testMean"] = np.average(test_group.plv.values)

                pwc = pwc.append(test)

    elif mode == "P3P4_vs_ACCtarget":

        df_p3p4 = data.loc[data["stimulation_site"] == "roast_P3P4Model"].groupby(["subject", "relFreq"]).mean().reset_index()

        df_acc = data.loc[data["stimulation_site"] == "roast_ACCtarget"].groupby(["subject", "relFreq"]).mean().reset_index()

        for f in relFreqs_sub:

            baseline = df_p3p4.loc[df_p3p4["relFreq"] == f]

            # average per subject df(f)
            test_group = df_acc.loc[df_acc["relFreq"] == f]

            ## CHECK PARAMETRIC ASSUMPTIONS
            # Samples' Normality
            n_base = stats.shapiro(baseline.plv.values)
            n_test = stats.shapiro(test_group.plv.values)

            assumptions = assumptions.append(
                [[f, n_base.statistic, n_base.pvalue, n_test.statistic, n_test.pvalue]])

            if stat_test == "t-test":
                test = pg.ttest(x=test_group.plv.values, y=baseline.plv.values, paired=True, alternative="less")
            else:
                test = pg.wilcoxon(x=test_group.plv.values, y=baseline.plv.values, alternative="less")

            test["assump_met"] = "yes" if (n_base.pvalue > 0.05) & (n_test.pvalue > 0.05) else "no"
            test["mode"] = mode
            test["relFreq"] = f
            test["maxVal"] = max(baseline.plv.values)
            test["baselineMean"] = np.average(baseline.plv.values)
            test["testMean"] = np.average(test_group.plv.values)

            pwc = pwc.append(test)

    # Correct with bonferroni
    pwc["p.corr"] = pg.multicomp(pwc["p-val"].values, method="bonf")[1]

    # Add asterisks
    pwc["sig"] = ["****" if row["p.corr"] <= 0.0001 else
                    "***" if row["p.corr"] <= 0.001 else
                    "**" if row["p.corr"] <= 0.01 else
                    "*" if row["p.corr"] <= 0.05 else "ns"
                    for index, row in pwc.iterrows()]

    return pwc, assumptions


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



######################## STRUCTURAL VULNERABILITY TO STIMULATION

######### Network analysis
# ¿Qué metricas incluyo? n paths total para cada roi y avg, porcentaje del total,
# media de tractos que entran por cada conexion, indegree, outdegree, etc.
from tvb.datatypes import connectivity
import networkx as nx

## Working over the averaged matrices
# Load structures
conn = connectivity.Connectivity.from_file("E://LCCN_Local/PycharmProjects/CTB_data2/NEMOS_AVG_AAL2.zip")

matrix = conn.weights

## Add up cerebellum weights to do single nodes plots
cer_rois = []
[cer_rois.append(roi) if "Cer" in roi else None for roi in conn.region_labels]
cer_ids = [list(conn.region_labels).index(roi) for roi in cer_rois]

ver_rois = []
[ver_rois.append(roi) if "Ver" in roi else None for roi in conn.region_labels]
ver_ids = [list(conn.region_labels).index(roi) for roi in ver_rois]


# sum cols, sum rows, delete excedent
matrix_single_nodes = matrix
# Left hemisphere
matrix_single_nodes[cer_ids[0], :] = np.sum(matrix[cer_ids[0::2], :], axis=0)
matrix_single_nodes[:, cer_ids[0]] = np.sum(matrix[:, cer_ids[0::2]], axis=1)

# Right hemisphere
matrix_single_nodes[cer_ids[1], :] = np.sum(matrix[cer_ids[1::2], :], axis=0)
matrix_single_nodes[:, cer_ids[1]] = np.sum(matrix[:, cer_ids[1::2]], axis=1)

# Vermis
matrix_single_nodes[ver_ids[0], :] = np.sum(matrix[ver_ids, :], axis=0)
matrix_single_nodes[:, ver_ids[0]] = np.sum(matrix[:, ver_ids], axis=1)


matrix_single_nodes = np.delete(matrix_single_nodes, ver_ids[1:], axis=0)
matrix_single_nodes = np.delete(matrix_single_nodes, ver_ids[1:], axis=1)
matrix_single_nodes = np.delete(matrix_single_nodes, cer_ids[2:], axis=0)
matrix_single_nodes = np.delete(matrix_single_nodes, cer_ids[2:], axis=1)


np.fill_diagonal(matrix_single_nodes, 0)

# region labels of 120 menos las del cer
regionLabels = []
[regionLabels.append(roi) if roi not in cer_rois+ver_rois else None for roi in conn.region_labels]
regionLabels.append("Cerebellum_L")
regionLabels.append("Cerebellum_R")
regionLabels.append("Vermis")

# Convert matrices to adj matrices
net = nx.convert_matrix.from_numpy_array(np.asarray(matrix_single_nodes))
    # This generates an undirected graph (Graph). Not a directed graph (DiGraph).


# label mapping
mapping = {i: roi for i, roi in enumerate(regionLabels)}
net = nx.relabel_nodes(net, mapping)

### NETWORK METRICS  # Compute metrics of interest for all nodes: append to dataframe

## Centrality
# 1. Degree normalized
degree = pd.DataFrame.from_dict(nx.degree_centrality(net), orient="index", columns=["degree"])


# 2. Node strength normalized
node_strength_norm = pd.DataFrame.from_dict({node: val/matrix_single_nodes.sum(axis=1).max()
                                        for (node, val) in net.degree(weight="weight")},
                                       orient="index", columns=["node_strength_norm"])

# 2b. Node strength
node_strength = pd.DataFrame.from_dict({node: round(val, 4)
                                        for (node, val) in net.degree(weight="weight")},
                                       orient="index", columns=["node_strength"])

# Specific connectivity Pre-ACC
matrix_single_nodes[regionLabels.index("Precuneus_L"):regionLabels.index("Precuneus_R")+1, regionLabels.index("Cingulate_Ant_L"):regionLabels.index("Cingulate_Ant_R")+1]
sum(sum(matrix_single_nodes[regionLabels.index("Precuneus_L"):regionLabels.index("Precuneus_R")+1, regionLabels.index("Cingulate_Ant_L"):regionLabels.index("Cingulate_Ant_R")+1]))

# 3. Closeness
closeness = pd.DataFrame.from_dict(nx.closeness_centrality(net), orient="index", columns=["closeness"])

# 4. Betweeness
betweeness = pd.DataFrame.from_dict(nx.betweenness_centrality(net), orient="index", columns=["betweeness"])


## Global Integration
# 5. Path length
path_length = pd.DataFrame.from_dict({source: np.average(list(paths.values()))
                                      for source, paths in nx.shortest_path_length(net)},
                                     orient="index", columns=["path_length"])
np.std(path_length.values)
nx.average_shortest_path_length(net)

# Specific path length ACC-Pre
nx.shortest_path_length(net, source="Precuneus_L", target="Cingulate_Ant_L")
nx.shortest_path_length(net, source="Precuneus_R", target="Cingulate_Ant_L")
nx.shortest_path_length(net, source="Precuneus_L", target="Cingulate_Ant_R")
nx.shortest_path_length(net, source="Precuneus_R", target="Cingulate_Ant_R")


## Local Segregation
# 6. Clustering coefficient
clustering = pd.DataFrame.from_dict(nx.clustering(net), orient="index", columns=["clustering"])
np.std(clustering.values)
nx.average_clustering(net)

# 7. Modularity (Newman approach)
comms = nx.community.greedy_modularity_communities(net)
nx.community.modularity(net, comms)



#### Gatering Results
network_analysis = pd.concat([degree, node_strength_norm, node_strength, closeness, betweeness, clustering, path_length], axis=1).reindex(degree.index)

# Subset dataframe
network_analysis_wide_l = network_analysis[0:-1:2]
network_analysis_wide_r = network_analysis[1::2]

# Add averages
columns = ["degree", "node_strength_norm", "node_strength", "closeness", "betweeness", "clustering", "path_length"]
network_analysis_wide_l = network_analysis_wide_l.append(
    pd.DataFrame([np.average(network_analysis_wide_l, axis=0)], columns=columns))
network_analysis_wide_r = network_analysis_wide_r.append(
    pd.DataFrame([np.average(network_analysis_wide_r, axis=0)], columns=columns))

# relabel Indexes
network_analysis_wide_l.index = [label[:-2] for label in network_analysis_wide_l.index[:-1]] + ["Average"]
network_analysis_wide_r.index = [label[:-2] for label in network_analysis_wide_r.index[:-1]] + ["Average"]

network_analysis_wide_avg = pd.concat((network_analysis_wide_l, network_analysis_wide_r))
network_analysis_wide_avg = network_analysis_wide_avg.groupby(network_analysis_wide_avg.index).mean()

# Rename columns
network_analysis_wide_l.columns = [col + "_l" for col in columns]
network_analysis_wide_r.columns = [col + "_r" for col in columns]
network_analysis_wide_avg.columns = [col + "_avg" for col in columns]

# join dataframes
network_analysis_wide = network_analysis_wide_l.join(network_analysis_wide_r)
network_analysis_wide = network_analysis_wide.join(network_analysis_wide_avg)


#### PLOT centrality measures
color_sub = ["darkslategray", "steelblue"]
color_all = ["lightslategray", "lightsteelblue"]

network_analysis_wide["color_l"] = [color_all[0]] * len(network_analysis_wide[:-1]) + [color_sub[0]]
network_analysis_wide["color_l"][["Precuneus", "Cingulate_Ant"]] = [color_sub[0]] * 2
network_analysis_wide["color_r"] = [color_all[1]] * len(network_analysis_wide[:-1]) + [color_sub[1]]
network_analysis_wide["color_r"][["Precuneus", "Cingulate_Ant"]] = [color_sub[1]] * 2

fig = make_subplots(rows=1, cols=3, column_titles=("Degree", "Betweeness", "Path Length"), horizontal_spacing=0.15,
                    specs=[[{}, {}, {}]])

# temp = network_analysis_wide.sort_values(by="degree_avg")
# fig.add_trace(go.Bar(x=-temp.degree_l.values, y=temp.index, orientation='h', marker=dict(color=temp.color_l.values), showlegend=False), row=1, col=1)
# fig.add_trace(go.Bar(x=temp.degree_r.values, y=temp.index, orientation='h', marker=dict(color=temp.color_r.values), showlegend=False), row=1, col=1)

temp = network_analysis_wide.sort_values(by="node_strength_norm_avg")
fig.add_trace(go.Bar(x=-temp.node_strength_norm_l.values, y=temp.index, orientation='h', marker=dict(color=temp.color_l.values),
                     customdata=temp.node_strength_l.values, hovertemplate="%{x}, %{y} <br> Tracts: %{customdata}", showlegend=False), row=1, col=1)
fig.add_trace(go.Bar(x=temp.node_strength_norm_r.values, y=temp.index, orientation='h', marker=dict(color=temp.color_r.values),
                     customdata=temp.node_strength_l.values, hovertemplate="%{x}, %{y} <br> Tracts: %{customdata}", showlegend=False), row=1, col=1)

temp = network_analysis_wide.sort_values(by="betweeness_avg")
fig.add_trace(go.Bar(x=-temp.betweeness_l.values, y=temp.index, orientation='h', marker=dict(color=temp.color_l.values), showlegend=False), row=1, col=2)
fig.add_trace(go.Bar(x=temp.betweeness_r.values, y=temp.index, orientation='h', marker=dict(color=temp.color_r.values), showlegend=False), row=1, col=2)

# temp = network_analysis_wide.sort_values(by="closeness_avg")
# fig.add_trace(go.Bar(x=-temp.closeness_l.values, y=temp.index, orientation='h', marker=dict(color=temp.color_l.values), showlegend=False), row=1, col=3)
# fig.add_trace(go.Bar(x=temp.closeness_r.values, y=temp.index, orientation='h', marker=dict(color=temp.color_r.values), showlegend=False), row=1, col=3)

temp = network_analysis_wide.sort_values(by="path_length_avg", ascending=False)
fig.add_trace(go.Bar(x=-temp.path_length_l.values, y=temp.index, orientation='h', marker=dict(color=temp.color_l.values), showlegend=False), row=1, col=3)
fig.add_trace(go.Bar(x=temp.path_length_r.values, y=temp.index, orientation='h', marker=dict(color=temp.color_r.values), showlegend=False), row=1, col=3)

fig.update_layout(template="plotly_white", barmode="relative", height=1000)
pio.write_html(fig, file=figures_folder + '/Network_metrics.html', auto_open=True)



######### Plot entraiment by SC
import statsmodels.api as sm
from tvb.simulator.lab import *

# Load data
results = pd.read_csv("E:\\LCCN_Local\PycharmProjects\\neuroStimulation\entrainment_bySC\PSE\PSEmpi_entrainment_bySC_indWP3-m02d25y2022-t03h.41m.56s\entrainment_bySC_results.csv")
df_ent_bySC = pd.read_csv("E:\\LCCN_Local\PycharmProjects\\neuroStimulation\entrainment_bySC\PSE\PSEmpi_entrainment_bySC_indWP3-m02d25y2022-t03h.41m.56s\entrainment_bySC_10subjs.csv")

ctb_folder = "E:\LCCN_Local\PycharmProjects\\CTB_data2\\"
conn = connectivity.Connectivity.from_file(ctb_folder + "NEMOS_035_AAL2.zip")
regionLabels = conn.region_labels

# color palette
cmap = px.colors.qualitative.Plotly

## Functions

def entrainment_bySC_plot(results, df_ent_bySC, disconn_metric="tracts_left", lowess_frac=0.1):

    fig = go.Figure()

    for i, target in enumerate(sorted(set(results.target.values))):

        # subset data
        df_2plot = df_ent_bySC.loc[(df_ent_bySC["target"]==target) & (df_ent_bySC["roi"]==target)]
        df_2plot = df_2plot.sort_values(by=[disconn_metric])

        # original scatter points
        fig.add_trace(go.Scatter(x=df_2plot[disconn_metric], y=df_2plot.ent_range, opacity=0.2,
                                 mode="markers", marker_symbol="circle-open", marker_color=cmap[i],
                                 name=regionLabels[target], legendgroup=regionLabels[target], showlegend=True))

        # smoothed line
        smoothed_line = sm.nonparametric.lowess(exog=df_2plot[disconn_metric], endog=df_2plot.ent_range, frac=lowess_frac)
        # savgol_filter(df_2plot.ent_range, window_length=25, polyorder=2)

        fig.add_trace(go.Scatter(x=smoothed_line[:, 0], y=smoothed_line[:, 1], marker_color=cmap[i],
                                 line=dict(width=4), name=regionLabels[target], legendgroup=regionLabels[target],
                                 showlegend=True))

    fig.update_xaxes(title="Number of tracts")
    fig.update_yaxes(title="Entrainment range (Hz)")
    fig.update_layout(template="plotly_white")
    pio.write_html(fig, file=figures_folder + '/ent_bySC.html', auto_open=True)

entrainment_bySC_plot(results, df_ent_bySC, disconn_metric="tracts_left", lowess_frac=0.15)



def entrainment_bySC_x4plot(results, df_ent_bySC, disconn_metric="tracts_left", lowess_frac=0.1):
    fig = make_subplots(rows=2, cols=2, subplot_titles=("disconnecting ACC_L","disconnecting Precuneus_L",
                                                        "disconnecting ACC_R", "disconnecting Precuneus_R"),
                        specs=[[{}, {}], [{}, {}]], shared_yaxes=True,
                        x_title=disconn_metric)

    for i, target in enumerate(sorted(set(results.target.values))):
        for ii, roi in enumerate(sorted(set(results.target.values))):
            sl = True if i<1 else False
            # subset data
            df_2plot = df_ent_bySC.loc[(df_ent_bySC["target"]==target) & (df_ent_bySC["roi"]==roi)]
            df_2plot = df_2plot.sort_values(by=[disconn_metric])

            # original scatter points
            fig.add_trace(go.Scatter(x=df_2plot[disconn_metric], y=df_2plot.ent_range,
                                     mode="markers", marker_symbol="circle-open", marker_color=cmap[ii], opacity=0.3,
                                     name=regionLabels[roi], legendgroup=regionLabels[roi], showlegend=sl), row=i%2+1, col=i//2+1)

            # smoothed line
            smoothed_line = sm.nonparametric.lowess(exog=df_2plot[disconn_metric], endog=df_2plot.ent_range,
                                                             frac=lowess_frac)

            # smoothed_y = savgol_filter(df_2plot.ent_range, window_length=25, polyorder=4)

            fig.add_trace(go.Scatter(x=smoothed_line[:, 0], y=smoothed_line[:,1], marker_color=cmap[ii],
                                     line=dict(width=4), name=regionLabels[roi], legendgroup=regionLabels[roi],
                                     showlegend=sl), row=i%2+1, col=i//2+1)

    pio.write_html(fig, file=figures_folder + '/ent_bySCx4.html', auto_open=True)

entrainment_bySC_x4plot(results, df_ent_bySC, disconn_metric="tracts_left", lowess_frac=0.3)



## PLOTS per SUBJECT
## X1 Plot
for subj in set(results.subject.values):
    subset_df_ent = df_ent_bySC.loc[df_ent_bySC["subj"]==subj]
    entrainment_bySC_plot(results, subset_df_ent, disconn_metric="tracts_left", lowess_frac=0.35)
## X4 plot
for subj in set(results.subject.values):
    subset_df_ent = df_ent_bySC.loc[df_ent_bySC["subject"]==subj]
    entrainment_bySC_x4plot(results, subset_df_ent, disconn_metric="tracts_left")




##############         WHOLE NETWORK IMPACT
import pickle

file_name = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\\4wholeNetwork_impact\wholeNetwork_impact.pkl"
open_file = open(file_name, "rb")
wn_impact_results = pickle.load(open_file)
open_file.close()

# Preparation
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


# plot AVERAGE
fftpeaks_baseline_avg = np.average(np.asarray([subject[0] for subject in wn_impact_results]), axis=0)
plv_baseline_avg = np.average(np.asarray([subject[1] for subject in wn_impact_results]), axis=0)
fftpeaks_stimulated_avg = np.average(np.asarray([subject[2] for subject in wn_impact_results]), axis=0)
plv_stimulated_avg = np.average(np.asarray([subject[3] for subject in wn_impact_results]), axis=0)

ef_mag_avg = np.average(np.asarray([subject[4] for subject in wn_impact_results]), axis=0)


fig = make_subplots(rows=2, cols=2, specs=[[{}, {}], [{"colspan": 2, "secondary_y":True}, {}]], row_heights=[0.7, 0.3], shared_yaxes=True,
                    vertical_spacing=0.35, column_titles=["Baseline", "During stimulation"])

fig.add_trace(go.Heatmap(z=plv_baseline_avg, x=regionLabels, y=regionLabels, zmax=1, zmin=0,
                         colorbar=dict(thickness=10, y=0.77, len=0.45), colorscale="Viridis"), row=1, col=1)
fig.add_trace(go.Heatmap(z=plv_stimulated_avg, x=regionLabels, y=regionLabels, zmax=1, zmin=0, colorscale="Viridis", showscale=False), row=1, col=2)


delta_fc = np.average(plv_stimulated_avg[acc_index, :] - plv_baseline_avg[acc_index, :], axis=0)
order = np.argsort(delta_fc)

fig.add_trace(go.Scatter(x=regionLabels[order], y=delta_fc[order], name="FC ACC-others", line=dict(width=3)), row=2, col=1)

delta_fft = (fftpeaks_stimulated_avg - fftpeaks_baseline_avg)
fig.add_trace(go.Scatter(x=regionLabels[order], y=delta_fft[order], name="FFT peak", line=dict(width=3)), row=2, col=1)

fig.add_trace(go.Scatter(x=regionLabels[order], y=ef_mag_avg[order], name="Electric field input", line=dict(width=3)), secondary_y=True, row=2, col=1)

fig.update_layout(legend=dict(orientation="h", y=0.23, x=0.05), template="plotly_white",
                  yaxis3=dict(title="Delta FC<br>Delta FFT-peak", zerolinewidth=3), yaxis4=dict(title="Electric Field<br>(mV/mm)", zerolinewidth=3),
                  xaxis=dict(tickangle=45), xaxis2=dict(tickangle=45), xaxis3=dict(tickangle=45), height=850)

pio.write_html(fig, file=figures_folder + '/wholeNetwork_impact.html', auto_open=True)






## Color array for pairs of nodes
import plotly.express as px

cmap = px.colors.qualitative.Plotly

## Colors
# colors = [px.colors.label_rgb(px.colors.hex_to_rgb(cmap[(i//2)-((i//2)//len(cmap))*len(cmap)])) for i, roi_ef_mag in enumerate(ef_mag_avg[:94])] +\
#          [(0, 0, 0) for roi_ef_mag in ef_mag_avg[94:112]] + [(128, 128, 128) for roi_ef_mag in ef_mag_avg[112:]]

## Colors w/ Opcity
## Opacity array by ef_mag
range_op = [0, 1]
ef_mag_avg_normalized = abs(ef_mag_avg)/max(abs(ef_mag_avg))
opacity = ef_mag_avg_normalized * (range_op[1]-range_op[0]) + (range_op[0] - np.min(ef_mag_avg_normalized))

colors_op = [px.colors.label_rgb(px.colors.hex_to_rgb(cmap[(i//2)-((i//2)//len(cmap))*len(cmap)]) + (op, )) for i, op in enumerate(opacity[:94])] +\
            [(0, 0, 0, op) for op in opacity[94:112]] + \
            [(128, 128, 128, op) for op in opacity[112:]]  ## (128,128,128) Gray ## (0,0,0) Black


## Size by node strength
weights = np.sum(conn.weights, axis=1)
weights_norm = weights/max(weights)
range_size = [7, 70]
size = weights_norm * (range_size[1]-range_size[0]) + (range_size[0] - np.min(weights_norm))

#### Correlation between delta-FFTpeak and avg_ef_mag
stats.pearsonr(delta_fft, abs(ef_mag_avg[SC_cortex_idx]))


hovertext=[roi + "<br>" + str(round(ef_mag_avg[i],5)) +
           " - (mV/mm) Electric field input<br>" + str(round(weights_norm[i],3)) +
           " - Normalized node strength" for i, roi in enumerate(regionLabels)]

## PLOT
fig = make_subplots(rows=1, cols=2, specs=[[{}, {"type": "scene"}]])
fig.add_trace(go.Scatter(x=abs(ef_mag_avg), y=delta_fft, mode="markers", hovertext=regionLabels,
                         marker=dict(size=size), showlegend=False), row=1, col=1)


fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hovertext=hovertext, mode="markers",
                           marker=dict(size=size, color=ef_mag_avg, cmin=-max(abs(ef_mag_avg)), opacity=1, line=dict(color="grey", width=30),
                                       cmax=max(abs(ef_mag_avg)), colorscale="RdBu", reversescale=True,
                                       colorbar=dict(thickness=10)),
                           showlegend=False), row=1, col=2)

camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=-1.25, y=1.25, z=0.5))

fig.update_layout(xaxis=dict(title="Electric field input (mV/mm)"), yaxis=dict(title="delta FFT peak"),
                  template="plotly_white", scene_camera=camera)
fig.show(renderer="browser")





## Chord graph
# # Dos mitades donde se vean los de cada hemisferio
# # Todas las regiones con sus tamaños bien definidos pero connexiones solo de las que nos interesan.
#
# # Selecciona los sources que te interesen
# df_subset = df.loc[
#     (df["source"].str.contains("Thal")) | (df["source"].str.contains("Cer")) | (df["target"].str.contains("Thal")) | (
#         df["target"].str.contains("Cer"))]
# # Put all cereb and thalamus as source
# df_sources = pd.DataFrame()
# for i, row in df_subset.iterrows():
#     if ("Thal" in row.target) or ("Cer" in row.target):
#         df_sources = df_sources.append({"source": row.target, "target": row.source, "weight": row.weight},
#                                        ignore_index=True)
#     else:
#         df_sources = df_sources.append({"source": row.source, "target": row.target, "weight": row.weight},
#                                        ignore_index=True)
#
# ## Plot with holoviews
# # fig = hv.Sankey(df_sources)
# # fig.opts(edge_color=hv.dim('target'), width=1800, height=1000)
# #
# # # Name and show holoviews
# # # look for hv figure size adapts to browser window.
# # output_file("net.html")
# # show(hv.render(fig))
#
# #### Try with plotly
# # convert back labels to index
# cmap = px.colors.qualitative.Plotly
# node_colors = [cmap[-(i - (i//len(cmap)*len(cmap)))] for i in range(len(regionLabels))]
#
#
# df_sources_ids = pd.DataFrame()
# for i, row in df_sources.iterrows():
#     df_sources_ids = df_sources_ids.append(
#         {"source": regionLabels.index(row["source"]), "target": regionLabels.index(row["target"]),
#          "weight": row["weight"], "color": node_colors[regionLabels.index(row["source"])]}, ignore_index=True)
#
# fig = go.Figure(data=[go.Sankey(
#     node=dict(
#         pad=15,
#         thickness=20,
#         line=dict(color="black", width=0.5),
#         label=regionLabels,
#         color=node_colors
#     ),
#     link=dict(
#         source=df_sources_ids.source.values,  # indices correspond to labels, eg A1, A2, A1, B1, ...
#         target=df_sources_ids.target,
#         value=df_sources_ids.weight,
#         color=df_sources_ids.color
#     ))])
#
# fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
# fig.show(renderer="browser")






## Ax. Line plot with std bands [UnFinished..]
#
# # Average percentages by subject and w; Add std
# resultsAAL_avg = resultsAAL.groupby(["Subject", "w"]).mean().reset_index()
# resultsAAL_avg["std"] = resultsAAL.groupby(["Subject", "w"]).std().percent.values
#
#
# ### Define interesting range of w vals
# resultsAAL_avg_sub = resultsAAL_avg.loc[resultsAAL_avg["w"] <= 0.2]
#
# # Plot
# fig = go.Figure()
#
# subjects = sorted(list(set(resultsAAL_avg_sub.Subject)))
# for subj in subjects:
#
#     data_temp = resultsAAL_avg_sub.loc[resultsAAL_avg_sub["Subject"] == subj]
#
#     fig.add_trace(go.Scatter(name='Measurement', x=data_temp['w'], y=data_temp['percent'], mode='lines'))
#
#     fig.add_trace(go.Scatter(name='Upper Bound', x=data_temp['w'], y=data_temp['percent']+data_temp['std'],
#                               mode='lines', line=dict(width=0), showlegend=False))
#
#     fig.add_trace(go.Scatter(name='Lower Bound', x=data_temp['w'], y=data_temp['percent']-data_temp['std'],
#                               line=dict(width=0), mode='lines', fill='tonexty', showlegend=False))
#
# fig.show(renderer="browser")