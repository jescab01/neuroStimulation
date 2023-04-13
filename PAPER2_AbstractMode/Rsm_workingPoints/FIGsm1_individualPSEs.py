
import pandas as pd
import time

import plotly.graph_objects as go  # for gexplore_data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px

# Define PSE folder
folder = 'E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\R2_CingulumBundle\R2a_workingPoints\\'

simulations_tag = "PSEmpi_JR_v2-m03d27y2023-t12h.08m.05s"
nmm_folder = folder + simulations_tag + '\\'

df = pd.read_csv(nmm_folder + "/results.csv")

# Average out repetitions
df_avg = df.groupby(["subject", "coup", "mode"]).mean().reset_index()

## NMM working points
df_avg_sub = df_avg.loc[df_avg["mode"] == "sigma0.11"]
working_points = df_avg_sub.loc[df_avg_sub.groupby('subject')['rPLV'].idxmax()]

##      Auxiliar figure to compare sigmas     ################
cmap_s2, opacity = px.colors.qualitative.Set2, 0.8

subjects, modes = sorted(set(df_avg.subject)), sorted(set(df_avg["mode"]))

fig = make_subplots(rows=10, cols=4, horizontal_spacing=0.075, x_title="Coupling factor (g)",
                    shared_xaxes=True, row_titles=["Subj" + str(i+1) for i, subj in enumerate(subjects)])

for ii, subj in enumerate(subjects):
    for c, mode in enumerate(modes):

        df_avg_subj = df_avg.loc[(df_avg["mode"] == mode) & (df_avg["subject"] == subj)]
        sl = True if ii == 0 else False

        # Add r PLV
        fig.add_trace(
            go.Scatter(x=df_avg_subj.coup, y=df_avg_subj.rPLV, mode="lines", name=mode, legendgroup=mode,
                       line=dict(width=c+1, color=cmap_s2[c]), opacity=opacity, showlegend=sl), row=1+ii, col=1)

        # Add Freq
        fig.add_trace(
            go.Scatter(x=df_avg_subj.coup, y=df_avg_subj.fpeak, mode="lines", legendgroup=mode,
                       line=dict(width=c+1, color=cmap_s2[c]), opacity=opacity, showlegend=False), row=ii+1, col=2)

        # Add Power
        fig.add_trace(
            go.Scatter(x=df_avg_subj.coup, y=df_avg_subj.amplitude_fpeak, mode="lines", legendgroup=mode,
                       line=dict(width=c+1, color=cmap_s2[c]), opacity=opacity, showlegend=False), row=ii+1, col=3)


        # Add plv mean
        fig.add_trace(
            go.Scatter(x=df_avg_subj.coup, y=df_avg_subj.plv_mean, mode="lines", legendgroup=mode,
                       line=dict(width=c+1, color=cmap_s2[c]), opacity=opacity, showlegend=False), row=ii+1, col=4)


##  Format Axis
yaxis_total, yaxis_inloop = 41, 4
# First y axis: rPLV
id_yaxis = 1
for i in range(id_yaxis, yaxis_total, yaxis_inloop):
    fig["layout"]["yaxis" + str(i)]["range"] = [-0.05, 0.55]
    if i == list(range(id_yaxis, yaxis_total, yaxis_inloop))[5]:
        fig["layout"]["yaxis" + str(i)]["title"] = "$r_{PLV}$"

# First y axis: Frequency
id_yaxis = 2
for i in range(id_yaxis, yaxis_total, yaxis_inloop):
    fig["layout"]["yaxis" + str(i)]["range"] = [6, 12]
    if i == list(range(id_yaxis, yaxis_total, yaxis_inloop))[5]:
        fig["layout"]["yaxis" + str(i)]["title"] = "Frequency (Hz)"

# First y axis: Power
id_yaxis = 3
for i in range(id_yaxis, yaxis_total, yaxis_inloop):
    fig["layout"]["yaxis" + str(i)]["range"] = [df_avg.amplitude_fpeak.min(), df_avg.amplitude_fpeak.max()]
    if i == list(range(id_yaxis, yaxis_total, yaxis_inloop))[5]:
        fig["layout"]["yaxis" + str(i)]["title"] = "Power (dB)"

# First y axis: mean PLV
id_yaxis = 4
for i in range(id_yaxis, yaxis_total, yaxis_inloop):
    fig["layout"]["yaxis" + str(i)]["range"] = [df_avg.plv_mean.min(), df_avg.plv_mean.max()]
    if i == list(range(id_yaxis, yaxis_total, yaxis_inloop))[5]:
        fig["layout"]["yaxis" + str(i)]["title"] = "mean PLV"

fig.update_layout(template="plotly_white", width=1200, height=900, font_family="Arial",
                   legend=dict(orientation="h", xanchor="right", x=0.95, yanchor="bottom", y=1.04))

pio.write_html(fig, nmm_folder+"/PSE_subjects_xsigmas_xmeasures.html", auto_open=True, include_mathjax="cdn")





#      FIGURE 1sm        ################
subjects = sorted(set(df_avg.subject))

# Prepare NMM data
df_avg_sub = df_avg.loc[(df_avg["mode"] == "sigma0.11")]

# Load SPK data
spk_df = pd.read_csv(folder + "cingulum_bundle_working_point_SPK@27-03-2022.txt", delimiter="\t", index_col=0)
spk_df_avg = spk_df.groupby(["subject", "coup"]).mean().reset_index()


# Figure
fig = make_subplots(rows=10, cols=2, horizontal_spacing=0.15, x_title="Coupling factor (g)",
                    specs=[[{"secondary_y": True}, {"secondary_y": True}]] * 10,
                    shared_xaxes=True, row_titles=["Subj" + str(i+1) for i, subj in enumerate(subjects)])
# Colours
c3, c1 = px.colors.qualitative.Set1[1], px.colors.qualitative.Set1[8]
opacity1, opacity2 = 0.9, 0.35
x_rowtitle=1.015

for ii, subj in enumerate(subjects):

    df_avg_subj = df_avg.loc[(df_avg["mode"] == "sigma0.11") & (df_avg["subject"]==subj)]
    sl = True if ii == 0 else False

    spk_df_subj = spk_df_avg.loc[spk_df_avg["subject"]==subj]

    ##       NMM      ####
    # Add r PLV
    fig.add_trace(
        go.Scatter(x=df_avg_subj.coup, y=df_avg_subj.rPLV, mode="lines", name="rPLV - NMM", legendgroup="rPLV - NMM",
                   line=dict(width=2, color=c3), opacity=opacity1, showlegend=sl), row=ii+1, col=1)
    # Add Freq
    fig.add_trace(
        go.Scatter(x=df_avg_subj.coup, y=df_avg_subj.fpeak, mode="lines", name="Frequency - NMM", legendgroup="Frequency - NMM",
                   line=dict(width=3, color=c1), opacity=opacity2, showlegend=sl), secondary_y=True, row=ii+1, col=1)

    ##       SPK     ####
    # Add r PLV
    fig.add_trace(
        go.Scatter(x=spk_df_subj.coup, y=spk_df_subj.rPLV, mode="lines", name="rPLV - SPK", legendgroup="rPLV - SPK",
                   line=dict(width=2, color=c3), opacity=opacity1, showlegend=sl), row=ii+1, col=2)

    # Add Freq
    fig.add_trace(
        go.Scatter(x=spk_df_subj.coup, y=spk_df_subj.fpeak, mode="lines", name="Frequency - SPK", legendgroup="Frequency - SPK",
                   line=dict(width=3, color=c1), opacity=opacity2, showlegend=sl), secondary_y=True, row=ii+1, col=2)

    fig["layout"]["annotations"][ii]["x"] = x_rowtitle

## FORMAT AXIS
yaxis_total, yaxis_inloop = 41, 4
# First y axis: rPLV
id_yaxis = 1
for i in range(id_yaxis, yaxis_total, yaxis_inloop):
    fig["layout"]["yaxis" + str(i)]["range"] = [-0.05, 0.65]
    fig["layout"]["yaxis" + str(i)]["color"] = c3
    if i == list(range(id_yaxis, yaxis_total, yaxis_inloop))[5]:
        fig["layout"]["yaxis" + str(i)]["title"] = "$r_{PLV}$"

# Second y axis: Frequency
id_yaxis=2
for i in range(id_yaxis, yaxis_total, yaxis_inloop):
    fig["layout"]["yaxis" + str(i)]["range"] = [2, 12]
    fig["layout"]["yaxis" + str(i)]["color"] = c1
    if i == list(range(id_yaxis, yaxis_total, yaxis_inloop))[5]:
        fig["layout"]["yaxis" + str(i)]["title"] = "Frequency (Hz)"

# Third y axis: rPLV (Spk)
id_yaxis=3
for i in range(id_yaxis, yaxis_total, yaxis_inloop):
    fig["layout"]["yaxis" + str(i)]["range"] = [-0.05, 0.65]
    fig["layout"]["yaxis" + str(i)]["color"] = c3
    if i == list(range(id_yaxis, yaxis_total, yaxis_inloop))[5]:
        fig["layout"]["yaxis" + str(i)]["title"] = "$r_{PLV}$"

# Fourth y axis: Frequency (spk)
id_yaxis=4
for i in range(id_yaxis, yaxis_total, yaxis_inloop):
    fig["layout"]["yaxis" + str(i)]["range"] = [2, 12]
    fig["layout"]["yaxis" + str(i)]["color"] = c1
    if i == list(range(id_yaxis, yaxis_total, yaxis_inloop))[5]:
        fig["layout"]["yaxis" + str(i)]["title"] = "Frequency (Hz)"

fig.update_layout(template="plotly_white", width=900, height=900, font_family="Arial",
                  legend=dict(orientation="h", xanchor="right", x=0.95, yanchor="bottom", y=1.04))

pio.write_image(fig, file=folder + "/PAPER-sm1_SUBJECTS-lineSpaces.svg", engine="kaleido")
pio.write_html(fig, file=folder + "/PAPER-sm1_SUBJECTS-lineSpaces.html", auto_open=True, include_mathjax="cdn")

