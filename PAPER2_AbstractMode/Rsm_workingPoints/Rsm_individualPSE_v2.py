
import pandas as pd
import time

import plotly.graph_objects as go  # for gexplore_data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px

# Define PSE folder
spk_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\output_SPK\\"
fig_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\Figures\\"



#      FIGURE 1sm        ################
# Load SPK data
spk_df = pd.read_csv(spk_folder + "cingulum_bundle_working_point.txt", delimiter="\t", index_col=0)
spk_df_avg = spk_df.groupby(["subject", "coup"]).mean().reset_index()

# Working points
wp_df = pd.read_csv(spk_folder + "cingulum_bundle_working_point_solution.txt", delimiter="\t", index_col=0)


subjects = sorted(set(spk_df.subject))


# Figure
fig = make_subplots(rows=10, cols=1, x_title="Coupling factor (g)",
                    specs=[[{"secondary_y": True}]] * 10,
                    shared_xaxes=True, row_titles=["Subj" + str(i+1) for i, subj in enumerate(subjects)])
# Colours
c3, c1 = px.colors.qualitative.Set1[1], px.colors.qualitative.Set1[8]
opacity1, opacity2 = 0.9, 0.35
x_rowtitle = 1.1

for ii, subj in enumerate(subjects):

    sl = True if ii == 0 else False

    spk_df_subj = spk_df_avg.loc[spk_df_avg["subject"] == subj]

    ##       SPK     ####
    # Add r PLV
    fig.add_trace(
        go.Scatter(x=spk_df_subj.coup, y=spk_df_subj.rPLV, mode="lines", name="$r_{PLV}$", legendgroup="rPLV",
                   line=dict(width=2, color=c3), opacity=opacity1, showlegend=sl), row=ii+1, col=1)

    # Add Freq
    fig.add_trace(
        go.Scatter(x=spk_df_subj.coup, y=spk_df_subj.fpeak, mode="lines", name="Frequency", legendgroup="Frequency",
                   line=dict(width=3, color=c1), opacity=opacity2, showlegend=sl), secondary_y=True, row=ii+1, col=1)

    # Add working point vertical line
    wp = wp_df.loc[wp_df["subject"]==subj, "coup"].values[0]

    fig.add_vline(x=wp, line=dict(dash="dash"), opacity=0.5, row=ii+1, col=1)

    fig["layout"]["annotations"][ii]["x"] = x_rowtitle

## FORMAT AXIS
yaxis_total, yaxis_inloop = 21, 2

# Third y axis: rPLV (Spk)
id_yaxis=1
for i in range(id_yaxis, yaxis_total, yaxis_inloop):
    fig["layout"]["yaxis" + str(i)]["range"] = [-0.15, 0.8]
    fig["layout"]["yaxis" + str(i)]["color"] = c3
    if i == list(range(id_yaxis, yaxis_total, yaxis_inloop))[5]:
        fig["layout"]["yaxis" + str(i)]["title"] = "$r_{PLV}$"

# Fourth y axis: Frequency (spk)
id_yaxis=2
for i in range(id_yaxis, yaxis_total, yaxis_inloop):
    fig["layout"]["yaxis" + str(i)]["range"] = [7, 12]
    fig["layout"]["yaxis" + str(i)]["color"] = c1
    if i == list(range(id_yaxis, yaxis_total, yaxis_inloop))[5]:
        fig["layout"]["yaxis" + str(i)]["title"] = "Frequency (Hz)"


fig.update_layout(template="plotly_white", width=500, height=900, font_family="Arial",
                  legend=dict(orientation="h", xanchor="right", x=0.95, yanchor="bottom", y=1.04))

pio.write_html(fig, file=fig_folder + "/Rsm2_WorkingPoints_allSubjs-lineSpaces.html", auto_open=True, include_mathjax="cdn")
pio.write_image(fig, file=fig_folder + "/Rsm2_WorkingPoints_allSubjs-lineSpaces.svg", engine="kaleido")

