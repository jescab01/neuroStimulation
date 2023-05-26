
import numpy as np
import os
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

spk_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\output_SPK\\"
fig_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\Figures\\"

# Load data
simtag = "stimulation_OzCz_densities_nodes_precuneus.txt"
stimPrec = pd.read_csv(spk_folder + simtag, sep="\t") # only precuneus stimulated

simtag = "stimulation_OzCz_densities_nodes_coupled.txt"
stimAll = pd.read_csv(spk_folder + simtag, sep="\t") # all nodes stimulated


stimPrec_avg = stimPrec[stimPrec["node"]=="Precuneus_R"].groupby(["w", "fex"])["fpeak", "amp_fpeak"].mean().reset_index()
stimAll_avg = stimAll[stimAll["node"]=="Precuneus_R"].groupby(["w", "fex"])["fpeak", "amp_fpeak"].mean().reset_index()


min_fpeak, max_fpeak = np.min([np.min(stimPrec_avg.fpeak), np.min(stimAll_avg.fpeak)]), np.max([np.max(stimPrec_avg.fpeak), np.max(stimAll_avg.fpeak)])
min_amp_fpeak, max_amp_fpeak = np.min([np.min(stimPrec_avg.amp_fpeak), np.min(stimAll_avg.amp_fpeak)]), np.max([np.max(stimPrec_avg.amp_fpeak), np.max(stimAll_avg.amp_fpeak)])


## Plotting

fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
                    column_titles=["Stimulating Precuneus", "Stimulating all nodes"],
                    row_titles=["Precuneus'<br>Frequency peak", "Precuneus'<br>Peak power"],
                    x_title="Stimulation Frequency (Hz)", y_title="Stimulation intensity")# (\u039b)")

fig.add_trace(go.Heatmap(x=stimPrec_avg.fex, y=stimPrec_avg.w, z=stimPrec_avg.fpeak, colorscale=px.colors.diverging.balance,
                         zmin=min_fpeak, zmax=max_fpeak, colorbar=dict(len=0.45, x=1.08, y=0.77, thickness=10, title="Hz")), row=1, col=1)
fig.add_trace(go.Heatmap(x=stimPrec_avg.fex, y=stimPrec_avg.w, z=stimPrec_avg.amp_fpeak, colorscale=px.colors.sequential.Sunsetdark,
                         zmin=min_amp_fpeak, zmax=max_amp_fpeak, colorbar=dict(len=0.45, x=1.08, y=0.23, thickness=10, title="dB")), row=2, col=1)

fig.add_trace(go.Heatmap(x=stimAll_avg.fex, y=stimAll_avg.w, z=stimAll_avg.fpeak, colorscale=px.colors.diverging.balance,
                         zmin=min_fpeak, zmax=max_fpeak, showscale=False), row=1, col=2)
fig.add_trace(go.Heatmap(x=stimAll_avg.fex, y=stimAll_avg.w, z=stimAll_avg.amp_fpeak, colorscale=px.colors.sequential.Sunsetdark,
                         zmin=min_amp_fpeak, zmax=max_amp_fpeak, showscale=False), row=2, col=2)

fig.update_layout(template="plotly_white", width=600, height=600)

pio.write_html(fig, fig_folder + "Rsm5_StimulationCommunication.html", auto_open=True)
pio.write_image(fig, fig_folder + "Rsm5_StimulationCommunication.svg")


