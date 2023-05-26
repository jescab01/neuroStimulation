
import pandas as pd
import numpy as np
from scipy.stats import rv_histogram

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

fig_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\Figures\\"
spk_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\\output_SPK\\"

sim_tag = "TheoreticalDists\\"
## DISTRIBUTIONS indexes
# 0: bimodal simétrica; 1-bimodal simétrica shifted; 2-bimodal asimétrica positiva;
# 3-bimodal asimétrica positiva shifted; 4-gaussiana; 5-gaussiana shifted
dists_vals = pd.read_csv(spk_folder + sim_tag + "theoretical_distributions.txt", delimiter="\t", index_col=0)
df = pd.read_csv(spk_folder + sim_tag + "one_node_tacs_theoretical_densities.txt", delimiter="\t", index_col=0)

df_spk = df.groupby(["histogram", "weight", "fex"]).mean().reset_index()


## SKETCH the distributions. by Jaime Sánchez-Claros
def bimodal(x, a1, mu1, sigma1, a2, mu2, sigma2):
    y = a1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)) + a2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))
    dx = np.diff(x)[0]
    return y/np.sum(y*dx)

x = np.linspace(-0.25,0.25,10001)
sigma = 0.05
mu = 0.075
shift = 0.050

# bimodal simetrica
y1  = bimodal(x, 1.0, -mu, sigma, 1.0, mu,  sigma)
y1s = bimodal(x, 1.0, -mu+shift, sigma, 1.0, mu+shift,  sigma)

# bimodal asimetrica
y2  = bimodal(x, 1.0, -mu-0.0415,       sigma, 3.5, mu-0.0415,  sigma)
y2s = bimodal(x, 1.0, -mu-0.0415+shift, sigma, 3.5, mu-0.0415+shift,  sigma)

# gaussian
y3  = bimodal(x, 1.0, 0.0, sigma, 1.0, 0.0, sigma)
y3s = bimodal(x, 1.0, shift, sigma, 1.0, shift, sigma)


g = []
dx = np.diff(x)[0]
xn = np.array(list(x-dx/2)+[x[-1]+dx/2])
np.random.seed(1993)
for i,y in enumerate([y1,y1s,y2,y2s,y3,y3s]):
    rv = rv_histogram((y,xn))
    g.append(rv.rvs(size=80))


freq_min, freq_max, freq_colorscale = df_spk["fpeak"].min(), df_spk["fpeak"].max(), px.colors.diverging.balance
pow_min, pow_max, pow_colorscale = df_spk["amplitude_fpeak"].min(), df_spk["amplitude_fpeak"].max(), px.colors.sequential.Sunsetdark

freq_colorbar = dict(title="Hz", thickness=10, x=1)
pow_colorbar = dict(title="dB", thickness=10, x=1.1)

cmap_p = px.colors.qualitative.Pastel2
cmap_s = px.colors.qualitative.Set2
cmap_d = px.colors.qualitative.Dark2

fig = make_subplots(rows=3, cols=5, column_titles=["Frequency", "Power", "", "Frequency", "Power"],
                    specs=[[{},{},{"secondary_y":True},{},{}], [{},{},{"secondary_y":True},{},{}],
                           [{},{},{"secondary_y":True},{},{}]])

for i, dist in enumerate([y1, y1s, y2, y2s, y3, y3s]):

    color = 0 if i % 2 == 0 else cmap_p[1]
    # Add histograms and theoretical distribution to the central column: blue left (centered), red right (shifted)
    fig.add_trace(go.Histogram(x=dists_vals.iloc[:, i].values, opacity=0.5, name="Dist " + str(i), marker=dict(color=cmap_p[i%2-1]), histnorm="probability", showlegend=True), row=i//2+1, col=3)
    fig.add_trace(go.Scatter(x=x, y=dist/sum(dist), line=dict(color=cmap_s[i%2-1]), name="Theo " + str(i), showlegend=True), secondary_y=True, row=i//2+1, col=3)
    if i%2==0:
        fig.add_vline(x=0, opacity=0.6, line=dict(color="black", width=1),  row=i//2+1, col=3)
    fig.add_vline(x=np.average(g[i]), opacity=0.6, line=dict(dash="dash", color=cmap_d[i%2-1]),  row=i//2+1, col=3)

    # Add arnold tongues
    sub = df_spk.loc[(df_spk["histogram"] == i)]

    s_col = 1 if i%2==0 else 4
    row = i*2//4
    ss = True if i%2 == 0 else None
    fig.add_trace(go.Heatmap(x=sub.fex, y=sub.weight, z=sub.fpeak, colorbar=dict(title="Hz", thickness=10, x=0.975), showscale=ss, colorscale=freq_colorscale, zmin=freq_min, zmax=freq_max), row=row+1, col=s_col)
    fig.add_trace(go.Heatmap(x=sub.fex, y=sub.weight, z=sub.amplitude_fpeak, colorbar=dict(title="dB", thickness=10, x=1.025), showscale=ss, colorscale=pow_colorscale, zmin=pow_min, zmax=pow_max), row=row+1, col=s_col+1)

fig.update_layout(barmode="overlay", template="plotly_white", height=550, width=1000, legend=dict(x=1.1),
                  yaxis7=dict(title="Stimulation intensity", anchor="free", title_standoff=0), yaxis11=dict(title="Stimulation intensity", title_standoff=0),
                  yaxis4=dict(visible=False),yaxis9=dict(title="Probability density", title_standoff=0), yaxis10=dict(visible=False),yaxis16=dict(visible=False),
                  xaxis11=dict(title="."+" "*40+"Stimulation Frequency (Hz)", anchor="free"), xaxis13=dict(title="EF component (V/m)"),
                  xaxis15=dict(title="Stimulation Frequency (Hz)"+" "*40+"."))

fig.add_annotation(text="Simulations with <br>zero-centered distributions", x=0.015, y=1.2, xref="paper", yref="paper", showarrow=False,
                   font=dict(size=16, color=cmap_d[-1]))
fig.add_annotation(text="Simulations with <br>right-shifted distributions", x=0.92, y=1.2, xref="paper", yref="paper", showarrow=False,
                   font=dict(size=16, color=cmap_s[0]))

pio.write_html(fig, file=fig_folder + "R2_SPK_theoreticalDistributions_wLegend.html", auto_open=True)

fig.update_traces(showlegend=False)
pio.write_html(fig, file=fig_folder + "R2_SPK_theoreticalDistributions.html", auto_open=True)
pio.write_image(fig, fig_folder + "R2_SPK_theoreticalDistributions.svg")






# fig = make_subplots(rows=3, cols=5, column_titles=["Frequency", "Power", "", "Frequency", "Power"],
#                     specs=[[{},{},{"secondary_y":True},{},{}], [{},{},{"secondary_y":True},{},{}],
#                            [{},{},{"secondary_y":True},{},{}]])
#
# for i, dist in enumerate([y1, y1s, y2, y2s, y3, y3s]):
#
#     color = 0 if i % 2 == 0 else cmap_p[1]
#     # Add histograms and theoretical distribution to the central column: blue left (centered), red right (shifted)
#     fig.add_trace(go.Histogram(x=dists_vals.iloc[:, i].values, opacity=0.5, name="Dist " + str(i), marker=dict(color=cmap_p[i%2-1]), histnorm="probability", showlegend=True), row=i//2+1, col=3)
#     fig.add_trace(go.Scatter(x=x, y=dist/sum(dist), line=dict(color=cmap_s[i%2-1]), name="Theo " + str(i), showlegend=True), secondary_y=True, row=i//2+1, col=3)
#     if i%2==0:
#         fig.add_vline(x=0, opacity=0.6, line=dict(color="black", width=1),  row=i//2+1, col=3)
#     fig.add_vline(x=np.average(g[i]), opacity=0.6, line=dict(dash="dash", color=cmap_d[i%2-1]),  row=i//2+1, col=3)
#
#     # Add arnold tongues
#     sub = df_spk.loc[(df_spk["histogram"] == i)]
#
#     s_col = 1 if i%2==0 else 4
#     row = i*2//4
#     ss = True if i%2 == 0 else None
#     fig.add_trace(go.Heatmap(x=sub.fex, y=sub.weight, z=sub.fpeak/sub.fex, showscale=ss), row=row+1, col=s_col)
#     fig.add_trace(go.Heatmap(x=sub.fex, y=sub.weight, z=sub.amplitude_fpeak, colorbar=dict(title="dB", thickness=10, x=1.025), showscale=ss, colorscale=pow_colorscale, zmin=pow_min, zmax=pow_max), row=row+1, col=s_col+1)
#
# # fig.update_layout(barmode="overlay", template="plotly_white", height=550, width=1000, legend=dict(x=1.1),
# #                   yaxis7=dict(title="Stimulation weight", anchor="free", title_standoff=0), yaxis11=dict(title="Stimulation weight", title_standoff=0),
# #                   yaxis4=dict(visible=False),yaxis9=dict(title="Probability density", title_standoff=0), yaxis10=dict(visible=False),yaxis16=dict(visible=False),
# #                   xaxis11=dict(title="."+" "*40+"Stimulation Frequency (Hz)", anchor="free"), xaxis13=dict(title="EF component (V/m)"),
# #                   xaxis15=dict(title="Stimulation Frequency (Hz)"+" "*40+"."))
#
# fig.add_annotation(text="Simulations with <br>zero-centered distributions", x=0.015, y=1.2, xref="paper", yref="paper", showarrow=False,
#                    font=dict(size=16, color=cmap_d[-1]))
# fig.add_annotation(text="Simulations with <br>right-shifted distributions", x=0.92, y=1.2, xref="paper", yref="paper", showarrow=False,
#                    font=dict(size=16, color=cmap_s[0]))
#
# pio.write_html(fig, file=fig_folder + "R0_SPK_theoreticalDistributions_wLegend.html", auto_open=True)
#
# fig.update_traces(showlegend=False)
# pio.write_html(fig, file=fig_folder + "R0_SPK_theoreticalDistributions.html", auto_open=True)
# pio.write_image(fig, fig_folder + "R0_SPK_theoreticalDistributions.svg")
#
