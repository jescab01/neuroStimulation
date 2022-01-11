import pandas as pd
import os
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px

# for i in [35,49,50,58,59,64,65,71,75,77]:

# emp_subj="NEMOS_035"  # "NEMOS_0"+str(i)
#
# specific_folder = "D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\stimulationCollab\NEMOS_fitW\PSE\PSEp_fittingW-" + emp_subj + ".1995JansenRitm06d29y2021\\"

def collectData(specific_folder):

    files = os.listspecific_folder(specific_folder)

    df_fft = pd.DataFrame()
    df_ar = pd.DataFrame()

    for file in files:
        if 'FFT' in file:
            df_fft = df_fft.append(pd.read_csv(specific_folder + file), ignore_index=True)

        else:
            df_ar = df_ar.append(pd.read_csv(specific_folder + file), ignore_index=True)

    df_fft.to_csv(specific_folder + '\\' + 'FFT_fullDF.csv')
    df_ar.to_csv(specific_folder + '\\' + 'alphaRise_fullDF.csv')

    return df_fft, df_ar


def boxPlot(df_ar, n_simulations):

    # calculate percentages
    df_ar_avg = df_ar.groupby('w').mean()

    df_ar["percent"] = [
        ((df_ar.peak_module[i] - df_ar_avg.peak_module[0]) / df_ar_avg.peak_module[0]) * 100 for i in
        range(len(df_ar))]

    df_ar_avg = df_ar.groupby('w').mean()
    df_ar_avg["sd"] = df_ar.groupby('w')[['w', 'peak_module']].std()


    fig = px.box(df_ar, x="w", y="peak_module",
                 title="Alpha peak module rise @ParietalComplex<br>(%i simulations | %s AAL2red)" % (n_simulations, emp_subj),
                 labels={  # replaces default labels by column name
                     "w": "Weight", "peak_module": "Alpha peak module"},
                 template="plotly")
    pio.write_html(fig, file=specific_folder + emp_subj + "AAL_alphaRise_modules_" + str(n_simulations) + "sim.html",
                   auto_open=False)

    fig = px.box(df_ar, x="w", y="percent",
                 title="Alpha peak module rise @ParietalComplex<br>(%i simulations | %s AAL2red)" % (n_simulations, emp_subj),
                 labels={  # replaces default labels by column name
                     "w": "Weight", "percent": "Percentage of alpha peak rise"},
                 template="plotly")
    pio.write_html(fig, file=specific_folder + emp_subj + "AAL_alphaRise_percent_" + str(n_simulations) + "sim.html",
                   auto_open=True)




### PLOT 3d surface of FFT curves by w (averaged by reps)
# def FFT3d():
#     fig = px.line_3d(df_fft_temp1, x="w", y="freq", z="fft_module", color='rep')
#     fig.show(renderer="browser")



# df_fft_temp=df_fft.loc[(df_fft["regLab"]=="Parietal_Sup_L") &  (df_fft["freq"]<15)]
#
# df_fft_temp1=df_fft_temp.loc[(df_fft["rep"]==6)]
#
#
# df_fft_temp1=df_fft_temp[["rep", "fft_module", "freq"]]
#
# df_fft_temp1=df_fft_temp1.pivot_table(index="rep", columns="freq", values="fft_module").reset_index()
# freqs=df_fft_temp1.columns.to_numpy()[1:] # Freqs
#
# df_fft_temp2=df_fft_temp1.to_numpy() # Matrix
#
# fig=go.Figure(data=go.Surface(z=df_fft_temp2[:, 1:], x=freqs, y=df_fft_temp2[:, 0]))
# fig.show(renderer="browser")
#
# fig = make_subplots(rows=1, cols=3, subplot_titles=("FFT peak", "rPLV (sim-emp)", "PLE - Phase Lag Entropy"),
#                     specs=[[{}, {}, {}]], shared_yaxes=True, shared_xaxes=True,
#                     x_title="Conduction speed (m/s)", y_title="Coupling factor")
#
# fig.add_trace(go.Heatmap(z=df_fft.mS_peak, x=df_fft.speed, y=df_fft.G, colorscale='Viridis',
#                          reversescale=False, showscale=True, colorbar=dict(x=0.30, thickness=7)), row=1, col=1)
# fig.add_trace(go.Heatmap(z=df_fc.Alpha, x=df_fc.speed, y=df_fc.G, colorscale='RdBu', reversescale=True, zmin=-0.5, zmax=0.5,
#                          showscale=True, colorbar=dict(x=0.66, thickness=7)), row=1, col=2)
# fig.add_trace(go.Heatmap(z=df_ple.Alpha, x=df_ple.speed, y=df_ple.G, colorscale='Plasma', colorbar=dict(thickness=7),
#                          reversescale=False, showscale=True), row=1, col=3)
#
#
# fig.update_layout(
#     title_text='Mix of measures in '+emp_subj)
# pio.write_html(fig, file=specific_folder + "mix_paramSpace-g&s.html", auto_open=True)
