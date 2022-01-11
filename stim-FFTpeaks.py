import os
import time
import subprocess

import numpy as np
import scipy.signal
import pandas as pd
import scipy.stats

from tvb.simulator.lab import *
from mne import time_frequency, filter
import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from toolbox import timeseriesPlot, FFTplot, FFTpeaks, AEC, PLV, PLI, epochingTool, paramSpace

# Choose a name for your simulation and define the empirical for SC
model_id = ".1973WilsonCowan"

# Structuring directory to organize outputs
wd=os.getcwd()
main_folder = wd+"\\"+"PSE"

if not os.path.isdir(main_folder):
    os.mkdir(main_folder)

specific_folder = main_folder + "\\""PSE" + model_id + "-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
os.mkdir(specific_folder)


simLength = 5000 # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1024 #Hz
transient = 1000 # ms to exclude from timeseries due to initial transient

# Parameters from Abeysuriya 2018. Using P=0.60 for the nodes to self oscillate at 9.75Hz.
m = models.WilsonCowan(P=np.array([0.60]), Q=np.array([0]),
                       a_e=np.array([4]), a_i=np.array([4]),
                       alpha_e=np.array([1]), alpha_i=np.array([1]),
                       b_e=np.array([1]), b_i=np.array([1]),
                       c_e=np.array([1]), c_ee=np.array([3.25]), c_ei=np.array([2.5]),
                       c_i=np.array([1]), c_ie=np.array([3.75]), c_ii=np.array([0]),
                       k_e=np.array([1]), k_i=np.array([1]),
                       r_e=np.array([0]), r_i=np.array([0]),
                       tau_e=np.array([10]), tau_i=np.array([20]),
                       theta_e=np.array([0]), theta_i=np.array([0]))


# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000/samplingFreq)
conn = connectivity.Connectivity.from_file("paupau_.zip") # ctb_folder+"CTB_connx66_"+emp_subj+".zip" |"paupau.zip"
# conn = connectivity.Connectivity.from_file(ctb_folder+"AVG_NEMOS_acc-pr.zip")
# conn = connectivity.Connectivity.from_file(ctb_folder+"AVG_NEMOS_CB.zip")
# conn = connectivity.Connectivity.from_file(ctb_folder+"AVG_NEMOS_AAL2red.zip")

## Choose coupling and speed following best working points in parameter space explorations
coup = coupling.Linear(a=np.array([0]))

mon = (monitors.Raw(),)

## Define the stimulus variable to loop over
stim_freqs = np.arange(1, 30, 1) #np.arange(1, 80, 1)
stim_amp = np.arange(0, 0.005, 0.0005) #np.arange(0, 0.05, 0.002)

data = list()

for f in stim_freqs:
    for a in stim_amp:

        tic0 = time.time()

        ## Sinusoid input
        eqn_t = equations.Sinusoid()
        eqn_t.parameters['amp'] = a
        eqn_t.parameters['frequency'] = f  # Hz
        eqn_t.parameters['onset'] = 0  # ms
        eqn_t.parameters['offset'] = 5000  # ms
        # if w != 0:
        #     eqn_t.parameters['DC'] = 0.0005 / w

        # Check the index of the region to stimulate and
        weighting = np.zeros((len(conn.weights),))
        weighting[[0]] = 1

        stimulus = patterns.StimuliRegion(temporal=eqn_t, connectivity=conn, weight=weighting)

        # Configure space and time
        stimulus.configure_space()
        stimulus.configure_time(np.arange(0, simLength, 1))

        # And take a look
        # plot_pattern(stimulus)

        # Run simulation
        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup,  integrator=integrator, monitors=mon, stimulus=stimulus)
        sim.configure()

        output = sim.run(simulation_length=simLength)
        print("Simulation time: %0.4f sec" % (time.time() - tic0,))
        # Extract data cutting initial transient
        raw_data = output[0][1][transient:, 0, :, 0].T
        raw_time = output[0][0][transient:]

        # Check initial transient and cut data
        # timeseriesPlot(raw_data, raw_time, conn.region_labels, main_folder, mode="html")
        avg_activation = np.average(raw_data, axis=1)

        # Fourier Analysis plot
        # FFTplot(raw_data, simLength-transient, regionLabels, main_folder, mode="html")
        fft_peaks = FFTpeaks(raw_data, simLength - transient)[0][:, 0]

        data.append([f, a] + list(avg_activation) + list(fft_peaks))

        print("LOOP ROUND REQUIRED %0.4f seconds.\n\n\n\n" % (time.time() - tic0,))



## GATHER RESULTS
simname = "isolated_nodes-DC"+time.strftime("m%md%dy%Y")

df = pd.DataFrame(data, columns=["stimFreq", "stimAmplitude", "tAvg_activity", "cAvg_activity",
                                 "tFFT_peak", "cFFT_peak"])
df.to_csv(specific_folder+"/PSE_"+simname+".csv", index=False)

# Load previously gathered data
# df01=pd.read_csv("C:\\Users\jesca\PycharmProjects\\brainModels\stimulationCollab\PSE\PSE.1973WilsonCowan-m02d19y2021-t12h.39m.10s-IsolatedNode01\PSE_isolated_nodes-m02d19y2021.csv")
# df01DC=pd.read_csv("C:\\Users\jesca\PycharmProjects\\brainModels\stimulationCollab\PSE\PSE.1973WilsonCowan-m02d24y2021-t00h.55m.41s-IsolatedNode01DC\PSE_isolated_nodes-DCm02d24y2021.csv")
# df02=pd.read_csv("C:\\Users\jesca\PycharmProjects\\brainModels\stimulationCollab\PSE\PSE.1973WilsonCowan-m02d22y2021-t22h.18m.50s-IsolatedNode02\PSE_isolated_nodes-m02d22y2021.csv")
# df02DC=pd.read_csv("C:\\Users\jesca\PycharmProjects\\brainModels\stimulationCollab\PSE\PSE.1973WilsonCowan-m02d23y2021-t20h.57m.35s-IsolatedNode02DC\PSE_isolated_nodes-DCm02d23y2021.csv")


import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots

def stimSpace(df, folder, auto_open="True"):
    fig = make_subplots(rows=2, cols=1, subplot_titles=(["FFT peak", "Average activity amplitude"]),
                        shared_yaxes=False, shared_xaxes=False)

    fig.add_trace(go.Heatmap(z=df.tFFT_peak, x=df.stimFreq, y=df.stimStrength,
                             colorscale='Viridis',
                             colorbar=dict(title="Hz", thickness=20, y=0.82, ypad=120),
                             zmin=np.min(df.tFFT_peak), zmax=np.max(df.tFFT_peak)), row=1, col=1)

    fig.add_trace(go.Heatmap(z=df.tAvg_activity, x=df.stimFreq, y=df.stimStrength,
                             colorscale='Inferno',
                             colorbar=dict(title="mV", thickness=20, y=0.2, ypad=120),
                             zmin=np.min(df.tAvg_activity), zmax=np.max(df.tAvg_activity)), row=2, col=1)

    # Update xaxis properties
    fig.update_xaxes(title_text="stimulation frequency", row=1, col=1)
    fig.update_xaxes(title_text="stimulation frequency", row=2, col=1)

    # Update yaxis properties
    fig.update_yaxes(title_text="stimulus amplitude", row=1, col=1)
    fig.update_yaxes(title_text="stimulus amplitude", row=2, col=1)

    fig.update_layout(title_text='Stimulation of an isolated node | default oscillation at 9.75Hz')
    pio.write_html(fig, file=folder+"/isolatedNodes-f&a.html", auto_open=auto_open)


# df1=df[df.stimStrength<=0.05]
# df=pd.read_csv("C:\\Users\jesca\PycharmProjects\\brainModels\stimulationCollab\PSE\PSE.1973WilsonCowan-m02d19y2021-t12h.39m.10s\PSE_isolated_nodes-m02d19y2021.csv")
stimSpace(df, specific_folder)


# df=df02
#
#
# import plotly.figure_factory as ff
#
# # Annotated heatmap
# data = []
# text = []
#
# for w in np.sort(np.array(list(set(df.stimStrength)))):
#     newRow = []
#     newRowText = []
#     for f in set(df.stimFreq):
#         point = df[(df["stimStrength"] == w) & (df["stimFreq"] == f)]
#         newRow.append(float(point.tFFT_peak))
#
#         if float(point.stimFreq) % float(point.tFFT_peak) == 0:
#             newRowText.append(str(int(float(point.stimFreq)/float(point.tFFT_peak))))
#
#         # elif round(float(point.stimFreq) / float(point.tFFT_peak))-float(point.stimFreq) / float(point.tFFT_peak) <= 0.065:
#         #     newRowText.append("."+str(int(round(float(point.stimFreq) / float(point.tFFT_peak)))))
#
#         else:
#             newRowText.append("")
#
#     data.append(newRow)
#     text.append(newRowText)
#
# # data1=data[::-1]
# # text1=text[::-1]
# fig = ff.create_annotated_heatmap(data, annotation_text=text, colorscale="Viridis", showscale=True)
# fig.show(renderer="browser")