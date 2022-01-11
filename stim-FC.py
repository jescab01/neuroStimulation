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
emp_subj = "subj04"

# Structuring directory to organize outputs
wd=os.getcwd()
main_folder = wd+"\\"+"PSE"

if not os.path.isdir(main_folder):
    os.mkdir(main_folder)

specific_folder = main_folder + "\\""PSE" + model_id + "-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
os.mkdir(specific_folder)

ctb_folder= wd + "\\CTB_data\\output\\"




simLength = 5000 # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1024 #Hz
transient = 1000 # ms to exclude from timeseries due to initial transient

# Parameters from Abeysuriya 2018. Good working point at s=13.5; g=0.375
m = models.WilsonCowan(P=np.array([0.31]), Q=np.array([0]),
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

conn = connectivity.Connectivity.from_file(ctb_folder+"CTB_connx66_"+emp_subj+".zip") # ctb_folder+"CTB_connx66_"+emp_subj+".zip" |"paupau.zip"
conn.weights = conn.scaled_weights(mode="tract")

## Choose coupling and speed following best working points in parameter space explorations
coup = coupling.Linear(a=np.array([0.375]))
conn.speed = np.array([13.5])

mon = (monitors.Raw(),)

## Define the stimulus variable to loop over
stim_freqs = np.arange(2,20,2)
stim_weights = np.arange(0,0.5,0.2)

plv_target = list()
plv_control = list()
for f in stim_freqs:
    for w in stim_weights:

        tic0 = time.time()

        ## Sinusoid input
        eqn_t = equations.Sinusoid()
        eqn_t.parameters['amp'] = 0.2
        eqn_t.parameters['frequency'] = f  # Hz
        eqn_t.parameters['onset'] = 0  # ms
        eqn_t.parameters['offset'] = 5000  # ms

        # Check the index of the region to stimulate and
        regionLabels = list(conn.region_labels)
        region_id=regionLabels.index("Left-cuneus")
        control_id=regionLabels.index("Right-bankssts")
        weighting = np.zeros((len(conn.weights),))
        weighting[[region_id]] = w

        stimulus = patterns.StimuliRegion(
            temporal=eqn_t,
            connectivity=conn,
            weight=weighting)

        # Configure space and time
        stimulus.configure_space()
        stimulus.configure_time(np.arange(0, simLength, 1))

        # And take a look
        # plot_pattern(stimulus)

        # Run simulation
        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup,  integrator=integrator, monitors=mon, stimulus=stimulus)
        sim.configure()



        # for i in range(5):
        output = sim.run(simulation_length=simLength)
        print("Simulation time: %0.2f sec" % (time.time() - tic0,))
        # Extract data cutting initial transient
        raw_data = output[0][1][transient:, 0, :, 0].T
        raw_time = output[0][0][transient:]


        # Check initial transient and cut data
        # timeseriesPlot(raw_data, raw_time, regionLabels, main_folder, mode="html")

        # Fourier Analysis plot
        # FFTplot(raw_data, simLength, regionLabels, main_folder, mode="html")

        # fft_peaks = FFTpeaks(data, simLength - transient)[0][:,0]

        newRow_t = [regionLabels[region_id], f, w]
        newRow_c = [regionLabels[control_id], f, w]
        bands = [["3-alfa"], [(8, 12)]]
        ## [["1-delta", "2-theta", "3-alfa", "4-beta", "5-gamma"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]

        for b in range(len(bands[0])):

            newRow_t.append(bands[0][b])
            newRow_c.append(bands[0][b])
            (lowcut, highcut) = bands[1][b]

            # Band-pass filtering
            filterSignals = filter.filter_data(raw_data, samplingFreq, lowcut, highcut)

            # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
            efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals")

            # Obtain Analytical signal
            efPhase = list()
            efEnvelope = list()
            for i in range(len(efSignals)):
                analyticalSignal = scipy.signal.hilbert(efSignals[i])
                # Get instantaneous phase and amplitude envelope by channel
                efPhase.append(np.unwrap(np.angle(analyticalSignal)))
                #efEnvelope.append(np.abs(analyticalSignal))

            #CONNECTIVITY MEASURES
            ## PLV
            plv = PLV(efPhase)
            # fname = ctb_folder+model_id+"\\"+bands[0][b]+"plv.txt"
            # np.savetxt(fname, plv)
            #
            # ## AEC
            # aec = AEC(efEnvelope)
            # fname = ctb_folder+model_id+"\\"+bands[0][b]+"corramp.txt"
            # np.savetxt(fname, aec)
            #
            # ## PLI
            # pli = PLI(efPhase)
            # fname = ctb_folder+model_id+"\\"+bands[0][b]+"pli.txt"
            # np.savetxt(fname, pli)
            plv_target.append(newRow_t+list(plv[region_id]))
            plv_control.append(newRow_c+list(plv[control_id]))

        print("LOOP ROUND REQUIRED %0.3f seconds.\n\n\n\n" % (time.time() - tic0,))


## GATHER RESULTS
simname = emp_subj+"-"+time.strftime("m%md%dy%Y")
# Working on FFT peak results
df_t = pd.DataFrame(plv_target, columns=["targetRegion", "stimFreq", "stimWeight", "band"] + regionLabels)
df_c = pd.DataFrame(plv_control, columns=["controlRegion", "stimFreq", "stimWeight", "band"] + regionLabels)

df_t.to_csv(specific_folder+"/PSE_stim_target"+simname+".csv", index=False)
df_c.to_csv(specific_folder+"/PSE_stim_control"+simname+".csv", index=False)

# Load previously gathered data
# df1=pd.read_csv("expResults/paramSpace_FFTpeaks-2d-subj4-8s-sim.csv")

import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots


# define number of rows: How many different stimulation weights do you want to plot?
w_subset = stim_weights

# Subset the dataframe deleting the regions that have no connection with target area
structure_t=conn.weights[region_id]
structure_c=conn.weights[control_id]


def stimulationFC(df, structure, stimulationWeights, regionLabels, folder, region=regionLabels[region_id],t_c="target", auto_open="True"):

    sp_titles=["Stimulation Weight = " + str(ws) for ws in stimulationWeights]

    fig = make_subplots(rows=len(stimulationWeights)+1, cols=1, subplot_titles=(sp_titles+["Structural Connectivity"]),
                        shared_yaxes=True, shared_xaxes=True, y_title="Stimulation Frequency")

    for i, ws in enumerate(stimulationWeights):
        subset = df[df["stimWeight"] == ws]


        if i == 0:
            fig.add_trace(go.Heatmap(z=subset[regionLabels], x=regionLabels, y=subset.stimFreq, colorscale='Viridis',
                                     colorbar=dict(title="PLV"), zmin=0, zmax=1), row=i+1, col=1)

        else:
            fig.add_trace(go.Heatmap(z=subset[regionLabels], x=regionLabels, y=subset.stimFreq, colorscale='Viridis',
                                     zmin=0, zmax=1, showscale=False), row=i+1, col=1)

    fig.add_trace(go.Scatter(x=regionLabels, y=structure), row=i + 2, col=1)

    fig.update_layout(
        title_text='FC of simulated signals by stimulation frequency and weight || ' + t_c + ' region: ' + region,
        template="simple_white")
    pio.write_html(fig, file=folder+"/stimSpace-f&a%s.html" % t_c, auto_open=auto_open)


stimSpace(df_t, structure_t, w_subset, regionLabels, folder=specific_folder, region=regionLabels[region_id], t_c="target", auto_open="True")
stimSpace(df_c, structure_c, w_subset, regionLabels, folder=specific_folder, region=regionLabels[control_id], t_c="control", auto_open="True")
