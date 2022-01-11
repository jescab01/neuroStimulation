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
import plotly.express as px

# Choose a name for your simulation and define the empirical for SC
model_id = ".1995JansenRit"

# Structuring directory to organize outputs
wd=os.getcwd()
main_folder = wd+"\\"+"PSE"

if not os.path.isdir(main_folder):
    os.mkdir(main_folder)

specific_folder = main_folder + "\\PSE_dyn" + model_id + "-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
os.mkdir(specific_folder)

ctb_folder = "D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\\CTB_data\\output\\"

simLength = 5000 # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1024 #Hz
transient = 1000 # ms to exclude from timeseries due to initial transient

# Create a dictionary to set stimulation conditions by connectome
# stimulate_indexes = {"AVG_NEMOS_acc.zip": [0],  # ACCleft
#                      "AVG_NEMOS_CB.zip": [4],
#                      "AVG_NEMOS_AAL2red.zip": [34]}

connectome="AVG_NEMOS_AAL2red.zip"


#########################
#########################
## WHAT dynamics do I want to explore?
aim = ["signals"]  # "fft" and/or "signals"



# if len(stimulate_indexes) == 1:
#     conn = connectivity.Connectivity.from_file(connectome)
# else:
conn = connectivity.Connectivity.from_file(ctb_folder+connectome)
conn.weights = conn.scaled_weights(mode="tract")


# # Parameters from Stefanovski 2019. Good working point at g=33, s=15.5 on AAL2red connectome.
m = models.JansenRit(A=np.array([3.25]), B=np.array([22]), J=np.array([1]),
                     a=np.array([0.1]), a_1=np.array([135]), a_2=np.array([108]),
                     a_3=np.array([33.75]), a_4=np.array([33.75]), b=np.array([0.06]),
                     mu=np.array([0.1085]), nu_max=np.array([0.0025]), p_max=np.array([0]), p_min=np.array([0]),
                     r=np.array([0.56]), v0=np.array([6]))

coup = coupling.SigmoidalJansenRit(a=np.array([33]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                   r=np.array([0.56]))
conn.speed = np.array([15.5])

# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000/samplingFreq)

mon = (monitors.Raw(),)

## Define the stimulus variable to loop over
stim_freqs = np.concatenate((np.array([0]), np.arange(8, 14, 0.25))) #np.arange(1, 25, 1) #[3,12,22,27,45] #Hz  -  np.arange(1, 60, 1)

# stim_weights = np.concatenate((np.arange(0, 0.0045, 0.0005),
#                                np.arange(0.0045, 0.005, 0.00001),
#                                np.arange(0.005, 0.02, 0.001),
#                                np.arange(0.02, 0.1, 0.01)))# np.arange(0, 0.1, 0.0005)

for f in stim_freqs:
    dynamic_fft_data = np.ndarray((1, 4))
    dynamic_signal_data = np.ndarray((1, 4))

    tic0 = time.time()

    ## Sinusoid input
    eqn_t = equations.Sinusoid()
    eqn_t.parameters['amp'] = 1
    eqn_t.parameters['frequency'] = f  # Hz
    eqn_t.parameters['onset'] = 0  # ms
    eqn_t.parameters['offset'] = 5000  # ms
    # if w != 0:
    #     eqn_t.parameters['DC'] = 0.0005 / w

    weighting = np.loadtxt('stimulationCollab/RoastModel-P3P4_efnorm_mag-AAL2red.txt') * 0.46  # reaching_Roast-P3P4 * w_fitted

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
    regionLabels = conn.region_labels
    regionLabels = list(regionLabels)
    regionLabels.insert(len(conn.weights[0]) + 1, "stimulus")

    # np.savetxt('signalSussi/signal_stim@'+str(f)+'.csv', raw_data, delimiter=",")



    # average signals to obtain mean signal frequency peak
    signals = np.concatenate((raw_data, stimulus.temporal_pattern[:, transient:]*0.1), axis=0)

    # save time, signals x3cols,
    if "signals" in aim:
        regLabs = list()
        fs = list()
        sign_tot = list()
        times_tot = list()

        for i in range(len(signals)):
            regLabs += [regionLabels[i]] * len(signals[i])
            fs += [f] * len(signals[i])
            sign_tot += list(signals[i])
            times_tot += list(raw_time)

        temp1 = np.asarray([regLabs, fs, sign_tot, times_tot]).transpose()
        dynamic_signal_data = np.concatenate((dynamic_signal_data, temp1))

    # Check initial transient and cut data
    # timeseriesPlot(raw_data, raw_time, conn.region_labels, main_folder, mode="html")

    # Fourier Analysis plot
    # FFTplot(raw_data, simLength-transient, regionLabels, main_folder, mode="html")

    #####

    if "fft" in aim:
        regLabs = list()
        fs = list()
        ws = list()
        fft_tot = list()
        freq_tot = list()

        for i in range(len(signals)):
            fft = abs(np.fft.fft(signals[i]))  # FFT for each channel signal
            fft = fft[range(np.int(len(signals[i]) / 2))]  # Select just positive side of the symmetric FFT
            freqs = np.arange(len(signals[i]) / 2)
            freqs = freqs / ((simLength - transient) / 1000)  # simLength (ms) / 1000 -> segs

            fft = fft[freqs > 0.5]  # remove undesired frequencies from peak analisis
            freqs = freqs[freqs > 0.5]

            regLabs += [regionLabels[i]] * len(fft)
            fs += [f] * len(fft)
            fft_tot += list(fft)
            freq_tot += list(freqs)

        temp = np.asarray([regLabs, fs, fft_tot, freq_tot]).transpose()
        dynamic_fft_data = np.concatenate((dynamic_fft_data, temp))
    #####
    print('Stimulation frequency = ' + str( f ))
    print("LOOP ROUND REQUIRED %0.4f seconds.\n\n\n" % (time.time() - tic0,))





## GATHER RESULTS
title = 'dyn_'  # "12Hz" | "0.004w"
simname = connectome+"-"+time.strftime("m%md%dy%Y")

print("saving...")
# dynamic ffts
if "fft" in aim:
    df_fft = pd.DataFrame(dynamic_fft_data[1:, ], columns=["name", "stimfreq", "fft", "freqs"])
    df_fft = df_fft.astype({"name": str, "stimfreq": float, "fft": float, "freqs": float})
    df_fft.to_csv(specific_folder+"/PSE_"+simname+"-dynamicFFTdf@"+title+".csv", index=False)

# dynamic signals
if "signals" in aim:
    df_s = pd.DataFrame(dynamic_signal_data[1:, ], columns=["name", "stimfreq", "signal", "time"])
    df_s = df_s.astype({"name": str, "stimfreq": float, "signal": float, "time": float})
    df_s.to_csv(specific_folder+"/PSE_"+simname+"-dynamicSignalsdf"+title+".csv", index=False)

# Load previously gathered data
# df1=pd.read_csv("expResults/paramSpace_FFTpeaks-2d-subj4-8s-sim.csv")


# Define frequency to explore
# Plot FFT dynamic
print("plotting...")
# improves plotting times
if "AAL2red" in connectome:
    indexes=[18, 19, 32, 33, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 62, 63, 64, 65, 70, 71, 76, 77, 92]
    labs=[regionLabels[i] for i in indexes]
    if "fft" in aim:
        df_fft=df_fft[df_fft["name"].isin(labs)]
    if "signals" in aim:
        df_s = df_s[df_s["name"].isin(labs)]



if "fft" in aim:
    fig = px.line(df_fft, x="freqs", y="fft", animation_frame="weight", animation_group="name", color="name",
                  title="Dynamic FFT @ " + title)
    pio.write_html(fig, file=specific_folder+"/"+connectome+"-f&w_dynFFT_@%s.html" % title)#, auto_open="False")

# Plot singals dynamic
if "signals" in aim:
    fig = px.line(df_s, x="time", y="signal", animation_frame="stimfreq", animation_group="name", color="name",
                  title="Dynamic Signals @ "+title)
    pio.write_html(fig, file=specific_folder+"/"+connectome+"-f&w_dynSIGNALS_@%s.html" % title, auto_open=True)
