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
from toolbox import timeseriesPlot, FFTplot, FFTpeaks, AEC, PLV, PLI, epochingTool, stimulation_fft

# Choose a name for your simulation and define the empirical for SC
model_id = ".1995JansenRit"

# Structuring directory to organize outputs
wd = os.getcwd()
main_folder = wd + "\\" + "PSE"

# if not os.path.isdir(main_folder):
#     os.mkdir(main_folder)

specific_folder = main_folder + "\\PSE_stimFFT" + model_id + "-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
# os.mkdir(specific_folder)

ctb_folder = "D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\\CTB_data\\output\\"


simLength = 10 * 1000  # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1024  # Hz
transient = 1000  # ms to exclude from timeseries due to initial transient

# Create a dictionary to set stimulation conditions by connectome
stimulate_indexes = {"AVG_NEMOS_CB.zip": [4],  # ACCleft
                     "AVG_NEMOS_AAL2red.zip": [34]}  # AACleft

control_idx = [1]
data=[]
for connectome in stimulate_indexes.keys():
    tic0 = time.time()
    # if len(stimulate_indexes) == 1:
    #     conn = connectivity.Connectivity.from_file(connectome)
    # else:
    conn = connectivity.Connectivity.from_file(ctb_folder+connectome)
    conn.weights = conn.scaled_weights(mode="tract")

    # # Parameters from Abeysuriya 2018. Using P=0.60 for the disconnected nodes to oscillate at 9.75Hz. Originally at 0.31.
    # # Good working point @ g=0.525 s=16.5 on AAL2red connectome.
    # m = models.WilsonCowan(P=np.array([0.60]), Q=np.array([0]),
    #                        a_e=np.array([4]), a_i=np.array([4]),
    #                        alpha_e=np.array([1]), alpha_i=np.array([1]),
    #                        b_e=np.array([1]), b_i=np.array([1]),
    #                        c_e=np.array([1]), c_ee=np.array([3.25]), c_ei=np.array([2.5]),
    #                        c_i=np.array([1]), c_ie=np.array([3.75]), c_ii=np.array([0]),
    #                        k_e=np.array([1]), k_i=np.array([1]),
    #                        r_e=np.array([0]), r_i=np.array([0]),
    #                        tau_e=np.array([10]), tau_i=np.array([20]),
    #                        theta_e=np.array([0]), theta_i=np.array([0]))
    #
    # coup = coupling.Linear(a=np.array([0.525]))
    # conn.speed = np.array([16.5])

    # # Parameters from Stefanovski 2019. Good working point at g=33, s=15.5 on AAL2red connectome.
    m = models.JansenRit(A=np.array([3.25]), B=np.array([22]), J=np.array([1]),
                         a=np.array([0.1]), a_1=np.array([135]), a_2=np.array([108]),
                         a_3=np.array([33.75]), a_4=np.array([33.75]), b=np.array([0.06]),
                         mu=np.array([0.1085]), nu_max=np.array([0.0025]), p_max=np.array([0]), p_min=np.array([0]),
                         r=np.array([0.56]), v0=np.array([6]))

    coup = coupling.SigmoidalJansenRit(a=np.array([33]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                       r=np.array([0.56]))
    conn.speed = np.array([15.5])

    # Parameters from Forrester 2019 - For single node oscillating. mu raised to 0.15 to get alpha oscillation
    # m = models.JansenRit(A=np.array([3.25]), B=np.array([22]), J=np.array([1]),
    #                      a=np.array([0.1]), a_1=np.array([135]), a_2=np.array([108]),
    #                      a_3=np.array([33.75]), a_4=np.array([33.75]), b=np.array([0.05]),
    #                      mu=np.array([0.15]), nu_max=np.array([0.0025]), p_max=np.array([0]), p_min=np.array([0]),
    #                      r=np.array([0.56]), v0=np.array([6]))
    #
    # coup = coupling.SigmoidalJansenRit(a=np.array([0]), cmax=np.array([0.005]), midpoint=np.array([6]),
    #                                    r=np.array([0.56]))
    # conn.speed = np.array([15.5])


    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)
    mon = (monitors.Raw(),)


    # Run simulation
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
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

    # indexes=[20, 21, 32, 33, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 62, 63, 70, 71, 76, 77]
    #
    # if "AAL2" in connectome:
    #     data.append(fft_peaks[indexes])
    #     data.append(conn.region_labels[indexes])
    # else:
    data.append(fft_peaks)
    data.append(conn.region_labels)

    print("LOOP ROUND REQUIRED %0.4f seconds.\n\n\n\n" % (time.time() - tic0,))


# ## GATHER RESULTS

data_wn = {"region": data[3], "fft_p": data[2], "struct": ["AAL2red"]*len(data[2])}
data_cb = {"region": data[1], "fft_p": data[0], "struct": ["CingulumBundle"]*len(data[0])}

df = pd.DataFrame(data_wn)
df = df.append(pd.DataFrame(data_cb))

import plotly.express as px
fig = px.bar(df, x="region", y="fft_p", color='struct', barmode='group')
fig.show("browser")



# simname = connectome + "-" + time.strftime("m%md%dy%Y")
#
# df = pd.DataFrame(data, columns=["stimFreq", "stimAmplitude", "tAvg_activity", "cAvg_activity",
#                                  "tFFT_peak", "cFFT_peak"])
# df.to_csv(specific_folder + "/PSE-FFT_" + simname + ".csv", index=False)
#
# stimulation_fft(df, specific_folder, title="FFT by stimulation frequency and amplitud - " + simname)


