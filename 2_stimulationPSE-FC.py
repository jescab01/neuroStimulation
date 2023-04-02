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
from toolbox import timeseriesPlot, FFTplot, FFTpeaks, AEC, PLV, PLI, epochingTool, stimulation_fc

# Choose a name for your simulation and define the empirical for SC
model_id = ".1995JansenRit"

# Structuring directory to organize outputs
wd = os.getcwd()
main_folder = wd + "\\" + "PSE"

if not os.path.isdir(main_folder):
    os.mkdir(main_folder)

specific_folder = main_folder + "\\PSE_stimFC" + model_id + "-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
os.mkdir(specific_folder)

ctb_folder= "D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\\CTB_data\\output\\"


simLength = 10 * 1000  # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1024  # Hz
transient = 1000  # ms to exclude from timeseries due to initial transient

# Create a dictionary to set stimulation conditions by connectome
stimulate_indexes = {"AVG_NEMOS_acc.zip": [0],
                    "AVG_NEMOS_CB.zip": [4],  # ACCleft
                    "AVG_NEMOS_AAL2red.zip": [34]}  # AACleft

control_idx = [1]


for connectome in stimulate_indexes.keys():

    conn = connectivity.Connectivity.from_file(ctb_folder+connectome)
    conn.weights = conn.scaled_weights(mode="tract")

    # conn = connectivity.Connectivity.from_file(ctb_folder+"AVG_NEMOS_CB.zip")
    # conn = connectivity.Connectivity.from_file(ctb_folder+"AVG_NEMOS_AAL2red.zip")
    # Parameters from Abeysuriya 2018. Using P=0.60 for the disconnected nodes to oscillate at 9.75Hz. Originally at 0.31.
    # Good working point @ g=0.525 s=16.5 on AAL2red connectome.
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


    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)
    mon = (monitors.Raw(),)

    ###### Analyze PSE-FFT results >> define what are the amplitude and frequencies of interest
    stim_freqs = np.arange(1, 20, 1)  # np.arange(1, 80, 1)
    stim_amp = [0.00, 0.02, 0.04]#np.arange(0, 0.05, 0.005)  # np.arange(0, 0.05, 0.002)

    plv_target = list()
    fft_target = list()
    plv_control = list()

    for f in stim_freqs:
        for a in stim_amp:
            tic0 = time.time()

            ## Sinusoid input
            eqn_t = equations.Sinusoid()
            eqn_t.parameters['amp'] = a
            eqn_t.parameters['frequency'] = f  # Hz
            eqn_t.parameters['onset'] = 0  # ms
            eqn_t.parameters['offset'] = 10000  # ms
            # if w != 0:
            #     eqn_t.parameters['DC'] = 0.0005 / w

            # Check the index of the region to stimulate and
            regionLabels = list(conn.region_labels)
            weighting = np.zeros((len(conn.weights),))
            weighting[stimulate_indexes[connectome]] = 1

            stimulus = patterns.StimuliRegion(temporal=eqn_t, connectivity=conn, weight=weighting)

            # Configure space and time
            stimulus.configure_space()
            stimulus.configure_time(np.arange(0, simLength, 1))

            # And take a look
            # plot_pattern(stimulus)

            # Run simulation
            sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon,
                                      stimulus=stimulus)
            sim.configure()

            output = sim.run(simulation_length=simLength)
            print("Simulation time: %0.4f sec" % (time.time() - tic0,))
            # Extract data cutting initial transient
            raw_data = output[0][1][transient:, 0, :, 0].T
            raw_time = output[0][0][transient:]

            # Fourier Analysis plot
            # FFTplot(raw_data, simLength-transient, regionLabels, main_folder, mode="html")
            fft_peaks = FFTpeaks(raw_data, simLength - transient)[0][:, 0]


            ##########
            ### Functional Connectivity PSE
            # Subset weights which you want to explore.
            newRow_t = [regionLabels[stimulate_indexes[connectome][0]], f, a]
            newRow_c = [regionLabels[control_idx[0]], f, a]
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
                    # efEnvelope.append(np.abs(analyticalSignal))

                # CONNECTIVITY MEASURES
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

                plv_target.append(newRow_t + list(plv[stimulate_indexes[connectome][0]]))
                fft_target.append(newRow_t + list(fft_peaks))
                plv_control.append(newRow_c + list(plv[control_idx][0]))

            print("LOOP ROUND REQUIRED %0.4f seconds.\n\n\n\n" % (time.time() - tic0,))

    ## GATHER RESULTS
    simname = connectome + "-" + time.strftime("m%md%dy%Y")

    df_t = pd.DataFrame(plv_target, columns=["targetRegion", "stimFreq", "stimAmplitude", "band"] + regionLabels)
    df_fft = pd.DataFrame(fft_target, columns=["targetRegion", "stimFreq", "stimAmplitude", "band"] + regionLabels)
    df_c = pd.DataFrame(plv_control, columns=["controlRegion", "stimFreq", "stimAmplitude", "band"] + regionLabels)

    df_t.to_csv(specific_folder + "/PSE-FCtarget_" + simname + ".csv", index=False)
    df_fft.to_csv(specific_folder + "/PSE-FFTtarget_" + simname + ".csv", index=False)
    df_c.to_csv(specific_folder + "/PSE-FCcontrol_" + simname + ".csv", index=False)

    # Subset structural connectivity (from ROI to other regions) and plot
    structure_t = conn.weights[stimulate_indexes[connectome][0]]
    structure_c = conn.weights[control_idx[0]]

    stimulation_fc(df_t, structure_t, regionLabels, folder=specific_folder, region=regionLabels[stimulate_indexes[connectome][0]], title=simname, t_c="target", auto_open="True")
    stimulation_fc(df_fft, structure_t, regionLabels, folder=specific_folder, region=regionLabels[stimulate_indexes[connectome][0]]+" FFT peaks", title=simname, t_c="target", auto_open="True")
    stimulation_fc(df_c, structure_c, regionLabels, folder=specific_folder, region=regionLabels[control_idx[0]], title=simname, t_c="control", auto_open="True")



    # Load previously gathered data
    # df01=pd.read_csv("C:\\Users\jesca\PycharmProjects\\brainModels\stimulationCollab\PSE\PSE.1973WilsonCowan-m02d19y2021-t12h.39m.10s-IsolatedNode01\PSE_isolated_nodes-m02d19y2021.csv")
    # df01DC=pd.read_csv("C:\\Users\jesca\PycharmProjects\\brainModels\stimulationCollab\PSE\PSE.1973WilsonCowan-m02d24y2021-t00h.55m.41s-IsolatedNode01DC\PSE_isolated_nodes-DCm02d24y2021.csv")
    # df02=pd.read_csv("C:\\Users\jesca\PycharmProjects\\brainModels\stimulationCollab\PSE\PSE.1973WilsonCowan-m02d22y2021-t22h.18m.50s-IsolatedNode02\PSE_isolated_nodes-m02d22y2021.csv")
    # df02DC=pd.read_csv("C:\\Users\jesca\PycharmProjects\\brainModels\stimulationCollab\PSE\PSE.1973WilsonCowan-m02d23y2021-t20h.57m.35s-IsolatedNode02DC\PSE_isolated_nodes-DCm02d23y2021.csv")


    # df1=df[df.stimStrength<=0.05]
    # df=pd.read_csv("C:\\Users\jesca\PycharmProjects\\brainModels\stimulationCollab\PSE\PSE.1973WilsonCowan-m02d19y2021-t12h.39m.10s\PSE_isolated_nodes-m02d19y2021.csv")


