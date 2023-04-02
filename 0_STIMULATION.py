from tvb.simulator.lab import *
import numpy as np

# temp package
import sys
sys.path.append("D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\")
from toolbox.signals import timeseriesPlot
from toolbox.fft import FFTplot, FFTpeaks

simLength = 5*1000 # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1000 #Hz

## STRUCTURE
f = os.getcwd()+"\\CTB_data\\output\\CTB_connx66_subj04.zip"
conn = connectivity.Connectivity.from_file() # f | paupau_

## MODEL
# m = models.Generic2dOscillator(I=np.array([5]))

# coup = coupling.Linear(a=np.array([0]), b=np.array([0]))

# # Parameters from Abeysuriya 2018. Good working point (for AAL2red connectivity) at s=16.5; g=0.525. P originally 0.31.
# m = models.WilsonCowan(P=np.array([0.31]), Q=np.array([0]),
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

# Parameters from Forrester 2019 - For single node oscillating. mu raised to 0.15 to get alpha oscillation
m = models.JansenRit(A=np.array([3.25]), B=np.array([22]), J=np.array([1]),
                     a=np.array([0.1]), a_1=np.array([135]), a_2=np.array([108]),
                     a_3=np.array([33.75]), a_4=np.array([33.75]), b=np.array([0.05]),
                     mu=np.array([0.15]), nu_max=np.array([0.0025]), p_max=np.array([0]), p_min=np.array([0]),
                     r=np.array([0.56]), v0=np.array([6]))

coup = coupling.SigmoidalJansenRit(a=np.array([33]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                   r=np.array([0.56]))
conn.speed = np.array([15.5])


# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000/samplingFreq)

mon = (monitors.Raw(),)
transient = 1000

##############
##### STIMULUS
## Pulse train
# eqn_t = equations.PulseTrain()
# eqn_t.parameters['onset'] = 1535
# eqn_t.parameters['T'] = 100.0
# eqn_t.parameters['tau'] = 50.0

## Sinusoid input
eqn_t = equations.Sinusoid()
eqn_t.parameters['amp'] = 1
eqn_t.parameters['frequency'] = 10 #Hz
eqn_t.parameters['onset'] = 0 #ms
eqn_t.parameters['offset'] = 5000 #ms

## Drifting Sinusoid input
# eqn_t = equations.DriftSinusoid()
# eqn_t.parameters['amp'] = 0.1
# eqn_t.parameters['f_init'] = 10 #Hz
# eqn_t.parameters['f_end'] = 20 #Hz
# eqn_t.parameters['onset'] = 500 #ms
# eqn_t.parameters['offset'] = 2000 #ms
# eqn_t.parameters['feedback'] = True # Grow&Shrink (True) - Grow|Shrink(False)
# eqn_t.parameters['sim_length'] = simLength # Grow&Shrink (True) - Grow|Shrink(False)
# eqn_t.parameters['dt'] = 1000/samplingFreq # in ms
# eqn_t.parameters['avg'] = 0.5 #


weighting = np.zeros((76, ))
weighting[[0]] = 0.04

stimulus = patterns.StimuliRegion(
    temporal=eqn_t,
    connectivity=conn,
    weight=weighting)

#Configure space and time
stimulus.configure_space()
stimulus.configure_time(np.arange(0, simLength, 1))
#And take a look
# plot_pattern(stimulus)

# Run simulation
sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup,  integrator=integrator, monitors=mon, stimulus=stimulus)
sim.configure()
output = sim.run(simulation_length=simLength)
# Extract data cutting initial transient
raw_data = output[0][1][:, 0, :, 0].T
raw_time = output[0][0][:]
regionLabels = conn.region_labels
regionLabels = list(regionLabels)
regionLabels.insert(76, "stimulus")

# average signals to obtain mean signal frequency peak
data = np.concatenate((raw_data, stimulus.temporal_pattern), axis=0)  # concatenate mean signal: data[0]; with raw_data: data[1:end]
# Check initial transient and cut data
timeseriesPlot(data, raw_time, regionLabels, title= "20HzStim", mode="html")


# Fourier Analysis plot
FFTplot(data, simLength-transient, regionLabels,  mode="html")



fft_peaks = FFTpeaks(raw_data, simLength-transient)[:, 0]

# history=output[0][1][-22:]
# plt.plot(np.arange(0,22,1),history[:,0,0,0].T)