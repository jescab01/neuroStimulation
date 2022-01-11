# Edited from https://gist.github.com/maedoc/448b21d51371c1631c8bd0b4360a00f4
import numpy as np
from tvb.simulator.lab import *
from toolbox import timeseriesPlot, FFTplot, FFTpeaks

simLength = 2000 # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1000 #Hz
m = models.Generic2dOscillator(I=np.array([5]), variables_of_interest=("V"))
coup = coupling.Linear(a=np.array([0]), b=np.array([0]))

# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000/samplingFreq)

conn = connectivity.Connectivity.from_file("paupau.zip")
mon = (monitors.Raw(),)

# Run simulation
sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup,  integrator=integrator, monitors=mon)
sim.configure()
output = sim.run(simulation_length=simLength)
# Extract data cutting initial transient
raw_data = output[0][1][:, 0, :, 0].T
raw_time = output[0][0][:]
regionLabels = conn.region_labels

# Check initial transient and cut data
timeseriesPlot(raw_data, raw_time, regionLabels, mode="html")


class Stim(patterns.StimuliRegion):
    def __init__(self, *args, **kwargs):
        super(Stim, self).__init__(*args, **kwargs)
        self.threshold = -4

    def __call__(self, time, space=None):
        n_node = self.state.shape[1]
        self.stimulus = np.zeros(n_node)
        # implement custom logic here
        # NB time is in steps, not ms
        if time > 1e3:
            self.stimulus = 10.0
        return self.stimulus

    def set_state(self, state):
        self.state = state

    def configure_time(self, t):
        pass

eqn_t = equations.Sinusoid()
eqn_t.parameters['amp'] = 1
eqn_t.parameters['frequency'] = 0.02 #kHz

weighting = np.zeros((4, ))
weighting[[0]] = 0.05

stimulus=Stim(connectivity=conn, weight=weighting, temporal=eqn_t)
stimulus.configure_time(np.arange(0, simLength, 1))
stimulus.configure_space()


sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup,  integrator=integrator, monitors=mon, stimulus=stimulus)
sim.configure()
output = sim.run(simulation_length=simLength)
# Extract data cutting initial transient
raw_data = output[0][1][:, 0, :, 0].T
raw_time = output[0][0][:]
regionLabels = conn.region_labels

# Check initial transient and cut data
timeseriesPlot(raw_data, raw_time, regionLabels, mode="html")