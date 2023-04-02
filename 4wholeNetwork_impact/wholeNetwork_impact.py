
import time
import numpy as np
import pandas as pd
import scipy
from mne import time_frequency, filter

from tvb.simulator.lab import *
from mpi4py import MPI
import datetime
import glob

ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data2\\"
import sys
sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
from toolbox.fft import multitapper, FFTpeaks
from toolbox.fc import PLV
from toolbox.signals import epochingTool

## Plotting
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

working_points = [("jr", "NEMOS_035", 6, 23.5),  # JR pass @ 10/06/2022
                  ("jr", "NEMOS_049", 8, 17.5),  # manipulated: original 49, 20.5 - the only wp out of limit cycle
                  ("jr", "NEMOS_050", 17, 24.5),
                  ("jr", "NEMOS_058", 17, 18.5),
                  ("jr", "NEMOS_059", 5, 11.5),
                  ("jr", "NEMOS_064", 5, 24.5),
                  ("jr", "NEMOS_065", 5, 24.5),
                  ("jr", "NEMOS_071", 10, 22.5),
                  ("jr", "NEMOS_075", 6, 24.5),
                  ("jr", "NEMOS_077", 8, 21.5)]


stimulation_site, w = "roast_F3F4Model", 0.29


# Prepare simulation parameters
simLength = 120 * 1000  # ms
samplingFreq = 1000  # Hz
transient = 4000  # ms


# COMMON SIMULATION PARAMETERS   ###
# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

mon = (monitors.Raw(),)

tic = time.time()

results = list()
for wp in working_points:

    mode, emp_subj, g, s = wp

    # STRUCTURAL CONNECTIVITY      #########################################
    if '_pTh' in mode:
        conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2pTh_pass.zip")
    else:
        conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2_pass.zip")
    conn.weights = conn.scaled_weights(mode="tract")

    # Define regions implicated in Functional analysis: remove  Cerebelum, Thalamus, Caudate (i.e. subcorticals)
    cortical_rois = ['Precentral_L', 'Precentral_R', 'Frontal_Sup_2_L',
                     'Frontal_Sup_2_R', 'Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                     'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L',
                     'Frontal_Inf_Tri_R', 'Frontal_Inf_Orb_2_L', 'Frontal_Inf_Orb_2_R',
                     'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L',
                     'Supp_Motor_Area_R', 'Olfactory_L', 'Olfactory_R',
                     'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R',
                     'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R', 'Rectus_L', 'Rectus_R',
                     'OFCmed_L', 'OFCmed_R', 'OFCant_L', 'OFCant_R', 'OFCpost_L',
                     'OFCpost_R', 'OFClat_L', 'OFClat_R', 'Insula_L', 'Insula_R',
                     'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Mid_L',
                     'Cingulate_Mid_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                     'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                     'ParaHippocampal_R', 'Calcarine_L',
                     'Calcarine_R', 'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R',
                     'Occipital_Sup_L', 'Occipital_Sup_R', 'Occipital_Mid_L',
                     'Occipital_Mid_R', 'Occipital_Inf_L', 'Occipital_Inf_R',
                     'Fusiform_L', 'Fusiform_R', 'Postcentral_L', 'Postcentral_R',
                     'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                     'Parietal_Inf_R', 'SupraMarginal_L', 'SupraMarginal_R',
                     'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R',
                     'Paracentral_Lobule_L', 'Paracentral_Lobule_R', 'Heschl_L', 'Heschl_R',
                     'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L',
                     'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R',
                     'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L',
                     'Temporal_Inf_R']
    cingulum_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                     'Insula_L', 'Insula_R',
                     'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                     'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                     'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R',
                     'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                     'Parietal_Inf_R', 'Precuneus_L', 'Precuneus_R',
                     'Thalamus_L', 'Thalamus_R']

    # load text with FC rois; check if match SC
    FClabs = list(np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/roi_labels_rms.txt", dtype=str))
    FC_cortex_idx = [FClabs.index(roi) for roi in
                     cortical_rois]  # find indexes in FClabs that matches cortical_rois
    SClabs = list(conn.region_labels)
    SC_cortex_idx = [SClabs.index(roi) for roi in cortical_rois]

    # Subset for Cingulum Bundle
    if "cb" in mode:
        FC_cb_idx = [FClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cortical_rois
        SC_cb_idx = [SClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cortical_rois
        conn.weights = conn.weights[:, SC_cb_idx][SC_cb_idx]
        conn.tract_lengths = conn.tract_lengths[:, SC_cb_idx][SC_cb_idx]
        conn.region_labels = conn.region_labels[SC_cb_idx]


    # NEURAL MASS MODEL    #########################################################
    if "jrd" in mode:  # JANSEN-RIT-DAVID
        if "_def" in mode:
            sigma_array = 0.022
            p_array = 0.22
        else:  # for jrd_pTh and jrd modes
            sigma_array = np.asarray([0.022 if 'Thal' in roi else 0 for roi in conn.region_labels])
            p_array = np.asarray([0.22 if 'Thal' in roi else 0.13 for roi in conn.region_labels])

        # Parameters edited from David and Friston (2003).
        # m = JansenRitDavid2003_N(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
        #                          tau_e1=np.array([10.8]), tau_i1=np.array([22.0]),
        #                          He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
        #                          tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),
        #
        #                          w=np.array([0.8]), c=np.array([135.0]),
        #                          c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
        #                          c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
        #                          v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
        #                          p=np.array([p_array]), sigma=np.array([sigma_array]))

        # # Remember to hold tau*H constant.
        # m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
        # m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])

    else:  # JANSEN-RIT
        # Parameters from Stefanovski 2019. Good working point at g=33, s=15.5 on AAL2red connectome.
        m = models.JansenRit(A=np.array([3.25]), B=np.array([22]), J=np.array([1]),
                             a=np.array([0.1]), a_1=np.array([135]), a_2=np.array([108]),
                             a_3=np.array([33.75]), a_4=np.array([33.75]), b=np.array([0.06]),
                             mu=np.array([0.1085]), nu_max=np.array([0.0025]), p_max=np.array([0]),
                             p_min=np.array([0]),
                             r=np.array([0.56]), v0=np.array([6]))


    # COUPLING FUNCTION   #########################################
    if "jrd" in mode:
        # coup = coupling.SigmoidalJansenRitDavid(a=np.array([g]), w=m.w, e0=m.e0, v0=m.v0, r=m.r)
        pass
    else:
        coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                           r=np.array([0.56]))
    conn.speed = np.array([s])



    ### First simulation set - baseline
    print("Baseline simulation set - Simulating for Coupling factor = %i and speed = %i" % (g, s))

    # Run simulation
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
    sim.configure()
    output = sim.run(simulation_length=simLength)

    # Extract data: "output[a][b][:,0,:,0].T" where:
    # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
    if "jrd" in mode:
        raw_data = m.w * (output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T) + \
                   (1 - m.w) * (output[0][1][transient:, 3, :, 0].T - output[0][1][transient:, 4, :, 0].T)
    else:
        raw_data = output[0][1][transient:, 0, SC_cortex_idx, 0].T

    raw_time = output[0][0][transient:]

    # Fourier Analysis plot
    # FFTplot(raw_data, simLength-transient, regionLabels, main_folder, mode="html")
    fftpeaks_baseline = FFTpeaks(raw_data, simLength-transient)[0]


    ##########
    ### Measure functional connectivity between regions of interest : line 60 - rois
    bands = [["3-alfa"], [(8, 12)]]
    ## [["1-delta", "2-theta", "3-alfa", "4-beta", "5-gamma"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]
    plv_baseline = list()
    for b in range(len(bands[0])):

        (lowcut, highcut) = bands[1][b]

        # Band-pass filtering
        filterSignals = filter.filter_data(raw_data, samplingFreq, lowcut, highcut)

        # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
        efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals")

        # Obtain Analytical signal
        efPhase = list()
        # efEnvelope = list()

        for i in range(len(efSignals)):
            analyticalSignal = scipy.signal.hilbert(efSignals[i])
            # Get instantaneous phase and amplitude envelope by channel
            efPhase.append(np.unwrap(np.angle(analyticalSignal)))
            # efEnvelope.append(np.abs(analyticalSignal))

        # CONNECTIVITY MEASURES
        ## PLV
        plv_baseline = PLV(efPhase)
        # fname = ctb_folder+model_id+"\\"+bands[0][b]+"plv.txt"
        # np.savetxt(fname, plv)


    ## SECOND ROUND: stimulating

    # Calculate IAF based on Precuneus
    precuneus_ids = [list(conn.region_labels).index(roi) for roi in conn.region_labels if "Precuneus" in roi]

    prec_base_hz = np.average(fftpeaks_baseline[precuneus_ids])

    print("Stimulation simulation set - Simulating for Coupling factor = %i and speed = %i" % (g, s))


    # STIMULUS ###############################

    eqn_t = equations.Sinusoid()
    eqn_t.parameters['amp'] = 1
    eqn_t.parameters['frequency'] = prec_base_hz + 1.6  # Hz
    eqn_t.parameters['onset'] = 0  # ms
    eqn_t.parameters['offset'] = simLength  # ms
    # if w != 0:
    #     eqn_t.parameters['DC'] = 0.0005 / w


    ## electric field; electrodes placed @ P3P4 to stimulate precuneus
    # weighting = np.loadtxt(ctb_folder + 'CurrentPropagationModels/' + emp_subj + '-roast_P3P4Model_ef_mag-AAL2red.txt') * w
    ## Focal stimulation on ACC electric field;
    weighting = np.loadtxt(glob.glob(
        ctb_folder + 'CurrentPropagationModels/' + emp_subj + '-efnorm_mag-' + stimulation_site + '*-AAL2.txt')[0],
                           delimiter=",") * w
    if "cb" in mode:
        weighting = weighting[SC_cb_idx]


    stimulus = patterns.StimuliRegion(temporal=eqn_t, connectivity=conn, weight=weighting)

    # Configure space and time
    stimulus.configure_space()
    stimulus.configure_time(np.arange(0, simLength, 1))


    print("Simulating for Coupling factor = %i and speed = %i" % (g, s))

    # Run simulation
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon, stimulus=stimulus)
    sim.configure()
    output = sim.run(simulation_length=simLength)

    # Extract data: "output[a][b][:,0,:,0].T" where:
    # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
    if "jrd" in mode:
        raw_data = m.w * (output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T) + \
                   (1 - m.w) * (output[0][1][transient:, 3, :, 0].T - output[0][1][transient:, 4, :, 0].T)
    else:
        raw_data = output[0][1][transient:, 0, SC_cortex_idx, 0].T

    raw_time = output[0][0][transient:]

    # Fourier Analysis plot
    # FFTplot(raw_data, simLength-transient, regionLabels, main_folder, mode="html")
    fftpeaks_stimulated = FFTpeaks(raw_data, simLength-transient)[0]


    ##########
    ### Measure functional connectivity between regions of interest : line 60 - rois
    bands = [["3-alfa"], [(8, 12)]]
    ## [["1-delta", "2-theta", "3-alfa", "4-beta", "5-gamma"], [(2, 4), (4, 8), (8, 12), (12, 30), (30, 45)]]
    plv_stimulated = list()
    for b in range(len(bands[0])):

        (lowcut, highcut) = bands[1][b]

        # Band-pass filtering
        filterSignals = filter.filter_data(raw_data, samplingFreq, lowcut, highcut)

        # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
        efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals")

        # Obtain Analytical signal
        efPhase = list()
        # efEnvelope = list()

        for i in range(len(efSignals)):
            analyticalSignal = scipy.signal.hilbert(efSignals[i])
            # Get instantaneous phase and amplitude envelope by channel
            efPhase.append(np.unwrap(np.angle(analyticalSignal)))
            # efEnvelope.append(np.abs(analyticalSignal))

        # CONNECTIVITY MEASURES
        ## PLV
        plv_stimulated = PLV(efPhase)
        # fname = ctb_folder+model_id+"\\"+bands[0][b]+"plv.txt"
        # np.savetxt(fname, plv)

    results.append([fftpeaks_baseline, plv_baseline, fftpeaks_stimulated, plv_stimulated, weighting])

    print("simulating stim_type = %s | stim_param = %0.2f - round = %i" % ("tACS", prec_base_hz + 1.6, 0))
    print("LOOP ROUND REQUIRED %0.4f seconds.\n\n\n\n" % (time.time() - tic,))



    # alpha PLV matrices

    # extract PLV ACC to all and plot as line; sorted by n connections of ACC (expected to be more altered).
    regionLabels = conn.region_labels[SC_cortex_idx]
    acc_index = [list(regionLabels).index(roi) for roi in regionLabels if "Cingulate_Ant" in roi]
    # acc_weights = conn.weights[acc_index, :]

    # sort by average weights
    # order = np.argsort(np.average(acc_weights, axis=0))

    # plot
    fig = make_subplots(rows=1, cols=3, column_widths=[0.25, 0.5, 0.25])

    fig.add_trace(go.Heatmap(z=plv_baseline, x=regionLabels, y=regionLabels, zmax=1, zmin=0), row=1, col=1)
    fig.add_trace(go.Heatmap(z=plv_stimulated, x=regionLabels, zmax=1, zmin=0), row=1, col=3)


    delta_fc = np.average(plv_stimulated[acc_index, :] - plv_baseline[acc_index, :], axis=0)
    order = np.argsort(delta_fc)

    fig.add_trace(go.Scatter(x=regionLabels[order], y=delta_fc[order]), row=1, col=2)

    delta_fft = (fftpeaks_stimulated - fftpeaks_baseline)
    fig.add_trace(go.Scatter(x=regionLabels[order], y=delta_fft[order]), row=1, col=2)

    pio.write_html(fig, "figures/" + emp_subj + ".html")


# Save output
import pickle

file_name = "wholeNetwork_impact" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss") + ".pkl"
open_file = open(file_name, "wb")
pickle.dump(results, open_file)
open_file.close()


# plot AVERAGE
fftpeaks_baseline_avg = np.average(np.asarray([subject[0] for subject in results]), axis=0)
plv_baseline_avg = np.average(np.asarray([subject[1] for subject in results]), axis=0)
fftpeaks_stimulated_avg = np.average(np.asarray([subject[2] for subject in results]), axis=0)
plv_stimulated_avg = np.average(np.asarray([subject[3] for subject in results]), axis=0)

fig = make_subplots(rows=1, cols=3, column_widths=[0.25, 0.5, 0.25])

fig.add_trace(go.Heatmap(z=plv_baseline_avg, x=regionLabels, y=regionLabels, zmax=1, zmin=0), row=1, col=1)
fig.add_trace(go.Heatmap(z=plv_stimulated_avg, x=regionLabels, zmax=1, zmin=0), row=1, col=3)


delta_fc = np.average(plv_stimulated_avg[acc_index, :] - plv_baseline_avg[acc_index, :], axis=0)
order = np.argsort(delta_fc)

fig.add_trace(go.Scatter(x=regionLabels[order], y=delta_fc[order]), row=1, col=2)

delta_fft = (fftpeaks_stimulated_avg - fftpeaks_baseline_avg)
fig.add_trace(go.Scatter(x=regionLabels[order], y=delta_fft[order]), row=1, col=2)

pio.write_html(fig, "figures/Subjects_Average.html")




