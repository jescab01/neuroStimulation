
import time
import numpy as np
import pandas as pd

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003
from mpi4py import MPI
import datetime


def stimWfit_parallel(params, baseline_subj=None):

    datapoints = list()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    print("Hello world from rank", str(rank), "of", str(size), '__', datetime.datetime.now().strftime("%Hh:%Mm:%Ss"))

    ## Folder structure - Local
    if "LCCN_Local" in os.getcwd():
        ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data2\\"
        import sys
        sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
        from toolbox.fft import FFTpeaks

    ## Folder structure - CLUSTER
    else:
        from toolbox import FFTpeaks
        wd = "/home/t192/t192950/mpi/"
        ctb_folder = wd + "CTB_data2/"

    # Prepare simulation parameters
    simLength = 24 * 1000  # ms
    samplingFreq = 1000  # Hz
    transient = 4000  # ms

    for set in params:
        tic = time.time()
        print(set)

        emp_subj, mode, g, s, r, w = set

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
            FC_cb_idx = [FClabs.index(roi) for roi in
                         cingulum_rois]  # find indexes in FClabs that matches cortical_rois
            SC_cb_idx = [SClabs.index(roi) for roi in
                         cingulum_rois]  # find indexes in FClabs that matches cortical_rois
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
            m = JansenRitDavid2003(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
                                   tau_e1=np.array([10.8]), tau_i1=np.array([22.0]),
                                   He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
                                   tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),

                                   w=np.array([0.8]), c=np.array([135.0]),
                                   c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                                   c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                                   v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                                   p=np.array([p_array]), sigma=np.array([sigma_array]))

            # Remember to hold tau*H constant.
            m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
            m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])

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
            coup = coupling.SigmoidalJansenRitDavid(a=np.array([g]), w=m.w, e0=m.e0, v0=m.v0, r=m.r)
        else:
            coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                               r=np.array([0.56]))
        conn.speed = np.array([s])

        # STIMULUS ###############################
        if w != 0:

            # Reconstruct DataFrame
            baseline_subj = pd.DataFrame(baseline_subj, columns=["Subject", "IAF", "module", "bModule"])
            initialPeak = float(baseline_subj.loc[baseline_subj["Subject"] == emp_subj].IAF.values)

            ## Sinusoid input
            eqn_t = equations.Sinusoid()
            eqn_t.parameters['amp'] = 1  # Amplitud diferencial por áreas ajustada en stimWeights
            eqn_t.parameters['frequency'] = initialPeak  # Hz
            eqn_t.parameters['onset'] = 0  # ms
            eqn_t.parameters['offset'] = simLength  # ms

            # electric field * orthogonal to surface
            weighting = np.loadtxt(
                ctb_folder + 'CurrentPropagationModels/' + emp_subj + '-efnorm_mag-roast_OzCzModel-AAL2.txt',
                delimiter=",") * w

            if "cb" in mode:
                weighting = weighting[SC_cb_idx]

            stimulus = patterns.StimuliRegion(
                temporal=eqn_t,
                connectivity=conn,
                weight=weighting)

            # Configure space and time
            stimulus.configure_space()
            stimulus.configure_time(np.arange(0, simLength, 1))
            # And take a look
            # plot_pattern(stimulus)

        # OTHER PARAMETERS   ###
        # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
        # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
        integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

        mon = (monitors.Raw(),)

        print("Simulating for Coupling factor = %i and speed = %i" % (g, s))

        # Run simulation
        if w != 0:
            sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon,
                                      stimulus=stimulus)
        else:
            sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
        sim.configure()
        output = sim.run(simulation_length=simLength)

        # Extract data: "output[a][b][:,0,:,0].T" where:
        # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
        if "jrd" in mode:
            raw_data = m.w * (output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T) + \
                       (1 - m.w) * (output[0][1][transient:, 3, :, 0].T - output[0][1][transient:, 4, :, 0].T)
        else:
            raw_data = output[0][1][transient:, 0, :, 0].T
        regionLabels = conn.region_labels

        # ROIs of interest to measure alpha peak increase #i
        # All occipito-parietal regins w/ 0-indexing in Python.
        # occipital_rois = ['Calcarine_L', 'Calcarine_R', 'Cuneus_L', 'Cuneus_R',
        #                   'Lingual_L', 'Lingual_R', 'Occipital_Sup_L', 'Occipital_Sup_R',
        #                   'Occipital_Mid_L', 'Occipital_Mid_R', 'Occipital_Inf_L',
        #                   'Occipital_Inf_R', 'Parietal_Sup_L', 'Parietal_Sup_R',
        #                   'Parietal_Inf_L', 'Parietal_Inf_R', 'SupraMarginal_L',
        #                   'SupraMarginal_R', 'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R']
        #
        # if "cb" in mode:
        #     occ_cb = list(set(occipital_rois).intersection(set(cingulum_rois)))
        #     rois = [list(conn.region_labels).index(roi) for roi in occ_cb]
        # else:
        #     rois = [SClabs.index(roi) for roi in occipital_rois]

        ## ROIs implicated in the empirical effect @ 26/04/2022
        empCluster_rois = ['Precentral_L', 'Frontal_Sup_2_L', 'Frontal_Sup_2_R', 'Frontal_Mid_2_L',
                           'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L', 'Frontal_Inf_Tri_R',
                           'Frontal_Inf_Orb_2_L', 'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Frontal_Sup_Medial_L',
                           'Frontal_Sup_Medial_R', 'Rectus_L', 'OFCmed_L', 'Insula_L', 'Insula_R', 'Cingulate_Ant_L', 'Cingulate_Ant_R',
                           'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L', 'ParaHippocampal_R',
                           'Amygdala_L', 'Calcarine_L', 'Calcarine_R', 'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R',
                           'Occipital_Sup_R', 'Occipital_Mid_L', 'Occipital_Mid_R', 'Occipital_Inf_L',
                           'Occipital_Inf_R', 'Fusiform_L', 'Fusiform_R', 'Postcentral_L', 'Parietal_Sup_R',
                           'Parietal_Inf_R', 'Angular_R', 'Precuneus_R', 'Temporal_Sup_L',
                           'Temporal_Sup_R', 'Temporal_Pole_Sup_L', 'Temporal_Pole_Sup_R', 'Temporal_Mid_L',
                           'Temporal_Mid_R', 'Temporal_Pole_Mid_L', 'Temporal_Inf_L', 'Temporal_Inf_R']

        if "cb" in mode:
            occ_cb = list(set(empCluster_rois).intersection(set(cingulum_rois)))
            rois = [list(conn.region_labels).index(roi) for roi in empCluster_rois]
        else:
            rois = [SClabs.index(roi) for roi in empCluster_rois]

        ## Calculate IAF band power rise
        if w != 0:
            fft_peaks_hzAAL, fft_peaks_modulesAAL, fft_iafband_modulesAAL = \
                FFTpeaks(raw_data[rois], simLength - transient, initialPeak)
        else:
            fft_peaks_hzAAL, fft_peaks_modulesAAL, fft_iafband_modulesAAL = \
                FFTpeaks(raw_data[rois], simLength - transient)

        ## Gather results
        datapoints.append((emp_subj, mode, g, s, r, w,
                           np.average(fft_peaks_hzAAL),
                           np.average(fft_peaks_modulesAAL),
                           np.average(fft_iafband_modulesAAL)))

        print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))

    return np.asarray(datapoints, dtype=object)
