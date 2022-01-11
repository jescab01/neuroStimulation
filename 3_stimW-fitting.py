from tvb.simulator.lab import *
import tvb.datatypes.projections as projections
from toolbox import timeseriesPlot, FFTplot, FFTpeaks
import numpy as np
import pandas as pd
import time
import plotly.express as px
import plotly.io as pio

# Choose a name for your simulation and define the empirical for SC
model_id = ".1995JansenRit_NEMOS"

# Structuring directory to organize outputs
wd = os.getcwd()
main_folder = wd + "\\" + "PSE"
specific_folder = main_folder + "\\PSE_fittingW" + model_id + "-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
os.mkdir(specific_folder)

LCCN_folder = "D:\\Users\Jesus CabreraAlvarez\OneDrive - Universidad Complutense de Madrid (UCM)\LNCC\LCCN _data\\"
ctb_folder = "D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\\CTB_data\\output\\"

simLength = 10 * 1000  # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1000  # Hz
transient = 1000

n_simulations = 50

working_points = [('NEMOS_035', 65, 11.5),
                  ('NEMOS_049', 19, 18.5),
                  ('NEMOS_050', 38, 22.5),
                  ('NEMOS_058', 37, 16.5),
                  ('NEMOS_059', 35, 12.5),
                  ('NEMOS_064', 54, 6.5),
                  ('NEMOS_065', 47, 22.5),
                  ('NEMOS_071', 81, 6.5),
                  ('NEMOS_075', 14, 5.5),
                  ('NEMOS_077', 20, 20.5)]

for subj, g, s in working_points:

    ## STRUCTURE
    conn = connectivity.Connectivity.from_file(
        ctb_folder + subj + "_AAL2red.zip")  # ctb_folder + "AVG_NEMOS_AAL2red.zip"
    conn.weights = conn.scaled_weights(mode="tract")

    ## MODEL
    # Parameters from Stefanovski 2019.
    m = models.JansenRit(A=np.array([3.25]), B=np.array([22]), J=np.array([1]),
                         a=np.array([0.1]), a_1=np.array([135]), a_2=np.array([108]),
                         a_3=np.array([33.75]), a_4=np.array([33.75]), b=np.array([0.06]),
                         mu=np.array([0.1085]), nu_max=np.array([0.0025]), p_max=np.array([0]), p_min=np.array([0]),
                         r=np.array([0.56]), v0=np.array([6]))

    coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                       r=np.array([0.56]))
    conn.speed = np.array([s])

    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

    # region mapping optional for the surface simulation
    # rm = region_mapping.RegionMapping.from_file(LCCN_folder+"region_mapping_N35_AAL2red.txt")

    # Projections: Automatically done by monitor when provided region mapping
    # All gain matrices in tvb are computed for the surface.
    #  If simulating on surface, can immediately project from source to sensor
    #  if region source space, but gain matrix is for surface, you first need to
    #   reduce the gain matrix to a regional version >> this is automatically done by the monitor.
    #   https://groups.google.com/u/1/g/tvb-users/c/pEVSwMbKdp0/m/5ZPHcv_CAwAJ
    # pr = projections.ProjectionSurfaceMEG.from_file(LCCN_folder+"avg_projection_N35_AAL2red_MEG306.txt")    #"avg_projection_N35_AAL2red_MEG306.txt"
    # ss = sensors.SensorsMEG.from_file(LCCN_folder+"sensors_MEG306.txt")
    mon = (monitors.Raw(),)  # monitors.MEG(projection=pr, sensors=ss, region_mapping=rm, period=1))

    # Run simulation
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
    sim.configure()
    output = sim.run(simulation_length=simLength)
    # Extract data cutting initial transient
    raw_data = output[0][1][transient:, 0, :, 0].T
    raw_time = output[0][0][transient:]
    regionLabels = conn.region_labels

    # regionLabels = list(regionLabels) - If monitors include MEG
    # regionLabels.insert(5, "stimulus")

    # average signals to obtain mean signal frequency peak
    # data = np.concatenate((raw_data, stimulus.temporal_pattern), axis=0)  # concatenate mean signal: data[0]; with raw_data: data[1:end]
    # # Check initial transient and cut data
    # timeseriesPlot(raw_data, raw_time, regionLabels, title="raw", mode="html")
    #
    # # Fourier Analysis plot
    # FFTplot(raw_data, simLength-transient, regionLabels,  mode="html")

    # fft_peaks = FFTpeaks(raw_data, simLength-transient)[:, 0]

    # Applying FORWARD MODEL : sensors timeseries
    # meg = np.dot(pr.projection_data, raw_data)

    # timeseriesPlot(meg, raw_time, ss.labels, title="MEG", mode="html")
    # FFTplot(meg, simLength-transient, ss.labels, mode="html")

    # Channels of interest ~POZ in MEG
    # channels = ["MEG2011", "MEG2012", "MEG2013", "MEG2021", "MEG2022", "MEG2023", "MEG2041", "MEG2042", "MEG2043",
    #              "MEG2031", "MEG2032", "MEG2033"]
    # channels = ["MEG2041", "MEG2042", "MEG2043"]
    # channels = ["MEG1921", "MEG1922", "MEG1923"] # check over occipital
    # ch_ids = [int(np.where(ss.labels == ch)[0]) for ch in channels]
    #
    # fft_peaks_hz = FFTpeaks(meg, simLength - transient)[0][ch_ids]
    # fft_peaks_modules = FFTpeaks(meg, simLength - transient)[1][ch_ids]

    # ROIs of interest to measure alpha peak increase #i

    rois = [62, 63, 64, 65, 70, 71]  # Parietal complex (sup [63,64] & inf [65,66] parietal) + precuneus [71,72]. 0-indexing in Python.
    fft_peaks_hzAAL = FFTpeaks(raw_data, simLength - transient)[0][rois]
    fft_peaks_modulesAAL = FFTpeaks(raw_data, simLength - transient)[1][rois]

    resultsAAL = pd.DataFrame.from_dict(
        {"w": [0], "peak_hz": [np.average(fft_peaks_hzAAL)], "peak_module": [np.average(fft_peaks_modulesAAL)]})
    # resultsMEG=pd.DataFrame.from_dict({"w": [0], "peak_hz": [np.average(fft_peaks_hz)], "peak_module": [np.average(fft_peaks_modules)]})

    initialPeak = np.average(fft_peaks_hzAAL)


    ## Loop over weights to get a 14% raise in alpha peak
    weights = np.concatenate((np.array([0]), np.linspace(g/250, g/25, 20)))

    ##### STIMULUS

    for w in weights:
        for r in range(n_simulations):
            tic0 = time.time()
            ## Sinusoid input
            eqn_t = equations.Sinusoid()
            eqn_t.parameters['amp'] = 1  # Amplitud diferencial por áreas ajustada en stimWeights
            eqn_t.parameters['frequency'] = initialPeak  # Hz
            eqn_t.parameters['onset'] = 0  # ms
            eqn_t.parameters['offset'] = 10000  # ms

            # weighting = np.zeros((len(conn.weights), ))
            # first value: what arrives to cortex mA
            # second value: influence on brain activity
            # weighting[[55]] = 0.015 * w

            # electric field * orthogonal to surface
            weighting = np.loadtxt(ctb_folder + 'orthogonals/' + emp_subj + '-roast_OzCzModel_ef_mag-AAL2red.txt') * w

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
            sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon,
                                      stimulus=stimulus)
            sim.configure()
            output = sim.run(simulation_length=simLength)
            # Extract data cutting initial transient
            raw_data = output[0][1][transient:, 0, :, 0].T
            raw_time = output[0][0][transient:]
            regionLabels = conn.region_labels

            # Applying FORWARD MODEL : sensors timeseries
            # meg = np.dot(pr.projection_data, raw_data)

            # FFTplot(raw_data, simLength-transient, ss.labels, mode="html")
            # FFTplot(meg, simLength-transient, ss.labels, mode="html")

            # fft_peaks_hz = FFTpeaks(meg, simLength-transient)[0][ch_ids]
            # fft_peaks_modules = FFTpeaks(meg, simLength-transient)[1][ch_ids]

            # ROIs of interest to measure alpha peak increase
            fft_peaks_hzAAL = FFTpeaks(raw_data, simLength - transient)[0][rois]
            fft_peaks_modulesAAL = FFTpeaks(raw_data, simLength - transient)[1][rois]

            resultsAAL = resultsAAL.append(
                {"w": w, "peak_hz": np.average(fft_peaks_hzAAL), "peak_module": np.average(fft_peaks_modulesAAL)},
                ignore_index=True)
            # resultsMEG=resultsMEG.append({"w":w, "peak_hz":np.average(fft_peaks_hz), "peak_module":np.average(fft_peaks_modules)}, ignore_index=True)
            print("w = %0.2f - round = %i" % (w, r))
            print("LOOP ROUND REQUIRED %0.4f seconds.\n\n\n\n" % (time.time() - tic0,))

    # calculate percentages
    resultsAAL_avg = resultsAAL.groupby('w').mean()
    # resultsMEG_avg=resultsMEG.groupby('w').mean()

    resultsAAL["percent"] = [
        ((resultsAAL.peak_module[i] - resultsAAL_avg.peak_module[0]) / resultsAAL_avg.peak_module[0]) * 100 for i in
        range(len(resultsAAL))]
    # resultsMEG["percent"]=[((resultsMEG.peak_module[i]-resultsMEG.peak_module[0])/resultsMEG.peak_module[0])*100 for i in range(len(resultsMEG))]

    resultsAAL_avg = resultsAAL.groupby('w').mean()
    resultsAAL_avg["sd"] = resultsAAL.groupby('w')[['w', 'peak_module']].std()
    # resultsMEG_avg=resultsMEG.groupby('w').mean()

    #### Save results
    resultsAAL.to_csv(specific_folder + "\\"+subj+"_AAL2red-ParietalComplex_alphaRise.csv")
    # resultsMEG.to_csv(specific_folder+"\\MEG1921_alphaRise.csv")

    # resultsAAL=pd.read_csv("stimulationCollab/AAL_aument.csv")
    # resultsMEG=pd.read_csv("stimulationCollab/MEG_aument.csv")

    fig = px.box(resultsAAL, x="w", y="peak_module",
                 title="Alpha peak module rise @ParietalComplex<br>(%i simulations | %s AAL2red)" % (n_simulations, subj),
                 labels={  # replaces default labels by column name
                     "w": "Weight", "peak_module": "Alpha peak module"},
                 template="plotly")
    pio.write_html(fig, file=specific_folder + "/"+subj+"AAL_alphaRise_modules_" + str(n_simulations) + "sim.html",
                   auto_open=False)

    fig = px.box(resultsAAL, x="w", y="percent",
                 title="Alpha peak module rise @ParietalComplex<br>(%i simulations | %s AAL2red)" % (n_simulations, subj),
                 labels={  # replaces default labels by column name
                     "w": "Weight", "percent": "Percentage of alpha peak rise"},
                 template="plotly")
    pio.write_html(fig, file=specific_folder + "/"+subj+"AAL_alphaRise_percent_" + str(n_simulations) + "sim.html",
                   auto_open=True)

    # con MEG no sale nada, algo estaré haciendo mal al extraer la señal MEG.
    # fig = px.box(resultsMEG, x="w", y="peak_module",
    #              title="Alpha peak module rise<br>(%i simulations | MEG)" % n_simulations,
    #              labels={  # replaces default labels by column name
    #                  "w": "Weight", "peak_module": "Alpha peak module"},
    #              template="plotly")
    # pio.write_html(fig, file=specific_folder + "/MEG_alphaRise_modules_" + str(n_simulations) + "s.html", auto_open=True)
    #
    # fig = px.box(resultsMEG, x="w", y="percent",
    #              title="Alpha peak module rise<br>(%i simulations | MEG)" % n_simulations,
    #              labels={  # replaces default labels by column name
    #                  "w": "Weight", "percent": "Percentage: Alpha peak rise"},
    #              template="plotly")
    # pio.write_html(fig, file=specific_folder + "/MEG_alphaRise_percent_" + str(n_simulations) + "s.html", auto_open=True)

    # # Fourier Analysis plot
    # FFTplot(raw_data, simLength-transient, regionLabels,  mode="html")
