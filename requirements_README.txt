tvb-library # ~=2.0.5
plotly # ~=4.14.3
mne
numpy
matplotlib # ~=3.2.1
scipy
pandas>=1.0.3
unidip~=0.1.1
jupyter
scikit-learn
statsmodels
mpi4py
chord

## REMEMBER (for stimulation): I'm using a modified version of "tvb/datatypes/equations.py".
# Ask for it @Jescab01 - "/.Research/simulation tools/brainModels" and replace it in site-packages

## ON DEVELOPMENT: toolbox is by now a package in Local - to import its functions use:
# import sys
# sys.path.append("E:\\LCCN_Local\PycharmProjects\\")  # temporal append
# from toolbox.signals import timeseriesPlot
# from toolbox.fft import FFTplot, FFTpeaks

# Check version:  pip freeze | findstr plotly
#   or into python: import module; module.__version__.
# install manually tvb data; download from: https://zenodo.org/record/4263723#.YG8SzOgzaHs
