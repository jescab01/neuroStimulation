

import time
import numpy as np
import pandas as pd
import pickle
from tvb.simulator.lab import *

# Simulate baseline y stimulated for every subject, haz la media

emp_data = pd.read_csv("E:\OneDrive - Universidad Complutense de Madrid (UCM)\.Research\R - StimulationStudies _sim\R - StimulationStudies _emp\Experimental data - 26Abril2022\powerRise_allRois.txt", header=None)
emp_data.columns = "name", "avg_rise"

aal2_names = pd.read_csv("E:\\OneDrive - Universidad Complutense de Madrid (UCM)\LNCC\LCCN _pipelines\correspondenciasAAL2.csv", delimiter=";")
myRois = [aal2_names.abbrevAAL2.values[i] for i, roi in enumerate(aal2_names.namesAAL.values) if roi in emp_data.name.values]

with open("E:\LCCN_Local\PycharmProjects\\neuroStimulation\\1b_stim_OzCz\PSE"
          "\PSEmpi_stimOzCz_indWP-m05d06y2022-t19h.05m.59s\stimOzCz_results.pkl", 'rb') as f:
    sim_base, sim_stim = pickle.load(f)
    f.close()

sim_base = sim_base.groupby(["Subject"]).mean().reset_index()
sim_stim = sim_stim.groupby(["Subject"]).mean().reset_index()

sim_data = aal2_names[["abbrevAAL2", "Labels AAL2"]].copy()
sim_data.columns = "AbbrevAAL2", "bModuleRise"
sim_data["bModuleRise"] = sim_stim["bModule_2"].sub(sim_base["bModule_2"]).mean()

data = emp_data.dropna().copy()
data.columns = "name", "emp_bRise"

data["corr_AAL2"] = [aal2_names.abbrevAAL2.values[i] if roi in aal2_names.namesAAL.values else None for i, roi in enumerate(data.name.values)]
data["sim_bRise"] = [sim_data.loc[sim_data["AbbrevAAL2"] == roi, "bModuleRise"].values[0] if roi in sim_data.AbbrevAAL2.values else None for i, roi in enumerate(data.corr_AAL2.values) ]

data = data.dropna()

data.to_csv('dara_rise.csv')


import pingouin as pg
pg.corr(data.emp_bRise, data.sim_bRise)

import plotly.express as px
fig = px.scatter(data, "emp_bRise", "sim_bRise", trendline="ols")
fig.show(renderer="browser")

results = px.get_trendline_results(fig)
results.iloc[0]["px_fit_results"].summary()




## 3D Plots

import nibabel as nib

file = nib.load("E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER\c2MNI152_T1_1mm_ras_T1orT2.nii")



from PAPER.nii_2_mesh_conversion import nii_2_mesh

nii_2_mesh("E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER\c2MNI152_T1_1mm_ras_T1orT2.nii",
           "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER\c2MNI152_T1_Obj.stl", 46)

import vtk

# read the file
reader = vtk.vtkNIFTIImageReader()
reader.SetFileName("E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER\c2MNI152_T1_1mm_ras_T1orT2.nii")
reader.Update()

# apply marching cube surface generation
surf = vtk.vtkDiscreteMarchingCubes()
surf.SetInputConnection(reader.GetOutputPort())
surf.SetValue(0, 0.5)  # use surf.GenerateValues function if more than one contour is available in the file
surf.Update()

# smoothing the mesh
smoother = vtk.vtkWindowedSincPolyDataFilter()
if vtk.VTK_MAJOR_VERSION <= 5:
    smoother.SetInput(surf.GetOutput())
else:
    smoother.SetInputConnection(surf.GetOutputPort())
smoother.SetNumberOfIterations(30)
smoother.NonManifoldSmoothingOn()
smoother.NormalizeCoordinatesOn()  # The positions can be translated and scaled such that they fit within a range of [-1, 1] prior to the smoothing computation
smoother.GenerateErrorScalarsOn()
smoother.Update()

# save the output
writer = vtk.vtkSTLWriter()
writer.SetInputConnection(smoother.GetOutputPort())
writer.SetFileTypeToASCII()
writer.SetFileName("filename_stl.stl")
writer.Write()

