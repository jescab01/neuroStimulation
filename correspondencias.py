
import pandas as pd
import numpy as np


cbpt_rois = pd.read_csv("E:\\OneDrive - Universidad Complutense de Madrid (UCM)\.Research\R - StimulationStudies _sim\R - StimulationStudies _emp\Experimental data - 26Abril2022\clusterROIs_info.csv", delimiter=";")

aal2_names = pd.read_csv("E:\\OneDrive - Universidad Complutense de Madrid (UCM)\LNCC\LCCN _pipelines\correspondenciasAAL2.csv", delimiter=";")
match51 = [roi for roi in cbpt_rois.Name.values if roi in aal2_names.namesAAL.values]
unmatch = [roi for roi in cbpt_rois.Name.values if roi not in match51]

myRois = [aal2_names.abbrevAAL2.values[i] for i, roi in enumerate(aal2_names.namesAAL.values) if roi in cbpt_rois.Name.values]