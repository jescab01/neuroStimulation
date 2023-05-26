
import numpy as np

## 1. Define rois
# Este va a ser el orden de rois que vamos a usar siempre.
# Adaptaremos las matrices empiricas a este orden.

cingulum_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                 'Insula_L', 'Insula_R',
                 'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                 'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                 'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R',
                 'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                 'Parietal_Inf_R', 'Precuneus_L', 'Precuneus_R',
                 'Thalamus_L', 'Thalamus_R']


## 2. Load labels from FC and SC files

FClabs = list(np.loadtxt(folder + "/FCrms_" + emp_subj + "/roi_labels_rms.txt", dtype=str))
SClabs = list(conn.region_labels)  # Yo lo cargo así, tu lo sacarías del zip


## 3.Sacar indices para adaptar las matrices empiricas a nuestro orden
FC_cb_idx = [FClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cortical_rois
SC_cb_idx = [SClabs.index(roi) for roi in cingulum_rois]  # find indexes in SClabs that matches cortical_rois


## 4. Adaptar la SC
conn.weights = conn.weights[:, SC_cb_idx][SC_cb_idx]
conn.tract_lengths = conn.tract_lengths[:, SC_cb_idx][SC_cb_idx]
conn.region_labels = conn.region_labels[SC_cb_idx]


## 5. Adaptar la FC empirica
plv_emp = np.loadtxt(folder + "FCrms_" + emp_subj + "/" + band + "_plv_rms.txt", delimiter=',')[:, FC_cb_idx][FC_cb_idx]

