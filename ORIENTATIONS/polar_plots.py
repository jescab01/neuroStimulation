
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

orient_folder = 'D:\\Users\Jesus CabreraAlvarez\OneDrive - Universidad Complutense de Madrid (UCM)\LNCC\LCCN _data\\NEMOS_035\.roast\\roast_ORIENTATIONS_ACCtarget\\'

orientations = {'radial-in':'in','radial-out':'out','anterior':90,  'right-anterior':45,'right': 0,
                'right-posterior':315, 'posterior':270,'left-posterior':225,'left': 180,'left-anterior':135}

# load labels
roi_labels = list(np.loadtxt('D:\\Users\Jesus CabreraAlvarez\OneDrive - Universidad Complutense de Madrid (UCM)\LNCC\LCCN _pipelines\centresAAL.txt', dtype=str)[:,0])


# What ROIS do I want to plot?
rois = ['Frontal_Sup_L', 'Frontal_Sup_R','Insula_L', 'Insula_R', 'Cingulum_Ant_L', 'Cingulum_Ant_R',
        'Cingulum_Post_L','Cingulum_Post_R', 'Occipital_Mid_L', 'Occipital_Mid_R',  'Parietal_Sup_L', 'Parietal_Sup_R',
        'Precuneus_L', 'Precuneus_R', 'Temporal_Sup_L', 'Temporal_Sup_R']

roi_idx = [ roi_labels.index(roi) for roi in rois ]

df_lh=[]
df_rh=[]

for orient, degree in orientations.items():

    efnorm_mag = np.loadtxt(orient_folder + 'roast_ACCtarget_' + orient +'/orthogonalization_v2/NEMOS-035-efnorm_mag-AAL.txt', delimiter=',')
    ef_mag = np.loadtxt(orient_folder + 'roast_ACCtarget_' + orient +'/orthogonalization_v2/NEMOS-035-ef_mag-AAL.txt', delimiter=',')

    df_lh = df_lh + [[orient, degree, roi_labels[roi], efnorm_mag[roi], ef_mag[roi]] for roi in np.asarray(roi_idx)[::2]]
    df_rh = df_rh + [[orient, degree, roi_labels[roi], efnorm_mag[roi], ef_mag[roi]] for roi in np.asarray(roi_idx)[1::2]]


df_lh = pd.DataFrame(df_lh, columns=['orient', 'degree', 'roi', 'efnorm_mag', 'ef_mag'])
df_rh = pd.DataFrame(df_rh, columns=['orient', 'degree', 'roi', 'efnorm_mag', 'ef_mag'])



# fig_lh = px.line_polar(df_lh, r='efnorm_mag', theta='orient', color='roi', line_close=True)
# fig_lh.show(renderer='browser')
# fig_rh = px.line_polar(df_rh, r='efnorm_mag', theta='orient', color='roi', line_close=True)
# fig_rh.show(renderer='browser')

# PLOTTING
color_palette = px.colors.qualitative.Plotly

fig = make_subplots(rows=2, cols=4, specs=[[{'type': 'polar'}, {}]*2]*2,
                    column_titles=("", "left hemisphere _radial in|out", "", "Right hemisphere _radial in|out"),
                    row_titles=('efnorm_mag', 'ef_mag'))

for color, roi in enumerate(rois):

    color = int(str(color)[-1])

    # left HEMISFIER
    temp = df_lh.loc[(df_lh['roi']==roi) & (df_lh['orient']!='radial-in') & (df_lh['orient']!='radial-out')]
    temp = temp.append(temp.loc[temp['orient'] == 'anterior']) # to close lines

    #   polarplots: efmag efnormmag
    fig.add_trace(go.Scatterpolar(r=np.asarray(temp['efnorm_mag']), theta=np.asarray(temp['orient']), marker_color=color_palette[color], mode='lines', name=roi, legendgroup=roi), 1, 1)
    fig.add_trace(go.Scatterpolar(r=np.asarray(temp['ef_mag']), theta=np.asarray(temp['orient']), marker_color=color_palette[color], mode='lines', name=roi, legendgroup=roi, showlegend=False), 2, 1)

    #   barplots: efmag efnormmag
    temp = df_lh.loc[(df_lh['roi']==roi) & (df_lh['orient']=='radial-in') | (df_lh['roi']==roi) & (df_lh['orient']=='radial-out')]
    fig.add_trace(go.Bar(x=temp.orient, y=temp.efnorm_mag, marker_color=color_palette[color], name=roi, legendgroup=roi, showlegend=False),1,2)
    fig.add_trace(go.Bar(x=temp.orient, y=temp.ef_mag, marker_color=color_palette[color], name=roi, legendgroup=roi, showlegend=False),2,2)

    # right HEMISFIER
    temp = df_rh.loc[(df_rh['roi']==roi) & (df_rh['orient']!='radial-in') & (df_rh['orient']!='radial-out')]
    temp = temp.append(temp.loc[temp['orient']=='anterior'])

    #   polarplots: efmag efnormmag
    fig.add_trace(go.Scatterpolar(r=np.asarray(temp['efnorm_mag']), theta=np.asarray(temp['orient']), marker_color=color_palette[color], mode='lines', name=roi, legendgroup=roi), 1, 3)
    fig.add_trace(go.Scatterpolar(r=np.asarray(temp['ef_mag']), theta=np.asarray(temp['orient']), marker_color=color_palette[color], mode='lines', name=roi, legendgroup=roi, showlegend=False), 2, 3)

    #   barplots: efmag efnormmag
    temp = df_rh.loc[(df_rh['roi']==roi) & (df_rh['orient']=='radial-in') | (df_rh['roi']==roi) & (df_rh['orient']=='radial-out')]
    fig.add_trace(go.Bar(x=temp.orient, y=temp.efnorm_mag, marker_color=color_palette[color], name=roi, legendgroup=roi, showlegend=False),1,4)
    fig.add_trace(go.Bar(x=temp.orient, y=temp.ef_mag, marker_color=color_palette[color], name=roi, legendgroup=roi, showlegend=False),2,4)

## Add dashed reference at 0
temp = df_rh.loc[(df_rh['roi']==roi) & (df_rh['orient']!='radial-in') & (df_rh['orient']!='radial-out')]
temp = temp.append(temp.loc[temp['orient']=='anterior'])

fig.add_trace(go.Scatterpolar(r=np.array([0]*len(temp)), theta=np.asarray(temp['orient']),  marker_color='gray', mode='markers', name='ref', legendgroup='ref'), 1, 1)
fig.add_trace(go.Scatterpolar(r=np.array([0]*len(temp)), theta=np.asarray(temp['orient']),  marker_color='gray', mode='markers', showlegend=False, legendgroup='ref'), 1, 3)
fig.add_trace(go.Scatterpolar(r=np.array([0]*len(temp)), theta=np.asarray(temp['orient']),  marker_color='gray', mode='markers', showlegend=False, legendgroup='ref'), 2, 1)
fig.add_trace(go.Scatterpolar(r=np.array([0]*len(temp)), theta=np.asarray(temp['orient']), marker_color='gray', mode='markers', showlegend=False, legendgroup='ref'), 2, 3)

fig.update_layout(title='Electric field magnitude by field orientation :: targeting ACC', barmode='group',
                  polar=dict(radialaxis_range=[-0.1, 0.1], angularaxis=dict(direction='clockwise')),
                  polar2=dict(radialaxis_range=[-0.1, 0.1], angularaxis=dict(direction='clockwise')),
                  polar3=dict(radialaxis_range=[-0.2, 0.2], angularaxis=dict(direction='clockwise')),
                  polar4=dict(radialaxis_range=[-0.2, 0.2], angularaxis=dict(direction='clockwise')))

pio.write_html(fig, file="ORIENTATIONS/figures/ef_by_orientation_targetACC.html")



