
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
import glob

figures_folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER\FIGURES\\"


######################## STRUCTURAL VULNERABILITY TO STIMULATION

######### Network analysis
# ¿Qué metricas incluyo? n paths total para cada roi y avg, porcentaje del total,
# media de tractos que entran por cada conexion, indegree, outdegree, etc.
from tvb.datatypes import connectivity
import networkx as nx

## Working over the averaged matrices
# Load structures
conn = connectivity.Connectivity.from_file("E://LCCN_Local/PycharmProjects/CTB_dataOLD2/NEMOS_AVG_AAL2_pass.zip")

matrix = conn.weights

## Add up cerebellum weights to do single nodes plots
cer_rois = []
[cer_rois.append(roi) if "Cer" in roi else None for roi in conn.region_labels]
cer_ids = [list(conn.region_labels).index(roi) for roi in cer_rois]

ver_rois = []
[ver_rois.append(roi) if "Ver" in roi else None for roi in conn.region_labels]
ver_ids = [list(conn.region_labels).index(roi) for roi in ver_rois]


# sum cols, sum rows, delete excedent
matrix_single_nodes = matrix
# Left hemisphere
matrix_single_nodes[cer_ids[0], :] = np.sum(matrix[cer_ids[0::2], :], axis=0)
matrix_single_nodes[:, cer_ids[0]] = np.sum(matrix[:, cer_ids[0::2]], axis=1)

# Right hemisphere
matrix_single_nodes[cer_ids[1], :] = np.sum(matrix[cer_ids[1::2], :], axis=0)
matrix_single_nodes[:, cer_ids[1]] = np.sum(matrix[:, cer_ids[1::2]], axis=1)

# Vermis
matrix_single_nodes[ver_ids[0], :] = np.sum(matrix[ver_ids, :], axis=0)
matrix_single_nodes[:, ver_ids[0]] = np.sum(matrix[:, ver_ids], axis=1)


matrix_single_nodes = np.delete(matrix_single_nodes, ver_ids[1:], axis=0)
matrix_single_nodes = np.delete(matrix_single_nodes, ver_ids[1:], axis=1)
matrix_single_nodes = np.delete(matrix_single_nodes, cer_ids[2:], axis=0)
matrix_single_nodes = np.delete(matrix_single_nodes, cer_ids[2:], axis=1)


np.fill_diagonal(matrix_single_nodes, 0)

# region labels of 120 menos las del cer
regionLabels = []
[regionLabels.append(roi) if roi not in cer_rois+ver_rois else None for roi in conn.region_labels]
regionLabels.append("Cerebellum_L")
regionLabels.append("Cerebellum_R")
regionLabels.append("Vermis")

# Convert matrices to adj matrices
net = nx.convert_matrix.from_numpy_array(np.asarray(matrix_single_nodes))
    # This generates an undirected graph (Graph). Not a directed graph (DiGraph).


# label mapping
mapping = {i: roi for i, roi in enumerate(regionLabels)}
net = nx.relabel_nodes(net, mapping)

### NETWORK METRICS  # Compute metrics of interest for all nodes: append to dataframe

## Centrality
# 1. Degree normalized
degree = pd.DataFrame.from_dict(nx.degree_centrality(net), orient="index", columns=["degree"])


# 2. Node strength normalized
node_strength_norm = pd.DataFrame.from_dict({node: val/matrix_single_nodes.sum(axis=1).max()
                                        for (node, val) in net.degree(weight="weight")},
                                       orient="index", columns=["node_strength_norm"])

# 2b. Node strength
node_strength = pd.DataFrame.from_dict({node: round(val, 4)
                                        for (node, val) in net.degree(weight="weight")},
                                       orient="index", columns=["node_strength"])

# Specific connectivity Pre-ACC
matrix_single_nodes[regionLabels.index("Precuneus_L"):regionLabels.index("Precuneus_R")+1, regionLabels.index("Cingulate_Ant_L"):regionLabels.index("Cingulate_Ant_R")+1]
sum(sum(matrix_single_nodes[regionLabels.index("Precuneus_L"):regionLabels.index("Precuneus_R")+1, regionLabels.index("Cingulate_Ant_L"):regionLabels.index("Cingulate_Ant_R")+1]))

# 3. Closeness
closeness = pd.DataFrame.from_dict(nx.closeness_centrality(net), orient="index", columns=["closeness"])

# 4. Betweeness
betweeness = pd.DataFrame.from_dict(nx.betweenness_centrality(net), orient="index", columns=["betweeness"])


## Global Integration
# 5. Path length
path_length = pd.DataFrame.from_dict({source: np.average(list(paths.values()))
                                      for source, paths in nx.shortest_path_length(net)},
                                     orient="index", columns=["path_length"])
np.std(path_length.values)
nx.average_shortest_path_length(net)

# Specific path length ACC-Pre
nx.shortest_path_length(net, source="Precuneus_L", target="Cingulate_Ant_L")
nx.shortest_path_length(net, source="Precuneus_R", target="Cingulate_Ant_L")
nx.shortest_path_length(net, source="Precuneus_L", target="Cingulate_Ant_R")
nx.shortest_path_length(net, source="Precuneus_R", target="Cingulate_Ant_R")


## Local Segregation
# 6. Clustering coefficient
clustering = pd.DataFrame.from_dict(nx.clustering(net), orient="index", columns=["clustering"])
np.std(clustering.values)
nx.average_clustering(net)

# 7. Modularity (Newman approach)
comms = nx.community.greedy_modularity_communities(net)
nx.community.modularity(net, comms)


#### Load VBM results to standardize node strengths (number of paths) by structure size
vbm = np.loadtxt("E:\OneDrive - Universidad Complutense de Madrid (UCM)\LNCC\LCCN _data\AVG_NEMOS\\anat\VoxelBasedMorphometry_AAL2fromAAL3_NEMOS_AVG.txt")
vbm_red = vbm.copy()

# Sum up cerebellum and vermis
# Left hemisphere
vbm_red[94] = np.sum(vbm[cer_ids[0::2]], axis=0)
# Right hemisphere
vbm_red[95] = np.sum(vbm[cer_ids[1::2]], axis=0)
# Vermis
vbm_red[96] = np.sum(vbm[ver_ids], axis=0)

vbm_red = np.delete(vbm_red, np.arange(97, 120, 1))
vbm_red_df = pd.DataFrame(vbm_red, columns=["vbm"], index=regionLabels)


#### Gatering Results
network_analysis = pd.concat([degree, node_strength_norm, node_strength, closeness, betweeness, clustering, path_length, vbm_red_df], axis=1).reindex(degree.index)

network_analysis["node_strength/vbm"] = [(node_strength_norm.values[i] / vbm_red_df.values[i])[0] for i, roi in enumerate(regionLabels)]

# Subset dataframe
network_analysis_wide_l = network_analysis[0:-1:2]
network_analysis_wide_r = network_analysis[1::2]

# Add averages
columns = ["degree", "node_strength_norm", "node_strength", "closeness", "betweeness", "clustering", "path_length", "vbm", "node_strength/vbm"]
network_analysis_wide_l = network_analysis_wide_l.append(
    pd.DataFrame([np.average(network_analysis_wide_l, axis=0)], columns=columns))
network_analysis_wide_r = network_analysis_wide_r.append(
    pd.DataFrame([np.average(network_analysis_wide_r, axis=0)], columns=columns))

# relabel Indexes
network_analysis_wide_l.index = [label[:-2] for label in network_analysis_wide_l.index[:-1]] + ["Average"]
network_analysis_wide_r.index = [label[:-2] for label in network_analysis_wide_r.index[:-1]] + ["Average"]

network_analysis_wide_avg = pd.concat((network_analysis_wide_l, network_analysis_wide_r))
network_analysis_wide_avg = network_analysis_wide_avg.groupby(network_analysis_wide_avg.index).mean()

# Rename columns
network_analysis_wide_l.columns = [col + "_l" for col in columns]
network_analysis_wide_r.columns = [col + "_r" for col in columns]
network_analysis_wide_avg.columns = [col + "_avg" for col in columns]

# join dataframes
network_analysis_wide = network_analysis_wide_l.join(network_analysis_wide_r)
network_analysis_wide = network_analysis_wide.join(network_analysis_wide_avg)


#### PLOT centrality measures
color_sub = ["darkslategray", "steelblue"]
color_all = ["lightslategray", "lightsteelblue"]

network_analysis_wide["color_l"] = [color_all[0]] * len(network_analysis_wide[:-1]) + [color_sub[0]]
network_analysis_wide["color_l"][["Precuneus", "Cingulate_Ant"]] = [color_sub[0]] * 2
network_analysis_wide["color_r"] = [color_all[1]] * len(network_analysis_wide[:-1]) + [color_sub[1]]
network_analysis_wide["color_r"][["Precuneus", "Cingulate_Ant"]] = [color_sub[1]] * 2

fig = make_subplots(rows=1, cols=3, column_titles=("Node Strength", "Betweenness", "Path Length"),
                    horizontal_spacing=0.15, specs=[[{}, {}, {}]])


# temp = network_analysis_wide.sort_values(by="node_strength/vbm_avg")
# fig.add_trace(go.Bar(x=-temp["node_strength/vbm_l"].values, y=temp.index, orientation='h', marker=dict(color=temp.color_l.values), showlegend=False), row=1, col=1)
# fig.add_trace(go.Bar(x=temp["node_strength/vbm_r"].values, y=temp.index, orientation='h', marker=dict(color=temp.color_r.values), showlegend=False), row=1, col=1)

# temp = network_analysis_wide.sort_values(by="degree_avg")
# fig.add_trace(go.Bar(x=-temp.degree_l.values, y=temp.index, orientation='h', marker=dict(color=temp.color_l.values), showlegend=False), row=1, col=1)
# fig.add_trace(go.Bar(x=temp.degree_r.values, y=temp.index, orientation='h', marker=dict(color=temp.color_r.values), showlegend=False), row=1, col=1)

temp = network_analysis_wide.sort_values(by="node_strength_norm_avg")
fig.add_trace(go.Bar(x=-temp.node_strength_norm_l.values, y=temp.index, orientation='h', marker=dict(color=temp.color_l.values),
                     customdata=temp.node_strength_l.values, hovertemplate="%{x}, %{y} <br> Tracts: %{customdata}", showlegend=False), row=1, col=1)
fig.add_trace(go.Bar(x=temp.node_strength_norm_r.values, y=temp.index, orientation='h', marker=dict(color=temp.color_r.values),
                     customdata=temp.node_strength_l.values, hovertemplate="%{x}, %{y} <br> Tracts: %{customdata}", showlegend=False), row=1, col=1)

temp = network_analysis_wide.sort_values(by="betweeness_avg")
fig.add_trace(go.Bar(x=-temp.betweeness_l.values, y=temp.index, orientation='h', marker=dict(color=temp.color_l.values), showlegend=False), row=1, col=2)
fig.add_trace(go.Bar(x=temp.betweeness_r.values, y=temp.index, orientation='h', marker=dict(color=temp.color_r.values), showlegend=False), row=1, col=2)

# temp = network_analysis_wide.sort_values(by="closeness_avg")
# fig.add_trace(go.Bar(x=-temp.closeness_l.values, y=temp.index, orientation='h', marker=dict(color=temp.color_l.values), showlegend=False), row=1, col=3)
# fig.add_trace(go.Bar(x=temp.closeness_r.values, y=temp.index, orientation='h', marker=dict(color=temp.color_r.values), showlegend=False), row=1, col=3)

temp = network_analysis_wide.sort_values(by="path_length_avg", ascending=False)
fig.add_trace(go.Bar(x=-temp.path_length_l.values, y=temp.index, orientation='h', marker=dict(color=temp.color_l.values), showlegend=False), row=1, col=3)
fig.add_trace(go.Bar(x=temp.path_length_r.values, y=temp.index, orientation='h', marker=dict(color=temp.color_r.values), showlegend=False), row=1, col=3)

fig.update_layout(template="plotly_white", barmode="relative", height=1000, width=1000)
pio.write_html(fig, file=figures_folder + '/Network_metrics_pass.html', auto_open=True)
pio.write_image(fig, file=figures_folder + '/Network_metrics_pass.svg')


