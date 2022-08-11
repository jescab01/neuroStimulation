import plotly.express as px
import numpy as np
import scipy.io
import pandas as pd

mat = scipy.io.loadmat('E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER\emp_power_bw2_occpar_ACS.mat')["power"][0, 0]


all_pre_mean = mat["pre_mean"][:, 0]
all_post_mean = mat["pos_mean"][:, 0]
bool_tacs = mat["tACS"][:, 0]

labs_subj = ["Subject " + str(i+1).zfill(2) for i in range(len(all_pre_mean))]
labs_pre_mean = ["pre" for i in range(len(all_pre_mean))]
labs_post_mean = ["post" for i in range(len(all_pre_mean))]
labs_cond = ["Stimulated" if b == 1 else "Control"for b in bool_tacs]

df = pd.DataFrame(np.asarray([labs_subj, labs_pre_mean, all_pre_mean, bool_tacs, labs_cond]).T,
                  columns=["subject", "stage", "power", "bool_cond", "condition"]).append(
    pd.DataFrame(np.asarray([labs_subj, labs_post_mean, all_post_mean, bool_tacs, labs_cond]).T,
                            columns=["subject", "stage", "power", "bool_cond", "condition"]))

df_diffs = pd.DataFrame(np.asarray([labs_subj, all_post_mean-all_pre_mean, bool_tacs, labs_cond]).T,
                  columns=["subject", "power_diff", "bool_cond", "condition"])
df_diffs = df_diffs.astype({"power_diff": float})

fig = px.violin(df_diffs, x="condition", y="power_diff", points="all")
fig.show(renderer="browser")

df_diffs.groupby("condition").mean()