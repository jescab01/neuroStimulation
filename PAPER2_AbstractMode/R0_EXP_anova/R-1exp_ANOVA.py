
import pandas as pd
import pingouin as pg

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots



## Load results
folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\R0_EXP_anova\\"

data = pd.read_csv(folder + "Power_ParietoOcc_tACS_IAF_1.csv")

# From wide to long dataframe
df = pd.DataFrame([(row["IDMEG"], row["Condicion"], time[-4:], row[time])
                   for i, row in data.iterrows() for time in ["PowPre1", "PowPre2", "PowPost"]],
                  columns=["IDMEG", "condition", "time", "power"])
aov = pg.mixed_anova(data=df, dv="power", within="time", subject="IDMEG", between="condition")
aov

pg.plot_paired(df.loc[df["condition"] == "tACS_"], dv="power", within="time", subject="IDMEG", order=["Pre1", "Pre2", "Post"])
pg.plot_paired(df.loc[df["condition"] == "Sham"], dv="power", within="time", subject="IDMEG", order=["Pre1", "Pre2", "Post"])


# Try multiple comparisons
pg.pairwise_tests(data=df, dv="power", within="time", subject="IDMEG", between="condition")

# Separte the effects by condition
pg.pairwise_tests(df.loc[df["condition"] == "tACS_"], dv="power", within="time", subject="IDMEG")
pg.pairwise_tests(df.loc[df["condition"] == "Sham"], dv="power", within="time", subject="IDMEG")




###### Work on deltas

# Load results
folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\R0_EXP_anova\\"
data = pd.read_csv(folder + "Power_ParietoOcc_tACS_IAF_3.csv")

# From wide to long dataframe
df = pd.DataFrame([(row["IDMEG"], row["Condicion"], time[-4:], row[time]/row["PowPre1"])
                   for i, row in data.iterrows() for time in ["PowPre2", "PowPost"]],
                  columns=["IDMEG", "condition", "time", "delta_pow"])

# Plot scatter
fig = px.violin(df, x="time", y="delta_pow", color="condition", hover_name="IDMEG")
fig.update_traces(meanline_visible=True,
                  points='all', # show all points
                  jitter=0.05,  # add some jitter on points for better visibility
                  scalemode='count') #scale violin plot area with total count
fig.show("browser")


###### Work on deltas

# Load results
folder = "E:\LCCN_Local\PycharmProjects\\neuroStimulation\PAPER2_AbstractMode\R0_EXP_anova\\"
data = pd.read_csv(folder + "Power_ParietoOcc_tACS_IAF_3.csv")

# From wide to long dataframe
df = pd.DataFrame([(row["IDMEG"], row["Condicion"], time[-4:], row[time]/row["PowPre2"])
                   for i, row in data.iterrows() for time in ["PowPost"]],
                  columns=["IDMEG", "condition", "time", "delta_pow"])

# Plot scatter
fig = px.violin(df, x="condition", y="delta_pow", hover_name="IDMEG")
fig.update_traces(meanline_visible=True,
                  points='all', # show all points
                  jitter=0.05,  # add some jitter on points for better visibility
                  scalemode='count') #scale violin plot area with total count
fig.show("browser")
