#### ==========================================================================
#### Dissertation chapter 7
#### Author: Simon Schramm
#### 13.06.2024
#### --------------------------------------------------------------------------
""" 
This contains the code for chapter 7 of the dissertation.
""" 
### ---------------------------------------------------------------------------
#%% Preamble.
### ---------------------------------------------------------------------------
import os 
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from py2neo import Graph, Node, Relationship
#
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from constants import FP_DATA, FP_FIGURES
from figures import set_graph_options, DPI, pink, green, light_blue
from launch_db import construct_onedirectional_ekg, construct_temporal_knowledge_graph
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from pykeen.pipeline import pipeline
#
bmw_color, bmw_colors = set_graph_options()
### ---------------------------------------------------------------------------
#%% Pre-Processing
### ---------------------------------------------------------------------------
# Get monthly automotive sales data
df_monthly = pd.read_csv(FP_DATA+"data_chapter_2_monthly.csv", delimiter=",")
# Convert the Timestamp column to datetime format
df_monthly['Timestamp'] = pd.to_datetime(df_monthly['Timestamp'], format='%d.%m.%y')
# Set the timestamp column as the index
df_monthly = df_monthly.set_index('Timestamp')
df_monthly.index.freq = pd.infer_freq(df_monthly.index)
list_econ_cols = [
    'Unemployment Rate in Germany Monthly',
    'Consumer Confidence in Germany Monthly',
    'Car Production in Germany Monthly',
    'Car Registrations in Germany Monthly',
    'Capacity Utilization in Germany Monthly', 
    'Business Confidence in Germany Monthly',
    'Bankruptcies in Germany Monthly'
    ]
# Index economic columns to their earliest value
for col in list_econ_cols:
    if col in df_monthly.columns:
        earliest_value = df_monthly[col].dropna().iloc[0]
        df_monthly[col + '_indexed'] = df_monthly[col] / earliest_value * 100
column = 'BMW i3 Sales in Germany'
# Filter data to 2010 - 2018
df_monthly_filtered = df_monthly[(df_monthly.index >= '2014-01-01') & (df_monthly.index <= '2018-12-31')]
df_events = pd.read_excel(FP_DATA + 'data_events.xlsx')
### ---------------------------------------------------------------------------
# CAHPTER 6
### ---------------------------------------------------------------------------
#%% SECTION 6.1.1
### ---------------------------------------------------------------------------
# Index all values to 100% = 2010
base_year = '2014'
for col in list_econ_cols:
    if col in df_monthly_filtered.columns:
        base_value = df_monthly_filtered.loc[base_year, col].mean()
        df_monthly_filtered[col] = df_monthly_filtered[col] / base_value * 100

# Select the columns to cluster
columns_to_cluster = ['BMW i3 Sales in Germany'] + [col + '_indexed' for col in list_econ_cols if col + '_indexed' in df_monthly.columns]

# Drop rows with NaN values
df_cluster = df_monthly_filtered[columns_to_cluster].fillna(0)

# Standardize the data
scaler = StandardScaler()
df_cluster_scaled = scaler.fit_transform(df_cluster)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(df_cluster_scaled)

# Add the cluster labels to the dataframe
df_monthly_filtered = df_monthly_filtered.loc[df_cluster.index]
df_monthly_filtered['Cluster'] = kmeans.labels_
#%%
# Plot the clusters
df_events_filtered = df_events[(df_events['start'] >= '2014-01-01') & (df_events['end'] <= '2019-12-31')]

# Plot the clusters
fig, ax = plt.subplots(figsize=(24, 8))

# Plot the original data
ax.plot(df_monthly_filtered.index, df_monthly_filtered['BMW i3 Sales in Germany'] / 1000, color=bmw_color, label='BMW i3 Sales (Thousands)')

# Color the points according to their cluster
colors = [pink, green, light_blue]
lighter_colors = [plt.cm.Purples(0.3), plt.cm.Blues(0.3), plt.cm.Greens(0.3)]
for cluster in range(3):
    df_cluster = df_monthly_filtered[df_monthly_filtered['Cluster'] == cluster]
    if not df_cluster.empty:
        ax.scatter(df_cluster.index, df_cluster['BMW i3 Sales in Germany'] / 1000, color=lighter_colors[cluster], label=f'Cluster {cluster}', s=1500)  

for i, (_, event) in enumerate(df_events_filtered[df_events_filtered['detected_in'].str.contains('i3', na=False)].iterrows()):
    start_date = pd.to_datetime(event['start'])
    end_date = pd.to_datetime(event['end'])
    label = event['name']
    if start_date == end_date:
        ax.axvline(start_date, color='black', linestyle='--', label=label)
        ax.text(start_date, ax.get_ylim()[1] * 0.4, label, rotation=90, verticalalignment='bottom', fontsize=20, color='black')
    else:
        ax.axvspan(start_date, end_date, color=color, alpha=0.3, label=label)
        ax.text(start_date, ax.get_ylim()[1] * 0.65, label, rotation=90, verticalalignment='bottom', fontsize=20, color='Black')

ax.set_title('BMW i3 Sales in Germany clusters based on $k$ means clustering ($k=3$)')
ax.set_xlabel('Date')
ax.set_ylabel('Monthly Sales Volume (in Thousands)')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(FP_FIGURES+"/chap06_clusters_and_economic_indicators_filtered.png", dpi=DPI)
plt.show()
### ---------------------------------------------------------------------------
#%% SECTION 6.2.
### ---------------------------------------------------------------------------
# Construct TKG Graph in Neo4j.
graph = construct_temporal_knowledge_graph(df_monthly, list_econ_cols)
# Define the pipeline for training the DistMult model.
result = pipeline(
    model='DistMult',
    dataset='nations',
    training_loop='lcwa',
    training_kwargs=dict(num_epochs=100),
)
# Get the trained model.
model = result.model
# Get the embeddings for the nodes.
node_embeddings = model.entity_representations[0](indices=None).detach().numpy()
# Perform KMeans clustering on the node embeddings.
kmeans = KMeans(n_clusters=3, random_state=0).fit(node_embeddings)
# Add the cluster labels to the nodes in the graph.
for i, node_id in enumerate(graph.nodes):
    node = graph.nodes[node_id]
    node['Cluster'] = int(kmeans.labels_[i])
# Save the graph with cluster labels.
graph.save(FP_DATA+'graph_with_clusters')
### ---------------------------------------------------------------------------
### End.
#### ==========================================================================
# %%
