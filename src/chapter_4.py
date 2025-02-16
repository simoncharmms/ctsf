#### ==========================================================================
#### Dissertation chapter 4
#### Author: Simon Schramm
#### 13.06.2024
#### --------------------------------------------------------------------------
""" 
This contains the code for chapter 4 of the dissertation.
""" 
### ---------------------------------------------------------------------------
#%% Preamble.
### ---------------------------------------------------------------------------
import os 
import sys
import pandas as pd
import matplotlib.pyplot as plt
#
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from constants import FP_DATA, FP_FIGURES
from figures import set_graph_options
from py2neo import Graph, Node, Relationship
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
list_econ_cols = [
    'GDP in Germany Annually',
    'Average Income in Germany Annually', 
    'Expenditures for alcoholic beverages in Germany Annually', 
    'Personal Savings in Germany Quarterly',
    'Disposable Personal Income in Germany Quarterly',
    'Consumer Spending in Germany Quarterly',
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
### ---------------------------------------------------------------------------
# CAHPTER 4
### ---------------------------------------------------------------------------
#%% SECTION 4.2
### ---------------------------------------------------------------------------
### ---------------------------------------------------------------------------
#%% Plot running example with timestamps
### ---------------------------------------------------------------------------
column = 'BMW i3 Sales in Germany'
df_monthly_filtered = df_monthly[column]
# Plot the data
fig, ax = plt.subplots(figsize=(24, 10))
ax.plot(df_monthly_filtered.index, df_monthly_filtered, color=bmw_color)

# Create secondary y-axis for shaded areas
ax3 = ax.twinx()
ax3.set_ylim(0, 6000)
ax3.set_ylabel('$s(EUR)$')

# Define subsidy periods
subsidy_periods = [
    (pd.Timestamp('2016-05-18'), pd.Timestamp('2020-06-30'), 4000, '#dfacfa', 'Subsidy phase 1'),
    (pd.Timestamp('2020-07-01'), pd.Timestamp('2022-12-31'), 6000, '#794694', 'Subsidy phase 2'),
    (pd.Timestamp('2023-01-01'), pd.Timestamp('2024-12-14'), 4500, '#ac79c7', 'Subsidy phase 3')
]

purple = '#4B277B'

# Add shaded areas and compute means
for start, end, height, color, label in subsidy_periods:
    ax3.fill_betweenx([0, height], start, end, color=color, alpha=0.3, label=label)
    period_mean = df_monthly_filtered[start:end].mean()
    ax3.text(start, height-500, f'$start:${start.date()}\n$s(EUR):${height}\n$end:${end.date()}', color='black', ha='left', va='top')

# Add events for the first and last sale (excluding zeros)
first_sale_date = df_monthly_filtered[df_monthly_filtered > 0].dropna().index.min()
last_sale_date = df_monthly_filtered[df_monthly_filtered > 0].dropna().index.max()

ax.axvline(first_sale_date, color=purple, linestyle='--', label='Start of Sales')
ax.axvline(last_sale_date, color=purple, linestyle='--', label='End of Sales')
ax.text(first_sale_date, 2000, f'First Sale: \n{first_sale_date.date()}', color=purple, ha='right', va='top')
ax.text(last_sale_date, 2000, f'Last Sale: \n{last_sale_date.date()}', color=purple, ha='left', va='top')

ax3.set_ylim(0, 7000)
ax.set_ylim(0, 2500)
ax.set_title(f'{column} with governmental subsidies $s$ in $EUR$ for electric vehicles in Germany')
ax.set_ylabel('BMW i3 sales volume in Germany')
ax.set_xlabel('Year')

plt.savefig(FP_FIGURES+"chap04_bmw_i3_sales_events.png")
plt.show()
### ---------------------------------------------------------------------------
#%% SECTION 4.3
### ---------------------------------------------------------------------------
# Connect to Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))

# Clear existing data
graph.delete_all()

# Create nodes
bmw_i3 = Node("Vehicle", name="BMW i3", start_of_sales=first_sale_date.date(), end_of_sales=last_sale_date.date())
tesla_model_3 = Node("Vehicle", name="Tesla Model 3", start_of_sales="2019-02-01", end_of_sales="2024-12-31")  # Example dates
germany = Node("Country", name="Germany")
subsidy = Node("Subsidy", name="Governmental Subsidy")

# Create relationships
bmw_i3_sales = Relationship(bmw_i3, "hasSales", germany)
tesla_model_3_sales = Relationship(tesla_model_3, "hasSales", germany)
bmw_i3_impact = Relationship(subsidy, "hasImpact", bmw_i3)
tesla_model_3_impact = Relationship(subsidy, "hasImpact", tesla_model_3)

# Add nodes and relationships to the graph
graph.create(bmw_i3 | tesla_model_3 | germany | subsidy)
graph.create(bmw_i3_sales | tesla_model_3_sales | bmw_i3_impact | tesla_model_3_impact)
### ---------------------------------------------------------------------------
### End.
#### ==========================================================================
# %%
