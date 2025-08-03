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
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from py2neo import Graph, Node, Relationship
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
#from torch_geometric_temporal.signal import StaticGraphTemporalSignal
#
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from constants import FP_PROJECT, FP_DATA, FP_FIGURES
from figures import set_graph_options, bmw_color, petrol, DPI, purple, light_purple
from launch_db import construct_onedirectional_ekg, construct_temporal_knowledge_graph, search_gdelt_for_umweltbonus_events
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
import pandas as pd
from sklearn.metrics import mean_squared_error
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
column = 'BMW i3 Sales in Germany'
df_monthly_filtered = df_monthly[column][(df_monthly.index >= pd.to_datetime('2014-01-01')) & (df_monthly.index < pd.to_datetime('2023-01-01'))]
# Add events for the first and last sale (excluding zeros)
first_sale_date = df_monthly_filtered[df_monthly_filtered > 0].dropna().index.min()
last_sale_date = df_monthly_filtered[df_monthly_filtered > 0].dropna().index.max()
df_events = pd.read_excel(FP_DATA + 'data_events.xlsx')
### ---------------------------------------------------------------------------
#%% CHAPTER 7
### ---------------------------------------------------------------------------

# Prepare data for LSTM
df_lstm = df_monthly_filtered.dropna()
data = df_lstm.values.astype(float)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.reshape(-1, 1))

# Create sequences for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 24  # Use past 12 months to predict the next month
X, y = create_sequences(data, seq_length)

# Split data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define LSTM model
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index, edge_attr):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).requires_grad_()
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h_0.detach(), c_0.detach()))
        out = self.fc(out[:, -1, :])
        return out

input_size = 1
hidden_size = 50
num_layers = 2
output_size = 1

model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Make predictions
model.eval()
train_predict = model(X_train)
test_predict = model(X_test)

# Inverse transform the predictions
train_predict = scaler.inverse_transform(train_predict.detach().numpy())
test_predict = scaler.inverse_transform(test_predict.detach().numpy())
y_train = scaler.inverse_transform(y_train.detach().numpy().reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.detach().numpy().reshape(-1, 1))
#%%
# Plot the results
fig, ax = plt.subplots(figsize=(16, 9))
plt.plot(df_lstm.index[:len(y_train)], y_train, label='Train Data')
plt.plot(df_lstm.index[len(y_train):len(y_train)+len(y_test)], y_test, label='Test Data', color=bmw_color, linestyle='--')
plt.plot(df_lstm.index[:len(train_predict)], train_predict, label='Train Predict', color='green')
plt.plot(df_lstm.index[len(y_train):len(y_train)+len(test_predict)], test_predict, label='Test Predict', color=purple)
# Add event for Tesla Model 3 Life Cycle Impulse
event_date = pd.Timestamp('2020-07-01')
ax.axvline(event_date, color='black', linestyle='--', label='Post-hoc adjustment')
ax.text(event_date, ax.get_ylim()[1] - 150, 'Subsidy Phase 2', color='black', verticalalignment='top', horizontalalignment='left')

# Set x-axis limits to 2018-2022
ax.set_xlim(pd.Timestamp('2018-01-01'), pd.Timestamp('2020-12-31'))

ax.legend(loc='upper left')
ax.set_xlabel('Date')
ax.set_ylabel('BMW i3 Sales in Germany')
plt.title('LSTM Prediction of BMW i3 Sales in Germany (2018-2022)')
#plt.savefig(FP_FIGURES+"/chap07_lstm_prediction_benchmark_2018_2022.png", dpi=DPI)
plt.show()

### ---------------------------------------------------------------------------
#%% Plot multivariate covariates.
### ---------------------------------------------------------------------------
# Covariate data.
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
    'Business Confidence in Germany Monthly',
    'Bankruptcies in Germany Monthly'
    ]
# Filter data to 2010 - 2018
df_monthly_filtered = df_monthly[(df_monthly.index >= '2010-01-01') & (df_monthly.index <= '2018-12-31')]

# Index all values to 100% = 2010
base_year = '2010'
for col in list_econ_cols:
    if col in df_monthly_filtered.columns:
        base_value = df_monthly_filtered.loc[base_year, col].mean()
        df_monthly_filtered[col] = df_monthly_filtered[col] / base_value * 100

# Plot BMW i3 Sales in Germany above the economic indicators
fig, axes = plt.subplots(nrows=len(list_econ_cols) + 1, ncols=1, figsize=(27, 2.5 * (len(list_econ_cols) + 1)), sharex=True)
fig.suptitle('BMW i3 Sales and Economic Indicators Over Time (Interpolated and Indexed to 2010)', fontsize=31)

# Plot BMW i3 Sales in Germany
df_monthly_filtered['BMW i3 Sales in Germany'].plot(ax=axes[0], color=bmw_color)
axes[0].set_title('BMW i3 Sales in Germany')
axes[0].legend().remove()
# Round up to the nearest hundred
def round_up_to_hundred(x):
    return int(math.ceil(x / 50.0)) * 50

# Plot BMW i3 Sales in Germany
max_ylim = round_up_to_hundred(axes[0].get_ylim()[1])
axes[0].set_ylim(bottom=0, top=max_ylim)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
axes[0].set_yticks([axes[0].get_ylim()[0], axes[0].get_ylim()[1]])

# Plot each economic indicator
for ax, col in zip(axes[1:], list_econ_cols):
    if col in df_monthly_filtered.columns:
        df_monthly_filtered[col].plot(ax=ax, color=petrol)
        ax.set_title(col)
        ax.legend().remove()
        max_ylim = round_up_to_hundred(ax.get_ylim()[1])
        ax.set_ylim(bottom=0, top=max_ylim)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
        ax.set_yticks([ax.get_ylim()[0], ax.get_ylim()[1]])

plt.xlabel('Date')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(FP_FIGURES+"/chap07_dependent_variable_and_covariates.png", dpi=DPI)
plt.show()
#%## ---------------------------------------------------------------------------
#%% SECTION 7.1
### ---------------------------------------------------------------------------
# Construct EKG Graph in Neo4j.
graph = construct_onedirectional_ekg(first_sale_date, last_sale_date)
### ---------------------------------------------------------------------------
#%% SECTION 7.2
### ---------------------------------------------------------------------------
# Construct TKG Graph in Neo4j.
graph = construct_temporal_knowledge_graph(df_monthly, list_econ_cols)
### ---------------------------------------------------------------------------
#%% SECTION 7.2.2
### ---------------------------------------------------------------------------
#
class CustomGATConv(GATConv):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(CustomGATConv, self).__init__(in_channels, out_channels, heads=heads, concat=concat, negative_slope=negative_slope, dropout=dropout, bias=bias, **kwargs)
#
    def forward(self, x, edge_index, return_attention_weights=False):
        out, (edge_index, alpha) = super().forward(x, edge_index, return_attention_weights=True)
        if return_attention_weights:
            return out, alpha
        else:
            return out
#
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = CustomGATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = CustomGATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data, return_attention_weights=False):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        if return_attention_weights:
            x, alpha1 = self.conv1(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            x = F.dropout(x, p=0.6, training=self.training)
            x, alpha2 = self.conv2(x, edge_index, return_attention_weights=True)
            return F.log_softmax(x, dim=1), (alpha1, alpha2)
        else:
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)
# Initialize the model, optimizer, and loss function
model = GAT(in_channels=3, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
#
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data_tensor)
    event_class_mask = data_tensor.y != -1  # Assuming -1 is used for unlabeled data
    loss = criterion(out[event_class_mask], data_tensor.y[event_class_mask])
    loss.backward()
    optimizer.step()
    return loss.item()
# Sample labels for training (for demonstration purposes)
data_tensor = Data(x=torch.tensor(np.random.rand(5, 3), dtype=torch.float), edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long), y=torch.tensor([0, 1, 0, 1, 0], dtype=torch.long))
# Train the model for a few epochs.
for epoch in range(200):
    loss = train()
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')
# Save the trained model.
model.export(os.path.join(FP_PROJECT, " models/gat_model_best.pth"))
#
def rank_event_instances(model):
    model.eval()
    with torch.no_grad():
        out, (alpha1, alpha2) = model(data_tensor, return_attention_weights=True)
        attention_weights = alpha1.mean(dim=1)  # Average attention weights across heads
        if attention_weights.numel() > 0:
            rank_scores = attention_weights.sum(dim=0)  # Sum over the correct dimension
            ranked_indices = rank_scores.argsort(descending=True)
        else:
            rank_scores = torch.tensor([])
            ranked_indices = torch.tensor([])
        return attention_weights, ranked_indices
# Get the ranked event instances.
attention_weights, ranked_event_instances = rank_event_instances(model)
#
print("Ranked Event Instances:", attention_weights)
print("Ranked Event Instances:", ranked_event_instances)
### ---------------------------------------------------------------------------
#%% SECTION 7.3
### ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(in_channels, hidden_channels, num_layers=num_layers, batch_first=True)
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

    def forward(self, x):
        batch_size = x.size(0)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_channels).to(x.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_channels).to(x.device)
        x = x.unsqueeze(1)  # Add sequence length dimension
        x, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        return x.squeeze(1), (h_n, c_n)  # Remove sequence length dimension
    
class Decoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=3):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_channels, hidden_channels, num_layers=num_layers, batch_first=True)
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, hidden):
        x = x.unsqueeze(1)  # Add sequence length dimension
        x, (h, c) = self.lstm(x, hidden)
        x = self.linear(x.squeeze(1))  # Apply linear layer to the output
        return x

class GNN(nn.Module):
    def __init__(self, node_features, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.encoder = Encoder(node_features, hidden_channels)
        self.decoder = Decoder(hidden_channels, out_channels, num_layers=3)

    def forward(self, x, edge_index, edge_attr):
        x, hidden = self.encoder(x)
        x = self.decoder(x, hidden)
        return x[:, -1]  # Return the last element in the sequence

# Initialize model, loss, and optimizer
model = GNN(node_features=1, hidden_channels=16, out_channels=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Extract data from Neo4j graph
query = """
MATCH (v:Vehicle)-[r:hasSales]->(c:Country)
RETURN v.name AS vehicle, c.name AS country, r
"""
results = graph.run(query).data()

# Prepare data for PyTorch Geometric
nodes = {}
edges = []
edge_attrs = []
for result in results:
    vehicle = result['vehicle']
    country = result['country']
    relationship = result['r']
    if vehicle not in nodes:
        nodes[vehicle] = len(nodes)
    if country not in nodes:
        nodes[country] = len(nodes)
    edges.append((nodes[vehicle], nodes[country]))
    edge_attrs.append([relationship[date] for date in sorted(relationship.keys()) if isinstance(date, str)])

# Convert to PyTorch Geometric format
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

# Create temporal signal
class TemporalGraphDataset:
    def __init__(self, edge_index, edge_attr, node_features):
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.node_features = node_features

    def __len__(self):
        return self.edge_attr.size(1)

    def __getitem__(self, idx):
        return self.node_features, self.edge_index, self.edge_attr[:, idx]

# Create temporal graph dataset
node_features = torch.tensor(np.random.rand(len(nodes), 1), dtype=torch.float)  # Example node features
temporal_dataset = TemporalGraphDataset(edge_index=edge_index, edge_attr=edge_attr, node_features=node_features)
#%%
# Define the parameter grid
param_grid = {
    'hidden_size': [16, 32, 64],
    'num_layers': [1, 2, 3],
    'learning_rate': [0.001, 0.01, 0.1],
    'epochs': [50]
}

# Initialize variables to store the best parameters and the corresponding loss
best_params = None
best_loss = float('inf')

# Store the grid search steps
grid_search_results = []

for params in ParameterGrid(param_grid):
    # Initialize model with current parameters
    model = GNN(node_features=1, hidden_channels=params['hidden_size'], out_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    losses = []
    for epoch in range(params['epochs']):
        epoch_loss = 0
        for t in range(len(temporal_dataset)):
            x, edge_index, edge_attr = temporal_dataset[t]
            optimizer.zero_grad()
            out = model(x, edge_index, edge_attr)
            loss = criterion(out, torch.zeros_like(out))  # Assuming snapshot.y is not defined, using zeros as placeholder
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(temporal_dataset)
        losses.append(epoch_loss)

    # Check if the current parameters yield a better loss
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_params = params

    grid_search_results.append((params, epoch_loss))
    print(f'Params: {params}, Loss: {epoch_loss}')

print(f'Best Params: {best_params}, Best Loss: {best_loss}')

# Save the best model
model.export(os.path.join(FP_PROJECT, " models/gnn_model_best.pth"))

# Print LaTeX table with the results

results_df = pd.DataFrame(grid_search_results, columns=['Parameters', 'Loss'])
results_df['Hidden Size'] = results_df['Parameters'].apply(lambda x: x['hidden_size'])
results_df['Num Layers'] = results_df['Parameters'].apply(lambda x: x['num_layers'])
results_df['Learning Rate'] = results_df['Parameters'].apply(lambda x: x['learning_rate'])
results_df['Epochs'] = results_df['Parameters'].apply(lambda x: x['epochs'])
results_df = results_df.drop(columns=['Parameters'])

latex_table = results_df.to_latex(index=False, float_format="%.4f")
print(latex_table)
print(f'Best Params: {best_params}, Best Loss: {best_loss}')

# Save the best model
model.export(os.path.join(FP_PROJECT, " models/gnn_model_best.pth"))

# Plot the loss over epochs for the best model
fig, ax = plt.subplots(figsize=(16, 9))
plt.plot(range(1, len(losses)+1), losses, color=bmw_color)
plt.xlabel('Epoch', fontsize=24)
plt.ylabel('Loss', fontsize=24)
plt.title('Loss over Epochs for Best Model', fontsize=28)
plt.ylim(0, 0.0001)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()


### ---------------------------------------------------------------------------
#%% Experiment 1
### ---------------------------------------------------------------------------
# Prepare data for GNN with time series node features
node_features = []
for node in nodes:
    if node in df_monthly.index:
        node_features.append(df_monthly.loc[node].values)
    else:
        node_features.append(np.zeros(len(df_monthly.columns)))

node_features = torch.tensor(node_features, dtype=torch.float)

# Create temporal graph dataset with time series node features
class TemporalGraphDataset:
    def __init__(self, edge_index, edge_attr, node_features):
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.node_features = node_features

    def __len__(self):
        return self.edge_attr.size(1)

    def __getitem__(self, idx):
        return self.node_features, self.edge_index, self.edge_attr[:, idx]

temporal_dataset = TemporalGraphDataset(edge_index=edge_index, edge_attr=edge_attr, node_features=node_features)

# Define the parameter grid
param_grid = {
    'hidden_size': [16, 32, 64],
    'num_layers': [1, 2, 3],
    'learning_rate': [0.001, 0.01, 0.1],
    'epochs': [50]
}

# Initialize variables to store the best parameters and the corresponding loss
best_params = None
best_loss = float('inf')

# Store the grid search steps
grid_search_results = []

for params in ParameterGrid(param_grid):
    # Initialize model with current parameters
    model = GNN(node_features=node_features.size(1), hidden_channels=params['hidden_size'], out_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    losses = []
    for epoch in range(params['epochs']):
        epoch_loss = 0
        for t in range(len(temporal_dataset)):
            x, edge_index, edge_attr = temporal_dataset[t]
            optimizer.zero_grad()
            out = model(x, edge_index, edge_attr)
            loss = criterion(out, torch.zeros_like(out))  # Assuming snapshot.y is not defined, using zeros as placeholder
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(temporal_dataset)
        losses.append(epoch_loss)

    # Check if the current parameters yield a better loss
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_params = params

    grid_search_results.append((params, epoch_loss))
    print(f'Params: {params}, Loss: {epoch_loss}')

print(f'Best Params: {best_params}, Best Loss: {best_loss}')

# Save the best model
model.export(os.path.join(FP_PROJECT, " models/gnn_model_best.pth"))

# Print LaTeX table with the results
results_df = pd.DataFrame(grid_search_results, columns=['Parameters', 'Loss'])
results_df['Hidden Size'] = results_df['Parameters'].apply(lambda x: x['hidden_size'])
results_df['Num Layers'] = results_df['Parameters'].apply(lambda x: x['num_layers'])
results_df['Learning Rate'] = results_df['Parameters'].apply(lambda x: x['learning_rate'])
results_df['Epochs'] = results_df['Parameters'].apply(lambda x: x['epochs'])
results_df = results_df.drop(columns=['Parameters'])

latex_table = results_df.to_latex(index=False, float_format="%.4f")
print(latex_table)
### ---------------------------------------------------------------------------
#%% Experiment 2
### ---------------------------------------------------------------------------
# Load the anomaly detection model
anomaly_model = torch.load('models/anomaly_model_best.pth')
anomaly_model.eval()

# Function to detect anomalies in a time series
def detect_anomalies(time_series, model, threshold=0.5):
    time_series_tensor = torch.tensor(time_series, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        anomaly_scores = model(time_series_tensor).squeeze(0).numpy()
    anomalies = anomaly_scores > threshold
    return anomalies

# Detect and remove anomalies in each edge time series
cleaned_edge_attrs = []
for edge_attr in edge_attrs:
    anomalies = detect_anomalies(edge_attr, anomaly_model)
    cleaned_edge_attr = np.array(edge_attr)[~anomalies]
    cleaned_edge_attrs.append(cleaned_edge_attr)

# Convert cleaned edge attributes back to tensor
cleaned_edge_attr = torch.tensor(cleaned_edge_attrs, dtype=torch.float)
# Re-initialize the model, optimizer, and loss function
model = GNN(node_features=1, hidden_channels=16, out_channels=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Create temporal graph dataset with cleaned edge attributes
temporal_dataset_cleaned = TemporalGraphDataset(edge_index=edge_index, edge_attr=cleaned_edge_attr, node_features=node_features)

# Training loop with cleaned edge attributes
model.train()
losses_cleaned = []
for epoch in range(100):  # Adjust the number of epochs as needed
    epoch_loss = 0
    for t in range(len(temporal_dataset_cleaned)):
        x, edge_index, edge_attr = temporal_dataset_cleaned[t]
        optimizer.zero_grad()
        out = model(x, edge_index, edge_attr)
        loss = criterion(out, torch.zeros_like(out))  # Assuming snapshot.y is not defined, using zeros as placeholder
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(temporal_dataset_cleaned)
    losses_cleaned.append(epoch_loss)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {epoch_loss:.4f}')

# Save the trained model with cleaned edge attributes
model.export(os.path.join(FP_PROJECT, "models/gnn_model_cleaned_best.pth"))

# Plot the loss over epochs for the cleaned model
fig, ax = plt.subplots(figsize=(16, 9))
plt.plot(range(1, len(losses_cleaned)+1), losses_cleaned, color=bmw_color)
plt.xlabel('Epoch', fontsize=24)
plt.ylabel('Loss', fontsize=24)
plt.title('Loss over Epochs for Cleaned Model', fontsize=28)
plt.ylim(0, 0.0001)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()

# Print LaTeX table comparing both results
results_df_cleaned = pd.DataFrame(grid_search_results, columns=['Parameters', 'Loss'])
results_df_cleaned['Hidden Size'] = results_df_cleaned['Parameters'].apply(lambda x: x['hidden_size'])
results_df_cleaned['Num Layers'] = results_df_cleaned['Parameters'].apply(lambda x: x['num_layers'])
results_df_cleaned['Learning Rate'] = results_df_cleaned['Parameters'].apply(lambda x: x['learning_rate'])
results_df_cleaned['Epochs'] = results_df_cleaned['Parameters'].apply(lambda x: x['epochs'])
results_df_cleaned = results_df_cleaned.drop(columns=['Parameters'])

# Combine both results
combined_results_df = pd.merge(results_df, results_df_cleaned, on=['Hidden Size', 'Num Layers', 'Learning Rate', 'Epochs'], suffixes=('_original', '_cleaned'))

# Print LaTeX table
latex_table_combined = combined_results_df.to_latex(index=False, float_format="%.4f")
print(latex_table_combined)
### ---------------------------------------------------------------------------
#%% Experiment 3
### ---------------------------------------------------------------------------

### ---------------------------------------------------------------------------
#%% SECTION 7.3.2: BENCHMARKING
### ---------------------------------------------------------------------------
#%% Prediction with the model
# Prepare the data for prediction
future_steps = 12  # Predict the next 12 months
last_sequence = data[-seq_length:]  # Get the last sequence from the data

# Convert to PyTorch tensor
last_sequence = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)

# Make prediction
model.eval()
with torch.no_grad():
    future_predictions = []
    for _ in range(future_steps):
        prediction = model(last_sequence.squeeze(0), edge_index, edge_attr)
        future_predictions.append(prediction.item())
        last_sequence = torch.cat((last_sequence[:, 1:, :], prediction.unsqueeze(0).unsqueeze(2)), dim=1)

# Inverse transform the predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Create a date range for the future predictions
last_date = df_lstm.index[-1]
future_dates = pd.date_range(last_date, periods=future_steps + 1, freq='M')[1:]

# Plot the future predictions
fig, ax = plt.subplots(figsize=(16, 9))
plt.plot(df_lstm.index, df_lstm.values, label='Historical Data')
plt.plot(future_dates, future_predictions, label='Future Predictions', color='red')
plt.xlabel('Date')
plt.ylabel('BMW i3 Sales in Germany')
plt.title('LSTM Future Predictions of BMW i3 Sales in Germany')
plt.legend()
plt.show()

#%%
# Get real world data.
df_real = pd.read_csv(FP_DATA + 'real_world_data.csv')
# Melt the df_real DataFrame to have a single column for dates and values.
df_real_melted = df_real.melt(id_vars=['SOURCE', 'BRAND', 'BODY_STYLE', 'SEGMENT', 'MODEL_NAME_DETAIL'], var_name='Date', value_name='Human Forecasted Sales')
# Filter out non-date values.
df_real_melted = df_real_melted[pd.to_numeric(df_real_melted['Date'], errors='coerce').notnull()]
# Convert the Date column to datetime format.
df_real_melted['Date'] = pd.to_datetime(df_real_melted['Date'], format='%Y')
df_real_melted = df_real_melted.set_index('Date')
# Filter to BMW i3.
df_real_melted_i3 = df_real_melted[df_real_melted['MODEL_NAME_DETAIL'] == 'I3']
# Aggregate df_lstm data by year.
df_actuals = df_monthly[column].resample('Y').sum()
df_lstm_yearly = df_lstm.resample('Y').sum()

#%% Initialize a dictionary to store the errors
errors = {'Source': [], 'Year': [], 'MAPE': []}
# Compute the mean percentage error for each Human-created forecast compared to df_actuals
for source in df_real_melted_i3['SOURCE'].unique():
    df_source = df_real_melted_i3[df_real_melted_i3['SOURCE'] == source]
    df_source = df_source.sort_index()  # Ensure the data is sorted by date
    df_source = df_source[df_source['Human Forecasted Sales'] > 0]  # Exclude zeros
#
    for year in df_source.index.year.unique():
        if year in df_actuals.index.year:
            actual_value = df_actuals[df_actuals.index.year == year].values[0]
            forecast_value = df_source[df_source.index.year == year]['Human Forecasted Sales'].values[0]
            MAPE = (forecast_value - actual_value) / actual_value * 100
            errors['Source'].append(source)
            errors['Year'].append(year)
            errors['MAPE'].append(MAPE)
# Create a new dataframe to store the error values.
df_errors = pd.DataFrame(errors)
df_errors['Source Year'] = df_errors['Source'].apply(lambda x: x.split('(')[-1].strip(')'))
print(df_errors)
# Restructure df_errors to provide the MAPE by forecast year.
df_errors['Source Year'] = df_errors['Source Year'].astype(int)
df_errors['Forecast Year'] = df_errors['Year'] - df_errors['Source Year']
# Group by forecast year and calculate the mean MAPE.
df_errors_grouped = df_errors.groupby('Forecast Year')['MAPE'].mean().reset_index()
# Plot the real-world forecasts and MAPE by forecast year and source
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(32, 9))
# Plot the real-world forecasts.
axes[0].plot(df_actuals.index, df_actuals.values, label='Actual Data', color=bmw_color)
# Define a colormap for the real-world data.
cmap = plt.get_cmap('Purples')
colors = [cmap(i) for i in np.linspace(0.4, 1, len(df_real_melted_i3['SOURCE'].unique()))]
for i, source in enumerate(df_real_melted_i3['SOURCE'].unique()):
    df_source = df_real_melted_i3[df_real_melted_i3['SOURCE'] == source]
    df_source = df_source.sort_index()  # Ensure the data is sorted by date
    df_source = df_source[df_source['Human Forecasted Sales'] > 0]  # Exclude zeros
    axes[0].plot(df_source.index, df_source['Human Forecasted Sales'], label=source, color=colors[i], marker='o')
# Add events to the plot with different shades of light purple
shades = [light_purple, '#D8BFD8', '#E6E6FA', '#EEE8AA', '#F0E68C']
for i, (_, event) in enumerate(df_events[df_events['detected_in'].str.contains('i3', na=False)].iterrows()):
    start_date = pd.to_datetime(event['start'])
    end_date = pd.to_datetime(event['end'])
    label = event['name']
    color = shades[i % len(shades)]
    if start_date == end_date:
        axes[0].axvline(start_date, color='black', linestyle='--', label=label)
        axes[0].text(start_date, axes[0].get_ylim()[0], label, rotation=90, verticalalignment='bottom', fontsize=20, color='black')
    else:
        #axes[0].axvspan(start_date, end_date, color=color, alpha=0.3, label=label)
        axes[0].axvline(start_date, color='black', linestyle='--', label=label)
        axes[0].text(start_date, axes[0].get_ylim()[1]*0.6, label, rotation=90, verticalalignment='bottom', fontsize=20, color='Black')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('BMW i3 Sales in Germany')
axes[0].set_title('Actual BMW i3 Sales in Germany and Human-Created Forecasts 2013 - 2023 with Events')
# Plot the MAPE by forecast year and source.
for i, source in enumerate(df_errors['Source'].unique()):
    df_source_errors = df_errors[df_errors['Source'] == source]
    axes[1].plot(df_source_errors['Forecast Year'], df_source_errors['MAPE'], label=source, color=colors[i], marker='o')
axes[1].set_xlabel('Forecast Year')
axes[1].set_ylabel('Mean Absolute Percentage Error (MAPE)')
axes[1].set_title('MAPE by Forecast Year and Source')
axes[1].legend()
plt.tight_layout()
plt.savefig(FP_FIGURES+"/chap7_bmw_3-series_fc_errors.png")
plt.show()

#%%
top_down_events = search_gdelt_for_umweltbonus_events()
# %%


df_real = pd.read_csv(FP_DATA + 'pred.csv')
# Melt the df_real DataFrame to have a single column for dates and values.
df_real_melted = df_real.melt(id_vars=['SOURCE', 'BRAND', 'BODY_STYLE', 'SEGMENT', 'MODEL_NAME_DETAIL'], var_name='Date', value_name='Human Forecasted Sales')
# Filter out non-date values.
df_real_melted = df_real_melted[pd.to_numeric(df_real_melted['Date'], errors='coerce').notnull()]
# Convert the Date column to datetime format.
df_real_melted['Date'] = pd.to_datetime(df_real_melted['Date'], format='%Y')
df_real_melted = df_real_melted.set_index('Date')
# Filter to BMW i3.
df_real_melted_i3 = df_real_melted[df_real_melted['MODEL_NAME_DETAIL'] == 'I3']
# Aggregate df_lstm data by year.
df_actuals = df_monthly[column].resample('Y').sum()
df_lstm_yearly = df_lstm.resample('Y').sum()

#%% Initialize a dictionary to store the errors
errors = {'Source': [], 'Year': [], 'MAPE': []}
# Compute the mean percentage error for each Human-created forecast compared to df_actuals
for source in df_real_melted_i3['SOURCE'].unique():
    df_source = df_real_melted_i3[df_real_melted_i3['SOURCE'] == source]
    df_source = df_source.sort_index()  # Ensure the data is sorted by date
    df_source = df_source[df_source['Human Forecasted Sales'] > 0]  # Exclude zeros
#
    for year in df_source.index.year.unique():
        if year in df_actuals.index.year:
            actual_value = df_actuals[df_actuals.index.year == year].values[0]
            forecast_value = df_source[df_source.index.year == year]['Human Forecasted Sales'].values[0]
            MAPE = (forecast_value - actual_value) / actual_value * 100
            errors['Source'].append(source)
            errors['Year'].append(year)
            errors['MAPE'].append(MAPE)
# Create a new dataframe to store the error values.
df_errors = pd.DataFrame(errors)
df_errors['Source Year'] = df_errors['Source'].apply(lambda x: x.split('(')[-1].strip(')'))
print(df_errors)
# Restructure df_errors to provide the MAPE by forecast year.
df_errors['Source Year'] = df_errors['Source Year'].astype(int)
df_errors['Forecast Year'] = df_errors['Year'] - df_errors['Source Year']
# Group by forecast year and calculate the mean MAPE.
df_errors_grouped = df_errors.groupby('Forecast Year')['MAPE'].mean().reset_index()
#%% Plot the real-world forecasts and MAPE by forecast year and source
# Plot the real-world forecasts.
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(df_actuals.index, df_actuals.values, label='Actual Data', color=bmw_color)
# Define a colormap for the real-world data.
cmap = plt.get_cmap('Purples')
colors = ['black' if 'Human-created' in source else 'purple' if 'Graph-Encoder-Decoder' in source else light_purple for source in df_real_melted_i3['SOURCE'].unique()]
linestyles = ['-' if 'Human-created' in source else '-' if 'Graph-Encoder-Decoder' in source else '--' for source in df_real_melted_i3['SOURCE'].unique()]
for i, source in enumerate(df_real_melted_i3['SOURCE'].unique()):
    df_source = df_real_melted_i3[df_real_melted_i3['SOURCE'] == source]
    df_source = df_source.sort_index()  # Ensure the data is sorted by date
    df_source = df_source[df_source['Human Forecasted Sales'] > 0]  # Exclude zeros
    ax.plot(df_source.index, df_source['Human Forecasted Sales'], label=source, color=colors[i], linestyle=linestyles[i], marker='o')
# Add events to the plot with different shades of light purple
shades = [light_purple, '#D8BFD8', '#E6E6FA', '#EEE8AA', '#F0E68C']
for i, (_, event) in enumerate(df_events[df_events['detected_in'].str.contains('i3', na=False)].iterrows()):
    start_date = pd.to_datetime(event['start'])
    end_date = pd.to_datetime(event['end'])
    label = event['name']
    color = shades[i % len(shades)]
    if start_date == end_date:
        ax.axvline(start_date, color='black', linestyle='--', label=label)
        ax.text(start_date, ax.get_ylim()[0], label, rotation=90, verticalalignment='bottom', fontsize=20, color='black')
    else:
        ax.axvline(start_date, color='black', linestyle='--', label=label)
        ax.text(start_date, ax.get_ylim()[1]*0.6, label, rotation=90, verticalalignment='bottom', fontsize=20, color='Black')
# Add another event line for "Elektromobilitätsgesetz"
elektromobilitaetsgesetz_date = pd.Timestamp('2015-06-12')
ax.axvline(elektromobilitaetsgesetz_date, color='blue', linestyle='--', label='Elektromobilitätsgesetz')
ax.text(elektromobilitaetsgesetz_date, ax.get_ylim()[1]*0.6, 'Elektromobilitätsgesetz', rotation=90, verticalalignment='bottom', fontsize=20, color='blue')
ax.set_xlabel('Date')
ax.set_ylabel('BMW i3 Sales in Germany')
ax.set_title('Actual BMW i3 Sales in Germany and Human-Created Forecasts 2013 - 2023 with Events')
plt.tight_layout()
plt.savefig(FP_FIGURES+"/chap7_bmw_3-series_fc_lstm_vs_graph_errors_forecasts.png")
plt.show()

# Plot the MAPE by forecast year and source.
fig, ax = plt.subplots(figsize=(16, 9))
for i, source in enumerate(df_errors['Source'].unique()):
    df_source_errors = df_errors[df_errors['Source'] == source]
    df_source_errors = df_source_errors[df_source_errors['Forecast Year'] <= 7]  # Limit to seven forecast years
    mean_mape = df_source_errors['MAPE'].mean()
    ax.plot(df_source_errors['Forecast Year'], df_source_errors['MAPE'], label=f'{source} (Mean MAPE: {mean_mape:.2f}%)', color=colors[i], linestyle=linestyles[i], marker='o')
ax.set_xlabel('Forecast Year')
ax.set_ylabel('Mean Absolute Percentage Error (MAPE)')
ax.set_title('MAPE by Forecast Year and Algorithm')
ax.legend()
plt.tight_layout()
plt.savefig(FP_FIGURES+"/chap7_bmw_3-series_fc_errors_lstm_vs_graph.png")
plt.show()

# %%
