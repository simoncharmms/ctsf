#### ==========================================================================
#### Dissertation chapter 5
#### Author: Simon Schramm
#### 13.06.2024
#### --------------------------------------------------------------------------
""" 
This contains the code for chapter 5 of the dissertation.
""" 
### ---------------------------------------------------------------------------
#%% Preamble.
### ---------------------------------------------------------------------------
import os 
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from py2neo import Graph, Node, Relationship
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from py2neo import NodeMatcher
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from node2vec import Node2Vec
from keras.utils import plot_model
from IPython.display import display, SVG
from sklearn.model_selection import ParameterGrid
from keras.regularizers import l2
#
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from constants import FP_PROJECT, FP_DATA, FP_FIGURES
from figures import set_graph_options, DPI, purple, light_purple, pink, blue
from launch_db import construct_temporal_knowledge_graph, add_events_to_tkg
from models import create_dataset, anomaly_detection_lstm_with_regularization
from utils import get_data, get_data_filtered
#
bmw_color, bmw_colors = set_graph_options()
### ---------------------------------------------------------------------------
#%% Pre-Processing.
### ---------------------------------------------------------------------------
# Get data.
df_monthly, df_events = get_data(FP_DATA)
# Get filtered data.
start = '2013-10-01'
end = '2018-12-31'
df_monthly_filtered = get_data_filtered(df_monthly, start, end)
# Define the example vehicle.
column = 'BMW i3 Sales in Germany'
# Add events for the first and last sale (excluding zeros)
first_sale_date = df_monthly_filtered[df_monthly_filtered > 0].dropna().index.min()
last_sale_date = df_monthly_filtered[df_monthly_filtered > 0].dropna().index.max()
### ---------------------------------------------------------------------------
# CAHPTER 5
### ---------------------------------------------------------------------------
#%% SECTION 5.1: Intro.
### ---------------------------------------------------------------------------
# Conduct time series anomaly detection using Isolation Forest as MVP.
model = IsolationForest(contamination=0.05)
df_monthly_isolation = df_monthly_filtered
df_monthly_isolation['anomaly'] = model.fit_predict(df_monthly_filtered.values.reshape(-1, 1))
# Plot the results for the intro.
fig, ax = plt.subplots(figsize=(24, 6))
ax.plot(df_monthly_isolation.index, df_monthly_isolation[column] / 1000, label='BMW i3 Sales (Thousands)', color=bmw_color)
anomalies = df_monthly_isolation[df_monthly_isolation['anomaly'] == -1]
ax.scatter(anomalies.index, anomalies[column] / 1000, color=purple, label='Anomaly', s=100, edgecolor='k')
ax.set_title('BMW i3 Sales in Germany with Isolation Forest Anomalies ($c=0.05$)')
ax.set_xlabel('Date')
ax.set_ylabel('Monthly Sales Volume (in Thousands)')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.savefig(FP_FIGURES+"chap05_bmw_i3_sales_anomalies.png", dpi=DPI)
plt.show()
### ---------------------------------------------------------------------------
#%% SECTION 5.2.1: LSTM based anomaly detection.
### ---------------------------------------------------------------------------
# Prepare the data for LSTM.
df_monthly_lstm = df_monthly_filtered
# Define the initial LSTM parameters.
time_step = 12
lstm_units = 50
batch_size = 32
epochs = 100
reg_lambda = 0.01
# Train first LSTM model with regularization.
model, history = anomaly_detection_lstm_with_regularization(df_monthly_lstm, column, time_step, lstm_units, batch_size, epochs, reg_lambda)

X, Y = create_dataset(df_monthly_lstm['scaled_sales'].values, time_step)
# Reshape input to be [samples, time steps, features].
X = X.reshape(X.shape[0], X.shape[1], 1)
# Create the LSTM.
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
# Train the model.
model.fit(X, Y, epochs=100, batch_size=32, verbose=1)
# Predict the values.
train_predict = model.predict(X)
# Invert the scaling.
train_predict = scaler.inverse_transform(train_predict)
original_data = scaler.inverse_transform(df_monthly_lstm['scaled_sales'].values.reshape(-1, 1))
# Calculate the anomalies with a higher threshold.
df_monthly_lstm['lstm_anomaly'] = 0
threshold = 0.3  # Increased the threshold for anomaly detection
for i in range(len(train_predict)):
    if abs(original_data[i + time_step] - train_predict[i]) > threshold * original_data[i + time_step]:
        df_monthly_lstm.iloc[i + time_step, df_monthly_lstm.columns.get_loc('lstm_anomaly')] = 1
#%% Plot the results.
fig, ax = plt.subplots(figsize=(24, 11))
ax.plot(df_monthly_lstm.index, df_monthly_lstm[column] / 1000, label='BMW i3 Sales (Thousands)', color=bmw_color)
anomalies = df_monthly_lstm[df_monthly_lstm['lstm_anomaly'] == 1]
ax.scatter(anomalies.index, anomalies[column] / 1000, color=purple, label='LSTM Anomaly', s=100, edgecolor='k')
# Add events to the plot with different shades of light purple.
shades = [light_purple, '#D8BFD8', '#E6E6FA', '#EEE8AA', '#F0E68C']
for i, (_, event) in enumerate(df_events[df_events['detected_in'].str.contains('i3', na=False)].iterrows()):
    start_date = pd.to_datetime(event['start'])
    end_date = pd.to_datetime(event['end'])
    label = event['name']
    color = shades[i % len(shades)]
    if start_date == end_date:
        ax.axvline(start_date, color='black', linestyle='-.', label=label)
        ax.text(start_date, ax.get_ylim()[1] * 0.4, label, rotation=90, verticalalignment='bottom', fontsize=20, color='black')
    else:
        ax.axvspan(start_date, end_date, color=color, alpha=0.3, linestyle='--', label=label)
        ax.text(start_date, ax.get_ylim()[1] * 0.65, label, rotation=90, verticalalignment='bottom', fontsize=20, color='Black')
ax.set_title('BMW i3 Sales in Germany with LSTM (initial state) Anomalies and Events $\\tau = 0.3$')
ax.set_xlabel('Date')
ax.set_ylabel('Monthly Sales Volume (in Thousands)')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.savefig(FP_FIGURES+"chap05_bmw_i3_sales_lstm_anomalies_events.png", dpi=DPI)
plt.show()
#%%
# Plot the LSTM loss over epochs.
history = model.fit(X, Y, epochs=100, batch_size=32, verbose=1)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.title('LSTM Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(FP_FIGURES + "chap05_lstm_loss_over_epochs.png", dpi=DPI)
plt.show()
### ---------------------------------------------------------------------------
#%% GAN based anomaly detection.
### ---------------------------------------------------------------------------
# Define the generator model.
def build_generator():
    input_layer = Input(shape=(time_step, 1))
    hidden_layer = Dense(50, activation='relu')(input_layer)
    output_layer = Dense(1)(hidden_layer)
    return Model(input_layer, output_layer)
# Define the discriminator model.
def build_discriminator():
    input_layer = Input(shape=(time_step,))
    hidden_layer = Dense(50, activation='relu')(input_layer)
    output_layer = Dense(1, activation='sigmoid')(hidden_layer)
    return Model(input_layer, output_layer)
# Compile the GAN.
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
# Combine the generator and discriminator into a GAN model.
discriminator.trainable = False
gan_input = Input(shape=(time_step,))
generated_data = generator(gan_input)
gan_output = discriminator(generated_data)
gan = Model(gan_input, gan_output)
gan.compile(optimizer=Adam(), loss='binary_crossentropy')
# Train the GAN.
epochs = 10000
batch_size = 32
for epoch in range(epochs):
    # Train the discriminator.
    idx = np.random.randint(0, X.shape[0], batch_size)
    real_data = X[idx]
    fake_data = generator.predict(real_data)
    d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    # Train the generator.
    g_loss = gan.train_on_batch(real_data, np.ones((batch_size, 1)))
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Discriminator Loss: {d_loss[0]}, Generator Loss: {g_loss}")
# Generate predictions.
generated_data = generator.predict(X)
generated_data = generated_data.reshape(generated_data.shape[0], generated_data.shape[1])
generated_data = scaler.inverse_transform(generated_data)
# Calculate the anomalies.
df_monthly_lstm['gan_anomaly'] = 0
threshold = 0.3  # Define a threshold for anomaly detection
for i in range(len(generated_data)):
    if np.any(abs(original_data[i + time_step] - generated_data[i]) > threshold * original_data[i + time_step]):
        df_monthly_lstm.iloc[i + time_step, df_monthly_lstm.columns.get_loc('gan_anomaly')] = 1
# Plot the results.
fig, ax = plt.subplots(figsize=(24, 6))
ax.plot(df_monthly_lstm.index, df_monthly_lstm[column] / 1000, label='BMW i3 Sales (Thousands)', color=bmw_color)
anomalies = df_monthly_lstm[df_monthly_lstm['gan_anomaly'] == 1]
ax.scatter(anomalies.index, anomalies[column] / 1000, color=purple, label='GAN Anomaly', s=100, edgecolor='k')
ax.set_title('BMW i3 Sales in Germany with GAN Anomalies')
ax.set_xlabel('Date')
ax.set_ylabel('Monthly Sales Volume (in Thousands)')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.savefig(FP_FIGURES+"chap05_bmw_i3_sales_gan_anomalies.png", dpi=DPI)
plt.show()
### ---------------------------------------------------------------------------
#%% SECTION 5.2.2
### ---------------------------------------------------------------------------
# Define the parameter grid.
param_grid = {
    'time_step': [1, 6, 12, 24],
    'lstm_units': [50, 100, 200],
    'batch_size': [16, 32, 64],
    'epochs': [50, 100, 500, 1000]
}
# Function to create and train LSTM model.
def train_lstm_model(time_step, lstm_units, batch_size, epochs):
    X, Y = create_dataset(df_monthly_lstm['scaled_sales'].values, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    model = Sequential()
    model.add(LSTM(lstm_units, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(lstm_units, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model, history
# Grid search.
best_model = None
best_loss = float('inf')
best_params = None
for params in ParameterGrid(param_grid):
    model, history = train_lstm_model(params['time_step'], params['lstm_units'], params['batch_size'], params['epochs'])
    loss = history.history['loss'][-1]
    if loss < best_loss:
        best_loss = loss
        best_model = model
        best_params = params
print(f"Best parameters: {best_params}")
print(f"Best loss: {best_loss}")
# Use the best model to predict and detect anomalies.
time_step = best_params['time_step']
X, Y = create_dataset(df_monthly_lstm['scaled_sales'].values, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)
train_predict = best_model.predict(X)
train_predict = scaler.inverse_transform(train_predict)
original_data = scaler.inverse_transform(df_monthly_lstm['scaled_sales'].values.reshape(-1, 1))
#
df_monthly_lstm['lstm_anomaly'] = 0
threshold = 0.2
for i in range(len(train_predict)):
    if abs(original_data[i + time_step] - train_predict[i]) > threshold * original_data[i + time_step]:
        df_monthly_lstm.iloc[i + time_step, df_monthly_lstm.columns.get_loc('lstm_anomaly')] = 1
# Plot the results.
fig, ax = plt.subplots(figsize=(24, 11))
ax.plot(df_monthly_lstm.index, df_monthly_lstm[column] / 1000, label='BMW i3 Sales (Thousands)', color=bmw_color)
anomalies = df_monthly_lstm[df_monthly_lstm['lstm_anomaly'] == 1]
ax.scatter(anomalies.index, anomalies[column] / 1000, color=purple, label='LSTM Anomaly', s=100, edgecolor='k')
# Add events to the plot with different shades of light purple
shades = [light_purple, '#D8BFD8', '#E6E6FA', '#EEE8AA', '#F0E68C']
for i, (_, event) in enumerate(df_events[df_events['detected_in'].str.contains('i3', na=False)].iterrows()):
    start_date = pd.to_datetime(event['start'])
    end_date = pd.to_datetime(event['end'])
    label = event['name']
    color = shades[i % len(shades)]
    if start_date == end_date:
        ax.axvline(start_date, color='black', linestyle='--', label=label)
        ax.text(start_date, ax.get_ylim()[1] * 0.4, label, rotation=90, verticalalignment='bottom', fontsize=20, color='black')
    else:
        ax.axvspan(start_date, end_date, color=color, alpha=0.3, label=label)
        ax.text(start_date, ax.get_ylim()[1] * 0.65, label, rotation=90, verticalalignment='bottom', fontsize=20, color='Black')
ax.set_title('BMW i3 Sales in Germany with LSTM Anomalies (Grid Search) $\\tau = 0.1$')
ax.set_xlabel('Date')
ax.set_ylabel('Monthly Sales Volume (in Thousands)')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.savefig(FP_FIGURES+"chap05_bmw_i3_sales_lstm_anomalies_grid_search_tau_01.png", dpi=DPI)
plt.show()
# Plot the LSTM loss over epochs.
history = model.fit(X, Y, epochs=500, batch_size=16, verbose=1)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.title('LSTM Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(FP_FIGURES + "chap05_lstm_loss_over_epochs.png", dpi=DPI)
plt.show()
#%%
# Define the parameter grid with regularization
param_grid_with_reg = {
    'time_step': [1, 6, 12, 24],
    'lstm_units': [50, 100, 200],
    'batch_size': [16, 32, 64],
    'epochs': [50, 100, 500, 1000],
    'reg_lambda': [0.001, 0.01, 0.1]
}
# Grid search with regularization
best_model_reg = None
best_loss_reg = float('inf')
best_params_reg = None
#
for params in ParameterGrid(param_grid_with_reg):
    model, history = anomaly_detection_lstm_with_regularization(params['time_step'], params['lstm_units'], params['batch_size'], params['epochs'], params['reg_lambda'])
    loss = history.history['loss'][-1]
    #
    if loss < best_loss_reg:
        best_loss_reg = loss
        best_model_reg = model
        best_params_reg = params
#
print(f"Best parameters with regularization: {best_params_reg}")
print(f"Best loss with regularization: {best_loss_reg}")
### ---------------------------------------------------------------------------
# Use the best model with regularization to predict and detect anomalies
time_step = best_params_reg['time_step']
X, Y = create_dataset(df_monthly_lstm['scaled_sales'].values, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)
train_predict = best_model_reg.predict(X)
train_predict = scaler.inverse_transform(train_predict)
original_data = scaler.inverse_transform(df_monthly_lstm['scaled_sales'].values.reshape(-1, 1))
#
df_monthly_lstm['lstm_anomaly_reg'] = 0
threshold = 0.2
for i in range(len(train_predict)):
    if abs(original_data[i + time_step] - train_predict[i]) > threshold * original_data[i + time_step]:
        df_monthly_lstm.iloc[i + time_step, df_monthly_lstm.columns.get_loc('lstm_anomaly_reg')] = 1
### ---------------------------------------------------------------------------
# Save the best LSTM model with regularization to a file
best_model_reg.export(os.path.join(FP_PROJECT, "models/anomaly_model_best.pth"))
### ---------------------------------------------------------------------------
# Plot the results with regularization
fig, ax = plt.subplots(figsize=(24, 11))
ax.plot(df_monthly_lstm.index, df_monthly_lstm[column] / 1000, label='BMW i3 Sales (Thousands)', color=bmw_color)
anomalies = df_monthly_lstm[df_monthly_lstm['lstm_anomaly_reg'] == 1]
ax.scatter(anomalies.index, anomalies[column] / 1000, color=purple, label='LSTM Anomaly with Regularization', s=100, edgecolor='k')
#
# Add events to the plot with different shades of light purple
shades = [light_purple, '#D8BFD8', '#E6E6FA', '#EEE8AA', '#F0E68C']
for i, (_, event) in enumerate(df_events[df_events['detected_in'].str.contains('i3', na=False)].iterrows()):
    start_date = pd.to_datetime(event['start'])
    end_date = pd.to_datetime(event['end'])
    label = event['name']
    color = shades[i % len(shades)]
    if start_date == end_date:
        ax.axvline(start_date, color='black', linestyle='--', label=label)
        ax.text(start_date, ax.get_ylim()[1] * 0.4, label, rotation=90, verticalalignment='bottom', fontsize=20, color='black')
    else:
        ax.axvspan(start_date, end_date, color=color, alpha=0.3, label=label)
        ax.text(start_date, ax.get_ylim()[1] * 0.65, label, rotation=90, verticalalignment='bottom', fontsize=20, color='Black')
#
ax.set_title('BMW i3 Sales in Germany with LSTM Anomalies (Regularization) $\\tau = 0.2$')
ax.set_xlabel('Date')
ax.set_ylabel('Monthly Sales Volume (in Thousands)')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.savefig(FP_FIGURES+"chap05_bmw_i3_sales_lstm_anomalies_regularization.png", dpi=DPI)
plt.show()
### ---------------------------------------------------------------------------
### End.
#### ==========================================================================