#### ==========================================================================
#### Dissertation chapter 2
#### Author: Simon Schramm
#### 13.06.2024
#### --------------------------------------------------------------------------
""" 
This contains the code for chapter 2 of the dissertation.
""" 
### ---------------------------------------------------------------------------
#%% Preamble.
### ---------------------------------------------------------------------------
import os 
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from prophet import Prophet
#
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from constants import FP_DATA, FP_FIGURES
from utils import read_csv
from figures import set_graph_options, DPI, purple
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor
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
#%% SECTION 2.1
### ---------------------------------------------------------------------------
# BMW 3-Series plots.
column = "BMW i3 Sales in Germany"
# Plot time series for each vehicle sales volume column
fig, ax = plt.subplots(figsize=(24, 6))
# Convert sales volume to hundreds
sales_volume_hundreds = df_monthly[column] / 100
# Interpolate zeros.
sales_volume_hundreds = sales_volume_hundreds.replace(0, np.nan).interpolate()
# Plot the time series
ax.plot(df_monthly.index.year + (df_monthly.index.month - 1) / 12, sales_volume_hundreds, color=bmw_color)
ax.set_title(f'{column}')
ax.set_ylabel('Monthly Sales Volume (in Hundreds)')
# Set x-axis to display years correctly
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
# Rotate x-axis labels for better readability.
ax.tick_params(axis='x', which='both', labelbottom=True, labeltop=False)
ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)  # Display y ticks on the left side with labels, and on the right side without labels
# Set ylim
ax.set_ylim(bottom=0)
# Set xlim
ax.set_xlim(2010, 2024)
# Set x-label
ax.set_xlabel('Year')
plt.savefig(FP_FIGURES+"chap2_bmw_i3_sales.png")
plt.show()
### ---------------------------------------------------------------------------
#%% Conduct tests for linearity and stationarity
### ---------------------------------------------------------------------------
column = 'BMW i3 Sales in Germany'
df_monthly_filtered = df_monthly[column]
# Filter data for the years 2014 to 2023
df_filtered = df_monthly_filtered[(df_monthly_filtered.index >= '2014-01-01') & (df_monthly_filtered.index <= '2023-12-31')]
# Compute mean and variance for each year
mean_per_year = df_filtered.resample('Y').mean()
variance_per_year = df_filtered.resample('Y').var()
# Plot the data
fig, ax = plt.subplots(figsize=(24, 10))
ax2 = ax.twinx()  # Create secondary y-axis
ax.plot(df_filtered.index, df_filtered, color=bmw_color, label='Sales Volume')
# Plot mean and variance
for year in range(2014, 2024):
    mean_value = mean_per_year.loc[f'{year}-12-31']
    variance_value = variance_per_year.loc[f'{year}-12-31']
    year_data = df_filtered[df_filtered.index.year == year]
    ax2.axhline(y=mean_value / 10000, color='#794694', linestyle='--', label=f'{year} $\mu$' if year == 2014 else "")
    ax2.fill_between(year_data.index, (mean_value - variance_value) / 10000, (mean_value + variance_value) / 10000, color='#794694', alpha=0.1, label=f'{year} $\sigma^2$' if year == 2014 else "")
    ax2.text(pd.Timestamp(f'{year}-07-01'), mean_value / 1000 + 21, f'$\mu$: {mean_value / 1000:.2f}\n$\sigma^2$: {variance_value / 1000:.2f}', color='black', ha='center', va='bottom')

ax.set_title(f'{column} (2014-2023) with mean $\mu / 1000 $ and variance $\sigma^2 / 1000 $  per year.')
ax.set_ylim(0, 2500)
ax.set_ylabel('Sales Volume')
ax.set_xlabel('Year')
ax2.set_ylabel('Mean and Variance (in hundred thousand units)')
plt.savefig(FP_FIGURES+"chap2_bmw_i3_sales_mean_variance_2014_2023.png")
plt.show()
### ---------------------------------------------------------------------------
adf_result = adfuller(df_monthly_filtered)
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])
for key, value in adf_result[4].items():
    print('Critical Values:')
    print(f'   {key}, {value}')
# Test for stationarity using KPSS test
kpss_result = kpss(df_monthly_filtered, regression='c')
print('\nKPSS Statistic:', kpss_result[0])
print('p-value:', kpss_result[1])
for key, value in kpss_result[3].items():
    print('Critical Values:')
    print(f'   {key}, {value}')
### ---------------------------------------------------------------------------
#%% Showcase time series decomposition 2014 - 2016
### ---------------------------------------------------------------------------
column = 'BMW i3 Sales in Germany'
df_monthly_filtered = df_monthly[column]
df_monthly_filtered = df_monthly[(df_monthly.index >= '2014-01-01') & (df_monthly.index <= '2015-12-31')]
# Perform time series decomposition
result = seasonal_decompose(df_monthly_filtered[column], model='multiplicative', period=12)
# Perform additive decomposition
result_additive = seasonal_decompose(df_monthly_filtered[column], model='additive', period=12)
# Plot the decomposition
fig, axes = plt.subplots(4, 2, figsize=(24, 18), sharex=True)
# Multiplicative Decomposition
axes[0, 0].set_title('Multiplicative decomposition of BMW i3 sales 2014 - 2016')
axes[0, 0].plot(result.observed, color=bmw_color)
axes[0, 0].set_ylabel('Observed')
axes[1, 0].plot(result.trend, color=bmw_color)
axes[1, 0].set_ylabel('Trend (Multiplicative)')
axes[2, 0].plot(result.seasonal, color=bmw_color)
axes[2, 0].set_ylabel('Seasonal (Multiplicative)')
axes[3, 0].plot(result.resid, color=bmw_color)
axes[3, 0].set_ylabel('Residual (Multiplicative)')
axes[3, 0].set_xlabel('Year')

# Additive Decomposition
axes[0, 1].set_title('Additive decomposition of BMW i3 sales data 2014 - 2016')
axes[0, 1].plot(result_additive.observed, color=bmw_color)
axes[0, 1].set_ylabel('Observed')
axes[1, 1].plot(result_additive.trend, color=bmw_color)
axes[1, 1].set_ylabel('Trend (Additive)')
axes[2, 1].plot(result_additive.seasonal, color=bmw_color)
axes[2, 1].set_ylabel('Seasonal (Additive)')
axes[3, 1].plot(result_additive.resid, color=bmw_color)
axes[3, 1].set_ylabel('Residual (Additive)')
axes[3, 1].set_xlabel('Year')

# Use the same scale for seasonal and residual plots
seasonal_min = min(result.seasonal.min(), result_additive.seasonal.min())
seasonal_max = max(result.seasonal.max(), result_additive.seasonal.max())
resid_min = min(result.resid.min(), result_additive.resid.min())
resid_max = max(result.resid.max(), result_additive.resid.max())

axes[2, 0].set_ylim(seasonal_min, seasonal_max)
axes[2, 1].set_ylim(seasonal_min, seasonal_max)
axes[3, 0].set_ylim(resid_min, resid_max)
axes[3, 1].set_ylim(resid_min, resid_max)

# Set x-axis to display years correctly at the bottom plots only
axes[3, 0].xaxis.set_major_locator(mdates.YearLocator())
axes[3, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[3, 1].xaxis.set_major_locator(mdates.YearLocator())
axes[3, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(axes[3, 0].xaxis.get_majorticklabels(), ha='right')
plt.setp(axes[3, 1].xaxis.get_majorticklabels(), ha='right')

# Reduce margins
plt.tight_layout()

plt.savefig(FP_FIGURES+"chap2_bmw_i3_sales_decomposition_comparison_2014-2016.png")
plt.show()
### ---------------------------------------------------------------------------
#%% Showcase time series decomposition 2014 - 2019
### ---------------------------------------------------------------------------
column = 'BMW i3 Sales in Germany'
df_monthly_filtered = df_monthly[column]
df_monthly_filtered = df_monthly[(df_monthly.index >= '2014-01-01') & (df_monthly.index <= '2018-12-31')]
# Perform time series decomposition
result = seasonal_decompose(df_monthly_filtered[column], model='multiplicative', period=3)
# Perform additive decomposition
result_additive = seasonal_decompose(df_monthly_filtered[column], model='additive', period=3)

# Plot the decomposition
fig, axes = plt.subplots(4, 2, figsize=(24, 18), sharex=True)

# Increase font size
plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'axes.labelsize': 24, 'xtick.labelsize': 24, 'ytick.labelsize': 24})

# Multiplicative Decomposition
axes[0, 0].set_title('Multiplicative decomposition of BMW i3 sales 2014 - 2019')
axes[0, 0].plot(result.observed, color=bmw_color)
axes[0, 0].set_ylabel('Observed')
axes[1, 0].plot(result.trend, color=bmw_color)
axes[1, 0].set_ylabel('Trend (Multiplicative)')
axes[2, 0].plot(result.seasonal, color=bmw_color)
axes[2, 0].set_ylabel('Seasonal (Multiplicative)')
axes[3, 0].plot(result.resid, color=bmw_color)
axes[3, 0].set_ylabel('Residual (Multiplicative)')
axes[3, 0].set_xlabel('Year')

# Additive Decomposition
axes[0, 1].set_title('Additive decomposition of BMW i3 sales data 2014 - 2019')
axes[0, 1].plot(result_additive.observed, color=bmw_color)
axes[0, 1].set_ylabel('Observed')
axes[1, 1].plot(result_additive.trend, color=bmw_color)
axes[1, 1].set_ylabel('Trend (Additive)')
axes[2, 1].plot(result_additive.seasonal, color=bmw_color)
axes[2, 1].set_ylabel('Seasonal (Additive)')
axes[3, 1].plot(result_additive.resid, color=bmw_color)
axes[3, 1].set_ylabel('Residual (Additive)')
axes[3, 1].set_xlabel('Year')

# Use the same scale for seasonal and residual plots
seasonal_min = min(result.seasonal.min(), result_additive.seasonal.min())
seasonal_max = max(result.seasonal.max(), result_additive.seasonal.max())
resid_min = min(result.resid.min(), result_additive.resid.min())
resid_max = max(result.resid.max(), result_additive.resid.max())

axes[2, 0].set_ylim(seasonal_min, seasonal_max)
axes[2, 1].set_ylim(seasonal_min, seasonal_max)
axes[3, 0].set_ylim(resid_min, resid_max)
axes[3, 1].set_ylim(resid_min, resid_max)

# Set x-axis to display years correctly at the bottom plots only
axes[3, 0].xaxis.set_major_locator(mdates.YearLocator())
axes[3, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[3, 1].xaxis.set_major_locator(mdates.YearLocator())
axes[3, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(axes[3, 0].xaxis.get_majorticklabels(), ha='right')
plt.setp(axes[3, 1].xaxis.get_majorticklabels(), ha='right')

# Reduce margins
plt.tight_layout()

plt.savefig(FP_FIGURES+"chap2_bmw_i3_sales_decomposition_comparison_2014-2019.png")
plt.show()
### ---------------------------------------------------------------------------
#%% Showcase multivariate time series forecasting.
### ---------------------------------------------------------------------------
list_econ_col_indexed = [
    # 'GDP in Germany Annually_indexed',
    # 'Average Income in Germany Annually_indexed', 
    # 'Expenditures for alcoholic beverages in Germany Annually_indexed', 
    # 'Personal Savings in Germany Quarterly_indexed',
    # 'Disposable Personal Income in Germany Quarterly_indexed',
    # 'Consumer Spending in Germany Quarterly_indexed',
    # 'Unemployment Rate in Germany Monthly_indexed',
    'Consumer Confidence in Germany Monthly_indexed',
    'Car Production in Germany Monthly_indexed',
    'Car Registrations in Germany Monthly_indexed',
    # 'Capacity Utilization in Germany Monthly_indexed', 
    'Business Confidence in Germany Monthly_indexed',
    'Bankruptcies in Germany Monthly_indexed'
    ]

column = 'BMW i3 Sales in Germany'

# Filter data for the years 2016 to 2019
df_monthly_filtered = df_monthly[[column] + list_econ_col_indexed]
df_monthly_filtered = df_monthly_filtered[(df_monthly_filtered.index >= '2016-01-01') & (df_monthly_filtered.index <= '2019-12-31')]

# Filter data for the training period (2016-2017)
df_train = df_monthly_filtered[(df_monthly_filtered.index >= '2016-01-01') & (df_monthly_filtered.index <= '2017-12-31')]

# Aggregate the training data to quarterly
df_train_monthly = df_train
X_train_monthly = df_train[list_econ_col_indexed]
y_train_monthly = df_train[column]

# Define the new forecast period until 2019-12-31
forecast_period_long_term = pd.date_range(start='2018-01-01', end='2019-12-31', freq='ME')

# Create a DataFrame for the long-term forecast period with the economic indicators
df_forecast_long_term = pd.DataFrame(index=forecast_period_long_term)
for col in list_econ_col_indexed:
    df_forecast_long_term[col] = df_monthly_filtered[col].loc['2016-01-01':'2019-12-31']

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train_monthly)
y_train_scaled = scaler_y.fit_transform(y_train_monthly.values.reshape(-1, 1))

# Prepare the data for Prophet
df_prophet = df_train_monthly.reset_index()
df_prophet.rename(columns={'Timestamp': 'ds', column: 'y'}, inplace=True)

# Add economic indicators as regressors
for col in list_econ_col_indexed:
    df_prophet[col] = X_train_monthly[col].values

# Initialize and fit the Prophet model with additional parameters
model_prophet = Prophet()
for col in list_econ_col_indexed:
    model_prophet.add_regressor(col)
model_prophet.fit(df_prophet)

# Prepare forecast data for Prophet
df_forecast_prophet = df_forecast_long_term.reset_index()
df_forecast_prophet.rename(columns={'index': 'ds'}, inplace=True)

# Add economic indicators to the forecast data
for col in list_econ_col_indexed:
    df_forecast_prophet[col] = df_forecast_long_term[col].values

# Make long-term predictions for the forecast period
forecast = model_prophet.predict(df_forecast_prophet)
y_pred_prophet_long_term = forecast['yhat'].values
y_pred_prophet_long_term[0] = df_monthly_filtered[column][df_monthly_filtered.index == '2018-01-31']

# Filter data for the years 2016 to 2019
df_monthly_filtered = df_monthly[[column] + list_econ_col_indexed]
df_monthly_filtered = df_monthly_filtered[(df_monthly_filtered.index >= '2016-01-01') & (df_monthly_filtered.index <= '2018-12-31')]

# Filter data for the training period (2016-2017)
df_train = df_monthly_filtered[(df_monthly_filtered.index >= '2016-01-01') & (df_monthly_filtered.index <= '2018-07-31')]

# Aggregate the training data to quarterly
df_train_monthly = df_train
X_train_monthly = df_train[list_econ_col_indexed]
y_train_monthly = df_train[column]

# Define the new forecast period until 2019-12-31
forecast_period_short_term = pd.date_range(start='2018-07-31', end='2018-12-31', freq='ME')

# Create a DataFrame for the long-term forecast period with the economic indicators
df_forecast_short_term = pd.DataFrame(index=forecast_period_short_term)
for col in list_econ_col_indexed:
    df_forecast_short_term[col] = df_monthly_filtered[col].loc['2018-07-31':'2018-12-31']

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train_monthly)
y_train_scaled = scaler_y.fit_transform(y_train_monthly.values.reshape(-1, 1))

# Make predictions for the forecast period
sarima_model = SARIMAX(y_train_scaled, order=(0, 1, 0), seasonal_order=(0, 1, 0, 10), exog=X_train_scaled)
sarima_result = sarima_model.fit(disp=False)

# Make predictions for the forecast period
y_pred_scaled = sarima_result.predict(start=len(y_train_scaled), end=len(y_train_scaled) + len(df_forecast_short_term) - 1, exog=scaler_X.transform(df_forecast_short_term[list_econ_col_indexed]))
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_pred[0] = df_monthly_filtered.loc['2018-07-31', column]

#%%
# Plot the long-term and short-term forecasted data side by side
fig, axes = plt.subplots(1, 2, figsize=(32, 9))
# Long-term forecast plot
ax = axes[0]
ax.plot(df_monthly_filtered.index[df_monthly_filtered.index <= '2018-01-31'], 
    df_monthly_filtered[column][df_monthly_filtered.index <= '2018-01-31'], 
    label='Actual Data', color=bmw_color)
ax.plot(df_monthly_filtered.index[df_monthly_filtered.index > '2017-12-31'], 
    df_monthly_filtered[column][df_monthly_filtered.index > '2017-12-31'], 
    label='Actual Data (after 2017-12-31)', color=bmw_color, linestyle='--')
ax.plot(forecast_period_long_term, y_pred_prophet_long_term, label='Prophet Forecasted Data (Quarterly)', color='green', linestyle='-.')

# Create secondary y-axis for exogenous factors
ax2 = ax.twinx()
for col in list_econ_col_indexed:
    ax2.plot(df_monthly_filtered.index, df_monthly_filtered[col], linestyle=':', label=col)

ax.set_title(f'{column} Long-Term Forecast until Dec 2019')
ax.set_ylabel('Sales Volume')
ax.set_xlabel('Date')
ax.legend(loc='upper left')
ax2.set_ylabel('Exogenous Factors')
ax2.legend(loc='upper right')

# Short-term forecast plot
ax = axes[1]
ax.plot(df_monthly_filtered.index[df_monthly_filtered.index <= '2018-07-31'], 
    df_monthly_filtered[column][df_monthly_filtered.index <= '2018-07-31'], 
    label='Actual Data', color=bmw_color)
ax.plot(df_monthly_filtered.index[df_monthly_filtered.index > '2018-06-30'], 
    df_monthly_filtered[column][df_monthly_filtered.index > '2018-06-30'], 
    label='Actual Data (after 2018-06-30)', color=bmw_color, linestyle='--')
ax.plot(df_monthly_filtered.index[df_monthly_filtered.index > '2018-06-30'], y_pred, label='Forecasted Data', color='purple', linestyle=':')
# Shade the area between July 1st and December 31st in every year
for year in range(df_monthly_filtered.index.year.min(), df_monthly_filtered.index.year.max() + 1):
    start_date = pd.Timestamp(f'{year}-07-31')
    end_date = pd.Timestamp(f'{year}-12-31')
    ax.axvspan(start_date, end_date, color='purple', alpha=0.1)

ax.set_title(f'{column} Forecast for Oct-Dec 2018')
ax.set_ylabel('Sales Volume')
ax.set_xlabel('Date')
ax.legend()

plt.tight_layout()
plt.savefig(FP_FIGURES + "chap2_bmw_i3_sales_forecast_combined.png")
plt.show()

#%%
# check for multicolinearity
# Combine the target column with the economic indicators
df_vif = df_monthly_filtered

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["feature"] = df_vif.columns
vif_data["VIF"] = [variance_inflation_factor(df_vif.values, i) for i in range(len(df_vif.columns))]

print(vif_data)

#%%
# Plot indexed economic indicators together with BMW i3 sales
fig, axes = plt.subplots(2, 1, figsize=(24, 16), sharex=True)

# Plot BMW i3 Sales in Germany on the top subplot
ax1 = axes[0]
ax1.plot(df_monthly_filtered.index, df_monthly_filtered[column], color=bmw_color, linewidth=2, label=str(column))
ax1.set_title(str(column) + ' (2010-2019)')
ax1.set_ylabel(str(column))

# Plot economic indicators on the bottom subplot
ax2 = axes[1]
markers = ['o', 's', 'D', '^', 'v']
colors = ['#794694', '#8a5aa3', '#9b6eb2', '#ac82c1', '#bd96d0']
for col, color, marker in zip(list_econ_col_indexed, colors, markers):
    if col in df_monthly_filtered.columns:
        ax2.plot(df_monthly_filtered.index, df_monthly_filtered[col], color=color, marker=marker, linestyle='-', linewidth=2, label=col)

ax2.set_title('Selected Economic Indicators (2010-2019)')
ax2.set_ylabel('Economic Indicators ($1=2010$)')
ax2.set_xlabel('Year')
ax2.legend()

# Set x-axis to display years correctly
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Save and show the plot
plt.tight_layout()
plt.savefig(FP_FIGURES + "chap2_bmw_i3_sales_econ_indicators_2010_2019.png")
plt.show()
### ---------------------------------------------------------------------------
#%% SECTION 2.2
### ---------------------------------------------------------------------------
cols = ["BMW 3-Series Sales in Germany (annually)", "BMW 3-Series Sales in Germany (quarterly)", "BMW 3-Series Sales in Germany (monthly)"]
# Plot time series for each vehicle sales volume column
fig, axes = plt.subplots(nrows=len(cols), figsize=(24, 6*len(cols)), sharex=True)
for i, column in enumerate(cols):
    ax = axes[i] if len(cols) > 1 else axes
    # Convert sales volume to hundreds
    sales_volume_hundreds = df_monthly[column] / 100
    # Interpolate zeros.
    sales_volume_hundreds = sales_volume_hundreds.replace(0, np.nan).interpolate()
    # Plot the time series
    ax.plot(df_monthly.index.year + (df_monthly.index.month - 1) / 12, sales_volume_hundreds, color=bmw_color)
    ax.set_title(f'{column}')
    ax.set_ylabel('Monthly Sales Volume (in Hundreds)')
    # Set x-axis to display years correctly
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
    # Rotate x-axis labels for better readability.
    ax.tick_params(axis='x', which='both', labelbottom=True, labeltop=False)
    ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)
    # Set ylim
    ax.set_ylim(bottom=0)
    # Set xlim
    ax.set_xlim(2010, 2024)
# Set x-label for the bottom subplot
axes[-1].set_xlabel('Year')
# Adjust spacing between subplots
fig.subplots_adjust(hspace=0.25)
plt.savefig(FP_FIGURES+"chap2_bmw_3-series_sales.png")
plt.show()
### ---------------------------------------------------------------------------
#%% Showcase a short term TSF.
### ---------------------------------------------------------------------------
# Short-term ARIMA forecast for 2017

# Filter data for the year 2017
df_2017 = df_monthly[(df_monthly.index >= '2017-01-01') & (df_monthly.index <= '2017-12-31')]

# Fit SARIMAX model
# Optimize SARIMAX model using grid search

# Define the p, d, q parameters to take any value between 0 and 2
p = d = q = range(0, 3)

# Generate all different combinations of p, d and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, d and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

# Perform grid search to find the best parameters
best_aic = float("inf")
best_pdq = None
best_seasonal_pdq = None
best_model = None

# Prepare exogenous variables
exog_2017 = df_2017[list_econ_col_indexed]

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            temp_model = SARIMAX(df_2017[column], order=param, seasonal_order=param_seasonal, exog=exog_2017)
            temp_result = temp_model.fit(disp=False)
            if temp_result.aic < best_aic:
                best_aic = temp_result.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal
                best_model = temp_result
        except:
            continue
#%%
# Fit the best model
sarimax_model = SARIMAX(df_2017[column], order=(best_pdq), seasonal_order=(2,1,3,5), exog=exog_2017)
sarimax_result = sarimax_model.fit()

# Prepare exogenous variables for the forecast period
exog_forecast = df_monthly[(df_monthly.index >= '2018-01-01') & (df_monthly.index <= '2018-12-31')][list_econ_col_indexed]

# Make forecast
forecast_2017 = sarimax_result.get_forecast(steps=12, exog=exog_forecast).predicted_mean
# Get confidence intervals
forecast_2017_ci = sarimax_result.get_forecast(steps=12, exog=exog_forecast).conf_int()

# Plot the actual data, forecasted data, and confidence intervals
fig, ax = plt.subplots(figsize=(24, 6))
ax.plot(df_2017.index, df_2017[column], label='Actual Data', color=bmw_color)
ax.plot(forecast_2017.index, forecast_2017, label='ARIMA Forecast', color=purple, linestyle='--')
ax.fill_between(forecast_2017.index, forecast_2017_ci.iloc[:, 0], forecast_2017_ci.iloc[:, 1], color=purple, alpha=0.2, label='95% Confidence Interval')
# Also plot the actual data in the forecasting period
ax.plot(df_monthly[(df_monthly.index >= '2017-12-01') & (df_monthly.index <= '2018-12-31')].index, 
    df_monthly[(df_monthly.index >= '2017-12-01') & (df_monthly.index <= '2018-12-31')][column], 
    label='Actual Data (Forecast Period)', color=bmw_color, linestyle=':')
ax.set_title(f'{column} - ARIMA Forecast for 2018 with Confidence Intervals')
ax.set_ylabel('Sales Volume')
ax.set_xlabel('Date')
ax.legend()
plt.savefig(FP_FIGURES + "chap2_bmw_i3_sales_arima_forecast_2017_with_ci.png")
plt.show()
#%%
#%%
# Train LSTM model for forecasting 2018 and 2019 based on 2017 data

# Filter data for the year 2017
df_2017 = df_monthly[(df_monthly.index >= '2017-01-01') & (df_monthly.index <= '2017-12-31')]

# Prepare the data
X = df_2017[list_econ_col_indexed].values
y = df_2017[column].values

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Reshape input to be 3D [samples, timesteps, features]
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_scaled.shape[1], X_scaled.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Fit the model
model.fit(X_scaled, y_scaled, epochs=200, batch_size=1, verbose=0)

# Prepare the forecast data for 2018 and 2019
forecast_period = pd.date_range(start='2018-01-01', end='2019-12-31', freq='M')
df_forecast = pd.DataFrame(index=forecast_period)
for col in list_econ_col_indexed:
    df_forecast[col] = df_monthly[col].loc['2017-01-01':'2019-12-31']

# Scale the forecast data
X_forecast_scaled = scaler_X.transform(df_forecast[list_econ_col_indexed].values)
X_forecast_scaled = X_forecast_scaled.reshape((X_forecast_scaled.shape[0], 1, X_forecast_scaled.shape[1]))

# Make predictions
y_forecast_scaled = model.predict(X_forecast_scaled)
y_forecast = scaler_y.inverse_transform(y_forecast_scaled)
#%% Plot the actual data and forecasted data for 2017 - 2019
fig, ax = plt.subplots(figsize=(24, 6))
ax.plot(df_monthly[(df_monthly.index >= '2017-01-01') & (df_monthly.index <= '2019-12-31')].index, 
    df_monthly[(df_monthly.index >= '2017-01-01') & (df_monthly.index <= '2019-12-31')][column], 
    label='Actual Data', color=bmw_color)
ax.plot(forecast_period, y_forecast, label='LSTM Forecast', color='purple', linestyle='--')
ax.set_title(f'{column} - LSTM Forecast for 2018 and 2019')
ax.set_ylabel('Sales Volume')
ax.set_xlabel('Date')
ax.legend()
plt.savefig(FP_FIGURES + "chap2_bmw_i3_sales_lstm_forecast_2018_2019.png")
plt.show()


#%% Plot the actual data and forecasted data with confidence intervals for 2017 - 2019
fig, ax = plt.subplots(figsize=(24, 6))
ax.plot(df_monthly[(df_monthly.index >= '2017-01-01') & (df_monthly.index <= '2019-12-31')].index, 
    df_monthly[(df_monthly.index >= '2017-01-01') & (df_monthly.index <= '2019-12-31')][column], 
    label='Actual Data', color=bmw_color)
ax.plot(forecast_period, y_forecast, label='LSTM Forecast', color='purple', linestyle='--')
ax.fill_between(forecast_period, y_forecast - confidence_interval, y_forecast + confidence_interval, color='purple', alpha=0.2, label='95% Confidence Interval')
ax.set_title(f'{column} - LSTM Forecast for 2018 and 2019 with Confidence Intervals')
ax.set_ylabel('Sales Volume')
ax.set_xlabel('Date')
ax.legend()
plt.savefig(FP_FIGURES + "chap2_bmw_i3_sales_lstm_forecast_2018_2019_with_ci.png")
plt.show()

#%% Plot the ARIMA and LSTM forecasted data side by side
fig, axes = plt.subplots(1, 2, figsize=(32, 9))

# ARIMA forecast plot
ax = axes[0]
ax.plot(df_2017.index, df_2017[column], label='Actual Data', color=bmw_color)
ax.plot(forecast_2017.index, forecast_2017, label='ARIMA Forecast', color=purple, linestyle='--')
ax.fill_between(forecast_2017.index, forecast_2017_ci.iloc[:, 0], forecast_2017_ci.iloc[:, 1], color=purple, alpha=0.2, label='95% Confidence Interval')
ax.plot(df_monthly[(df_monthly.index >= '2017-12-01') & (df_monthly.index <= '2018-12-31')].index, 
    df_monthly[(df_monthly.index >= '2017-12-01') & (df_monthly.index <= '2018-12-31')][column], 
    label='Actual Data (Forecast Period)', color=bmw_color, linestyle=':')
ax.set_title(f'{column} - ARIMA Forecast for 2018 with Confidence Intervals')
ax.set_ylabel('Sales Volume')
ax.set_xlabel('Date')
ax.legend()

# LSTM forecast plot
ax = axes[1]
ax.plot(df_monthly[(df_monthly.index >= '2017-01-01') & (df_monthly.index <= '2019-12-31')].index, 
    df_monthly[(df_monthly.index >= '2017-01-01') & (df_monthly.index <= '2019-12-31')][column], 
    label='Actual Data', color=bmw_color)
ax.plot(forecast_period, y_forecast, label='LSTM Forecast', color='purple', linestyle='--')
ax.set_title(f'{column} - LSTM Forecast for 2018 and 2019')
ax.set_ylabel('Sales Volume')
ax.set_xlabel('Date')
ax.legend()

plt.tight_layout()
plt.savefig(FP_FIGURES + "chap2_bmw_i3_sales_arima_lstm_forecast_comparison.png")
plt.show()
#%% Conduct tests for linearity and stationarity
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

# Add shaded areas and compute means
for start, end, height, color, label in subsidy_periods:
    ax3.fill_betweenx([0, height], start, end, color=color, alpha=0.3, label=label)
    period_mean = df_monthly_filtered[start:end].mean()
    ax3.text(start + (end - start) / 2, height-500, f'$s(EUR):${height} EUR\n$\mu$(Sales): {period_mean:.2f}', color='black', ha='center', va='top')
ax.set_ylim(0, 2500)
ax.set_title(f'{column} with governmental subsidies $s$ in $EUR$ for electric vehicles in Germany')
ax.set_ylabel('BMW i3 sales volume in Germany')
ax.set_xlabel('Year')

plt.savefig(FP_FIGURES+"chap2_bmw_i3_sales_shaded_periods.png")
plt.show()
### ---------------------------------------------------------------------------
#%% SECTION 2.2
### ---------------------------------------------------------------------------
# Read library.
df_lib = read_csv(FP_DATA+"data_chapter_2_kg_survey")
df_rank = read_csv(FP_DATA+"data_chapter_2_rankings")
#
df_rank = df_rank[["Title", "Rank"]]
# df_rank.rename(columns={'Title': 'Publication Title'}, inplace=True)
### ---------------------------------------------------------------------------
#### Pre-processing.
### ---------------------------------------------------------------------------
# Merge with rankings.
df_lib = pd.merge(df_lib, df_rank, left_on='Publication Title', 
                  right_on='Title', how='outer')
df_lib['Rank'] = df_lib['Rank'].fillna(0).astype(int)
# Drop duplicates, rows without year and format years.
df_lib = df_lib.drop_duplicates()
df_lib = df_lib.dropna(subset=['Publication Year'])
df_lib['Publication Year'] = df_lib['Publication Year'].fillna(0).astype(int)
# Compute publications by year.
pub = df_lib['Publication Year'].value_counts().sort_index().astype(int)
pub.index = pub.index.astype(int)
# Compute cumulated publications.
pub_cum = pub.cumsum()
# Create common dataframe.
df_pub = pd.merge(pub, pub_cum, right_index = True, 
                  left_index = True).rename(columns = {'Publication Year_x':'Publications per year',
                                                       'Publication Year_y':'Cumulated publications per year'})                                 
# Define item type groups.
Conference_proceedings = ["conferencePaper"]
Journal_articles = ["journalArticle"]
Books_book_sections = ["book","bookSection"]
Others = ["document","thesis"]
# Consolidate item types.
df_lib["Item Type"]= np.where(df_lib["Item Type"].isin(Conference_proceedings), "Conference proceedings",
np.where(df_lib["Item Type"].isin(Journal_articles), "Journal articles",
np.where(df_lib["Item Type"].isin(Journal_articles), "Journal articles",
np.where(df_lib["Item Type"].isin(Books_book_sections), "Books and book sections",
np.where(df_lib["Item Type"].isin(Others), "Others", "Others" )))))
# Combine Publication Title and Publisher.
df_lib['Journal, Conference or Publisher'] = df_lib['Publisher']
df_lib.loc[df_lib['Journal, Conference or Publisher'].isnull(),
           'Journal, Conference or Publisher'] = df_lib['Publication Title']
df_lib['Reviewed'] = np.where(df_lib['Manual Tags']=='Reviewed', 'Reviewed in depth', 'Abstract reviewed')
# Pivotize table of journals and publication types.
df_item_type = df_lib.pivot_table('Key','Publication Year',['Item Type','Reviewed'],
                                  aggfunc='count').fillna(0).astype(int)
# Create ranking table.
df_pub_rank = df_lib[df_lib['Rank'] != 0]
df_pub_rank = df_pub_rank.groupby(['Journal, Conference or Publisher'])['Rank'].value_counts().unstack().fillna(0).astype(int)
### ---------------------------------------------------------------------------
#### Tables.
### ---------------------------------------------------------------------------
# Print table of publications by year.
print(pub.to_latex(escape=False))
# Print pivot table of item types.
print(df_item_type.to_latex(escape=False))
# Print table with rankings.
print(df_pub_rank.to_latex(escape=False))
os.chdir(fp_fig)
# Abbreviate.
years = df_pub.index
pubs = df_pub['Publications per year']
pubs = df_lib.groupby('Publication Year')['Item Type'].count()
pubs_cum = df_pub['Cumulated publications per year']
book = df_item_type['Books and book sections','Abstract reviewed']
conf = df_item_type['Conference proceedings','Abstract reviewed']
jour = df_item_type['Journal articles','Abstract reviewed']
oth = df_item_type['Others','Abstract reviewed']
rev_book = df_item_type['Books and book sections','Reviewed in depth']
rev_conf = df_item_type['Conference proceedings','Reviewed in depth']
rev_jour = df_item_type['Journal articles','Reviewed in depth']
# rev_oth = df_item_type['Others','Reviewed in depth']
### ---------------------------------------------------------------------------
# Plot publications by year and journal.
fig, ax1 = plt.subplots(figsize=figsize)
ax1.bar(years, book, label='Books and book sections', color=green, edgecolor='black')
ax1.bar(years, conf, label='Conference proceedings', color=yellow, edgecolor='black')
ax1.bar(years, jour, label='Journal articles', color=petrol, edgecolor='black')
ax1.bar(years, oth, label='Others', color=light_blue, edgecolor='black')
ax1.bar(years, rev_book, label='Books and book sections reviewed in depth', color=green,
                 hatch='//',
                 edgecolor='white')
ax1.bar(years, rev_conf, label='Conference proceedings reviewed in depth', color=yellow,
                 hatch='//',
                 edgecolor='white')
ax1.bar(years, rev_conf, label='Journal articles reviewed in depth', color=petrol,
                 hatch='//',
                 edgecolor='white')
# ax1.bar(years, rev_oth, label='Others reviewed in depth', color=light_blue,
#                  hatch='x')
ax2 = ax1.twinx()
ax2.plot(years, pubs_cum, color = grey)
ax1.set_title('Count of publications and cumulated publications by year')
ax1.set_ylabel('Count of publications (bars)')
ax2.set_ylabel('Cumulated publications (line)')
ax1.set_xlabel('Years')
ax1.legend()
ax1.ticklabel_format(axis="y", style="sci")
plt.savefig('akt_pub_by_year.png', dpi=DPI)
### ---------------------------------------------------------------------------
# Plot publications by item type.
pub_type = df_lib.groupby(['Publication Year'])['Item Type'].value_counts().unstack().fillna(0)
#
fig = plt.figure(figsize=figsize)
ax1 = pub_type.plot(kind='bar', stacked=True, figsize=figsize, colormap=bmw_colors)
ax2 = pubs_cum.plot(color = grey, secondary_y=True)
ax1.set_title('Count of publications by type')
ax1.set_ylabel('Count of publications')
ax1.ticklabel_format(axis="y", style="sci")
ax1.legend()
plt.savefig('akt_pub_by_type.png', dpi=DPI)
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
### End.
#### ==========================================================================
# %%
