#### ==========================================================================
#### Dissertation models
#### Author: Simon Schramm
#### 01.06.2024
#### --------------------------------------------------------------------------
""" 
This script contains classes and methods for all models used in the dissertation.
""" 
### ---------------------------------------------------------------------------
#%% Preamble.
### ---------------------------------------------------------------------------
import numpy as np
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
### ---------------------------------------------------------------------------
#%% CHAPTER 5.
### ---------------------------------------------------------------------------
#
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step):
        a = data[i:(i + time_step)]
        X.append(a)
        Y.append(data[i + time_step])
    return np.array(X), np.array(Y)
#
def anomaly_detection_lstm_with_regularization(df_monthly_lstm, column, time_step, lstm_units, batch_size, epochs, reg_lambda):
    # Normalize the data.
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_monthly_lstm['scaled_sales'] = scaler.fit_transform(df_monthly_lstm[[column]])
    X, Y = create_dataset(df_monthly_lstm['scaled_sales'].values, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    #
    model = Sequential()
    model.add(LSTM(lstm_units, return_sequences=True, input_shape=(time_step, 1), kernel_regularizer=l2(reg_lambda)))
    model.add(LSTM(lstm_units, return_sequences=False, kernel_regularizer=l2(reg_lambda)))
    model.add(Dense(1, kernel_regularizer=l2(reg_lambda)))
    model.compile(optimizer='adam', loss='mean_squared_error')
    #
    history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model, history

### ---------------------------------------------------------------------------
### End.
#### ==========================================================================