#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, LSTM
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings("ignore")
import os
new_directory ='/Users/pankajyadav/Downloads/final project/'
os.chdir(new_directory)

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Load the data
data = pd.read_csv('^NSEBANK.csv')


# Convert 'Date' to datetime format
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Select the 'Close' price for prediction
df = data[['Close']]

# Handle missing values by forward filling
df.fillna(method='ffill', inplace=True)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Plot seasonal decomposition
result = seasonal_decompose(df, model='multiplicative', period=252)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 10))
result.observed.plot(ax=ax1, title='Observed')
result.trend.plot(ax=ax2, title='Trend')
result.seasonal.plot(ax=ax3, title='Seasonal')
result.resid.plot(ax=ax4, title='Residual')
plt.tight_layout()
plt.show()

# Plot ACF
plot_acf(data['Close'].dropna(), lags=50)
plt.show()

# Plot distribution of daily returns
daily_returns = data['Close'].pct_change().dropna()
sns.histplot(daily_returns, kde=True)
plt.title('Distribution of Daily Returns')
plt.show()

# Plot candlestick chart
fig = make_subplots(rows=1, cols=1)
candlestick = go.Candlestick(x=data.index,
                             open=data['Open'],
                             high=data['High'],
                             low=data['Low'],
                             close=data['Close'])
fig.add_trace(candlestick)
fig.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
fig.show()

# Create sequences for RNN
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

# Define sequence length
seq_length = 60

# Create sequences
X, y = create_sequences(scaled_data, seq_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Flatten data for Random Forest, Decision Tree, and Linear Regression
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)


# Create and train the model
rf_model = RandomForestRegressor()
rf_model.fit(X_train_flat, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test_flat)


# Define the parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# Perform Grid Search with Cross-Validation for Random Forest
rf_model_gs = RandomForestRegressor(random_state=42)
rf_grid_search = GridSearchCV(estimator=rf_model_gs, param_grid=rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
rf_grid_search.fit(X_train_flat, y_train)
rf_best_model = rf_grid_search.best_estimator_


# Create and train the model
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train_flat, y_train)

# Make predictions
y_pred_dt = dt_model.predict(X_test_flat)

# Define the parameter grid for Decision Tree
dt_param_grid = {
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# Perform Grid Search with Cross-Validation for Decision Tree
dt_model_gs = DecisionTreeRegressor(random_state=42)
dt_grid_search = GridSearchCV(estimator=dt_model_gs, param_grid=dt_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
dt_grid_search.fit(X_train_flat, y_train)
dt_best_model = dt_grid_search.best_estimator_


# Create and train the model
lr_model = LinearRegression()
lr_model.fit(X_train_flat, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test_flat)

# Define the parameter grid for Linear Regression
lr_param_grid = {
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'positive': [True, False]
}
# Linear Regression with Cross-Validation
lr_model_gs = LinearRegression()
lr_grid_search = GridSearchCV(estimator = lr_model_gs, param_grid=lr_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
lr_grid_search.fit(X_train_flat, y_train)
lr_best_model = lr_grid_search.best_estimator_

# Ridge Regression
ridge_model = Ridge(random_state=42)
ridge_model.fit(X_train_flat, y_train)
y_pred_ridge = ridge_model.predict(X_test_flat)

# Create and train the SVR model
svr_model = SVR(kernel='rbf', C=500, epsilon=0.2)
svr_model.fit(X_train_flat, y_train)
y_pred_svr = svr_model.predict(X_test_flat)

# Define the parameter grid for SVR
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.2, 0.5],
    'gamma': ['scale', 'auto']  # Only used for 'rbf', 'poly' and 'sigmoid'
}

# Perform Grid Search with Cross-Validation for SVR
svr_gs = SVR()
svr_grid_search = GridSearchCV(estimator=svr_gs, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
svr_grid_search.fit(X_train_flat, y_train)
svr_best_model = svr_grid_search.best_estimator_

# LGBMR Model
lgbm_model = LGBMRegressor(random_state=42)
lgbm_model.fit(X_train_flat, y_train)
y_pred_lgbm = lgbm_model.predict(X_test_flat)

# LSTM
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train_flat.shape[1], 1)))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train_flat, y_train, epochs=50, batch_size=32, verbose=0)
y_pred_lstm = lstm_model.predict(X_test_flat)

# Build and train the RNN model
rnn_model = Sequential()
rnn_model.add(SimpleRNN(units=50, input_shape=(X_train_flat.shape[1], 1)))
rnn_model.add(Dense(1))
rnn_model.compile(optimizer='adam', loss='mean_squared_error')
rnn_model.fit(X_train_flat, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
y_pred_rnn = rnn_model.predict(X_test_flat)


# Evaluate the models
def evaluate_model(model, X_test, y_test, model_type='rnn'):
    if model_type == 'rnn':
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(np.concatenate((predictions.reshape(-1, 1), np.zeros((predictions.shape[0], 1))), axis=1))[:, 0]
    else:
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(np.concatenate((predictions.reshape(-1, 1), np.zeros((predictions.shape[0], 1))), axis=1))[:, 0]
    
    actual_prices = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 1))), axis=1))[:, 0]
    rmse = np.sqrt(mean_squared_error(actual_prices, predictions))
    r2 = r2_score(actual_prices, predictions)
    bias = np.mean(predictions - actual_prices)
    return rmse, r2, bias

# Predict next 3 days close price
def predict_next_days(model, data, days=3):
    predictions = []
    current_sequence = data[-seq_length:].reshape(1, -1)  # Start with the last known sequence

    for _ in range(days):
        prediction = model.predict(current_sequence)
        predictions.append(prediction[0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[0, -1] = prediction[0]

    return np.array(predictions)

# RANDOM FOREST
rf_rmse, rf_r2, rf_bias = evaluate_model(rf_model, X_test_flat, y_test, model_type='rf')
print(f'Random Forest Before CV and GS - RMSE: {rf_rmse}, R-squared: {rf_r2}, Bias: {rf_bias}')

# Plot the results
plt.figure(figsize=(6, 5))
plt.plot(y_test, color='blue', label='Actual Prices')
plt.plot(y_pred_rf, color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Evaluate Random Forest
rf_rmse, rf_r2, rf_bias = evaluate_model(rf_best_model, X_test_flat, y_test, model_type='rf')
print(f'Best Random Forest After CV and GS - RMSE: {rf_rmse}, R-squared: {rf_r2}, Bias: {rf_bias}')

# Plot the results
plt.figure(figsize=(6, 5))
plt.plot(y_test, color='blue', label='Actual Prices')
plt.plot(rf_grid_search.predict(X_test_flat), color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# DECISION TREE
dt_rmse, dt_r2, dt_bias = evaluate_model(dt_model, X_test_flat, y_test, model_type='dt')
print(f'Decision Tree Before CV and GS - RMSE: {dt_rmse}, R-squared: {dt_r2}, Bias: {dt_bias}')

# Plot the results
plt.figure(figsize=(6, 5))
plt.plot(y_test, color='blue', label='Actual Prices')
plt.plot(y_pred_dt, color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Evaluate Decision Tree
dt_rmse, dt_r2, dt_bias = evaluate_model(dt_best_model, X_test_flat, y_test, model_type='dt')
print(f'Best Decision Tree after CV and GS- RMSE: {dt_rmse}, R-squared: {dt_r2}, Bias: {dt_bias}')

plt.figure(figsize=(6, 5))
plt.plot(y_test, color='blue', label='Actual Prices')
plt.plot(dt_grid_search.predict(X_test_flat), color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# LINEAR REGRESSION
lr_rmse, lr_r2, lr_bias = evaluate_model(lr_model, X_test_flat, y_test, model_type='lr')
print(f'linear regression Before CV and GS - RMSE: {lr_rmse}, R-squared: {lr_r2}, Bias: {lr_bias}')

# Plot the results
plt.figure(figsize=(6, 5))
plt.plot(y_test, color='blue', label='Actual Prices')
plt.plot(y_pred_lr, color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Evaluate Linear Regression
lr_rmse, lr_r2, lr_bias = evaluate_model(lr_best_model, X_test_flat, y_test, model_type='lr')
print(f'Best Linear Regression After CV and GS - RMSE: {lr_rmse}, R-squared: {lr_r2}, Bias: {lr_bias}')

plt.figure(figsize=(6, 5))
plt.plot(y_test, color='blue', label='Actual Prices')
plt.plot(lr_model.predict(X_test_flat), color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

ridge_rmse, ridge_r2, ridge_bias = evaluate_model(ridge_model, X_test_flat, y_test, model_type='ridge')
print(f'Ridge Regression - RMSE: {ridge_rmse}, R-squared: {ridge_r2}, Bias: {ridge_bias}')

plt.figure(figsize=(6, 5))
plt.plot(y_test, color='blue', label='Actual Prices')
plt.plot(y_pred_ridge, color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

svr_rmse, svr_r2, svr_bias = evaluate_model(svr_model, X_test_flat, y_test, model_type='svr')
print(f'SVR - RMSE: {svr_rmse}, R-squared: {svr_r2}, Bias: {svr_bias}')

plt.figure(figsize=(6, 5))
plt.plot(y_test, color='blue', label='Actual Prices')
plt.plot(y_pred_svr, color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Evaluate Linear Regression
svr_rmse, svr_r2, svr_bias = evaluate_model(svr_best_model, X_test_flat, y_test, model_type='svr')
print(f'Best SVR After CV and GS - RMSE: {svr_rmse}, R-squared: {svr_r2}, Bias: {svr_bias}')

plt.figure(figsize=(6, 5))
plt.plot(y_test, color='blue', label='Actual Prices')
plt.plot(lr_model.predict(X_test_flat), color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

lgbm_rmse, lgbm_r2, lgbm_bias = evaluate_model(lgbm_model, X_test_flat, y_test, model_type='lgbm')
print(f'LGBMR - RMSE: {lgbm_rmse}, R-squared: {lgbm_r2}, Bias: {lgbm_bias}')

plt.figure(figsize=(6, 5))
plt.plot(y_test, color='blue', label='Actual Prices')
plt.plot(y_pred_lgbm, color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

lstm_rmse, lstm_r2, lstm_bias = evaluate_model(lstm_model, X_test_flat, y_test, model_type='lstm')
print(f'LSTM - RMSE: {lstm_rmse}, R-squared: {lstm_r2}, Bias: {lstm_bias}')

plt.figure(figsize=(6, 5))
plt.plot(y_test, color='blue', label='Actual Prices')
plt.plot(y_pred_lstm, color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

rnn_rmse, rnn_r2, rnn_bias = evaluate_model(rnn_model, X_test_flat, y_test, model_type='rnn')
print(f'RNN - RMSE: {rnn_rmse}, R-squared: {rnn_r2}, Bias: {rnn_bias}')

plt.figure(figsize=(6, 5))
plt.plot(y_test, color='blue', label='Actual Prices')
plt.plot(y_pred_rnn, color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Predict next day close price
rf_predictions = predict_next_days(rf_best_model, scaled_data, days=1)
dt_predictions = predict_next_days(dt_best_model, scaled_data, days=1)
lr_predictions = predict_next_days(lr_best_model, scaled_data, days=1)
ridge_predictions = predict_next_days(ridge_model, scaled_data, days=1)
svr_predictions = predict_next_days(svr_best_model, scaled_data, days=1)
lgbm_predictions = predict_next_days(lgbm_model, scaled_data, days=1)
rnn_predictions = predict_next_days(rnn_model, scaled_data, days=1)
lstm_predictions = predict_next_days(lstm_model, scaled_data, days=1)

# Inverse transform the predictions to get the actual price values
rf_predictions = scaler.inverse_transform(rf_predictions.reshape(-1, 1))
dt_predictions = scaler.inverse_transform(dt_predictions.reshape(-1, 1))
lr_predictions = scaler.inverse_transform(lr_predictions.reshape(-1, 1))
ridge_predictions = scaler.inverse_transform(ridge_predictions.reshape(-1, 1))
svr_predictions = scaler.inverse_transform(svr_predictions.reshape(-1, 1))
lgbm_predictions = scaler.inverse_transform(lgbm_predictions.reshape(-1, 1))
rnn_predictions = scaler.inverse_transform(rnn_predictions.reshape(-1, 1))
lstm_predictions = scaler.inverse_transform(lstm_predictions.reshape(-1, 1))

# Create a DataFrame to display the predictions
predictions_df = pd.DataFrame({
    'Random Forest': rf_predictions.flatten(),
    'Decision Tree': dt_predictions.flatten(),
    'Linear Regression': lr_predictions.flatten(),
    'Ridge Regression': ridge_predictions.flatten(),
    'SVR': svr_predictions.flatten(),
    'LGBM': lgbm_predictions.flatten(),
    'RNN': rnn_predictions.flatten(),
    'LSTM': lstm_predictions.flatten()
}, index=pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=1, freq='B'))

# Display the predictions
print(predictions_df)



# # 
