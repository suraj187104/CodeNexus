import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('synthetic_electricity_demand.csv')

# Preprocessing
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month
df['weekday'] = df['timestamp'].dt.weekday

# Train models
def train_models():
    X = df[['temperature', 'humidity', 'holiday', 'real_estate_growth', 'hour', 'day', 'month', 'weekday']]
    y = df['demand']
    X_train = X.copy()
    y_train = y.copy()

    # Random Forest Model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # LSTM Model
    X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=72, verbose=0)

    # XGBoost Model
    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)

    return rf, lstm_model, xgb_model

# Function to make predictions
def preprocess_input(input_data):
    input_df = pd.DataFrame(input_data, index=[0])
    
    # Preprocess timestamp
    input_df['timestamp'] = pd.to_datetime(input_df['timestamp'])
    input_df['hour'] = input_df['timestamp'].dt.hour
    input_df['day'] = input_df['timestamp'].dt.day
    input_df['month'] = input_df['timestamp'].dt.month
    input_df['weekday'] = input_df['timestamp'].dt.weekday
    
    # Drop timestamp as it is not needed for prediction
    input_df = input_df.drop(columns=['timestamp'])
    return input_df

def predict_demand(rf, lstm_model, xgb_model, input_data):
    input_df = preprocess_input(input_data)

    # Get predictions from the models
    rf_pred = rf.predict(input_df)
    xgb_pred = xgb_model.predict(input_df)

    # Reshape for LSTM
    input_lstm = input_df.values.reshape((input_df.shape[0], 1, input_df.shape[1]))
    lstm_pred = lstm_model.predict(input_lstm)

    # Ensemble predictions (average of the three models)
    ensemble_pred = (rf_pred + lstm_pred.squeeze() + xgb_pred) / 3
    return ensemble_pred
