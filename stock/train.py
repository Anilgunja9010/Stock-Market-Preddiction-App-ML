import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle
import os

# ----------------------------------------------------
# Download Data
# ----------------------------------------------------
stock = "GOOG"
start = "2012-01-01"
end = "2024-12-31"

data = yf.download(stock, start, end)
close_prices = data[['Close']]

# ----------------------------------------------------
# Scaling
# ----------------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# ----------------------------------------------------
# Create Training Data
# ----------------------------------------------------
x_train = []
y_train = []

for i in range(100, len(scaled_data)):
    x_train.append(scaled_data[i-100:i])
    y_train.append(scaled_data[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

# ----------------------------------------------------
# Build LSTM Model
# ----------------------------------------------------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

# ----------------------------------------------------
# Train Model
# ----------------------------------------------------
model.fit(x_train, y_train, epochs=10, batch_size=32)

# ----------------------------------------------------
# Save Model & Scaler
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model.save(os.path.join(BASE_DIR, "Stock_Predictions_Model.keras"))

with open(os.path.join(BASE_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print(" Training completed")
print(" Stock_Predictions_Model.keras saved")
print(" scaler.pkl saved")
