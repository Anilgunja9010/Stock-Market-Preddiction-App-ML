import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Stock Market Predictor",
    layout="centered"
)

# ----------------------------------------------------
# Custom Page Styling
# ----------------------------------------------------
st.markdown("""
<style>
.stApp { background-color: #0f172a; }
html, body, [class*="css"]  { color: white !important; }
h1, h2, h3, h4, h5, h6 { color: white !important; }
label { color: white !important; }
.stTabs [data-baseweb="tab-list"] {  padding: 10px; border-radius: 12px; }
.stTabs [data-baseweb="tab"] { 
    background-color: #1e293b !important; 
    color: white !important; 
    font-size: 18px !important; 
    border-radius: 10px !important; 
    margin-right: 8px; 
    padding: 10px 20px; 
}
.stTabs [aria-selected="true"] { 
    background-color: #38bdf8 !important; 
    color: black !important; 
    font-weight: bold; 
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Load Machine Learning Model
# ----------------------------------------------------
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "Stock_Predictions_Model.keras")

if not os.path.exists(model_path):
    st.error("Model file not found. Please upload 'Stock_Predictions_Model.keras'.")
    st.stop()

model = load_model(model_path)

# ----------------------------------------------------
# App Header
# ----------------------------------------------------
st.markdown("""
<h1 style="text-align: center;">Stock Market Predictor</h1>
<h4 style="text-align: center;">Machine Learning Based Forecasting</h4>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# User Input Section
# ----------------------------------------------------
stock = st.text_input("Enter Stock Symbol (example: GOOG, HDFCBANK.NS)", "GOOG")
start = '2012-01-01'
end = '2024-12-31'

# Fetch historical price data
data = yf.download(stock, start, end)

# ----------------------------------------------------
# Tabs Layout
# ----------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Stock Data", "Charts", "Prediction"])

# ------------------ TAB 1: Raw Data ------------------
with tab1:
    st.subheader("Historical Price Data")
    st.write(data)

# ----------------------------------------------------
# Calculate Moving Averages
# ----------------------------------------------------
ma_50 = data.Close.rolling(50).mean()
ma_100 = data.Close.rolling(100).mean()
ma_200 = data.Close.rolling(200).mean()

# Chart 1 – Price + MA50
fig1 = plt.figure(figsize=(8, 6))
plt.plot(data.Close, label="Close Price")
plt.plot(ma_50, label="50-Day Moving Average")
plt.legend()

# Chart 2 – Price + MA50 + MA100
fig2 = plt.figure(figsize=(8, 6))
plt.plot(data.Close, label="Close Price")
plt.plot(ma_50, label="50-Day MA")
plt.plot(ma_100, label="100-Day MA")
plt.legend()

# Chart 3 – Price + MA100 + MA200
fig3 = plt.figure(figsize=(8, 6))
plt.plot(data.Close, label="Close Price")
plt.plot(ma_100, label="100-Day MA")
plt.plot(ma_200, label="200-Day MA")
plt.legend()

# ------------------ TAB 2: Charts ------------------
with tab2:
    st.subheader("Price and Moving Average Charts")
    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)

# ----------------------------------------------------
# Prepare Data for ML Prediction
# ----------------------------------------------------
data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])

scaler = MinMaxScaler(feature_range=(0, 1))

# Combine last 100 days of training with test data
past_100 = data_train.tail(100)
data_test = pd.concat([past_100, data_test], ignore_index=True)

# Scale data
scaled_data = scaler.fit_transform(data_test)

# Create testing sequences
x_test = []
y_test = []

for i in range(100, len(scaled_data)):
    x_test.append(scaled_data[i - 100:i])
    y_test.append(scaled_data[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)

# ----------------------------------------------------
# Make Predictions
# ----------------------------------------------------
predicted = model.predict(x_test)

# Reverse scaling
scale_factor = 1 / scaler.scale_[0]
predicted = predicted * scale_factor
y_test = y_test * scale_factor

# Prediction chart
fig4 = plt.figure(figsize=(8, 6))
plt.plot(predicted, label="Predicted Price")
plt.plot(y_test, label="Actual Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()

# ------------------ TAB 3: Prediction ------------------
with tab3:
    st.subheader("Predicted vs Actual Prices")
    st.pyplot(fig4)

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.markdown("""
<br><br>
<hr style="border:1px solid #334155">
<div style="text-align:center; color:white; font-size:16px;">
    © 2025 | Stock Market Prediction System <br>
    Developed by <b style="color:#38bdf8;">Anil Gunja</b>
</div>
""", unsafe_allow_html=True)
