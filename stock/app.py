import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ------------------ PAGE CONFIG (FIRST LINE AFTER IMPORTS) ------------------
st.set_page_config(page_title="Stock Market Predictor", layout="centered")

# ------------------ APP STYLE ------------------
st.markdown("""
<style>
.stApp { background-color: #0f172a; }
html, body, [class*="css"]  { color: white !important; }
h1, h2, h3, h4, h5, h6 { color: white !important; }
label { color: white !important; }
.stTabs [data-baseweb="tab-list"] { background-color: #020617; padding: 10px; border-radius: 12px; }
.stTabs [data-baseweb="tab"] { background-color: #1e293b !important; color: white !important; font-size: 18px !important; border-radius: 10px !important; margin-right: 8px; padding: 10px 20px; }
.stTabs [aria-selected="true"] { background-color: #38bdf8 !important; color: black !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL (✅ FIXED) ------------------
model = load_model("Stock_Predictions_Model.keras")

# ------------------ TITLE ------------------
st.markdown("""
<h1 style='text-align: center;'>Stock Market Predictor</h1>
<h4 style='text-align: center;'>AI Powered Stock Forecasting App</h4>
""", unsafe_allow_html=True)

# ------------------ USER INPUT ------------------
stock = st.text_input("Enter Stock Symbol (Example: GOOG, HDFCBANK.NS)", "GOOG")
start = '2012-01-01'
end = '2024-12-31'
data = yf.download(stock, start, end)

# ------------------ TABS ------------------
tab1, tab2, tab3 = st.tabs(["📊 Stock Data", "📈 Charts", "🤖 Prediction"])

with tab1:
    st.subheader("📊 Stock Data")
    st.write(data)

# ------------------ MOVING AVERAGE CHARTS ------------------
ma_50 = data.Close.rolling(50).mean()
ma_100 = data.Close.rolling(100).mean()
ma_200 = data.Close.rolling(200).mean()

fig1 = plt.figure(figsize=(8,6))
plt.plot(data.Close, label="Close Price")
plt.plot(ma_50, label="MA50")
plt.legend()
st.pyplot(fig1)

fig2 = plt.figure(figsize=(8,6))
plt.plot(data.Close, label="Close Price")
plt.plot(ma_50, label="MA50")
plt.plot(ma_100, label="MA100")
plt.legend()
st.pyplot(fig2)

fig3 = plt.figure(figsize=(8,6))
plt.plot(data.Close, label="Close Price")
plt.plot(ma_100, label="MA100")
plt.plot(ma_200, label="MA200")
plt.legend()
st.pyplot(fig3)

with tab2:
    st.subheader("📈 Charts")
    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)

# ------------------ PREDICTION ------------------
data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):])

scaler = MinMaxScaler(feature_range=(0,1))
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test)

x_test, y_test = [], []

for i in range(100, data_test_scaled.shape[0]):
    x_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

predicted = model.predict(x_test)

scale = 1/scaler.scale_[0]
predicted = predicted * scale
y_test = y_test * scale

with tab3:
    st.subheader("🤖 Model Prediction")
    st.markdown("Predicted vs Original Prices")
    fig4 = plt.figure(figsize=(8,6))
    plt.plot(predicted, label="Predicted Price")
    plt.plot(y_test, label="Original Price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig4)

# ------------------ FOOTER ------------------
st.markdown("""
<br><br>
<hr style="border:1px solid #334155">
<div style="text-align:center; color:white; font-size:16px;">
    © 2025 | Stock Market Prediction System <br>
    Developed by <b style="color:#38bdf8;">Anil Gunja</b> 
</div>
""", unsafe_allow_html=True)