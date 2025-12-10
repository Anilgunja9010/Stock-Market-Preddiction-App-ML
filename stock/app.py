import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Prediction App", layout="wide")

# ------------------ MODEL PATH ------------------
BASE_DIR = os.path.dirname(os.path.abspath(_file_))
model_path = os.path.join(BASE_DIR, "Stock_Predictions_Model.keras")

if not os.path.exists(model_path):
    st.error("❌ Model file missing! Upload Stock_Predictions_Model.keras")
    st.stop()

model = load_model(model_path)

# ------------------ UI ------------------
st.title("📈 Stock Market Prediction App (ML Based)")

stock = st.text_input("Enter Stock Symbol (Example: GOOG, HDFCBANK.NS)")
start = "2012-01-01"
end = "2024-12-31"

# ------------------ FETCH DATA SAFELY ------------------
if stock:
    try:
        data = yf.download(stock, start=start, end=end)

        if data.empty:
            st.error("❌ Invalid stock symbol or no data found.")
            st.stop()

    except Exception as e:
        st.error(f"⚠ Error fetching data: {e}")
        st.stop()

    # ------------------ TABS ------------------
    tab1, tab2, tab3 = st.tabs(["📊 Stock Data", "📈 Charts", "🤖 Prediction"])

    # ======== TAB 1: RAW DATA =========
    with tab1:
        st.subheader("📊 Stock Data")
        st.dataframe(data)

    # ======== TAB 2: CHARTS =========
    with tab2:
        st.subheader("📈 Closing Price History")

        fig = plt.figure(figsize=(12, 5))
        plt.plot(data['Close'])
        plt.xlabel("Date")
        plt.ylabel("Closing Price")
        plt.title(f"{stock} Closing Price History")
        st.pyplot(fig)

    # ======== TAB 3: PREDICTION =========
    with tab3:
        st.subheader("🤖 Stock Price Prediction")

        # Prepare data
        close_prices = data["Close"].values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        last_100 = scaled_data[-100:]
        X_test = np.array([last_100])
        X_test = X_test.reshape((1, 100, 1))

        # Prediction
        pred = model.predict(X_test)
        predicted_price = scaler.inverse_transform(pred)[0][0]

        st.success(f"📌 *Predicted closing price for next day: ₹{predicted_price:.2f}*")
