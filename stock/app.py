import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Stock Market Predictor",
    layout="centered"
)

# ---------------- STREAMLIT DARK UI ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
}

html, body, div, span, p, li, a, label,
h1, h2, h3, h4, h5, h6,
.stMarkdown, .stText, .stCaption,
.stMetric, .stMarkdownContainer {
    color: white !important;
}

input, textarea, select {
    background-color: #1e293b !important;
    color: white !important;
}

.stTabs [data-baseweb="tab"] {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
}

.stTabs [aria-selected="true"] {
    background-color: #38bdf8 !important;
    color: #0f172a !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL & SCALER ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = load_model(os.path.join(BASE_DIR, "Stock_Predictions_Model.keras"))

with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>Stock Market Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>ML-based Price Forecasting</h4>", unsafe_allow_html=True)

# ---------------- USER INPUT ----------------
stock = st.text_input("Enter Stock Symbol (example: GOOG, HDFCBANK.NS)", "GOOG")

start = "2012-01-01"
end = "2024-12-31"

data = yf.download(stock, start, end)

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Stock Data", "Charts", "Prediction", "Model Info", "About"]
)

# ---------------- TAB 1: STOCK DATA ----------------
with tab1:
    st.subheader("Recent Stock Data")
    st.write(data.tail())

# ---------------- MOVING AVERAGES ----------------
ma50 = data.Close.rolling(50).mean()
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()

def style_white_bg_black_text():
    ax = plt.gca()
    ax.set_facecolor("white")

    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")

    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    for spine in ax.spines.values():
        spine.set_color("black")

    ax.grid(True, color="lightgray", alpha=0.6)

# ---------------- TAB 2: CHARTS ----------------
with tab2:
    st.subheader("Stock Charts with Moving Averages")

    fig1 = plt.figure(figsize=(8,5))
    plt.plot(data.Close, label="Close", color="#2563eb")
    plt.plot(ma50, label="MA50", color="#dc2626")
    plt.xlabel("Time")
    plt.ylabel("Price")
    style_white_bg_black_text()
    plt.legend(facecolor="white", edgecolor="black", labelcolor="black")
    st.pyplot(fig1)

    fig2 = plt.figure(figsize=(8,5))
    plt.plot(data.Close, label="Close", color="#2563eb")
    plt.plot(ma50, label="MA50", color="#dc2626")
    plt.plot(ma100, label="MA100", color="#ca8a04")
    plt.xlabel("Time")
    plt.ylabel("Price")
    style_white_bg_black_text()
    plt.legend(facecolor="white", edgecolor="black", labelcolor="black")
    st.pyplot(fig2)

    fig3 = plt.figure(figsize=(8,5))
    plt.plot(data.Close, label="Close", color="#2563eb")
    plt.plot(ma100, label="MA100", color="#ca8a04")
    plt.plot(ma200, label="MA200", color="#16a34a")
    plt.xlabel("Time")
    plt.ylabel("Price")
    style_white_bg_black_text()
    plt.legend(facecolor="white", edgecolor="black", labelcolor="black")
    st.pyplot(fig3)

# ---------------- PREPARE TEST DATA ----------------
data_train = pd.DataFrame(data.Close[:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):])

past_100 = data_train.tail(100)
final_df = pd.concat([past_100, data_test], ignore_index=True)

scaled_data = scaler.transform(final_df)

x_test, y_test = [], []

for i in range(100, len(scaled_data)):
    x_test.append(scaled_data[i-100:i])
    y_test.append(scaled_data[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)

# ---------------- PREDICTION ----------------
predicted = model.predict(x_test)
predicted = scaler.inverse_transform(predicted)
y_test = scaler.inverse_transform(y_test.reshape(-1,1))

# ---------------- TAB 3: PREDICTION ----------------
with tab3:
    st.subheader("Predicted vs Actual Prices")

    fig4 = plt.figure(figsize=(8,5))
    plt.plot(predicted, label="Predicted Price", color="#2563eb")
    plt.plot(y_test, label="Actual Price", color="#dc2626")
    plt.xlabel("Time")
    plt.ylabel("Price")
    style_white_bg_black_text()
    plt.legend(facecolor="white", edgecolor="black", labelcolor="black")
    st.pyplot(fig4)

# ---------------- TAB 4: MODEL INFO ----------------
with tab4:
    st.subheader("Model Information")
    st.write("This LSTM model is trained on historical stock data.")
    st.markdown("""
- *Input:* Past 100 days closing prices  
- *Output:* Next day price  
- *Optimizer:* Adam  
- *Loss:* Mean Squared Error  
""")

# ---------------- TAB 5: ABOUT ----------------
with tab5:
    st.subheader("About the Project")
    st.markdown("""
*Stock Market Predictor* is an ML-based Streamlit app.

- TensorFlow (LSTM)
- Yahoo Finance API
- Dark UI + Clean Charts

*Developed by Anil Gunja — 2025*
""")

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<p style="text-align:center;">
© 2025 | Developed by <b style="color:#38bdf8;">Anil Gunja</b>
</p>
""", unsafe_allow_html=True)
