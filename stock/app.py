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

# ---------------- GLOBAL STYLING ----------------
st.markdown("""
<style>
.stApp { background-color: #0f172a; }
html, body, [class*="css"]  { color: white !important; }
h1, h2, h3, h4, h5, h6 { color: white !important; }

.stTabs [data-baseweb="tab-list"] {
    padding: 10px;
    border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #1e293b !important;
    color: white !important;
    font-size: 16px !important;
    border-radius: 10px !important;
    margin-right: 8px;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background-color: #38bdf8 !important;
    color: black !important;
    font-weight: bold;
}

label, p, span, div { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- HELPER FUNCTION ----------------
def style_dark_axes(ax):
    ax.set_facecolor("#0f172a")
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_color('white')

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
    st.dataframe(data.tail())

# ---------------- MOVING AVERAGES ----------------
ma50 = data.Close.rolling(50).mean()
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()

# ---------------- TAB 2: CHARTS ----------------
with tab2:
    st.subheader("Stock Charts with Moving Averages")

    fig1, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(data.Close, label="Close")
    ax1.plot(ma50, label="MA50")
    ax1.set_title("Closing Price with MA50")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.legend(facecolor="#0f172a", labelcolor="white")
    style_dark_axes(ax1)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.plot(data.Close, label="Close")
    ax2.plot(ma50, label="MA50")
    ax2.plot(ma100, label="MA100")
    ax2.set_title("Closing Price with MA50 & MA100")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price")
    ax2.legend(facecolor="#0f172a", labelcolor="white")
    style_dark_axes(ax2)
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(8,5))
    ax3.plot(data.Close, label="Close")
    ax3.plot(ma100, label="MA100")
    ax3.plot(ma200, label="MA200")
    ax3.set_title("Closing Price with MA100 & MA200")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Price")
    ax3.legend(facecolor="#0f172a", labelcolor="white")
    style_dark_axes(ax3)
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
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# ---------------- TAB 3: PREDICTION ----------------
with tab3:
    st.subheader("Predicted vs Actual Prices")

    fig4, ax4 = plt.subplots(figsize=(8,5))
    ax4.plot(predicted, label="Predicted Price", color="#38bdf8")
    ax4.plot(y_test, label="Actual Price", color="#f87171")
    ax4.set_title("Predicted vs Actual Prices")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Price")
    ax4.legend(facecolor="#0f172a", labelcolor="white")
    style_dark_axes(ax4)
    st.pyplot(fig4)

# ---------------- TAB 4: MODEL INFO ----------------
with tab4:
    st.subheader("Model Information")
    st.markdown("""
    *Model Type:* LSTM Neural Network  
    *Input:* Past 100 days closing prices  
    *Output:* Next day price prediction  
    *Optimizer:* Adam  
    *Loss Function:* Mean Squared Error  
    """)

# ---------------- TAB 5: ABOUT ----------------
with tab5:
    st.subheader("About the Project")
    st.markdown("""
    *Stock Market Predictor* is an interactive ML web app built using:
    - Streamlit (Frontend)
    - TensorFlow / Keras (LSTM Model)
    - Yahoo Finance API (yfinance)
    - Custom Dark Theme UI  

    *Developed by Anil Gunja – 2025*
    """)

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<p style="text-align:center;">
© 2025 | Developed by <span style="color:#38bdf8;"><b>Anil Gunja</b></span>
</p>
""", unsafe_allow_html=True)
