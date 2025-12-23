import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
import os

# Page Configuration


st.set_page_config(
    page_title="Stock Market Predictor",
    layout="centered"
)


# Styling

st.markdown("""
<style>
.stApp { background-color: #0f172a; }
html, body, [class*="css"]  { color: white !important; }
h1, h2, h3, h4, h5{ color: white !important; }

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



label, label * { color: white !important; }
footer, footer * { color: white !important; }
svg text { fill: white !important; }
            

</style>
""", unsafe_allow_html=True)


# Load Model & Scaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR, "Stock_Predictions_Model.keras"))

with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# Header

st.markdown("<h1 style='text-align:center;'>Stock Market Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>ML-based Price Forecasting</h4>", unsafe_allow_html=True)


# User Input

stock = st.text_input("Enter Stock Symbol (example: GOOG, HDFCBANK.NS)", "GOOG")
start = "2012-01-01"
end = "2024-12-31"

data = yf.download(stock, start, end)


# Tabs

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Stock Data", "Charts", "Prediction", "Model Info", "About"
])

#  TAB 1: Stock Data 
with tab1:
    st.subheader("Recent Stock Data")
    st.write(data.tail())


# Moving Averages

ma50 = data.Close.rolling(50).mean()
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()

fig1 = plt.figure(figsize=(8,5))
plt.plot(data.Close, label="Close")
plt.plot(ma50, label="MA50")
plt.legend()

fig2 = plt.figure(figsize=(8,5))
plt.plot(data.Close, label="Close")
plt.plot(ma50, label="MA50")
plt.plot(ma100, label="MA100")
plt.legend()

fig3 = plt.figure(figsize=(8,5))
plt.plot(data.Close, label="Close")
plt.plot(ma100, label="MA100")
plt.plot(ma200, label="MA200")
plt.legend()

#TAB 2: Charts 
with tab2:
    st.subheader("Stock Charts with Moving Averages")
    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)


# Prepare Test Data

data_train = pd.DataFrame(data.Close[:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):])

past_100 = data_train.tail(100)
final_df = pd.concat([past_100, data_test], ignore_index=True)

# Transform only (no fit)
scaled_data = scaler.transform(final_df)

x_test = []
y_test = []

for i in range(100, len(scaled_data)):
    x_test.append(scaled_data[i-100:i])
    y_test.append(scaled_data[i,0])

x_test = np.array(x_test)
y_test = np.array(y_test)


# Prediction

predicted = model.predict(x_test)

# Inverse scaling
predicted = scaler.inverse_transform(predicted)
y_test = scaler.inverse_transform(y_test.reshape(-1,1))


# Plot Prediction

fig4 = plt.figure(figsize=(8,5))
plt.plot(predicted, label="Predicted Price", color="#38bdf8")
plt.plot(y_test, label="Actual Price", color="#f87171")
plt.xlabel("Time", color="black")
plt.ylabel("Price", color="black")
plt.legend(facecolor="#0f172a", labelcolor="white")
plt.xticks(color="black")
plt.yticks(color="black")

# TAB 3: Prediction
with tab3:
    st.subheader("Predicted vs Actual Prices")
    st.pyplot(fig4)

# TAB 4: Model Info 
with tab4:
    st.subheader("Model Information")
    st.write("This LSTM model is trained on historical stock data to predict future prices.")
    st.write("*Model file used:*", os.path.basename("Stock_Predictions_Model.keras"))
    st.write("*Scaler file used:*", os.path.basename("scaler.pkl"))
    st.markdown("""
    *Features:*
    - Input: Past 100 days' closing prices  
    - Output: Next day price prediction  
    - Optimizer: Adam  
    - Loss: Mean Squared Error  
    """)

#TAB 5: About
with tab5:
    st.subheader("About the Project")
    st.markdown("""
    *Stock Market Predictor* is an interactive ML web app built using:
    - *Streamlit* for the front-end
    - *TensorFlow (Keras)* for LSTM model
    -  *Yahoo Finance API (yfinance)* for data
    -  Custom *Dark Theme UI*

    Developed by *Anil Gunja*, 2025  
    """, unsafe_allow_html=True)


# Footer

st.markdown("""
<hr>
<p style='text-align:center; color:white;'>
© 2025 | Developed by <b style='color:#38bdf8;'>Anil Gunja</b>
</p>
""", unsafe_allow_html=True)
