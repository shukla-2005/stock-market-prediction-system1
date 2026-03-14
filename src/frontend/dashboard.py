import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.title("AI Powered Stock Market Prediction Dashboard")

# Health check
try:
    response = requests.get(f"{API_BASE}/docs", timeout=5)  # or /health if exists
    if response.status_code != 200:
        st.error("Backend API is not available. Please start the backend server.")
        st.stop()
except requests.exceptions.RequestException:
    st.error("Cannot connect to backend API. Please check if the server is running.")
    st.stop()

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
model = st.selectbox("Select Model", ["lr", "rf", "xgb", "lstm", "arima"])
days = st.slider("Days Ahead", 1, 30, 1)

if st.button("Predict"):
    try:
        response = requests.post(f"{API_BASE}/predict", json={"ticker": ticker, "model": model, "days_ahead": days})
        data = response.json()
        
        st.subheader("Prediction Results")
        st.write(f"Current Price: ${data['current']:.2f}")
        st.write(f"Predicted Price: ${data['prediction']:.2f}")
        st.write(f"Signal: {data['signal']}")
        
        # Chart
        stock_data = requests.get(f"{API_BASE}/stock/{ticker}").json()
        df = pd.DataFrame(stock_data)
        df.index = pd.to_datetime(df.index)
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close']))
        fig.update_layout(title=f"{ticker} Stock Price", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error: {e}")

st.sidebar.header("About")
st.sidebar.info("This dashboard uses AI models to predict stock prices and provide buy/sell signals.")