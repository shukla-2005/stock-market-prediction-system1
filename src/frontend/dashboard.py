import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="AI Stock Prediction Dashboard", page_icon="📈", layout="wide")

st.title("🚀 AI Powered Stock Market Prediction Dashboard")

# Health check
try:
    response = requests.get(f"{API_BASE}/docs", timeout=5)
    if response.status_code != 200:
        st.error("Backend API is not available. Please start the backend server.")
        st.stop()
except requests.exceptions.RequestException:
    st.error("Cannot connect to backend API. Please check if the server is running.")
    st.stop()

# Sidebar for inputs
st.sidebar.header("📊 Prediction Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
model = st.sidebar.selectbox("Select Model", ["lr", "rf", "xgb", "lstm", "arima"])
days = st.sidebar.slider("Days Ahead", 1, 30, 1)

# Load historical data (using local data for demo)
@st.cache_data
def load_stock_data():
    df = pd.read_csv('data/AAPL_stock_data.csv', index_col='Date', parse_dates=True)
    return df

df = load_stock_data()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Historical Stock Price")
    fig_historical = go.Figure()
    fig_historical.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
    fig_historical.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='orange', dash='dash')))
    fig_historical.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='red', dash='dash')))
    fig_historical.update_layout(title=f"{ticker} Historical Price & Moving Averages", xaxis_title="Date", yaxis_title="Price ($)")
    st.plotly_chart(fig_historical, use_container_width=True)

with col2:
    st.subheader("📊 Technical Indicators")
    fig_indicators = go.Figure()
    fig_indicators.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
    fig_indicators.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_indicators.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig_indicators.update_layout(title="RSI Indicator", xaxis_title="Date", yaxis_title="RSI")
    st.plotly_chart(fig_indicators, use_container_width=True)

    # Volume chart
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='lightblue'))
    fig_volume.update_layout(title="Trading Volume", xaxis_title="Date", yaxis_title="Volume")
    st.plotly_chart(fig_volume, use_container_width=True)

# Prediction section
st.header("🔮 Price Prediction")

if st.button("🚀 Generate Prediction", type="primary"):
    try:
        response = requests.post(f"{API_BASE}/predict", json={"ticker": ticker, "model": model, "days_ahead": days})
        data = response.json()
        
        # Results cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${data['current']:.2f}")
        with col2:
            st.metric("Predicted Price", f"${data['prediction']:.2f}", 
                     delta=f"{((data['prediction'] - data['current']) / data['current'] * 100):.2f}%")
        with col3:
            signal_color = "🟢" if data['signal'] == "BUY" else "🔴" if data['signal'] == "SELL" else "🟡"
            st.metric("Signal", f"{signal_color} {data['signal']}")
        
        # Prediction chart
        st.subheader("📉 Prediction Visualization")
        recent_data = df.tail(30).copy()  # Last 30 days
        future_dates = [recent_data.index[-1] + timedelta(days=i) for i in range(1, days+1)]
        
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=recent_data.index, y=recent_data['Close'], mode='lines', name='Historical', line=dict(color='blue')))
        fig_pred.add_trace(go.Scatter(x=future_dates, y=[data['prediction']] * days, mode='lines+markers', name='Prediction', line=dict(color='red', dash='dash')))
        fig_pred.add_annotation(x=future_dates[0], y=data['prediction'], text=f"Predicted: ${data['prediction']:.2f}", showarrow=True, arrowhead=1)
        fig_pred.update_layout(title=f"{ticker} Price Prediction ({model.upper()} Model)", xaxis_title="Date", yaxis_title="Price ($)")
        st.plotly_chart(fig_pred, use_container_width=True)
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Model Comparison
st.header("⚖️ Model Comparison")
models = ["lr", "rf", "xgb", "lstm", "arima"]
if st.button("Compare All Models"):
    predictions = {}
    for m in models:
        try:
            response = requests.post(f"{API_BASE}/predict", json={"ticker": ticker, "model": m, "days_ahead": 1})
            if response.status_code == 200:
                data = response.json()
                predictions[m.upper()] = data['prediction']
        except:
            predictions[m.upper()] = None
    
    # Filter out None values
    valid_preds = {k: v for k, v in predictions.items() if v is not None}
    
    if valid_preds:
        fig_compare = px.bar(x=list(valid_preds.keys()), y=list(valid_preds.values()), 
                           title="Model Predictions Comparison",
                           labels={'x': 'Model', 'y': 'Predicted Price ($)'})
        fig_compare.add_hline(y=df['Close'].iloc[-1], line_dash="dash", line_color="red", annotation_text="Current Price")
        st.plotly_chart(fig_compare, use_container_width=True)
    else:
        st.error("Could not get predictions from any model")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, FastAPI, and AI models")
        st.error(f"Error: {e}")

st.sidebar.header("About")
st.sidebar.info("This dashboard uses AI models to predict stock prices and provide buy/sell signals.")