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

# Load historical data (use preprocessed file if available)
@st.cache_data
def load_stock_data():
    # Prefer preprocessed data with indicators
    preprocessed_path = 'data/AAPL_preprocessed.csv'
    if os.path.exists(preprocessed_path):
        df = pd.read_csv(preprocessed_path, index_col='Date', parse_dates=True)
    else:
        df = pd.read_csv('data/AAPL_stock_data.csv', index_col='Date', parse_dates=True)

    # If indicators missing, compute on the fly
    if 'SMA_20' not in df.columns or 'RSI' not in df.columns:
        try:
            import ta
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
            df['MACD'] = ta.trend.macd_diff(df['Close'])
        except Exception:
            pass

    return df

df = load_stock_data()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Candlestick + Volume")
    last_n = st.slider("Days to display", 30, 180, 90, key="candles")
    df_tail = df.tail(last_n)

    from plotly.subplots import make_subplots

    fig_candle = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
    fig_candle.add_trace(go.Candlestick(
        x=df_tail.index,
        open=df_tail['Open'],
        high=df_tail['High'],
        low=df_tail['Low'],
        close=df_tail['Close'],
        name='Price'), row=1, col=1)

    fig_candle.add_trace(go.Bar(x=df_tail.index, y=df_tail['Volume'], name='Volume', marker_color='lightblue'), row=2, col=1)

    fig_candle.update_layout(
        title=f"{ticker} Candlestick + Volume", 
        xaxis=dict(rangeslider=dict(visible=False)),
        yaxis_title="Price ($)",
        yaxis2_title="Volume",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig_candle, use_container_width=True)

with col2:
    st.subheader("📊 Technical Indicators")

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig_rsi.update_layout(title="RSI Indicator", xaxis_title="Date", yaxis_title="RSI")
    st.plotly_chart(fig_rsi, use_container_width=True)

    fig_macd = go.Figure()
    if 'MACD' in df.columns:
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='green')))
        fig_macd.update_layout(title="MACD", xaxis_title="Date", yaxis_title="MACD")
        st.plotly_chart(fig_macd, use_container_width=True)
    else:
        st.info("MACD not available in dataset.")

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

st.sidebar.header("About")
st.sidebar.info("This dashboard uses AI models to predict stock prices and provide buy/sell signals.")