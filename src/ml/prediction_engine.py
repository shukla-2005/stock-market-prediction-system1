import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMAResults
import yfinance as yf
from datetime import datetime, timedelta
import os

class PredictionEngine:
    def __init__(self):
        self.models = {}
        self.load_models()

    def load_models(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            self.models['lr'] = joblib.load(os.path.join(base_dir, 'models', 'lr_model.pkl'))
            self.models['rf'] = joblib.load(os.path.join(base_dir, 'models', 'rf_model.pkl'))
            self.models['xgb'] = joblib.load(os.path.join(base_dir, 'models', 'xgb_model.pkl'))
            self.models['lstm'] = load_model(os.path.join(base_dir, 'models', 'lstm_model.h5'))
            self.models['arima'] = ARIMAResults.load(os.path.join(base_dir, 'models', 'arima_model.pkl'))
            self.scaler = joblib.load(os.path.join(base_dir, 'models', 'scaler.pkl'))
        except FileNotFoundError as e:
            raise RuntimeError(f"Model file not found: {e}. Please run training first.")

    def get_real_time_data(self, ticker):
        data = yf.download(ticker, period='1d', interval='1m')
        return data

    def predict(self, ticker, model_name, days_ahead=1):
        # Get recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Preprocess (match preprocessing.py)
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        # Add technical indicators
        import ta
        data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
        data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
        data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
        data['MACD'] = ta.trend.macd_diff(data['Close'])
        data.dropna(inplace=True)
        data = data.tail(60)
        
        scaled_data = self.scaler.transform(data)
        
        if model_name in ['lr', 'rf', 'xgb']:
            pred = self.models[model_name].predict(scaled_data.reshape(1, -1))
        elif model_name == 'lstm':
            pred = self.models[model_name].predict(scaled_data.reshape(1, 60, 9))
        elif model_name == 'arima':
            pred = self.models[model_name].forecast(steps=days_ahead)
        
        # Inverse scale - create full feature array with prediction at Close position
        if model_name != 'arima':
            pred_array = np.zeros((1, 9))
            pred_array[0, 3] = pred[0] if model_name == 'lstm' else pred  # Close is at index 3
            pred = self.scaler.inverse_transform(pred_array)[0, 3]
        return pred

    def get_buy_sell_signal(self, current_price, predicted_price):
        if predicted_price > current_price * 1.02:
            return 'BUY'
        elif predicted_price < current_price * 0.98:
            return 'SELL'
        else:
            return 'HOLD'

if __name__ == "__main__":
    engine = PredictionEngine()
    pred = engine.predict('AAPL', 'lstm')
    print(f"Predicted price: {pred}")