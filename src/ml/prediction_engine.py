import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMAResults
import yfinance as yf
from datetime import datetime, timedelta

class PredictionEngine:
    def __init__(self):
        self.models = {}
        self.load_models()

    def load_models(self):
        self.models['lr'] = joblib.load('models/lr_model.pkl')
        self.models['rf'] = joblib.load('models/rf_model.pkl')
        self.models['xgb'] = joblib.load('models/xgb_model.pkl')
        self.models['lstm'] = load_model('models/lstm_model.h5')
        self.models['arima'] = ARIMAResults.load('models/arima_model.pkl')

    def get_real_time_data(self, ticker):
        data = yf.download(ticker, period='1d', interval='1m')
        return data

    def predict(self, ticker, model_name, days_ahead=1):
        # Get recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Preprocess (simplified)
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(60)
        # Assume scaler is saved
        scaler = joblib.load('models/scaler.pkl')
        scaled_data = scaler.transform(data)
        
        if model_name in ['lr', 'rf', 'xgb']:
            pred = self.models[model_name].predict(scaled_data.reshape(1, -1))
        elif model_name == 'lstm':
            pred = self.models[model_name].predict(scaled_data.reshape(1, 60, 5))
        elif model_name == 'arima':
            pred = self.models[model_name].forecast(steps=days_ahead)
        
        # Inverse scale
        pred = scaler.inverse_transform([[0,0,0,pred[0],0]])[0][3]
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