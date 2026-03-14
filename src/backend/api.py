from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.ml.prediction_engine import PredictionEngine
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json

app = FastAPI(title="Stock Prediction API")

engine = PredictionEngine()

class PredictionRequest(BaseModel):
    ticker: str
    model: str
    days_ahead: int = 1

@app.get("/stock/{ticker}")
def get_stock_data(ticker: str):
    try:
        data = yf.download(ticker, period='1y')
        return data.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        pred = engine.predict(request.ticker, request.model, request.days_ahead)
        current = yf.Ticker(request.ticker).history(period='1d')['Close'].iloc[-1]
        signal = engine.get_buy_sell_signal(current, pred)
        return {"prediction": pred, "current": current, "signal": signal}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/models")
def get_models():
    return {"models": ["lr", "rf", "xgb", "lstm", "arima"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)