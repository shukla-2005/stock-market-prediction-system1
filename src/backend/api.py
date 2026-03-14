from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from ml.prediction_engine import PredictionEngine
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json
import re

app = FastAPI(title="Stock Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = PredictionEngine()

class PredictionRequest(BaseModel):
    ticker: str = Field(..., pattern=r'^[A-Z]{1,5}$')
    model: str = Field(..., pattern=r'^(lr|rf|xgb|lstm|arima)$')
    days_ahead: int = Field(default=1, gt=0, le=100)

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
        history = yf.Ticker(request.ticker).history(period='1d')
        if history.empty:
            raise HTTPException(status_code=400, detail="No data available for this ticker")
        current = history['Close'].iloc[-1]
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