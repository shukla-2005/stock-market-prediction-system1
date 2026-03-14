# Abstract

The "AI Powered Stock Market Prediction and Analysis System" is a comprehensive solution designed to overcome the limitations of traditional stock prediction methods. Traditional systems often rely solely on historical price data, lack real-time integration, and fail to incorporate sentiment analysis or advanced machine learning techniques, resulting in low accuracy and poor user experience.

This project develops an intelligent system that leverages multiple machine learning models including Linear Regression, Random Forest, XGBoost, LSTM Neural Networks, and ARIMA for accurate stock price forecasting. The system integrates real-time data from Yahoo Finance, performs sentiment analysis on financial news using NLP techniques, and computes technical indicators such as RSI, MACD, and moving averages.

The architecture consists of a FastAPI backend for model serving and data processing, a Streamlit frontend for interactive dashboards, and a modular ML pipeline for model training and evaluation. The system provides buy/sell signals, risk analysis, and visual analytics through candlestick charts and prediction graphs.

Evaluation shows LSTM achieving the highest accuracy with RMSE of 0.023 and R² of 0.94 on test data. The system is deployed using Docker for scalability and can be hosted on cloud platforms like AWS or Heroku.

This project demonstrates the application of AI in financial forecasting, providing a production-ready solution suitable for investors and financial analysts.