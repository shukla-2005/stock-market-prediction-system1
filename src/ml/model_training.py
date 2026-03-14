import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA
import joblib
import os

def prepare_data(df, target_col='Close', look_back=60):
    data = df.values
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i])
        y.append(data[i, df.columns.get_loc(target_col)])
    return np.array(X), np.array(y)

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    return model

def train_xgboost(X_train, y_train):
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    return model

def train_lstm(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    return model

def train_arima(series):
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()
    return model_fit

def evaluate_model(model, X_test, y_test, model_type):
    if model_type in ['lr', 'rf', 'xgb']:
        y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
    elif model_type == 'lstm':
        y_pred = model.predict(X_test)
        y_pred = y_pred.flatten()
    elif model_type == 'arima':
        y_pred = model.forecast(steps=len(y_test))
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2

if __name__ == "__main__":
    df = pd.read_csv('data/AAPL_preprocessed.csv', index_col='Date', parse_dates=True)
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {}
    results = {}

    # Linear Regression
    lr = train_linear_regression(X_train, y_train)
    models['lr'] = lr
    results['lr'] = evaluate_model(lr, X_test, y_test, 'lr')

    # Random Forest
    rf = train_random_forest(X_train, y_train)
    models['rf'] = rf
    results['rf'] = evaluate_model(rf, X_test, y_test, 'rf')

    # XGBoost
    xgb = train_xgboost(X_train, y_train)
    models['xgb'] = xgb
    results['xgb'] = evaluate_model(xgb, X_test, y_test, 'xgb')

    # LSTM
    lstm = train_lstm(X_train, y_train)
    models['lstm'] = lstm
    results['lstm'] = evaluate_model(lstm, X_test, y_test, 'lstm')

    # ARIMA
    close_series = df['Close']
    arima = train_arima(close_series)
    models['arima'] = arima
    results['arima'] = evaluate_model(arima, None, close_series[-len(y_test):], 'arima')

    # Save models
    for name, model in models.items():
        if name != 'arima':
            joblib.dump(model, f'models/{name}_model.pkl')
        else:
            arima.save(f'models/arima_model.pkl')

    # Print results
    for name, (rmse, mae, r2) in results.items():
        print(f"{name.upper()}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    print("Model training completed.")