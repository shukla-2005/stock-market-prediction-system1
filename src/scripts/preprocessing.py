import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import joblib
import os

nltk.download('vader_lexicon')

def preprocess_stock_data(file_path):
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Add technical indicators
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['Close'])

    # Drop NaN
    df.dropna(inplace=True)

    # Scale
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

    return df_scaled, scaler

def preprocess_news_data(file_path):
    df = pd.read_csv(file_path)
    sia = SentimentIntensityAnalyzer()

    def get_sentiment(text):
        if pd.isna(text):
            return 0
        scores = sia.polarity_scores(text)
        return scores['compound']

    df['sentiment'] = df['description'].apply(get_sentiment)
    return df[['publishedAt', 'sentiment']]

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    stock_df, scaler = preprocess_stock_data(os.path.join(base_dir, 'data', 'AAPL_stock_data.csv'))
    stock_df.to_csv(os.path.join(base_dir, 'data', 'AAPL_preprocessed.csv'))
    joblib.dump(scaler, os.path.join(base_dir, 'models', 'scaler.pkl'))

    news_df = preprocess_news_data(os.path.join(base_dir, 'data', 'AAPL_news_data.csv'))
    news_df.to_csv(os.path.join(base_dir, 'data', 'AAPL_news_sentiment.csv'))

    print("Preprocessing completed.")