import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
from newsapi import NewsApiClient
import os
from dotenv import load_dotenv

load_dotenv()

# Yahoo Finance for stock data
def get_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            # Fallback to ticker.history if no data returned
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"WARNING: failed to download {ticker} via yfinance: {e}")
        # Try a more direct approach in case download fails
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period='2y')
            return data
        except Exception as inner:
            print(f"WARNING: fallback ticker.history failed: {inner}")
            return pd.DataFrame()

# NewsAPI for news data
def get_news_data(query, from_date, to_date, api_key, max_days=25):
    if api_key is None:
        raise ValueError("NEWS_API_KEY is not set in the environment.")

    # NewsAPI free plan only supports recent articles. Limit to last max_days.
    to_dt = datetime.strptime(to_date, "%Y-%m-%d")
    min_allowed = to_dt - timedelta(days=max_days)
    from_dt = datetime.strptime(from_date, "%Y-%m-%d")
    if from_dt < min_allowed:
        from_dt = min_allowed

    print(f"Fetching news from {from_dt.strftime('%Y-%m-%d')} to {to_dt.strftime('%Y-%m-%d')} (max {max_days} days)")

    newsapi = NewsApiClient(api_key=api_key)
    all_articles = newsapi.get_everything(
        q=query,
        from_param=from_dt.strftime("%Y-%m-%d"),
        to=to_dt.strftime("%Y-%m-%d"),
        language='en',
        sort_by='relevancy',
        page_size=100,
    )
    return all_articles.get('articles', [])

if __name__ == "__main__":
    # Example: Get data for AAPL
    ticker = 'AAPL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years

    # Ensure output directory exists
    os.makedirs('data', exist_ok=True)

    stock_data = get_stock_data(ticker, start_date, end_date)
    stock_data.to_csv(f'data/{ticker}_stock_data.csv')

    # News data
    api_key = os.getenv('NEWS_API_KEY')  # Set in .env
    try:
        news_data = get_news_data('Apple stock', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), api_key)
        news_df = pd.DataFrame(news_data)
        news_df.to_csv(f'data/{ticker}_news_data.csv', index=False)
        print(f"Saved news data for {ticker} to data/{ticker}_news_data.csv")
    except Exception as e:
        print(f"ERROR collecting news data: {e}")

    print("Data collection completed.")