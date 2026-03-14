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
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# NewsAPI for news data
def get_news_data(query, from_date, to_date, api_key):
    newsapi = NewsApiClient(api_key=api_key)
    all_articles = newsapi.get_everything(q=query,
                                          from_param=from_date,
                                          to=to_date,
                                          language='en',
                                          sort_by='relevancy',
                                          page_size=100)
    return all_articles['articles']

if __name__ == "__main__":
    # Example: Get data for AAPL
    ticker = 'AAPL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years

    stock_data = get_stock_data(ticker, start_date, end_date)
    stock_data.to_csv(f'data/{ticker}_stock_data.csv')

    # News data
    api_key = os.getenv('NEWS_API_KEY')  # Set in .env
    news_data = get_news_data('Apple stock', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), api_key)
    news_df = pd.DataFrame(news_data)
    news_df.to_csv(f'data/{ticker}_news_data.csv', index=False)

    print("Data collection completed.")