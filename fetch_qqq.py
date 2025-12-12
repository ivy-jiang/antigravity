import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime, timedelta
import os

def generate_csv():
    print("--- Starting Data Fetch for Antigravity Project ---")
    
    # 1. Define Timeframe
    # We fetch more than 60 days initially to calculate the 100-Day Moving Average correctly
    end_date = datetime.now()
    start_date = end_date - timedelta(days=250) 
    
    print(f"1. Fetching raw data from {start_date.date()} to {end_date.date()}...")
    
    # 2. Fetch Data
    # QQQ: Price
    # ^VIX: Market Sentiment (Volatility Index)
    # ^TNX: 10-Year Treasury Yield
    tickers = ["QQQ", "^VIX", "^TNX"]
    raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    # Handle yfinance data structure
    try:
        df = raw_data['Close'].copy()
    except KeyError:
        df = raw_data['Adj Close'].copy()

    # 3. Fetch 2-Year Treasury Yield (DGS2) from FRED
    # If FRED fails, we fallback to ^FVX (5-year) as a proxy
    print("2. Fetching 2-Year Treasury Yields...")
    try:
        us_2y = web.DataReader('DGS2', 'fred', start_date, end_date)
        df = df.join(us_2y)
        df.rename(columns={'DGS2': 'US_2y_yield'}, inplace=True)
    except Exception as e:
        print(f"   (FRED source unavailable, using Yahoo backup for Yields)")
        fvx = yf.download("^FVX", start=start_date, end=end_date, progress=False)['Close']
        df = df.join(fvx)
        df.rename(columns={'^FVX': 'US_2y_yield'}, inplace=True)

    # Fill missing weekends/holidays for yields
    df = df.ffill().dropna()

    # 4. Calculate Indicators
    print("3. Calculating Technical Indicators (RSI, MACD, 100D MA)...")
    
    # Prepare columns
    df['price'] = df['QQQ']
    df['US_10y_yield'] = df['^TNX']
    df['market_sentiment'] = df['^VIX']
    
    # 100-Day Moving Average
    df['100D_MA'] = df['price'].rolling(window=100).mean()
    
    # RSI (14-Day)
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    ema12 = df['price'].ewm(span=12, adjust=False).mean()
    ema26 = df['price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26

    # 5. Clean and Export Last 60 Days
    # Select only the requested columns
    final_cols = ['price', 'RSI', '100D_MA', 'MACD', 'US_10y_yield', 'US_2y_yield', 'market_sentiment']
    final_df = df[final_cols].tail(60)
    
    # Save to CSV (without date index, as requested)
    filename = 'qqq_data_60days.csv'
    final_df.to_csv(filename, index=False)
    
    print(f"\nSUCCESS: Generated '{filename}' with {len(final_df)} rows.")
    print(f"Location: {os.path.abspath(filename)}")
    print("\nFirst 3 rows preview:")
    print(final_df.head(3))

if __name__ == "__main__":
    generate_csv()