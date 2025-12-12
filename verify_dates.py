import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def find_dates_for_prices():
    # Prices from the end of qqq_market_data.csv
    target_prices = [
        608.8599853515625,
        603.6599731445312,
        596.3099975585938,
        599.8699951171875
    ]
    
    print("🔍 Searching for these closing prices:")
    for p in target_prices:
        print(f"  ${p:.2f}")
        
    # Fetch QQQ data for late 2024 (wide range to be safe)
    print("\n⏳ Fetching QQQ data from Yahoo Finance (Wide Range)...")
    start_date = "2023-01-01"
    end_date = "2026-01-01"
    
    df = yf.download("QQQ", start=start_date, end=end_date, progress=False)
    
    # Handle multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        try:
            prices = df['Close']['QQQ']
        except KeyError:
            prices = df['Close'] # Fallback
    else:
        prices = df['Close']
        
    print(f"✅ Fetched {len(prices)} days of data")
    
    # Find matches
    print("\n🎯 MATCH RESULTS:")
    found_match = False
    
    # Check the last few days of our target list against the fetched data
    # We look for a sequence match
    
    for i in range(len(prices) - len(target_prices) + 1):
        subset = prices.iloc[i : i+len(target_prices)]
        
        # Check if values match (allowing for small floating point differences)
        match = True
        for j, price in enumerate(subset):
            if abs(price - target_prices[j]) > 0.01: # 1 cent tolerance
                match = False
                break
        
        if match:
            found_match = True
            print("\n✅ FOUND EXACT SEQUENCE MATCH!")
            for date, price in subset.items():
                print(f"  {date.strftime('%Y-%m-%d')} ({date.strftime('%A')}): ${price:.2f}")
            
            last_date = subset.index[-1]
            print(f"\n📅 The last row of your data is: {last_date.strftime('%A, %B %d, %Y')}")
            return last_date.strftime('%A')

    if not found_match:
        print("\n❌ No exact sequence match found. Trying individual matches...")
        # Try to match just the last price
        last_target = target_prices[-1]
        for date, price in prices.items():
            if abs(price - last_target) < 0.01:
                print(f"  Possible match for last row: {date.strftime('%Y-%m-%d')} ({date.strftime('%A')}): ${price:.2f}")

if __name__ == "__main__":
    find_dates_for_prices()
