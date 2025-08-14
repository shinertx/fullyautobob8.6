import pandas as pd
import sys

# We need to add the project root to the path to import the lakehouse
sys.path.insert(0, '.')

from v26meme.data.lakehouse import Lakehouse

def verify_data(symbol, timeframe):
    try:
        lh = Lakehouse()
        df = lh.get_data(symbol, timeframe)

        if df.empty:
            print(f"No data found for {symbol} in timeframe {timeframe}.")
            return

        print(f"--- Verification for {symbol} [{timeframe}] ---")
        print(f"Total rows: {len(df)}")
        print(f"Data starts: {df.index.min()}")
        print(f"Data ends:   {df.index.max()}")
        print("\n--- First 5 rows (Oldest Data) ---")
        print(df.head())
        print("\n--- Last 5 rows (Most Recent Data) ---")
        print(df.tail())
        print("\n" + "="*40 + "\n")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Verify a few key symbols
    verify_data("BTC_USD_SPOT", "1h")
    verify_data("ETH_USD_SPOT", "1h")
