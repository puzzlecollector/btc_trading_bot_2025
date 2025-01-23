# fetch Russell 2000 index using yfinance
import yfinance as yf
import pandas as pd

def get_russell_2000_history(start_date, end_date):
    try:
        ticker = "^RUT"
        russell_index=  yf.Ticker(ticker)
        hist = russell_index.history(start=start_date, end=end_date)
        if hist.empty:
            print("No data available for the Russell 2000 Index in the specified range.")
            return pd.DataFrame()
        hist = hist[["Close"]]
        print(f"Historical data (datetime and close) for the Russell 2000 Index from {start_date} to {end_date} as a DataFrame.")
        print(hist)
        return hist
    except Exception as e:
        print(f"An error occured: {e}")
        return pd.DataFrame()

russell_history = get_russell_2000_history("2019-01-01", "2025-01-23")
russell_history["Date"] = russell_history.index
russell_history.to_csv("russell_2000_index_20250123.csv", index=False)  
