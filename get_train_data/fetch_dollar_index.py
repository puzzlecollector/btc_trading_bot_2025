# fetch us dollar index using yfinance
import yfinance as yf
import pandas as pd

def get_us_dollar_index_history(start_date, end_date):
    try:
        # Fetch the data for the US Dollar Index (ticker: "DX-Y.NYB")
        ticker = "DX-Y.NYB"
        usd_index = yf.Ticker(ticker)

        # Get the historical data for the specified date range
        hist = usd_index.history(start=start_date, end=end_date)

        if hist.empty:
            print("No data available for the US Dollar Index in the specified range.")
            return pd.DataFrame()

        # Extract only the datetime index and Close column
        hist = hist[['Close']]

        print(f"Historical data (datetime and close) for the US Dollar Index from {start_date} to {end_date} as a DataFrame:")
        print(hist)
        return hist
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()

# Call the function to get the US Dollar Index history
dxy_history = get_us_dollar_index_history("2019-01-01", "2025-01-23")

dxy_history["Date"] = dxy_history.index
dxy_history.to_csv("us_dollar_index_20250123.csv",index=False)
