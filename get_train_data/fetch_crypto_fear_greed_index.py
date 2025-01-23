# Get crypto fear and greed index
import requests
import pandas as pd
from datetime import datetime

def get_crypto_fear_greed_index(start_date=None, end_date=None):
    """
    Fetch Crypto Fear and Greed Index historical data.
    Parameters:
        start_date (str): Start date in 'YYYY-MM-DD' format (optional)
        end_date (str): End date in 'YYYY-MM-DD' format (optional)
    Returns:
        pd.DataFrame: DataFrame containing the index data.
    """
    try:
        url = "https://api.alternative.me/fng/?limit=0&format=json"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Failed to fetch data. Http Status Code: {response.status_code}")
            return pd.DataFrame()

        data = response.json().get("data", [])

        # Convert to DataFrame
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")  # Convert timestamp
        df["value"] = pd.to_numeric(df["value"])  # Convert value to numeric
        df.rename(columns={"value": "Fear_Greed_Index", "value_classification": "Classification"}, inplace=True)

        # Filter by start_date and end_date if provided
        if start_date:
            df = df[df["timestamp"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["timestamp"] <= pd.to_datetime(end_date)]

        print("Fetched Crypto Fear and Greed Index data:")
        print(df.head())
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()

# Fetch the data
fng_history = get_crypto_fear_greed_index("2019-01-01", "2025-01-23")

# Save to CSV (ensure 'timestamp' is included as a column)
fng_history.to_csv("crypto_fear_greed_index_20250123.csv", index=False)

print(fng_history)
