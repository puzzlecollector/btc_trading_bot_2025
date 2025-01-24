import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import datetime 
import pytz 

btc_df = pd.read_feather("/content/BTC_USDT-1d.feather") 
russell_df = pd.read_csv("/content/russell_2000_index_20250123.csv") 
fear_greed_df = pd.read_csv("/content/crypto_fear_greed_index_20250123.csv") 
dollar_index_df = pd.read_csv("/content/us_dollar_index_20250123.csv") 
btc_news = pd.read_csv("/content/BTC_news_filtered_20250124.csv")

# Fix SettingWithCopyWarning and ensure successful conversion
btc_news["datetimes"] = pd.to_datetime(btc_news["datetimes"], errors='coerce')

# Define Korean timezone
korean_timezone = pytz.timezone("Asia/Seoul")

# Convert all datetime columns to Korean timezone
btc_df["date"] = pd.to_datetime(btc_df["date"], utc=True).dt.tz_convert(korean_timezone)
fear_greed_df["date"] = pd.to_datetime(fear_greed_df["timestamp"], utc=True).dt.tz_convert(korean_timezone)
russell_df["Date"] = pd.to_datetime(russell_df["Date"], utc=True).dt.tz_convert(korean_timezone)
dollar_index_df["Date"] = pd.to_datetime(dollar_index_df["Date"], utc=True).dt.tz_convert(korean_timezone)

# Ensure btc_news datetimes are timezone-aware for comparison
btc_news["datetimes"] = btc_news["datetimes"].apply(lambda x: x.tz_convert(korean_timezone) if x.tzinfo else x.tz_localize("UTC").tz_convert(korean_timezone))

# Extract numpy arrays for news data
datetimes = btc_news["datetimes"].dt.date.to_numpy()
titles = btc_news["titles"].astype(str).to_numpy()
contents = btc_news["contents"].astype(str).to_numpy()

# Initialize the grouped_news dictionary
grouped_news = defaultdict(str)

# Function to get the past 5 days of data
def get_past_5_days_data(current_date, data_df, date_column, value_columns):
    # Convert current_date to pd.Timestamp for compatibility
    current_date = pd.Timestamp(current_date).tz_localize("Asia/Seoul")
    start_date = current_date - pd.Timedelta(days=5)
    past_data = data_df[
        (data_df[date_column] >= start_date) & (data_df[date_column] < current_date)
    ][[date_column] + value_columns]
    return past_data.values.tolist()

# Process each date
unique_dates = sorted(set(datetimes))
for date in tqdm(unique_dates, desc="Processing dates"):
    # Fetch past 5 days of data
    fear_greed_data = get_past_5_days_data(
        date, fear_greed_df, "date", ["Fear_Greed_Index", "Classification"]
    )
    russell_data = get_past_5_days_data(date, russell_df, "Date", ["Close"])
    dollar_data = get_past_5_days_data(date, dollar_index_df, "Date", ["Close"])
    btc_data = get_past_5_days_data(date, btc_df, "date", ["close", "volume"])

    # Format the past data
    fear_greed_str = " | ".join(
        [f"{row[0].date()}: {row[1]}, {row[2]}" for row in fear_greed_data]
    ) if len(fear_greed_data) > 0 else "No data"
    russell_str = ", ".join(
        [f"{row[0].date()}: {row[1]}" for row in russell_data]
    ) if len(russell_data) > 0 else "No data"
    dollar_str = ", ".join(
        [f"{row[0].date()}: {row[1]}" for row in dollar_data]
    ) if len(dollar_data) > 0 else "No data"
    btc_str = ", ".join(
        [f"{row[0].date()}: Close={row[1]}, Volume={row[2]}" for row in btc_data]
    ) if len(btc_data) > 0 else "No data"

    # Append external data for the date
    grouped_news[date] += (
        f"US Dollar Index:\n{dollar_str}\n"
        f"Russell 2000:\n{russell_str}\n"
        f"Fear and Greed Index:\n{fear_greed_str}\n"
        f"BTC Chart Data:\n{btc_str}\n\n"
    )

    # Append all news for the date
    daily_news = btc_news[btc_news["datetimes"].dt.date == date]
    grouped_news[date] += "News:\n" 
    for title, content in zip(daily_news["titles"], daily_news["contents"]):
        grouped_news[date] += f"{title}: {content}\n\n"

# Convert to a regular dictionary
grouped_news = dict(grouped_news)

# Example output for a specific date
example_date = datetime.date(2020, 1, 1)
print(f"Data for {example_date}:\n{grouped_news.get(example_date, 'No news available')}")


# print sample 
# Convert the string "2019-03-23" to a datetime.date object
date_key = datetime.date(2025, 1, 8)
# Access the grouped_news dictionary using the correct date object
print(grouped_news[date_key])   
sample_text = grouped_news[date_key]
