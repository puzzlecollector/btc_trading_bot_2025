# Ensure imports are available
import pandas as pd
import numpy as np

# Convert btc_data 'date' column to pandas datetime and align with grouped_news date format
btc_data["date"] = pd.to_datetime(btc_data["date"], utc=True).dt.date  # Convert to datetime.date format

# Get the min and max dates from grouped_news
min_date = min(loaded_grouped_news.keys())
max_date = max(loaded_grouped_news.keys())

# Filter btc_data to include only dates within the grouped_news date range
filtered_btc_data = btc_data[(btc_data["date"] >= min_date) & (btc_data["date"] <= max_date)].reset_index(drop=True)

# Add 'label' column: long (0) if the next day's close price is higher, short (1) otherwise
filtered_btc_data["label"] = np.where(
    filtered_btc_data["close"].shift(-1) > filtered_btc_data["close"], 0, 1
)

# Remove the last row because it has no "next day" to compare
filtered_btc_data = filtered_btc_data[:-1]

# Get grouped_news dates and ensure they are in datetime.date format
grouped_news_dates = set(loaded_grouped_news.keys())

# Filter btc_data to include only dates that are present in grouped_news
filtered_btc_data = filtered_btc_data[filtered_btc_data["date"].isin(grouped_news_dates)].reset_index(drop=True)

# Save the filtered DataFrame with labels
filtered_btc_data.to_csv(
    "/content/drive/MyDrive/filtered_btc_data_with_labels_for_baseline_train.csv", index=False
)

print(f"Filtered BTC data saved successfully with {filtered_btc_data.shape[0]} rows.")
