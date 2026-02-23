import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

DATA_ROOT = "data/maec_repo/MAEC_Dataset"
OUTPUT_FILE = "yfinance_labeled_AAPL.csv"
STOCK_SYMBOL = "AAPL"

# threshold for neutral zone
THRESHOLD = 0.003  # 0.3%


def extract_dates_from_folders(root):
    dates = []

    for folder in os.listdir(root):
        try:
            date_str = folder.split("_")[0]
            date = datetime.strptime(date_str, "%Y%m%d")
            dates.append(date)
        except:
            continue

    dates = sorted(list(set(dates)))
    return dates


def assign_label(change):
    if change > THRESHOLD:
        return "positive"
    elif change < -THRESHOLD:
        return "negative"
    else:
        return "neutral"


def download_and_label(symbol, dates):

    start_date = min(dates) - timedelta(days=10)
    end_date = max(dates) + timedelta(days=10)

    print(f"Downloading {symbol} data...")
    data = yf.download(
        symbol,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        progress=True
    )

    data.reset_index(inplace=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data["FutureClose"] = data["Close"].shift(-1)
    data["Return"] = (data["FutureClose"] - data["Close"]) / data["Close"]

    #give labels
    data["Label"] = data["Return"].apply(assign_label)
    data.dropna(inplace=True)
    data.to_csv(OUTPUT_FILE, index=False)
    print("\nSaved labeled data →", OUTPUT_FILE)
    print(data[["Date", "Close", "FutureClose", "Return", "Label"]].head())


if __name__ == "__main__":
    dates = extract_dates_from_folders(DATA_ROOT)
    print(f"Found {len(dates)} unique dataset dates")

    download_and_label(STOCK_SYMBOL, dates)
