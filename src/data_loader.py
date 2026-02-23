import os
import pandas as pd
import numpy as np
from datetime import datetime


LABEL_FILE = "data/yfinance_labeled_AAPL.csv"

LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}


def load_dataset(root):
    X_audio = []
    texts = []
    labels = []
    dates = []
    label_df = pd.read_csv(LABEL_FILE)
    label_df["Date"] = pd.to_datetime(label_df["Date"])

    label_lookup = dict(zip(label_df["Date"], label_df["Label"]))

    for root_dir, _, files in os.walk(root):
        if "features_clean.csv" in files and "text.txt" in files:

            folder_name = os.path.basename(root_dir)

            try:
                date_str = folder_name.split("_")[0]
                folder_date = pd.to_datetime(
                    datetime.strptime(date_str, "%Y%m%d")
                )
            except:
                continue

            if folder_date not in label_lookup:
                continue

            label_text = label_lookup[folder_date]
            label = LABEL_MAP[label_text]

            df = pd.read_csv(os.path.join(root_dir, "features_clean.csv"))
            X_audio.append(df.values.flatten())

            with open(os.path.join(root_dir, "text.txt"), "r", encoding="utf-8") as f:
                texts.append(f.read())

            labels.append(label)
            dates.append(folder_date)


    X_audio = np.array(X_audio)

    print(f"Loaded {len(X_audio)} samples")
    print(f"Audio feature dim: {X_audio.shape[1]}")
    print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

    return X_audio, texts, labels, dates

