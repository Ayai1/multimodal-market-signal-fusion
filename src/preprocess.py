import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def clean_features_csv(csv_path, expected_columns=None):
    df = pd.read_csv(csv_path)

    # force numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)

    # enforce schema
    if expected_columns is not None:
        df = df.reindex(columns=expected_columns, fill_value=0.0)

    # statistical pooling
    features = {}
    for col in df.columns:
        features[f"{col}_mean"] = df[col].mean()
        features[f"{col}_std"]  = df[col].std()
        features[f"{col}_min"]  = df[col].min()
        features[f"{col}_max"]  = df[col].max()

    return features, list(df.columns)


def preprocess_dataset(data_root, output_name="features_clean.csv"):
    all_columns = set()

    # pass 1: collect schema
    for root, _, files in os.walk(data_root):
        if "features.csv" in files:
            _, cols = clean_features_csv(os.path.join(root, "features.csv"))
            all_columns.update(cols)

    all_columns = sorted(list(all_columns))

    # pass 2: clean + save
    for root, _, files in tqdm(list(os.walk(data_root))):
        if "features.csv" in files:
            csv_path = os.path.join(root, "features.csv")
            clean_feats, _ = clean_features_csv(
                csv_path, expected_columns=all_columns
            )

            out_path = os.path.join(root, output_name)
            pd.DataFrame([clean_feats]).to_csv(out_path, index=False)

    print("Preprocessing complete.")


if __name__ == "__main__":
    DATA_ROOT = "data/maec_repo/MAEC_Dataset"
    preprocess_dataset(DATA_ROOT)
