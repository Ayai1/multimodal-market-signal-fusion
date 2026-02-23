import pandas as pd

LABEL_FILE ="data/yfinance_labeled_AAPL.csv"

def load_market_features():

    df = pd.read_csv(LABEL_FILE)

    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    df["ReturnLag1"] = df["Return"].shift(1)
    df["ReturnLag2"] = df["Return"].shift(2)
    df["Volatility3"] = df["Return"].rolling(3).std()

    df["Momentum5"] = df["Close"].pct_change(5)
    df["Momentum10"] = df["Close"].pct_change(10)

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA_ratio"] = df["MA5"] / df["MA10"]

    df["VolumeChange"] = df["Volume"].pct_change()
    df["Volatility5"] = df["Return"].rolling(5).std()

    df["RSI14"] = 100 - (100 / (1 + df["Return"].rolling(14).mean() / df["Return"].rolling(14).std()))
    df["Price_vs_MA10"] = df["Close"] / df["MA10"]
    df["ReturnAbs"] = df["Return"].abs()

    df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)

    df.dropna(inplace=True)

    X_market = df[
    [
        "ReturnLag1",
        "ReturnLag2",
        "Volatility3",
        "Momentum5",
        "Momentum10",
        "MA_ratio",
        "VolumeChange",
        "Volatility5",
        "RSI14",
        "Price_vs_MA10",
        "ReturnAbs"
        ]
    ].values


    label_map = {"negative":0,"neutral":1,"positive":2}
    y_market = df["Label"].map(label_map).values

    dates = df["Date"].values

    return X_market, y_market, dates
