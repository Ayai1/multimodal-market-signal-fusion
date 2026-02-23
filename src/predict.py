import os
import joblib
import pandas as pd
import numpy as np


def predict_folder(folder_path):
    model = joblib.load("models/maec_model.pkl")
    audio_scaler = joblib.load("models/audio_scaler.pkl")
    text_vectorizer = joblib.load("models/text_vectorizer.pkl")

    df = pd.read_csv(os.path.join(folder_path, "features.csv"))
    audio = df.select_dtypes(include="number").values.flatten()
    audio = audio_scaler.transform([audio])

    with open(os.path.join(folder_path, "text.txt"), "r", encoding="utf-8") as f:
        text = f.read()

    text_vec = text_vectorizer.transform([text]).toarray()
    X = np.hstack([audio, text_vec])

    pred = model.predict(X)[0]
    conf = model.predict_proba(X).max()

    return pred, conf
