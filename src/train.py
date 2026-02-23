import os
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from xgboost import XGBClassifier

from src.data_loader import load_dataset
from src.features import process_audio, process_text
from src.market_features import load_market_features

def entropy(p):
    p = np.clip(p, 1e-9, 1.0)
    return -np.sum(p * np.log(p), axis=1, keepdims=True)

def adaptive_gate(audio_probs, market_probs, audio_entropy, market_entropy):
    # Higher confidence = lower entropy
    audio_conf  = 1.0 / (1.0 + audio_entropy)
    market_conf = 1.0 / (1.0 + market_entropy)

    # Normalize into a gate weight
    gate = audio_conf / (audio_conf + market_conf + 1e-9)

    # gated probabilities
    gated_audio  = audio_probs * gate
    gated_market = market_probs * (1 - gate)

    return gated_audio, gated_market, gate





def train_model(data_root):

    # -----------------------
    # LOAD AUDIO + TEXT
    # -----------------------
    X_audio, texts, y, dates = load_dataset(data_root)

    X_audio, audio_scaler = process_audio(X_audio)
    X_text, text_vectorizer = process_text(texts)

    X_audio_text = np.hstack([X_audio, X_text])

    audio_df = pd.DataFrame({
        "date": pd.to_datetime(dates).normalize(),
        "label": y
    })

    # -----------------------
    # LOAD MARKET DATA
    # -----------------------
    X_market, y_market, market_dates = load_market_features()

    market_df = pd.DataFrame({
        "date": pd.to_datetime(market_dates).normalize(),
        "X_market": list(X_market)
    })

    # -----------------------
    # ALIGN BY DATE
    # -----------------------
    audio_df["X_audio"] = list(X_audio_text)

    merged = audio_df.merge(market_df, on="date", how="inner")

    merged = merged.sort_values("date").reset_index(drop=True)

    print(f"Aligned samples: {len(merged)}")

    # -----------------------
    # TIME-BASED SPLIT (NO LEAKAGE)
    # -----------------------
    split_idx = int(len(merged) * 0.8)

    train_df = merged.iloc[:split_idx]
    test_df  = merged.iloc[split_idx:]

    X_audio_train = np.vstack(train_df["X_audio"].values)
    X_audio_test  = np.vstack(test_df["X_audio"].values)

    X_market_train = np.vstack(train_df["X_market"].values)
    X_market_test  = np.vstack(test_df["X_market"].values)

    y_train = train_df["label"].values
    y_test  = test_df["label"].values

    # -----------------------
    # MODEL A — AUDIO CLASSIFIER
    # -----------------------
    audio_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss"
    )

    from sklearn.utils.class_weight import compute_sample_weight

    weights = compute_sample_weight(class_weight="balanced", y=y_train)
    weights = np.sqrt(weights)   # 🔥 soft balancing

    audio_model.fit(
    X_audio_train,
    y_train,
    sample_weight=weights
)


    audio_probs_train = audio_model.predict_proba(X_audio_train)
    audio_probs_test  = audio_model.predict_proba(X_audio_test)

    # -----------------------
    # MODEL B — MARKET CLASSIFIER
    # -----------------------
    market_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss"
    )

    market_model.fit(X_market_train, y_train)

    market_probs_train = market_model.predict_proba(X_market_train)
    market_probs_test  = market_model.predict_proba(X_market_test)

    # -----------------------
    # FUSION FEATURES
    # -----------------------
    audio_entropy_train  = entropy(audio_probs_train)
    audio_entropy_test   = entropy(audio_probs_test)

    market_entropy_train = entropy(market_probs_train)
    market_entropy_test  = entropy(market_probs_test)

    g_audio_train, g_market_train, gate_train = adaptive_gate(
    audio_probs_train,
    market_probs_train,
    audio_entropy_train,
    market_entropy_train
)

    fusion_train = np.hstack([
    g_audio_train,
    g_market_train,
    audio_entropy_train,
    market_entropy_train,
    gate_train
])

    g_audio_test, g_market_test, gate_test = adaptive_gate(
    audio_probs_test,
    market_probs_test,
    audio_entropy_test,
    market_entropy_test
)

    fusion_test = np.hstack([
    g_audio_test,
    g_market_test,
    audio_entropy_test,
    market_entropy_test,
    gate_test
])


    from sklearn.utils.class_weight import compute_sample_weight

    fusion_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss"
    )

    fusion_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    fusion_model.fit(
        fusion_train,
        y_train,
        sample_weight=fusion_weights    
    )

    # -----------------------
    # EVALUATION (REAL METRICS)
    # -----------------------
    y_pred = fusion_model.predict(fusion_test)
    
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec  = recall_score(y_test, y_pred, average="weighted")
    f1   = f1_score(y_test, y_pred, average="weighted")

    print("\n===== FUSION MODEL METRICS (LEAK-FREE) =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\n===== CONFUSION MATRIX =====")

    disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["negative","neutral","positive"],
    xticks_rotation=45
    )

    plt.title("Fusion Model Confusion Matrix")
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()



    # -----------------------
    # SAVE MODELS
    # -----------------------
    os.makedirs("models", exist_ok=True)

    joblib.dump(audio_model,"models/audio_model.pkl")
    joblib.dump(market_model,"models/market_model.pkl")
    joblib.dump(fusion_model,"models/fusion_model.pkl")
    joblib.dump(audio_scaler,"models/audio_scaler.pkl")
    joblib.dump(text_vectorizer,"models/text_vectorizer.pkl")

    print("Training complete. Models saved in /models")

if __name__ == "__main__":
    DATA_ROOT = "data/maec_repo/MAEC_Dataset"
    train_model(DATA_ROOT)