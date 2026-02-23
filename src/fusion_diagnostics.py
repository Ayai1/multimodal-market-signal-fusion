import joblib
import numpy as np

fusion_model = joblib.load("models/fusion_model.pkl")

print("fusion diagnostics")

#logistic reg
if hasattr(fusion_model, "coef_"):

    weights = np.abs(fusion_model.coef_).mean(axis=0)

    audio_importance  = weights[:3].sum()
    market_importance = weights[3:].sum()

    print("\nModel Type: LogisticRegression")

    print("\nAudio influence :", round(float(audio_importance),4))
    print("Market influence:", round(float(market_importance),4))

    print("\nPer-feature influence:")
    names = [
        "Audio_neg","Audio_neutral","Audio_pos",
        "Market_neg","Market_neutral","Market_pos"
    ]

    for n, w in zip(names, weights):
        print(f"{n:15s} -> {float(w):.4f}")

elif hasattr(fusion_model, "feature_importances_"):

    weights = fusion_model.feature_importances_

    audio_importance  = weights[:3].sum()
    market_importance = weights[3:].sum()

    print("\nModel Type: XGBClassifier")

    print("\nAudio influence :", round(float(audio_importance),4))
    print("Market influence:", round(float(market_importance),4))

    print("\nPer-feature importance:")
    names = [
        "Audio_neg","Audio_neutral","Audio_pos",
        "Market_neg","Market_neutral","Market_pos"
    ]

    for n, w in zip(names, weights):
        print(f"{n:15s} -> {float(w):.4f}")

else:
    print("Unsupported fusion model type.")
