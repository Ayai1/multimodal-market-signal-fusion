import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

def process_audio(X_audio):
    scaler = StandardScaler()
    X_audio = scaler.fit_transform(X_audio)
    return X_audio, scaler



def process_text(texts):
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    X_text = vectorizer.fit_transform(texts).toarray()
    return X_text, vectorizer
