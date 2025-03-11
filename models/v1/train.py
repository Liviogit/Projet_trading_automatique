"""
train.py
Entraîne un modèle LSTM pour prédire un signal Achat/Vente (binaire).
Sauvegarde :
 - les poids et l'architecture du modèle dans models/v1/model_v1.h5
 - le scaler dans models/v1/scaler_v1.pkl
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# ---------------------------
# Paramètres & chemins
# ---------------------------
DATA_PATH = "data/processed/cac40_clean_format.csv"
MODEL_H5_PATH = "models/v1/model_v1.h5"      # Fichier .h5
SCALER_PATH = "models/v1/scaler_v1.pkl"      # Fichier .pkl

# Paramètres d'entraînement
TEST_SIZE = 0.2
EPOCHS = 10
BATCH_SIZE = 32
SEQ_LENGTH = 30  # Fenêtre historique pour le LSTM (30 jours)

def load_data(data_path):
    """
    Charge le CSV nettoyé et renvoie un DataFrame.
    """
    df = pd.read_csv(data_path, parse_dates=["Datetime"])
    df.sort_values(by="Datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def create_labels(df, horizon=5, threshold=0.0):
    """
    Crée un label binaire (Achat=1 / Vente=0) en comparant le prix futur
    (Close t+horizon) vs le prix actuel (Close t).
    threshold peut être un % de variation minimale pour valider l'achat.
    """
    df["Future_Close"] = df["Close"].shift(-horizon)
    df["Signal"] = ((df["Future_Close"] - df["Close"]) / df["Close"] > threshold).astype(int)
    df.dropna(inplace=True)
    return df

def create_sequences(features, labels, seq_length):
    """
    Transforme les données en séquences pour le LSTM.
    features.shape = (n_samples, n_features)
    labels.shape   = (n_samples,)
    Retourne X, y de forme (n_sequences, seq_length, n_features), (n_sequences,)
    """
    X, y = [], []
    for i in range(seq_length, len(features)):
        X.append(features[i - seq_length : i])
        y.append(labels[i])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """
    Construit un modèle LSTM simple.
    input_shape = (seq_length, nb_features)
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))  # binaire Achat/Vente
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def main():
    # 1) Charger le dataset
    df = load_data(DATA_PATH)

    # 2) Créer les labels Achat/Vente
    df = create_labels(df, horizon=5, threshold=0.0)

    # 3) Sélection des features
    feature_cols = ["Close", "Volume"]  # Ajouter RSI, MACD si dispos
    data = df[feature_cols].values
    labels = df["Signal"].values

    # 4) Scaler
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # 5) Créer les séquences pour LSTM
    X, y = create_sequences(data_scaled, labels, SEQ_LENGTH)

    # 6) Split en train / test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)

    # 7) Construire le modèle
    model = build_lstm_model((SEQ_LENGTH, X.shape[2]))

    # 8) Entraîner
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # 9) Évaluation rapide
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f} | Test Accuracy: {acc:.4f}")

    # 10) Sauvegarder le modèle au format H5
    model.save(MODEL_H5_PATH)
    print(f"Modèle sauvegardé dans {MODEL_H5_PATH}")

    # 11) Sauvegarder le scaler
    with open(SCALER_PATH, "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "feature_cols": feature_cols,
            "seq_length": SEQ_LENGTH
        }, f)
    print(f"Scaler sauvegardé dans {SCALER_PATH}")

if __name__ == "__main__":
    main()
