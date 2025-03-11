"""
evaluate.py
Évalue la performance du modèle LSTM (model_v1.h5) sur un dataset de test.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_H5_PATH = "models/v1/model_v1.h5"
SCALER_PATH = "models/v1/scaler_v1.pkl"
TEST_DATA_PATH = "data/processed/cac40_clean.csv"
SEQ_LENGTH = 30  # Va être chargé depuis SCALER_PATH, sauf si vous voulez le forcer

def create_sequences(features, labels, seq_length):
    X_seq, y_seq = [], []
    for i in range(seq_length, len(features)):
        X_seq.append(features[i - seq_length : i])
        y_seq.append(labels[i])
    return np.array(X_seq), np.array(y_seq)

def main():
    # 1) Vérifier la présence des fichiers
    if not os.path.exists(MODEL_H5_PATH):
        print(f"ERREUR : Le fichier {MODEL_H5_PATH} est introuvable.")
        return
    if not os.path.exists(SCALER_PATH):
        print(f"ERREUR : Le fichier {SCALER_PATH} est introuvable.")
        return

    # 2) Charger le modèle et le scaler
    model = load_model(MODEL_H5_PATH)
    with open(SCALER_PATH, "rb") as f:
        saved_data = pickle.load(f)

    scaler = saved_data["scaler"]
    feature_cols = saved_data["feature_cols"]
    seq_length = saved_data["seq_length"]  # Écrase la constante SEQ_LENGTH ci-dessus

    # 3) Charger le dataset de test (même structure que celui d'entraînement)
    df_test = pd.read_csv(TEST_DATA_PATH, parse_dates=["Date"])
    df_test.sort_values(by="Date", inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    # S'assurer qu'on a la colonne "Signal" (créée lors du cleaning ou identique au train)
    if "Signal" not in df_test.columns:
        # Sinon, on peut la recréer comme dans train.py :
        df_test["Future_Close"] = df_test["Close"].shift(-5)
        df_test["Signal"] = (df_test["Future_Close"] > df_test["Close"]).astype(int)
        df_test.dropna(inplace=True)

    # 4) Sélectionner les features + labels
    data_test = df_test[feature_cols].values
    data_test_scaled = scaler.transform(data_test)
    labels_test = df_test["Signal"].values

    # 5) Créer les séquences
    X_seq, y_seq = create_sequences(data_test_scaled, labels_test, seq_length)
    if len(X_seq) == 0:
        print("Pas assez de données pour créer des séquences de test.")
        return

    # 6) Prédictions
    y_pred_proba = model.predict(X_seq)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # 7) Métriques
    acc = accuracy_score(y_seq, y_pred)
    cm = confusion_matrix(y_seq, y_pred)
    report = classification_report(y_seq, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print("Matrice de confusion :\n", cm)
    print("Rapport de classification:\n", report)

if __name__ == "__main__":
    main()
