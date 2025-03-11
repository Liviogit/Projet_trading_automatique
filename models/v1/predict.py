"""
predict.py
Charger un modèle LSTM entraîné (model_v1.h5) et prédire un signal Achat/Vente
sur de nouvelles données.
"""

import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_H5_PATH = "models/v1/model_v1.h5"
SCALER_PATH = "models/v1/scaler_v1.pkl"
NEW_DATA_PATH = "data/raw/new_cac40_data.csv"  # Exemple de nouveau dataset

def create_sequences(features, seq_length):
    X_seq = []
    for i in range(seq_length, len(features)):
        X_seq.append(features[i - seq_length : i])
    return np.array(X_seq)

def main():
    # 1) Charger le modèle
    if not os.path.exists(MODEL_H5_PATH):
        print(f"ERREUR : Le fichier {MODEL_H5_PATH} est introuvable.")
        return
    model = load_model(MODEL_H5_PATH)
    print("Modèle chargé avec succès.")

    # 2) Charger le scaler et la config
    if not os.path.exists(SCALER_PATH):
        print(f"ERREUR : Le fichier {SCALER_PATH} est introuvable.")
        return
    with open(SCALER_PATH, "rb") as f:
        saved_data = pickle.load(f)

    scaler = saved_data["scaler"]
    feature_cols = saved_data["feature_cols"]
    seq_length = saved_data["seq_length"]

    # 3) Charger de nouvelles données
    df_new = pd.read_csv(NEW_DATA_PATH, parse_dates=["Date"])
    df_new.sort_values(by="Date", inplace=True)
    df_new.reset_index(drop=True, inplace=True)

    # 4) Sélectionner les features et scaler
    data_new = df_new[feature_cols].values
    data_new_scaled = scaler.transform(data_new)

    # 5) Créer les séquences
    X_new = create_sequences(data_new_scaled, seq_length)
    if len(X_new) == 0:
        print("Pas assez de données pour créer des séquences.")
        return

    # 6) Prédictions
    y_pred_proba = model.predict(X_new)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # 7) Associer les signaux aux dates correspondantes
    df_signals = df_new.iloc[seq_length:].copy()
    df_signals["Signal"] = y_pred

    print("Quelques prédictions:")
    print(df_signals[["Date", "Signal"]].head(10))

    # 8) Sauvegarder les prédictions si besoin
    df_signals.to_csv("models/v1/predictions_v1.csv", index=False)
    print("Prédictions sauvegardées dans models/v1/predictions_v1.csv.")

if __name__ == "__main__":
    main()
