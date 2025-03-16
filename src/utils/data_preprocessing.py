import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Définition du chemin vers le fichier CSV
CSV_PATH = "C:/Users/El Hammoumi/Desktop/Projet_trading_automatique/Data/cac40_clean_format.csv"

def load_data(file_path=CSV_PATH):
    """Charge et nettoie les données de trading."""
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    
    # Vérification et suppression des valeurs manquantes
    df = df.dropna()

    # Vérifier les colonnes disponibles
    print("Colonnes du dataset :", df.columns)

    return df

def preprocess_data(df):
    """Normalise les données pour l'agent RL."""
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])
    return df_scaled, scaler

# Exemple d'utilisation
if __name__ == "__main__":
    df = load_data()
    df_scaled, scaler = preprocess_data(df)
    print("Aperçu des données normalisées :\n", df_scaled[:5])
