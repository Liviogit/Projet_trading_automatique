import pandas as pd
import numpy as np
import ta  # Librairie pour les indicateurs techniques
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from trading_env import TradingEnv  # Assurez-vous que ce fichier est bien défini

# 📂 Charger les données du CAC40
file_path = "/Users/yassinf/GIT/Projet_trading_automatique/Data/cac40_clean_format.csv"
df_cac40 = pd.read_csv(file_path)

# Vérifier les colonnes disponibles
print("🔍 Colonnes initiales :", df_cac40.columns)

# 📊 Vérifier si les valeurs de "Price" sont correctes
print("🔍 Valeurs uniques de Price :", df_cac40["Price"].unique())

# ⚠️ S'assurer que seules les valeurs OHLCV sont utilisées
df_cac40 = df_cac40[df_cac40["Price"].isin(["Open", "High", "Low", "Close", "Volume"])]

# 🏗️ Reformater les données pour un accès rapide
df_cac40_restructured = df_cac40.pivot(index=["Datetime", "Ticker"], columns="Price", values="Value").reset_index()

# 🔍 Vérifier les colonnes après pivot
print("🔍 Colonnes après pivot :", df_cac40_restructured.columns)

# ⚠️ Renommer les colonnes si nécessaire
df_cac40_restructured.columns = [col if isinstance(col, str) else str(col) for col in df_cac40_restructured.columns]

# 🔄 Vérifier que les colonnes OHLCV existent
required_columns = ["Open", "High", "Low", "Close", "Volume"]
missing_columns = [col for col in required_columns if col not in df_cac40_restructured.columns]

if missing_columns:
    raise ValueError(f"⚠️ Les colonnes suivantes sont manquantes après la transformation : {missing_columns}")

# 🔄 Trier les données par date et ticker
df_cac40_restructured["Datetime"] = pd.to_datetime(df_cac40_restructured["Datetime"])
df_cac40_restructured = df_cac40_restructured.sort_values(by=["Datetime", "Ticker"]).reset_index(drop=True)

print("✅ Préparation des données terminée avec succès !")

# 📈 Ajouter des indicateurs techniques
def add_indicators(df):
    df = df.copy()
    
    # SMA (Moyenne mobile simple)
    df["SMA_10"] = ta.trend.sma_indicator(df["Close"], window=10)
    df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)

    # EMA (Moyenne mobile exponentielle)
    df["EMA_10"] = ta.trend.ema_indicator(df["Close"], window=10)
    df["EMA_50"] = ta.trend.ema_indicator(df["Close"], window=50)

    # RSI (Relative Strength Index)
    df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)

    # Bollinger Bands
    df["BB_high"] = ta.volatility.bollinger_hband(df["Close"], window=20)
    df["BB_low"] = ta.volatility.bollinger_lband(df["Close"], window=20)

    # Remplacer les NaN
    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)
    
    return df

# Appliquer les indicateurs techniques
df_cac40_restructured = add_indicators(df_cac40_restructured)

# 🎯 Sélectionner un ticker pour l'entraînement
selected_ticker = "AC.PA"  # Remplace ici par le ticker souhaité
print(f"✅ Entraînement sur le ticker : {selected_ticker}")

# 🔄 Fonction pour créer l'environnement avec les nouvelles données
def create_env():
    return TradingEnv(df_cac40_restructured, ticker=selected_ticker)

if __name__ == "__main__":
    vec_env = make_vec_env(create_env, n_envs=1)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="./ppo_trading_tensorboard/",
        gamma=0.99,  # Favoriser les gains à court terme
        learning_rate=3e-5,  # Légère augmentation pour forcer l’apprentissage
        clip_range=0.3,  # Réduire le clipping pour encourager l'exploration
        ent_coef=0.005,  # Encourager la diversité des actions
        n_steps=8192,  # Augmenter le nombre d'échantillons par itération
        batch_size=1024,  # Ajout d'un batch number pour stabiliser l'entraînement
    )

    print("\n🚀 Début de l'entraînement PPO...")
    model.learn(total_timesteps=50000)

    model.save("ppo_trading_model")
    print("\n✅ Entraînement terminé ! Modèle sauvegardé.")

    print("📅 Période couverte par les données :")
    print(f"Début : {df_cac40_restructured['Datetime'].min()}")
    print(f"Fin   : {df_cac40_restructured['Datetime'].max()}")

