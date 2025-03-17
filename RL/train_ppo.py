import gymnasium as gym  # ✅ Correction pour compatibilité avec Gymnasium
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from trading_env import TradingEnv  # ✅ Vérifie que le fichier `trading_env.py` est bien accessible

# 📂 Charger les données du CAC40
file_path = "/Users/yassinf/GIT/Projet_trading_automatique/Data/cac40_clean_format.csv"  # ✅ Vérifie que le chemin est correct
df_cac40 = pd.read_csv(file_path)

# 🏗️ Transformer les données au format OHLCV
df_cac40_pivot = df_cac40.pivot(index=["Datetime", "Ticker"], columns="Price", values="Value").reset_index()

# 🔄 Vérifier que toutes les colonnes nécessaires existent après le pivot
required_columns = ["Open", "High", "Low", "Close", "Volume"]
missing_columns = [col for col in required_columns if col not in df_cac40_pivot.columns]

if missing_columns:
    raise ValueError(f"Les colonnes suivantes sont manquantes après la transformation : {missing_columns}")

# 🗂️ Trier les données par date et ticker
df_cac40_pivot["Datetime"] = pd.to_datetime(df_cac40_pivot["Datetime"])
df_cac40_pivot = df_cac40_pivot.sort_values(by=["Datetime", "Ticker"]).reset_index(drop=True)

# 🎯 Sélectionner un ticker pour l'entraînement
selected_ticker = df_cac40_pivot["Ticker"].unique()[0]
print(f"✅ Entraînement sur le ticker : {selected_ticker}")

# 🔄 Fonction pour créer l'environnement, nécessaire pour `make_vec_env`
def create_env():
    return TradingEnv(df_cac40_pivot, ticker=selected_ticker)

if __name__ == "__main__":
    # 🏗️ Wrapper vectorisé pour PPO
    vec_env = make_vec_env(create_env, n_envs=1)

    # 🤖 Définir le modèle PPO
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./ppo_trading_tensorboard/")

    # 🚀 Entraîner l’agent PPO
    print("\n🚀 Début de l'entraînement PPO...")
    model.learn(total_timesteps=10000)  # Augmente cette valeur pour de meilleurs résultats

    # 💾 Sauvegarder le modèle entraîné
    model.save("ppo_trading_model")

    print("\n✅ Entraînement terminé ! Modèle sauvegardé.")

