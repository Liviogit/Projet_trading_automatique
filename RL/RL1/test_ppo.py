import gymnasium as gym  # ✅ Correction pour compatibilité avec Gymnasium
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from RL.RL1.trading_env import TradingEnv  # ✅ Vérifie que le fichier `trading_env.py` est bien accessible

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

# 🎯 Sélectionner un ticker pour le test
selected_ticker = df_cac40_pivot["Ticker"].unique()[0]
print(f"✅ Test du modèle sur le ticker : {selected_ticker}")

# 🔄 Charger l’environnement
env = TradingEnv(df_cac40_pivot, ticker=selected_ticker)

# 🔄 Charger le modèle PPO entraîné
model_path = "ppo_trading_model.zip"  # Vérifie que le fichier existe bien !
model = PPO.load(model_path)

# 🚀 Tester l'agent entraîné
obs, _ = env.reset()  # ✅ Correction : reset() retourne maintenant un tuple (obs, info)
done = False
total_reward = 0

print("\n🎬 Début du test PPO...\n")

for step in range(100):  # On teste seulement 10 étapes pour voir le comportement
    action, _ = model.predict(obs, deterministic=True)  # Choix d'une action
    obs, reward, terminated, truncated, _ = env.step(action)  # ✅ Correction
    done = terminated or truncated  # ✅ Fusion des flags pour compatibilité


    total_reward += reward

    env.render()  # Affiche l’état actuel du trading

    if done:
        print("\n💡 Fin de l'épisode")
        break

print(f"\n✅ Test terminé ! Récompense totale : {total_reward:.2f}")
