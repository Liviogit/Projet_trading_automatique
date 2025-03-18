import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from trading_env import TradingEnv  # Assure-toi que ton TradingEnv est bien importé

# 📂 Charger les nouvelles données pour le backtest (BNP.PA)
file_path = "/Users/yassinf/GIT/Projet_trading_automatique/Data/cac40_clean_format.csv"
df_cac40 = pd.read_csv(file_path)

# Reformater les données pour un accès rapide
df_cac40_restructured = df_cac40.pivot(index=["Datetime", "Ticker"], columns="Price", values="Value").reset_index()

# Filtrer pour BNP.PA uniquement
selected_ticker = "BNP.PA"
df_test = df_cac40_restructured[df_cac40_restructured["Ticker"] == selected_ticker].reset_index(drop=True)

if df_test.empty:
    raise ValueError(f"🚨 Pas de données disponibles pour {selected_ticker}. Vérifie ton fichier CSV.")

print(f"✅ Backtest en cours sur {selected_ticker}...")

# ✅ Créer l'environnement de backtest
env = TradingEnv(df_test, ticker=selected_ticker)

# ✅ Charger le modèle entraîné sur AC.PA
model = PPO.load("ppo_trading_model")

# 🔄 Exécuter le backtest
obs, _ = env.reset()  # ✅ Correction ici !
done = False

while not done:
    action, _ = model.predict(obs)  # Prédiction du modèle
    obs, reward, done, truncated, info = env.step(action)

# 📊 Résultats finaux
print("\n✅ Backtest terminé !")
print(f"📊 Balance finale: {env.balance:.2f}")
print(f"📈 Actions détenues: {env.shares_held}")
print(f"💰 Valeur totale du portefeuille: {env.total_value:.2f}")
