import optuna
from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd
import numpy as np
from multi_ticker_env import MultiTickerTradingEnv
import os
from datetime import datetime
from importlib import reload
import multi_ticker_env
reload(multi_ticker_env)

# 📁 Charger les données
file_path = "/Users/yassinf/GIT/Projet_trading_automatique/Data/cac40_clean_format.csv"
df = pd.read_csv(file_path)

# 📊 Pivot + nettoyage
df = df[df["Price"].isin(["Open", "High", "Low", "Close", "Volume"])]
df = df.pivot(index=["Datetime", "Ticker"], columns="Price", values="Value").reset_index()
df.columns.name = None
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values(by=["Datetime", "Ticker"]).reset_index(drop=True)
df = df.ffill().bfill()

print("✅ Données pivotées et triées")

# 🧠 Split entraînement = avant 2024-01-01
df_train = df[df["Datetime"] < pd.to_datetime("2024-01-01", utc=True)].copy()

tickers = df_train["Ticker"].unique().tolist()
print(f"🧠 Tickers utilisés pour l'entraînement : {tickers}")

# ✅ Créer l’environnement
env = DummyVecEnv([lambda: MultiTickerTradingEnv(df_train, tickers=tickers)])

# ⚙️ Créer le modèle PPO
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_multi_tensorboard")

# 🚀 Entraîner le modèle
print("\n🚀 Début de l'entraînement PPO multi-ticker (1 itération)...")
from tqdm import trange

# 🚀 Entraînement avec barre de progression
iterations = 10  # ou plus, selon ce que tu veux
timesteps_per_iter = 1024  # Nombre de pas de temps par itération
for i in trange(iterations, desc="🧠 Entraînement PPO multi-ticker"):
    model.learn(total_timesteps=timesteps_per_iter, reset_num_timesteps=False)
    model.save(f"ppo_multi_model_iter_{i+1}.zip")


# 💾 Sauvegarder le modèle
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"ppo_multi_model_iter_1.zip"
model.save(model_name)

print(f"\n✅ Entraînement terminé et modèle sauvegardé : {model_name}")
