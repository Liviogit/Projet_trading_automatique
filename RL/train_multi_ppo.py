from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from multi_ticker_env import MultiTickerTradingEnv
import pandas as pd
from datetime import datetime
from importlib import reload
import multi_ticker_env
reload(multi_ticker_env)

# 📁 Charger les données
file_path = "/Users/yassinf/GIT/Projet_trading_automatique/Data/cac40_clean_format.csv"
df = pd.read_csv(file_path)

# 🧼 Nettoyage & pivot
df = df[df["Price"].isin(["Open", "High", "Low", "Close", "Volume"])]
df = df.pivot(index=["Datetime", "Ticker"], columns="Price", values="Value").reset_index()
df.columns.name = None
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values(by=["Datetime", "Ticker"]).reset_index(drop=True)
df = df.ffill().bfill()

print("✅ Données pivotées et triées")

# 📆 Split en données d'entraînement
df_train = df[df["Datetime"] < pd.to_datetime("2024-01-01", utc=True)].copy()
tickers = df_train["Ticker"].unique().tolist()
print(f"🧠 Tickers utilisés pour l'entraînement : {tickers}")

# 🌍 Environnement
env = DummyVecEnv([lambda: MultiTickerTradingEnv(df_train, tickers=tickers)])

# 🔧 Hyperparamètres optimisés
best_params = {
    'learning_rate': 3e-5,
    'n_steps': 256,
    'gamma': 0.99,
    'gae_lambda': 0.9306,
    'ent_coef': 0.005
}

# 🧠 Entraînement du modèle avec les bons paramètres
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_multi_tensorboard", **best_params)

print("\n🚀 Début de l'entraînement avec hyperparamètres optimisés...")
model.learn(total_timesteps=10240)

# 💾 Sauvegarde
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model.save(f"ppo_multi_model_optimized_{timestamp}.zip")
print(f"\n✅ Modèle optimisé sauvegardé sous : ppo_multi_model_optimized_{timestamp}.zip")
