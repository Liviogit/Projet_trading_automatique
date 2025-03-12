import pandas as pd
import numpy as np
from trading_env import TradingEnv


# Charger un DataFrame de test (avec des données factices)
data = pd.DataFrame({
    "Datetime": pd.date_range(start="2025-03-01", periods=10, freq="c"),
    "Ticker": ["AC.PA"] * 10,
    "Open": np.random.uniform(50, 150, 10),
    "High": np.random.uniform(50, 150, 10),
    "Low": np.random.uniform(50, 150, 10),
    "Close": np.random.uniform(50, 150, 10),
    "Volume": np.random.uniform(1000, 5000, 10),
})

# Transformer les données comme dans le script principal
data["Datetime"] = pd.to_datetime(data["Datetime"])
data = data.sort_values(by=["Datetime", "Ticker"]).reset_index(drop=True)

# Importer l'environnement de trading
env = TradingEnv(data, ticker="AC.PA")

# ✅ Vérifier `reset()`
print("\n🔄 Réinitialisation de l'environnement...")
obs = env.reset()
print(f"Observation initiale: {obs}")

# ✅ Vérifier `step(action)`
actions = [0, 1, 1, 2, 0, 1, 2]  # Séquence d'actions : Hold, Buy, Buy, Sell...
print("\n🚀 Test des actions...")
for action in actions:
    obs, reward, done, _ = env.step(action)
    print(f"\nAction: {action} | Reward: {reward:.2f} | Done: {done}")
    env.render()

# ✅ Vérifier `done`
if done:
    print("\n✅ Fin de l'épisode atteinte avec succès !")
