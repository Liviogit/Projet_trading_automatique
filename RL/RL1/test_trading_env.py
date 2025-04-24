import pandas as pd
from RL.RL1.trading_env import TradingEnv  # Assure-toi que `trading_env.py` est dans le même dossier

# 📂 Charger les données du fichier CAC40
file_path = "/Users/yassinf/GIT/Projet_trading_automatique/Data/cac40_clean_format.csv"  # Mets le chemin correct si besoin
df_cac40 = pd.read_csv(file_path)

# 🏗️ Transformer les données au format OHLCV
df_cac40_pivot = df_cac40.pivot_table(index=["Datetime", "Ticker"], columns="Price", values="Value").reset_index()
df_cac40_pivot.columns = ["Datetime", "Ticker", "Close", "High", "Low", "Open", "Volume"]
df_cac40_pivot["Datetime"] = pd.to_datetime(df_cac40_pivot["Datetime"])
df_cac40_pivot = df_cac40_pivot.sort_values(by=["Datetime", "Ticker"]).reset_index(drop=True)

# 🎯 Sélectionner un ticker du CAC40 pour le test
selected_ticker = df_cac40_pivot["Ticker"].unique()[0]  # Prend le premier ticker disponible
print(f"✅ Test sur le ticker : {selected_ticker}")

# 🔄 Créer l’environnement de trading
env = TradingEnv(df_cac40_pivot, ticker=selected_ticker)

# 🏁 Tester l’environnement
print("\n🔄 Réinitialisation de l'environnement...")
obs = env.reset()
print(f"Observation initiale: {obs}")

# 🚀 Tester quelques actions
actions = [0, 1, 1, 2, 0, 1, 2]  # Hold, Buy, Buy, Sell, Hold, Buy, Sell
print("\n🚀 Test des actions...")
for action in actions:
    obs, reward, done, _ = env.step(action)
    print(f"\nAction: {action} | Reward: {reward:.2f} | Done: {done}")
    env.render()

# ✅ Vérifier si tout fonctionne
if done:
    print("\n✅ Test réussi : L'environnement fonctionne avec CAC40 !")
