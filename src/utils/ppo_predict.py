# 📄 src/utils/ppo_predict.py

from stable_baselines3 import PPO
import pandas as pd
import os
from src.utils.multi_ticker_env import MultiTickerTradingEnv
from pages.portefeuille import get_portfolio

ppo_model = None
MODEL_PATH = "Data/model/ppo.zip"

def load_ppo_model():
    global ppo_model
    if ppo_model is None:
        if os.path.exists(MODEL_PATH):
            print("✅ Loading PPO model...")
            ppo_model = PPO.load(MODEL_PATH)
            print("✅ PPO model loaded successfully!")
        else:
            print("❌ PPO model not found at", MODEL_PATH)
            ppo_model = None

def get_ppo_prediction(path_recent):
    global ppo_model

    if ppo_model is None:
        load_ppo_model()

    if ppo_model is None:
        raise ValueError("PPO model could not be loaded.")

    # 📥 Load recent.csv
    df = pd.read_csv(path_recent, parse_dates=["Datetime"])
    all_tickers = [
        "AC.PA", "ACA.PA", "AI.PA", "AIR.PA", "BN.PA", "BNP.PA", "CA.PA", "CAP.PA",
        "CS.PA", "DG.PA", "DSY.PA", "EDEN.PA", "EL.PA", "EN.PA", "ENGI.PA", "ERF.PA",
        "GLE.PA", "HO.PA", "KER.PA", "LR.PA", "MC.PA", "ML.PA", "MT.AS", "OR.PA",
        "ORA.PA", "PUB.PA", "RI.PA", "RMS.PA", "RNO.PA", "SAF.PA", "SAN.PA", "SGO.PA",
        "STLAP.PA", "SU.PA", "TEP.PA", "TTE.PA", "VIE.PA", "VIV.PA", "STMPA.PA", "URW.PA"
    ]

    portefeuille_tickers = get_portfolio()  # <- tickers choisis par l'utilisateur

    today = df["Datetime"].max()
    df_today = df[df["Datetime"] == today]

    if df_today.empty:
        raise ValueError("No today's data found in recent.csv.")

    # 🌍 Create environment with all tickers
    env = MultiTickerTradingEnv(df_today, tickers=all_tickers)
    obs = env.reset() # <-- CORRECT: Unpack (obs, info)

    # ⚡ Predict
    action, _ = ppo_model.predict(obs, deterministic=True)

    signals = []
    for i, act in enumerate(action):
        ticker = all_tickers[i]
        if ticker not in portefeuille_tickers:
            continue  # ⛔ On ne garde que les tickers du portefeuille

        decision = ("Hold", "Buy", "Sell")[act]
        last_close = df_today[df_today["Ticker"] == ticker]["Close"].values[-1] if not df_today[df_today["Ticker"] == ticker].empty else None

        signals.append({
            "Ticker": ticker,
            "Decision": decision,
            "Close": last_close
        })

    df_signals = pd.DataFrame(signals)
    return df_signals, today.strftime("%Y-%m-%d")