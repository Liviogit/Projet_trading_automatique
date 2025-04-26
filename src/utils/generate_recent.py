# 📄 generate_recent.py

import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import os

def generate_recent_data(filepath_out, tickers, days_interval=10):
    """
    Télécharge les données des tickers sur les X derniers jours et sauvegarde dans un fichier CSV.
    """
    # 📅 Dates
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days_interval)

    print(f"📥 Téléchargement des données du {start_date.date()} au {end_date.date()}...")

    # 📈 Télécharger avec yfinance
    data = yf.download(
        tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval="1d",  # données journalières
        group_by="ticker",
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        print("⚠️ Aucun résultat téléchargé.")
        return

    # 🎯 Formatage du dataframe
    if isinstance(data.columns, pd.MultiIndex):
        df_list = []
        for ticker in data.columns.levels[0]:
            df_ticker = data[ticker].copy()
            df_ticker["Ticker"] = ticker
            df_ticker["Datetime"] = df_ticker.index
            df_list.append(df_ticker)
        df_final = pd.concat(df_list)
        df_final.reset_index(drop=True, inplace=True)
    else:
        df_final = data.copy()
        df_final["Ticker"] = tickers[0]
        df_final["Datetime"] = df_final.index
        df_final.reset_index(drop=True, inplace=True)

    # 🛠 Nettoyage
    columns_to_keep = ["Datetime", "Ticker", "Open", "High", "Low", "Close", "Volume"]
    df_final = df_final[columns_to_keep]

    # 📂 Sauvegarder
    os.makedirs(os.path.dirname(filepath_out), exist_ok=True)
    df_final.to_csv(filepath_out, index=False)
    print(f"✅ Données sauvegardées dans {filepath_out}")



if __name__ == "__main__":
    tickers = [
        "AC.PA", "ACA.PA", "AI.PA", "AIR.PA", "BN.PA", "BNP.PA",
        "CA.PA", "CAP.PA", "CS.PA", "DG.PA", "DSY.PA", "EDEN.PA",
        "EL.PA", "EN.PA", "ENGI.PA", "ERF.PA", "GLE.PA", "HO.PA",
        "KER.PA", "LR.PA", "MC.PA", "ML.PA", "MT.AS", "OR.PA",
        "ORA.PA", "PUB.PA", "RI.PA", "RMS.PA", "RNO.PA", "SAF.PA",
        "SAN.PA", "SGO.PA", "STLAP.PA", "SU.PA", "TEP.PA", "TTE.PA",
        "VIE.PA", "VIV.PA", "STMPA.PA", "URW.PA"
    ]

    generate_recent_data(
        filepath_out="Data/recent.csv",
        tickers=tickers,
        days_interval=2  # Tu peux changer pour 5, 7, etc
    )
