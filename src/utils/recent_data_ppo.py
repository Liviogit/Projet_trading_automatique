# src/utils/generate_recent_data_for_ppo.py

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def generate_recent_data(filepath_out, tickers, days_interval=10):
    """
    Génère un fichier recent.csv avec les colonnes adaptées pour l'environnement PPO.
    
    Args:
        filepath_out (str): chemin pour sauvegarder le fichier recent.csv
        tickers (list): liste des tickers à télécharger
        days_interval (int): nombre de jours récents à récupérer (ex: 10 derniers jours)
    """

    # 1. 📆 Définir les dates
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days_interval)

    # 2. 📥 Télécharger les données avec yfinance
    print(f"Téléchargement des données du {start_date.date()} au {end_date.date()} pour les tickers : {tickers}")

    df = yf.download(
        tickers,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        interval="1d",
        group_by='ticker',
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        print("Erreur : aucune donnée téléchargée.")
        return

    # 3. 🔄 Reformater au bon format (Datetime, Ticker, Open, High, Low, Close, Volume)
    all_data = []
    for ticker in tickers:
        if ticker not in df.columns.get_level_values(0):
            print(f"⚠️ Données manquantes pour {ticker}")
            continue
        sub_df = df[ticker].copy()
        sub_df['Datetime'] = sub_df.index
        sub_df['Ticker'] = ticker
        all_data.append(sub_df)

    if not all_data:
        print("Erreur : aucune donnée utilisable.")
        return

    final_df = pd.concat(all_data)

    # Remettre dans l’ordre
    final_df = final_df[['Datetime', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # 4. 💾 Sauvegarder en CSV
    final_df.to_csv(filepath_out, index=False)
    print(f"✅ Fichier récent généré et sauvegardé ici : {filepath_out}")
