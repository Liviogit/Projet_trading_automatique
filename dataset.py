import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Fonction pour ajuster les dates
def adjust_dates(days_interval):
    # La date d'aujourd'hui
    end_date = datetime.today()
    
    # Calcul de la start date en soustrayant le nombre de jours à la end date
    start_date = end_date - timedelta(days=days_interval)
    
    # Retourne les deux dates au format 'YYYY-MM-DD'
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

# Nombre de jours que tu veux dans l'intervalle
days_interval = 729  # Exemple d'intervalle de 30 jours

# Ajustement des dates
start_date, end_date = adjust_dates(days_interval)

# Charger la liste des tickers depuis le fichier
with open("cac40_tickers.txt", "r") as f:
    tickers = [line.strip() for line in f.readlines()]

interval = "1h"  # Données horaires

# Télécharger les données avec yfinance pour l'intervalle spécifié
data = yf.download(tickers, start=start_date, end=end_date, interval=interval, group_by="ticker",multi_level_index=False)

# Sauvegarder les données sous format CSV
data.to_csv("cac40_stock_data.csv")

print(f" Données du CAC 40 récupérées entre {start_date} et {end_date} et enregistrées dans 'cac40_stock_data.csv'.")
