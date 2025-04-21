import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import ta

# Fonction pour ajuster les dates
def adjust_dates(days_interval=60):
    # La date d'aujourd'hui
    end_date = datetime.today()
    
    # Calcul de la start date en soustrayant le nombre de jours à la end date
    start_date = end_date - timedelta(days=days_interval)
    
    # Retourne les deux dates au format 'YYYY-MM-DD'
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def get_tickers(FilePath):
    # Charger la liste des tickers depuis le fichier
    with open(FilePath, "r") as f:
        tickers = [line.strip() for line in f.readlines()]
    return tickers


def clean_data(FilePath):
    # Charger le fichier CSV avec l'index 'Datetime'
    df = pd.read_csv(FilePath, header=[0, 1], index_col=0)

    # Réorganiser les données en format long en utilisant 'stack()' sur les colonnes multi-index
    df_long = df.stack(level=['Price', 'Ticker']).reset_index()

    # Renommer les colonnes pour plus de clarté
    df_long.columns = ['Datetime', 'Price', 'Ticker', 'Value']

    # Sauvegarder le DataFrame transformé si nécessaire
    df_long.to_csv(FilePath, index=False)


def get_data(FilePathinput, FilePathoutput,days_interval=60):
    # Obtenir les tickers
    tickers = get_tickers(FilePathinput)
    
    # Ajuster les dates
    start_date, end_date = adjust_dates(days_interval)
    
    # Spécifier l'intervalle de temps pour les données
    interval = "1d"  # Intervalle de 1 jour
    
    # Télécharger les données avec yfinance pour l'intervalle spécifié
    data = yf.download(tickers, start=start_date, end=end_date, interval=interval, prepost=True, multi_level_index=True)
    # Sauvegarder les données sous format CSV
    data.to_csv(FilePathoutput)
    clean_data(FilePathoutput)
    return None

def Xgb_process(FilePathinput):
    df = pd.read_csv(FilePathinput,parse_dates=["Datetime"])
    df.sort_values(by=["Ticker", "Datetime"], ascending=True, inplace=True)
    df_wide = df.pivot(index=["Datetime", "Ticker"], columns="Price", values="Value").reset_index()
    df_wide.columns.name = None  # Enlever le nom de l'index multi-colonne
    df_wide.rename(columns={"Open": "O", "High": "H", "Low": "L", "Close": "C", "Volume": "V"}, inplace=True)  # Renommer pour simplifier
    df_wide=df_wide[df_wide["Ticker"]!="^FCHI"]
    df=df_wide.copy()

    scalers = {}  # Stocker les scalers
    features = ["O", "H", "L", "C", "V"]

    # Normalisation par ticker
    for ticker in df["Ticker"].unique():
        scaler = MinMaxScaler()
        mask = df["Ticker"] == ticker
        df.loc[mask, features] = scaler.fit_transform(df.loc[mask, features])
        scalers[ticker] = scaler  # Stocker le scaler pour usage futur

    df["SMA_10"] = df.groupby("Ticker")["C"].transform(lambda x: ta.trend.sma_indicator(x, window=10))
    df["SMA_50"] = df.groupby("Ticker")["C"].transform(lambda x: ta.trend.sma_indicator(x, window=50))
    df["RSI_14"] = df.groupby("Ticker")["C"].transform(lambda x: ta.momentum.rsi(x, window=14))
    df["MACD"] = df.groupby("Ticker")["C"].transform(lambda x: ta.trend.macd(x))
    df["MACD_Signal"] = df.groupby("Ticker")["C"].transform(lambda x: ta.trend.macd_signal(x))
    df["Return_1h"] = df.groupby("Ticker")["C"].pct_change(1).shift(-1)  # Variation future du prix
    df["Ticker"]=df["Ticker"].astype("category")
    return df

def get_portfolio():
    with open("Data/Tickers/txt/portefeuille.txt", "r", encoding="utf-8") as f:
        contenu = f.read()
        contenu = contenu.splitlines()
    return contenu