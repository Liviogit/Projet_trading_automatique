import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import ta
import xgboost as xgb
from sklearn.model_selection import train_test_split

df = pd.read_csv("Data/DataForTrain/cac40_clean_format.csv", parse_dates=["Datetime"])
# Trier d'abord par Ticker, puis par Datetime
df.sort_values(by=["Ticker", "Datetime"], inplace=True)

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
df["Target"] = (df["Return_1h"] > 0).astype(int)  # 1 si hausse, 0 si baisse

df.dropna(inplace=True)  # Supprime les NaN

df["Ticker"]=df["Ticker"].astype("category")
X = df[["O", "H", "L", "C", "V", "SMA_10", "SMA_50", "RSI_14", "MACD", "MACD_Signal"]]
y = df["Target"]
print(scalers)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=False)

model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,enable_categorical=True)
model.fit(X_train, y_train,)

model.save_model("Data/model/xgboost.json")