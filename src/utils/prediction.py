import pandas as pd
import xgboost as xgb
from src.utils.data_process import get_data,Xgb_process

def get_model():
    model = xgb.XGBClassifier()
    model.load_model("Data/model/xgboost.json")
    return model

def get_prediction(FilePathinput, FilePathoutput,days_interval=729):
    model= get_model()
    get_data(FilePathinput,FilePathoutput,days_interval)
    data=Xgb_process(FilePathoutput)
    latest_date = data["Datetime"].max()

    # Garder uniquement les lignes de cette date
    df_latest = data[data["Datetime"] == latest_date].reset_index(drop=True)
    y_pred= model.predict(df_latest[["O", "H", "L", "C", "V", "SMA_10", "SMA_50", "RSI_14", "MACD", "MACD_Signal"]])
    df_latest["Prediction"] = y_pred
    df_latest=df_latest[["Ticker","Prediction"]]
    return df_latest