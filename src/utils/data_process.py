# src/utils/data_process.py

import yfinance as yf
import pandas as pd
import numpy as np # <<<--- ADD THIS IMPORT AT THE TOP
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler # <<<--- ADD StandardScaler
import ta
import pickle
import os # <<<--- ADD THIS IMPORT

# --- adjust_dates, get_tickers, clean_data, get_data --- (Keep these as they were, ensure clean_data works as expected)
def adjust_dates(days_interval=60):
    # La date d'aujourd'hui
    end_date = datetime.today()
    # Calcul de la start date en soustrayant le nombre de jours à la end date
    start_date = end_date - timedelta(days=days_interval)
    # Retourne les deux dates au format 'YYYY-MM-DD'
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def get_tickers(FilePath):
    # Charger la liste des tickers depuis le fichier
    try:
        with open(FilePath, "r", encoding="utf-8") as f: # Added encoding
            tickers = [line.strip() for line in f.readlines()]
            # Remove empty lines
            tickers = [ticker for ticker in tickers if ticker.strip()]
        return tickers
    except FileNotFoundError:
        print(f"Error: Ticker file not found at {FilePath}")
        return []
    except Exception as e:
        print(f"Error reading ticker file {FilePath}: {e}")
        return []

def clean_data(FilePath):
    # --- Use the robust cleaning logic that pivots data correctly ---
    # (Ensure this function correctly pivots the data into OHLCV columns per Ticker/Datetime)
    try:
        # Check if file exists and is not empty
        if not os.path.exists(FilePath) or os.path.getsize(FilePath) == 0:
            print(f"Warning: File {FilePath} is empty or missing for cleaning.")
            pd.DataFrame().to_csv(FilePath, index=False) # Create empty file
            return

        # Load with multi-index header
        df = pd.read_csv(FilePath, header=[0, 1], index_col=0)

        # Check if DataFrame is empty after loading
        if df.empty:
             print(f"Warning: DataFrame loaded from {FilePath} is empty.")
             pd.DataFrame().to_csv(FilePath, index=False) # Save empty file
             return

        df.index = pd.to_datetime(df.index) # Ensure index is datetime

        # Flatten columns: ('MSFT', 'Close') -> 'MSFT_Close'
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

        # Melt the DataFrame
        df_long = df.reset_index().melt(id_vars='Datetime', var_name='Ticker_Price', value_name='Value')

        # Split 'Ticker_Price' into 'Ticker' and 'Price'
        # Handle potential errors if split doesn't yield 2 parts
        try:
            split_cols = df_long['Ticker_Price'].str.split('_', expand=True)
            if split_cols.shape[1] == 2:
                df_long[['Ticker', 'Price']] = split_cols
            else:
                # Handle cases like index names or unexpected column formats
                print(f"Warning: Unexpected column format in {FilePath}. Attempting alternate split.")
                # Fallback or error handling logic here if needed
                # For now, we'll drop rows that didn't split correctly
                df_long = df_long[split_cols.shape[1] == 2]
                if df_long.empty:
                     print("Error: Could not extract Ticker and Price after alternate split.")
                     pd.DataFrame().to_csv(FilePath, index=False)
                     return
                df_long[['Ticker', 'Price']] = df_long['Ticker_Price'].str.split('_', expand=True)

        except Exception as split_err:
             print(f"Error splitting Ticker_Price column in {FilePath}: {split_err}")
             pd.DataFrame().to_csv(FilePath, index=False) # Save empty on error
             return

        df_long.drop('Ticker_Price', axis=1, inplace=True)

        # Pivot to get prices as columns
        # Use aggfunc='first' or 'last' if duplicates exist for Datetime/Ticker/Price combination
        df_final = df_long.pivot_table(index=['Datetime', 'Ticker'], columns='Price', values='Value', aggfunc='first').reset_index()


        # Select and rename standard columns if necessary (yfinance names usually okay)
        standard_cols = ['Datetime', 'Ticker', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
        # Check which columns actually exist after pivot
        existing_cols = [col for col in standard_cols if col in df_final.columns]
        df_final = df_final[existing_cols]

        # Handle potential 'Adj Close' if 'Close' is missing, or vice-versa
        if 'Adj Close' in df_final.columns and 'Close' not in df_final.columns:
             df_final.rename(columns={'Adj Close': 'Close'}, inplace=True)
             # Recalculate existing_cols if 'Close' was added
             existing_cols = [col for col in standard_cols if col in df_final.columns]
             df_final = df_final[existing_cols]
        elif 'Close' in df_final.columns and 'Adj Close' in df_final.columns:
             # Prefer 'Close' but keep 'Adj Close' if needed by models
             pass # Keep both for now, ensure models use the correct one


        # Ensure essential OHLCV columns are present
        essential_cols = ['Datetime', 'Ticker', 'Close', 'High', 'Low', 'Open', 'Volume']
        missing_essentials = [col for col in essential_cols if col not in df_final.columns]
        if missing_essentials:
             print(f"Warning: Essential columns missing after pivoting in {FilePath}: {missing_essentials}")
             # Optionally, try to fill basic ones if possible (e.g., Open=High=Low=Close if only Close exists)
             # For now, we proceed but models might fail

        # Sort and save cleaned data
        df_final = df_final.sort_values(['Ticker', 'Datetime'])
        df_final.to_csv(FilePath, index=False)
        print(f"Cleaned data saved to {FilePath} with columns: {df_final.columns.tolist()}")

    except FileNotFoundError:
         print(f"Error: File {FilePath} not found during cleaning.")
         pd.DataFrame().to_csv(FilePath, index=False) # Create empty file
    except Exception as e:
        print(f"Error cleaning data in {FilePath}: {e}")
        import traceback
        traceback.print_exc()
        # Handle case where file might be corrupted or empty, save empty df
        pd.DataFrame().to_csv(FilePath, index=False)


def get_data(FilePathinput, FilePathoutput, days_interval=60):
    # Obtenir les tickers
    tickers = get_tickers(FilePathinput)
    if not tickers:
        print("No tickers found. Cannot download data.")
        # Ensure an empty file exists at the output path
        try:
             pd.DataFrame().to_csv(FilePathoutput, index=False) # Save empty file
        except Exception as save_err:
             print(f"Error creating empty output file {FilePathoutput}: {save_err}")
        return None # Indicate failure

    # Adjuster les dates
    start_date, end_date = adjust_dates(days_interval)

    # Spécifier l'intervalle de temps pour les données
    interval = "1d" # Daily data is standard for these indicators/models

    try:
        print(f"Downloading data for: {tickers}")
        print(f"Start: {start_date}, End: {end_date}, Interval: {interval}")

        # Download data
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            interval=interval,
            group_by='ticker', # Groups by ticker, results in MultiIndex columns if >1 ticker
            auto_adjust=False, # Keep Open, High, Low, Close, Adj Close, Volume separate
            progress=False # Suppress progress bar in logs
        )

        if data.empty:
            print("Warning: yfinance download returned empty DataFrame.")
            data.to_csv(FilePathoutput) # Save empty dataframe
            return None

        # Ensure Datetime is the index before saving
        if not isinstance(data.index, pd.DatetimeIndex):
             data.index = pd.to_datetime(data.index)

        # Rename index to 'Datetime' if it's unnamed ('Date')
        if data.index.name != 'Datetime':
             data.index.name = 'Datetime'

        # Handle potential single ticker download (no multi-index columns)
        if len(tickers) == 1 and isinstance(data.columns, pd.Index) and not isinstance(data.columns, pd.MultiIndex):
             ticker = tickers[0]
             # Create MultiIndex
             data.columns = pd.MultiIndex.from_product([[ticker], data.columns])
             # Make sure column names are standard ('Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume')
             # yf.download usually returns standard names when auto_adjust=False

        # Ensure columns are a MultiIndex for multi-ticker downloads
        elif len(tickers) > 1 and not isinstance(data.columns, pd.MultiIndex):
             print(f"Warning: Expected MultiIndex columns for multiple tickers, but got flat index. Check yfinance output.")
             # Attempt to reconstruct if possible, otherwise error
             # This case is less common with group_by='ticker'
             pd.DataFrame().to_csv(FilePathoutput, index=False)
             return None


        # If columns are MultiIndex, ensure 'Ticker' is the top level
        if isinstance(data.columns, pd.MultiIndex):
             if data.columns.names[0] != 'Ticker':
                 # Find which level has the ticker names
                 ticker_level = -1
                 for i, level_name in enumerate(data.columns.names):
                     # Heuristic: Check if level values look like the tickers
                     if level_name and all(item in tickers for item in data.columns.get_level_values(i).unique()):
                         ticker_level = i
                         break
                     elif level_name is None and all(item in tickers for item in data.columns.get_level_values(i).unique()):
                         # Sometimes the ticker level might be unnamed
                         ticker_level = i
                         break

                 if ticker_level != -1 and ticker_level != 0:
                     print(f"Swapping column levels 0 and {ticker_level}")
                     data = data.swaplevel(0, ticker_level, axis=1)
                 elif ticker_level == -1:
                     print(f"Error: Could not identify the ticker level in MultiIndex columns: {data.columns.names}")
                     pd.DataFrame().to_csv(FilePathoutput, index=False)
                     return None

                 # Ensure level names are set correctly after swap
                 if data.columns.names[0] is None or data.columns.names[0] != 'Ticker':
                      current_names = list(data.columns.names)
                      current_names[0] = 'Ticker'
                      # Assign a default name like 'Price' to the second level if it's None
                      if len(current_names) > 1 and current_names[1] is None:
                           current_names[1] = 'Price'
                      data.columns.names = current_names


             # Sort columns by Ticker then Price for consistency
             data = data.sort_index(axis=1, level=[0, 1])


        # --- Save raw data (multi-index columns) ---
        try:
             data.to_csv(FilePathoutput)
             print(f"Raw data saved to {FilePathoutput}")
        except Exception as save_err:
             print(f"Error saving raw data to {FilePathoutput}: {save_err}")
             # Attempt to save empty df as fallback
             pd.DataFrame().to_csv(FilePathoutput, index=False)
             return None # Indicate failure

        # --- Clean the downloaded data using the function above ---
        clean_data(FilePathoutput) # This pivots the data to long format

    except Exception as e:
        print(f"Error in get_data: {e}")
        import traceback
        traceback.print_exc() # Print full error stack
        # Consider saving an empty file or handling the error upstream
        try:
            pd.DataFrame().to_csv(FilePathoutput, index=False)
        except Exception as final_save_err:
            print(f"Error saving empty file to {FilePathoutput} after failure: {final_save_err}")
        return None # Indicate failure

    # If successful, return None as per original structure (function modifies file in place)
    return None


# --- Xgb_process --- (Keep as it was - it uses different indicators/scaling)
def Xgb_process(FilePathinput):
    try:
        df = pd.read_csv(FilePathinput, parse_dates=["Datetime"])
    except FileNotFoundError:
        print(f"Error: Input file not found at {FilePathinput} for Xgb_process")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading file {FilePathinput} for Xgb_process: {e}")
        return pd.DataFrame()

    if df.empty:
        print(f"Warning: Input file {FilePathinput} is empty for Xgb_process.")
        return pd.DataFrame()

    # Ensure essential columns exist after reading the potentially cleaned file
    essential_cols = ['Datetime', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in essential_cols):
        missing = [col for col in essential_cols if col not in df.columns]
        print(f"Error: Missing essential columns {missing} in {FilePathinput} for Xgb_process after reading.")
        # Attempt to read the raw file again if cleaning failed? Or just return empty.
        return pd.DataFrame()

    df.sort_values(by=["Ticker", "Datetime"], ascending=True, inplace=True)

    # Check for '^FCHI' ticker presence before filtering
    if 'Ticker' in df.columns and '^FCHI' in df['Ticker'].unique():
        df = df[df["Ticker"] != "^FCHI"]
    elif 'Ticker' not in df.columns:
         print("Warning: 'Ticker' column not found for filtering '^FCHI' in Xgb_process")

    # Rename columns for XGBoost features
    df_wide = df.rename(columns={
        "Open": "O", "High": "H", "Low": "L", "Close": "C", "Volume": "V"
    }, errors='ignore') # Use ignore in case columns already renamed or missing

    # Select only the columns needed + identifiers
    xgb_cols = ["Datetime", "Ticker", "O", "H", "L", "C", "V"]
    cols_to_use = [col for col in xgb_cols if col in df_wide.columns]
    df_wide = df_wide[cols_to_use]

    # Drop rows where essential numeric features are missing
    numeric_essentials_xgb = ["O", "H", "L", "C", "V"]
    df_wide.dropna(subset=numeric_essentials_xgb, inplace=True)

    if df_wide.empty:
         print("DataFrame is empty after processing/filtering in Xgb_process.")
         return df_wide

    # Use a copy to avoid SettingWithCopyWarning
    df = df_wide.copy()

    scalers = {}
    features = ["O", "H", "L", "C", "V"] # Features to scale for XGB

    # Normalisation par ticker
    for ticker in df["Ticker"].unique():
        scaler = MinMaxScaler()
        mask = df["Ticker"] == ticker
        if mask.any():
             # Ensure features exist and are numeric before scaling
             valid_features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
             if valid_features:
                 # Check for NaNs or Infs before scaling
                 if df.loc[mask, valid_features].isnull().values.any() or np.isinf(df.loc[mask, valid_features].values).any():
                      print(f"Warning: NaNs or Infs found in features for ticker {ticker} before XGB scaling. Attempting to fill.")
                      # Simple fill - consider more sophisticated methods if needed
                      df.loc[mask, valid_features] = df.loc[mask, valid_features].fillna(method='ffill').fillna(method='bfill').fillna(0)
                      if df.loc[mask, valid_features].isnull().values.any(): # Check again after filling
                          print(f"Error: Could not fill NaNs for {ticker}. Skipping scaling.")
                          continue # Skip this ticker for scaling

                 df.loc[mask, valid_features] = scaler.fit_transform(df.loc[mask, valid_features])
                 scalers[ticker] = scaler
             else:
                  print(f"Warning: No numeric features {features} to scale for ticker {ticker} in Xgb_process")
        else:
             print(f"Warning: No data found for ticker {ticker} during XGB scaling.")

    # --- Add TA indicators safely using ta library ---
    # Ensure the 'C' (Close) column exists
    if 'C' in df.columns:
         print("Calculating XGBoost TA indicators...")
         # Use min_periods in rolling operations for robustness if needed
         # SMA
         df["SMA_10"] = df.groupby("Ticker")["C"].transform(lambda x: ta.trend.sma_indicator(x, window=10, fillna=True)) # fillna=True handles initial NaNs
         df["SMA_50"] = df.groupby("Ticker")["C"].transform(lambda x: ta.trend.sma_indicator(x, window=50, fillna=True))
         # RSI
         df["RSI_14"] = df.groupby("Ticker")["C"].transform(lambda x: ta.momentum.rsi(x, window=14, fillna=True))
         # MACD
         # ta library's MACD returns MACD line, macd_signal returns signal line, macd_diff returns histogram
         macd = df.groupby("Ticker")["C"].transform(lambda x: ta.trend.macd(x, window_slow=26, window_fast=12, fillna=True))
         macd_signal = df.groupby("Ticker")["C"].transform(lambda x: ta.trend.macd_signal(x, window_slow=26, window_fast=12, window_sign=9, fillna=True))
         df["MACD"] = macd # MACD Line
         df["MACD_Signal"] = macd_signal # Signal Line
         # df["MACD_Hist"] = df.groupby("Ticker")["C"].transform(lambda x: ta.trend.macd_diff(x, window_slow=26, window_fast=12, window_sign=9, fillna=True)) # Histogram if needed

         # Target calculation (optional here, done in training script)
         # df["Return_1h"] = df.groupby("Ticker")["C"].pct_change(1).shift(-1) # Shift(-1) looks into the future
         # df["Target"] = (df["Return_1h"] > 0).astype(int)
         print("XGBoost TA indicators calculated.")
    else:
         print("Warning: 'Close' column ('C') not found, cannot calculate TA indicators for XGBoost.")

    # Convert Ticker to category for potential model optimization (like LightGBM/XGBoost cat feature support)
    if 'Ticker' in df.columns:
        df["Ticker"] = df["Ticker"].astype("category")

    # Final check for NaNs introduced by indicators (especially at the start)
    # Option 1: Drop rows with any NaNs in feature columns
    # feature_cols_xgb = ["O", "H", "L", "C", "V", "SMA_10", "SMA_50", "RSI_14", "MACD", "MACD_Signal"]
    # df.dropna(subset=[col for col in feature_cols_xgb if col in df.columns], inplace=True)
    # Option 2: Keep rows, prediction function needs to handle latest row potentially having NaNs (e.g., use fillna(0) or predict only if no NaNs)
    # For prediction, we typically only need the *latest* row, so we handle NaNs there if needed.

    print(f"Xgb_process returning DataFrame with shape: {df.shape} and columns: {df.columns.tolist()}")
    return df


# --- get_portfolio --- (Keep as it was)
def get_portfolio():
    try:
        with open("Data/Tickers/txt/portefeuille.txt", "r", encoding="utf-8") as f:
            contenu = f.read()
            contenu = [line.strip() for line in contenu.splitlines() if line.strip()] # Read lines and remove empty ones
        return contenu
    except FileNotFoundError:
        print("Error: portefeuille.txt not found.")
        return []
    except Exception as e:
         print(f"Error reading portfolio file: {e}")
         return []


# --- >>> DEFINE add_lstm_technical_indicators BEFORE Lstm_process <<< ---
# Based on deepclaude2.ipynb's add_technical_indicators and prepare_ml_data
def add_lstm_technical_indicators(df_group):
    """Calculates technical indicators used in the LSTM model for a single ticker group."""
    # Make sure we have enough data (adjust minimum as needed)
    if len(df_group) < 30: # Original notebook check
        print(f"Warning: Insufficient data ({len(df_group)} rows) for LSTM indicator calculation in group.")
        # Add columns as NaN if they don't exist
        indicator_cols = [
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SMA_20', 'SMA_50', 'SMA_200',
            'ATR', 'Upper_Band', 'Middle_Band', 'Lower_Band', 'OBV',
            'Daily_Return', '5d_Return', '10d_Return',
            'Close_to_SMA20', 'Close_to_SMA50', 'SMA20_to_SMA50'
        ]
        for col in indicator_cols:
            if col not in df_group.columns:
                df_group[col] = np.nan
        return df_group # Return with NaNs

    # Ensure required base columns exist
    required_base_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    if not all(col in df_group.columns for col in required_base_cols):
        missing_bases = [c for c in required_base_cols if c not in df_group.columns]
        print(f"Warning: Missing base columns {missing_bases} for indicator calculation. Some indicators will be NaN.")
        # Create missing base columns as NaN to avoid errors below, though results will be poor
        for col in missing_bases:
             df_group[col] = np.nan

    # Convert columns to numeric, coercing errors
    for col in required_base_cols:
         df_group[col] = pd.to_numeric(df_group[col], errors='coerce')

    # --- Calculate Indicators (using logic similar to the notebook) ---

    # Basic momentum indicators - RSI (Manual calculation like notebook)
    delta = df_group['Close'].diff()
    gain = delta.where(delta > 0, 0).fillna(0) # Fill initial NaN for gain/loss
    loss = -delta.where(delta < 0, 0).fillna(0)
    # Use Simple Moving Average (SMA) for avg gain/loss as standard
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan) # Replace 0 with NaN, then fillna later
    df_group['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    # Fill NaNs resulting from 0 avg_loss or initial periods
    df_group['RSI'] = df_group['RSI'].fillna(50) # Fill NaNs with 50 (neutral)

    # MACD (using ewm like notebook)
    exp1 = df_group['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
    exp2 = df_group['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
    df_group['MACD'] = exp1 - exp2
    df_group['MACD_Signal'] = df_group['MACD'].ewm(span=9, adjust=False, min_periods=9).mean()
    df_group['MACD_Hist'] = df_group['MACD'] - df_group['MACD_Signal']

    # Moving averages (SMA)
    df_group['SMA_20'] = df_group['Close'].rolling(window=20, min_periods=1).mean()
    df_group['SMA_50'] = df_group['Close'].rolling(window=50, min_periods=1).mean()
    df_group['SMA_200'] = df_group['Close'].rolling(window=200, min_periods=1).mean()

    # Volatility indicators - ATR (Manual like notebook)
    high_low = df_group['High'] - df_group['Low']
    high_close = abs(df_group['High'] - df_group['Close'].shift()).fillna(0) # Fill first NaN
    low_close = abs(df_group['Low'] - df_group['Close'].shift()).fillna(0) # Fill first NaN
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    # Use Rolling Mean for ATR as in notebook (EMA is also common)
    df_group['ATR'] = true_range.rolling(window=14, min_periods=1).mean()

    # Bollinger Bands
    df_group['Middle_Band'] = df_group['SMA_20'] # Reuse SMA_20
    std_dev = df_group['Close'].rolling(window=20, min_periods=20).std() # Use min_periods=20 for std dev
    df_group['Upper_Band'] = df_group['Middle_Band'] + (std_dev * 2)
    df_group['Lower_Band'] = df_group['Middle_Band'] - (std_dev * 2)

    # Volume indicators - OBV (On-Balance Volume)
    # Ensure volume is numeric
    volume_numeric = pd.to_numeric(df_group['Volume'], errors='coerce').fillna(0)
    df_group['OBV'] = (np.sign(df_group['Close'].diff()) * volume_numeric).fillna(0).cumsum()

    # Returns
    df_group['Daily_Return'] = df_group['Close'].pct_change(periods=1).fillna(0) # Fill first NaN
    df_group['5d_Return'] = df_group['Close'].pct_change(periods=5).fillna(0)
    df_group['10d_Return'] = df_group['Close'].pct_change(periods=10).fillna(0)

    # Price-based features (relative to SMAs - handle division by zero)
    df_group['Close_to_SMA20'] = (df_group['Close'] / df_group['SMA_20'].replace(0, np.nan) - 1).fillna(0)
    df_group['Close_to_SMA50'] = (df_group['Close'] / df_group['SMA_50'].replace(0, np.nan) - 1).fillna(0)
    df_group['SMA20_to_SMA50'] = (df_group['SMA_20'] / df_group['SMA_50'].replace(0, np.nan) - 1).fillna(0)

    # Target variable (optional here, needed for training but not prediction)
    # df_group['Target'] = df_group['Close'].pct_change(periods=5).shift(-5)

    # Drop initial rows with NaNs created by longer lookback periods (like SMA200, 10d_Return)
    # This is crucial for LSTM sequence creation later. Find the first valid index.
    # first_valid_idx = df_group.dropna().index.min()
    # if pd.notnull(first_valid_idx):
    #     df_group = df_group.loc[first_valid_idx:]
    # Instead of dropping here, we'll handle NaNs just before scaling/sequence creation in get_lstm_prediction

    return df_group


# --- Lstm_process (Calls the function defined above) ---
def Lstm_process(FilePathInput):
    """
    Processes data by adding LSTM-specific technical indicators per ticker.
    Does NOT perform scaling.
    """
    try:
        df = pd.read_csv(FilePathInput, parse_dates=["Datetime"])
    except FileNotFoundError:
        print(f"Error: Input file not found at {FilePathInput} for Lstm_process")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading file {FilePathInput} for Lstm_process: {e}")
        return pd.DataFrame()

    if df.empty:
        print(f"Warning: Input file {FilePathInput} is empty for Lstm_process.")
        return pd.DataFrame()

    # Standardize column names expected by add_lstm_technical_indicators
    # Ensure case-insensitivity if needed, but yfinance is usually consistent
    column_map = {
        "Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume",
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"
        # Add 'Adj Close' if potentially used, map it to 'Close' if 'Close' is missing
    }
    df = df.rename(columns=column_map, errors='ignore')

    # If 'Close' is missing but 'Adj Close' exists, use 'Adj Close' as 'Close'
    if 'Adj Close' in df.columns and 'Close' not in df.columns:
         print("Using 'Adj Close' as 'Close' for LSTM processing.")
         df.rename(columns={'Adj Close': 'Close'}, inplace=True)


    required_cols = ["Datetime", "Ticker", "Open", "High", "Low", "Close", "Volume"]
    missing_essentials = [col for col in required_cols if col not in df.columns]
    if missing_essentials:
         print(f"Warning: Lstm_process missing essential columns {missing_essentials} in {FilePathInput}. Indicators may fail or be inaccurate.")
         # Attempt to proceed, but expect issues

    df.sort_values(by=["Ticker", "Datetime"], ascending=True, inplace=True)

    # Apply technical indicator calculations per group
    print("Calculating LSTM technical indicators per ticker...")
    # Use group_keys=False to avoid adding the group key as an index level
    # Handle potential errors during apply
    all_processed_dfs = []
    error_tickers = []
    for ticker, group in df.groupby('Ticker'):
        print(f"  Processing indicators for: {ticker}")
        try:
            # Pass a copy to avoid modifying the original group within the loop
            processed_group = add_lstm_technical_indicators(group.copy())
            all_processed_dfs.append(processed_group)
        except Exception as apply_err:
            print(f"  Error calculating indicators for {ticker}: {apply_err}")
            # Option: append the original group without indicators, or skip it
            # Appending original allows prediction attempt if base features exist
            # group['Error'] = str(apply_err) # Mark with error
            # all_processed_dfs.append(group)
            error_tickers.append(ticker) # Track tickers that failed


    if not all_processed_dfs:
         print("Error: No tickers processed successfully for LSTM indicators.")
         return pd.DataFrame()

    # Concatenate results
    df_enhanced = pd.concat(all_processed_dfs, ignore_index=True)

    print(f"Indicator calculation finished. {len(error_tickers)} tickers had errors.")

    # --- Define the *superset* of features potentially needed by *any* LSTM model ---
    # Based on features used in notebook's prepare_ml_data + base OHLCV
    potential_lstm_features = [
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'SMA_20', 'SMA_50', 'SMA_200', 'ATR',
        'Upper_Band', 'Middle_Band', 'Lower_Band', 'OBV',
        'Daily_Return', '5d_Return', '10d_Return',
        'Open', 'High', 'Low', 'Close', 'Volume', # Base features
        'Close_to_SMA20', 'Close_to_SMA50', 'SMA20_to_SMA50' # Price-based features
    ]

    # Add missing potential feature columns as NaN if they weren't generated
    for feature in potential_lstm_features:
        if feature not in df_enhanced.columns:
            df_enhanced[feature] = np.nan

    print(f"Lstm_process returning DataFrame with shape: {df_enhanced.shape}")
    print(f"Columns: {df_enhanced.columns.tolist()}")
    return df_enhanced


# --- create_sequences (Modified for robustness and single ticker) ---
def create_sequences(ticker_scaled_data, features, time_steps=10):
    """
    Extracts the last 'time_steps' sequence from already scaled data for a SINGLE ticker.

    Args:
        ticker_scaled_data (pd.DataFrame): DataFrame containing SCALED features for ONE ticker,
                                           sorted by time, WITH NaNs potentially handled.
        features (list): List of feature column names to include in the sequence.
        time_steps (int): The length of the sequence.

    Returns:
        np.array or None: A numpy array of shape (time_steps, n_features) or None if not enough data or features missing.
    """
    if ticker_scaled_data is None or ticker_scaled_data.empty:
         print("Error: Input data for sequence creation is None or empty.")
         return None

    # Ensure Ticker column exists for context in logs if available
    ticker_name = ticker_scaled_data['Ticker'].iloc[0] if 'Ticker' in ticker_scaled_data.columns and not ticker_scaled_data.empty else "Unknown Ticker"

    if len(ticker_scaled_data) < time_steps:
        print(f"Warning for {ticker_name}: Not enough data for sequence ({len(ticker_scaled_data)} rows, need {time_steps}).")
        return None

    # Check if all requested features exist in the scaled data
    missing_features = [f for f in features if f not in ticker_scaled_data.columns]
    if missing_features:
         print(f"Error for {ticker_name}: Features missing in scaled data for sequence creation: {missing_features}")
         return None

    # --- Select the required features and the last 'time_steps' rows ---
    sequence_data_df = ticker_scaled_data[features].iloc[-time_steps:]

    # --- CRITICAL: Check for NaNs or Infs within the final sequence window ---
    if sequence_data_df.isnull().values.any() or np.isinf(sequence_data_df.values).any():
        print(f"Warning for {ticker_name}: NaNs or Infs found within the final {time_steps} steps needed for sequence.")
        # Attempt imputation (forward fill, then backward fill, then fill with 0)
        # This should ideally not happen if NaNs were handled *before* scaling/sequence creation
        sequence_data_filled = sequence_data_df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Check again if imputation failed (e.g., all NaNs in a column)
        if sequence_data_filled.isnull().values.any() or np.isinf(sequence_data_filled.values).any():
            print(f"  Error for {ticker_name}: Could not impute all NaNs/Infs in the sequence window. Cannot create sequence.")
            return None
        else:
            print(f"  Info for {ticker_name}: NaNs/Infs imputed in sequence window.")
            sequence_data = sequence_data_filled.values # Use imputed data
    else:
        sequence_data = sequence_data_df.values # No NaNs/Infs found

    # Final shape check
    if sequence_data.shape != (time_steps, len(features)):
        print(f"Error for {ticker_name}: Final sequence shape mismatch. Expected {(time_steps, len(features))}, got {sequence_data.shape}")
        return None

    return sequence_data # Return as numpy array