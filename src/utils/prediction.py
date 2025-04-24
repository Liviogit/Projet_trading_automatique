import pandas as pd
import xgboost as xgb
import numpy as np
import tensorflow as tf # <<<--- ADD TF IMPORT
import pickle
import os
from datetime import datetime # <<<--- ADD datetime

# Import necessary functions from data_process
from src.utils.data_process import (
    get_data,
    Xgb_process,
    Lstm_process, # <<<--- ADD Lstm_process
    create_sequences, # <<<--- ADD create_sequences
    get_tickers # <<<--- ADD get_tickers
)

# --- XGBoost Functions ---
def get_model():
    """Loads the pre-trained XGBoost model."""
    try:
        model = xgb.XGBClassifier()
        model_path = "Data/model/xgboost.json" # Ensure this path is correct
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"XGBoost model file not found at {model_path}")
        model.load_model(model_path)
        print("XGBoost model loaded successfully.")
        return model
    except FileNotFoundError as fnf_err:
         print(f"Error loading XGBoost model: {fnf_err}")
         return None
    except Exception as e:
         print(f"An unexpected error occurred loading the XGBoost model: {e}")
         return None

def get_prediction(FilePathinput, FilePathoutput, days_interval=729):
    """
    Generates predictions using the XGBoost model.

    Args:
        FilePathinput (str): Path to the ticker list file (e.g., portefeuille.txt).
        FilePathoutput (str): Path to save temporary downloaded/processed data for XGBoost.
        days_interval (int): Number of past days of data to download.

    Returns:
        tuple: (pd.DataFrame, str)
            - DataFrame with columns ['Ticker', 'Prediction'] (0 or 1).
            - String representing the date of the prediction.
            Returns (empty DataFrame, current date) on failure.
    """
    print("--- Starting XGBoost Prediction ---")
    latest_date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S') # Default date
    empty_result = pd.DataFrame(columns=['Ticker', 'Prediction'])

    # 1. Load Model
    model = get_model()
    if model is None:
        print("Failed to load XGBoost model. Cannot predict.")
        return empty_result, latest_date_str

    # 2. Get and Process Data
    print(f"Fetching/Processing data for XGBoost: Input={FilePathinput}, Output={FilePathoutput}")
    # get_data returns None but modifies FilePathoutput
    get_data_result = get_data(FilePathinput, FilePathoutput, days_interval)
    # We don't strictly need to check get_data_result if subsequent steps handle file errors

    print("Processing data with Xgb_process...")
    data = Xgb_process(FilePathoutput) # Reads FilePathoutput

    if data.empty:
        print("No data available for XGBoost prediction after processing.")
        # Try to get tickers from input file to return empty results for them
        tickers = get_tickers(FilePathinput)
        empty_result = pd.DataFrame({'Ticker': tickers, 'Prediction': 'Data Error'})
        # Try to get latest date from the (potentially empty) processed file
        try:
            temp_df = pd.read_csv(FilePathoutput, parse_dates=['Datetime'])
            if not temp_df.empty and 'Datetime' in temp_df.columns:
                latest_date_str = temp_df['Datetime'].max().strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass # Keep default date on error
        return empty_result, latest_date_str

    # Check for Datetime column
    if 'Datetime' not in data.columns:
        print(f"Error: 'Datetime' column not found after Xgb_process. Columns: {data.columns.tolist()}")
        tickers = get_tickers(FilePathinput)
        empty_result = pd.DataFrame({'Ticker': tickers, 'Prediction': 'Processing Error'})
        return empty_result, latest_date_str

    # 3. Get Latest Data Row(s)
    latest_date = data["Datetime"].max()
    latest_date_str = latest_date.strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(latest_date) else datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Latest data date for XGBoost prediction: {latest_date_str}")

    df_latest = data[data["Datetime"] == latest_date].reset_index(drop=True)

    if df_latest.empty:
         print("No data found for the latest date. Cannot make XGBoost prediction.")
         tickers = get_tickers(FilePathinput)
         empty_result = pd.DataFrame({'Ticker': tickers, 'Prediction': 'No Latest Data'})
         return empty_result, latest_date_str

    # 4. Prepare Features for Prediction
    # Features used during training in model.py
    feature_cols_xgb = ["O", "H", "L", "C", "V", "SMA_10", "SMA_50", "RSI_14", "MACD", "MACD_Signal"]
    # Check if all required features exist
    missing_features = [col for col in feature_cols_xgb if col not in df_latest.columns]
    if missing_features:
        print(f"Error: Missing required features for XGBoost prediction: {missing_features}")
        df_latest['Prediction'] = 'Feature Error'
        return df_latest[['Ticker', 'Prediction']], latest_date_str

    # Handle potential NaNs in the latest row (e.g., indicators might be NaN if history is short)
    X_pred = df_latest[feature_cols_xgb].copy()
    if X_pred.isnull().values.any():
        print("Warning: NaNs detected in features for the latest date. Filling with 0 for prediction.")
        X_pred.fillna(0, inplace=True) # Simple imputation for prediction

    # 5. Predict
    try:
        print(f"Predicting for {len(X_pred)} tickers using XGBoost...")
        y_pred = model.predict(X_pred)
        y_proba = model.predict_proba(X_pred)[:, 1]  # Probability of class 1 (Hausse)
        df_latest["Prediction"] = y_pred
        df_latest["Confidence"] = y_proba
        print("XGBoost prediction complete.")
    except Exception as pred_err:
        print(f"Error during XGBoost prediction: {pred_err}")
        df_latest["Prediction"] = "Predict Error"
        df_latest["Confidence"] = None

    # 6. Format and Return Result
    df_final = df_latest[["Ticker", "Prediction", "Confidence"]].copy()

    print("--- XGBoost Prediction Finished ---")
    return df_final, latest_date_str


# --- >>> NEW LSTM Prediction Function <<< ---
def get_lstm_prediction(FilePathinput, FilePathoutput,
                        days_interval=300, # History needed for indicators + buffer
                        time_steps=10,     # Default from notebook, MUST match training
                        model_base_dir="model_results", # Directory containing ticker subfolders
                        prediction_threshold=0.0): # Threshold to convert predicted return -> binary
    """
    Generates predictions using ticker-specific LSTM models.

    Args:
        FilePathinput (str): Path to the file containing portfolio tickers (e.g., portefeuille.txt).
        FilePathoutput (str): Path to save temporary downloaded/processed data for LSTM.
        days_interval (int): Number of past days of data to download.
        time_steps (int): Number of time steps LSTM models expect (MUST match training).
        model_base_dir (str): Path to the base directory holding model results (e.g., "model_results").
        prediction_threshold (float): Threshold for converting LSTM output (predicted return)
                                       to binary signal (0=Down/Hold, 1=Up). E.g., 0.0 means any positive
                                       predicted return is classified as 'Up'.

    Returns:
        tuple: (pd.DataFrame, str)
            - DataFrame with columns ['Ticker', 'Prediction'] (Prediction is 0, 1, or error string).
            - String representing the date of the prediction.
            Returns (empty DataFrame with error messages, current date) on major failure.
    """
    print("--- Starting LSTM Prediction (Per-Ticker Models) ---")
    latest_date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S') # Default date
    empty_result = pd.DataFrame(columns=['Ticker', 'Prediction'])

    # 0. Get Tickers list first
    portfolio_tickers = get_tickers(FilePathinput)
    if not portfolio_tickers:
        print("No tickers found in input file for LSTM.")
        return empty_result, latest_date_str

    # 1. Fetch Data (Needs enough history for indicators like SMA200 + time_steps)
    # Based on notebook: SMA200, 10d_Return -> need at least 200 + 10 = 210 days
    # Add time_steps (10) and buffer (e.g., 50)
    required_days = 200 + time_steps + 50 # ~260 days minimum, use default 300 or more
    fetch_days = max(days_interval, required_days)
    print(f"Fetching {fetch_days} days of data for LSTM: Input={FilePathinput}, Output={FilePathoutput}")
    get_data_result = get_data(FilePathinput, FilePathoutput, days_interval=fetch_days)
    # We proceed even if get_data had issues, Lstm_process will handle file read errors

    # 2. Process Data (Calculate Indicators Globally using Lstm_process)
    print("Processing data with Lstm_process (calculating indicators)...")
    df_processed = Lstm_process(FilePathoutput) # Reads FilePathoutput

    if df_processed.empty:
        print("Error: No data available for LSTM prediction after Lstm_process.")
        # Return all portfolio tickers with 'Processing Error'
        results_df = pd.DataFrame({'Ticker': portfolio_tickers, 'Prediction': 'Processing Error'})
        # Try get date from raw file
        try:
            temp_df = pd.read_csv(FilePathoutput, parse_dates=['Datetime'])
            if not temp_df.empty and 'Datetime' in temp_df.columns:
                 latest_date_str = temp_df['Datetime'].max().strftime('%Y-%m-%d %H:%M:%S')
        except: pass
        return results_df, latest_date_str

    # Check for Datetime column
    if 'Datetime' not in df_processed.columns:
         print(f"Error: 'Datetime' column missing after Lstm_process. Columns: {df_processed.columns.tolist()}")
         results_df = pd.DataFrame({'Ticker': portfolio_tickers, 'Prediction': 'Process Error (No Date)'})
         return results_df, latest_date_str


    # 3. Get Latest Date
    latest_date = df_processed["Datetime"].max()
    if pd.isnull(latest_date):
         print("Error: Could not determine the latest date from processed data.")
         results_df = pd.DataFrame({'Ticker': portfolio_tickers, 'Prediction': 'Date Error'})
         return results_df, latest_date_str
    latest_date_str = latest_date.strftime('%Y-%m-%d %H:%M:%S')
    print(f"Latest data date for LSTM prediction: {latest_date_str}")

    results = [] # Store prediction results {'Ticker': ticker, 'Prediction': value/error}

    # 4. Loop Through Each Ticker in the Portfolio for Prediction
    for ticker in portfolio_tickers:
        ticker_result = {'Ticker': ticker} # Initialize result for this ticker
        print(f"\nProcessing LSTM prediction for ticker: {ticker}")

        # --- Load Ticker-Specific Artifacts ---
        ticker_model_dir = os.path.join(model_base_dir, ticker)
        if not os.path.isdir(ticker_model_dir):
            print(f"  Error: Model directory not found for {ticker} at {ticker_model_dir}")
            ticker_result['Prediction'] = 'Model Missing'
            results.append(ticker_result)
            continue # Move to the next ticker

        try:
            # Load Features list
            feature_path = os.path.join(ticker_model_dir, 'features.pkl')
            with open(feature_path, 'rb') as f:
                ticker_features = pickle.load(f)
            # print(f"  Loaded features for {ticker}: {ticker_features}")

            # Load Scaler
            scaler_path = os.path.join(ticker_model_dir, 'scaler.pkl')
            with open(scaler_path, 'rb') as f:
                ticker_scaler = pickle.load(f)
            # print(f"  Loaded scaler for {ticker}.")

            # Load Model
            model_path = os.path.join(ticker_model_dir, 'model.h5')
            if not os.path.exists(model_path):
                 raise FileNotFoundError(f"Model file model.h5 not found in {ticker_model_dir}")
            ticker_model = tf.keras.models.load_model(model_path, compile=False) # compile=False speeds up loading if not retraining
            # print(f"  Loaded model for {ticker}.")

            current_time_steps = time_steps # Use the global time_steps passed to the function

        except FileNotFoundError as fnf_err:
            print(f"  Error loading artifact for {ticker}: {fnf_err}")
            ticker_result['Prediction'] = 'Artifact Missing'
            results.append(ticker_result)
            continue
        except Exception as load_err:
            print(f"  Error loading model/scaler/features for {ticker}: {load_err}")
            import traceback
            traceback.print_exc()
            ticker_result['Prediction'] = 'Load Error'
            results.append(ticker_result)
            continue

        # --- Prepare Data for THIS Ticker ---
        # Get all historical data for this ticker from the processed df
        ticker_data_history = df_processed[df_processed['Ticker'] == ticker].copy()
        ticker_data_history = ticker_data_history.sort_values('Datetime')

        if ticker_data_history.empty:
             print(f"  Error: No processed data found for ticker {ticker}.")
             ticker_result['Prediction'] = 'Data Missing'
             results.append(ticker_result)
             continue

        # Check if all features needed by *this* ticker's model exist in the processed dataframe
        missing_data_features = [f for f in ticker_features if f not in ticker_data_history.columns]
        if missing_data_features:
             print(f"  Error: Required features {missing_data_features} are missing in the processed data for {ticker}.")
             ticker_result['Prediction'] = 'Feature Missing'
             results.append(ticker_result)
             continue

        # Select only the required features + Datetime/Ticker for potential later use
        ticker_data_features = ticker_data_history[ticker_features + ['Datetime', 'Ticker']].copy()

        # --- Handle NaNs before scaling ---
        # Drop rows with NaNs *only in the columns required by this specific model*
        # This ensures we have enough history for the sequence after dropping
        initial_rows = len(ticker_data_features)
        ticker_data_features.dropna(subset=ticker_features, inplace=True)
        rows_after_na = len(ticker_data_features)
        if initial_rows > rows_after_na:
             print(f"  Info: Dropped {initial_rows - rows_after_na} rows with NaNs in required features for {ticker}.")

        # Check if enough data remains for the sequence *after* dropping NaNs
        if len(ticker_data_features) < current_time_steps:
             print(f"  Error: Not enough non-NaN data points for {ticker} ({len(ticker_data_features)} found, need {current_time_steps} for sequence).")
             ticker_result['Prediction'] = 'Insufficient Data'
             results.append(ticker_result)
             continue

        # --- Scale the required features using the ticker's specific scaler ---
        try:
            # Ensure features are numeric before scaling
            numeric_features_to_scale = ticker_data_history[ticker_features].select_dtypes(include=np.number).columns.tolist()
            non_numeric = [f for f in ticker_features if f not in numeric_features_to_scale]
            if non_numeric:
                 print(f"  Warning: Non-numeric features {non_numeric} requested by {ticker}'s features.pkl. Skipping them for scaling.")

            if not numeric_features_to_scale:
                 raise ValueError("No numeric features available to scale for this ticker.")

            # Scale only the numeric features required by this ticker
            # Use .values to avoid sklearn warning about feature names
            ticker_data_history[numeric_features_to_scale] = ticker_scaler.transform(ticker_data_history[numeric_features_to_scale].values)
            # print("  Scaling complete.")

        except ValueError as ve:
             print(f"  Error scaling data for {ticker}: {ve}")
             print(f"  Scaler expected {ticker_scaler.n_features_in_} features. Trying to scale {len(numeric_features_to_scale)} features: {numeric_features_to_scale}")
             ticker_result['Prediction'] = 'Scaling Error'
             results.append(ticker_result)
             continue
        except Exception as scale_err:
            print(f"  Unexpected error scaling data for {ticker}: {scale_err}")
            ticker_result['Prediction'] = 'Scaling Error'
            results.append(ticker_result)
            continue


        # --- Create Sequence ---
        # Pass the dataframe containing the scaled features
        sequence = create_sequences(ticker_data_features, ticker_features, current_time_steps)

        if sequence is None:
            print(f"  Error: Failed to create prediction sequence for {ticker}.")
            # Reason should have been printed by create_sequences or previous checks
            if ticker_result.get('Prediction') is None: # Avoid overwriting a previous error
                 ticker_result['Prediction'] = 'Sequence Error'
            results.append(ticker_result)
            continue
        # Expected sequence shape: (time_steps, n_features)

        # --- Predict ---
        # Model expects input shape like (batch_size, time_steps, n_features)
        # We predict one ticker at a time, so batch_size is 1
        sequence_batch = np.expand_dims(sequence, axis=0) # Shape becomes (1, time_steps, n_features)
        # Verify the final input shape matches model expectation if possible
        # print(f"  Input sequence shape for prediction: {sequence_batch.shape}")

        try:
            raw_prediction = ticker_model.predict(sequence_batch, verbose=0)
            prediction_value = raw_prediction[0][0]
            binary_prediction = 1 if prediction_value > prediction_threshold else 0
            ticker_result['Prediction'] = binary_prediction
            ticker_result['Confidence'] = float(prediction_value)
            print(f"  Predicted class for {ticker}: {binary_prediction} (Threshold: {prediction_threshold}, Confidence: {prediction_value})")
        except Exception as pred_err:
            print(f"  Error during LSTM prediction for {ticker}: {pred_err}")
            import traceback
            traceback.print_exc()
            ticker_result['Prediction'] = 'Predict Error'
            ticker_result['Confidence'] = None

        results.append(ticker_result)
        # --- End of Ticker Loop ---

    # 5. Format Final DataFrame
    if not results:
         print("Warning: No results generated for any ticker.")
         df_final = pd.DataFrame({'Ticker': portfolio_tickers, 'Prediction': 'Unknown Error', 'Confidence': None})
    else:
         df_final = pd.DataFrame(results)

    # Ensure all original portfolio tickers are in the final df, even if processing failed early
    missing_tickers = set(portfolio_tickers) - set(df_final['Ticker'])
    if missing_tickers:
         print(f"Adding missing tickers to final result: {missing_tickers}")
         missing_rows = pd.DataFrame([{'Ticker': t, 'Prediction': 'Not Processed', 'Confidence': None} for t in missing_tickers])
         df_final = pd.concat([df_final, missing_rows], ignore_index=True)

    # Sort by ticker for consistency
    df_final = df_final.sort_values('Ticker').reset_index(drop=True)

    print("--- LSTM Prediction (Per-Ticker) Finished ---")
    return df_final, latest_date_str

