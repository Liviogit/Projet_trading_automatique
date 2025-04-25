# pages/predict.py

from dash import html, dcc, Input, Output, State, callback
# Import prediction functions from the UTILS file
from src.utils.prediction import get_prediction, get_lstm_prediction
import pandas as pd
import dash_bootstrap_components as dbc
import traceback # For detailed error logging
from datetime import datetime

# --- Initial Layout ---
def prediction_layout():
    # Initial display (e.g., XGBoost portfolio prediction by default)
    try:
        # Use a temporary file path for initial load if needed
        initial_df, initial_time = get_prediction(
            "Data/Tickers/txt/portefeuille.txt",
            "Data/Tickers/csv/portefeuille_xgb_temp.csv" # Use temp file
        )
        initial_display = generate_divs_from_df(initial_df)
        model_name = "XGBoost"
    except Exception as e:
        print(f"Error during initial prediction load: {e}")
        initial_display = dbc.Alert("Erreur lors du chargement initial des prédictions.", color="warning")
        initial_time = "N/A"
        model_name = "N/A"

    return html.Div([
        dbc.Row(dbc.Col(html.H3("Prédictions Actions"))),

        # Section for Portfolio Prediction (XGBoost vs LSTM)
        dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(html.Label("Choisir le Modèle pour le Portefeuille:"), width=4),
                    dbc.Col(
                        dcc.Dropdown(
                            id='model-choice',
                            options=[
                                {'label': 'XGBoost (Général)', 'value': 'xgboost'},
                                {'label': 'LSTM (Spécifique par Ticker)', 'value': 'lstm'},
                            ],
                            value='xgboost', # Default value
                            clearable=False
                        ), width=4
                    ),
                    dbc.Col(
                        dbc.Button("Actualiser Prédictions Portefeuille", id='run-prediction', n_clicks=0, color="primary"),
                        width="auto"
                    )
                ], align="center", class_name="mb-3"), # Added margin bottom

                # Output area for portfolio predictions
                html.Div(id='prediction-output', children=[
                     html.P(f"Prédictions initiales ({initial_time}) via {model_name}:", style={"textAlign":"center","fontSize":"16px","fontWeight":"bold"}),
                     html.Div(children=initial_display, style={
                         "padding": "15px",
                         "backgroundColor": "#f8f9fa",
                         "margin": "10px 0",
                         "borderRadius": "8px"
                     })
                 ])
            ]), className="mb-4" # Added margin bottom to the card
        ),

        # Section for Single Ticker Prediction (using XGBoost only in current setup)
        dbc.Card(
            dbc.CardBody([
                 html.H5("Prédiction pour un Ticker Spécifique (via XGBoost)"),
                 dbc.Row([
                     dbc.Col(
                         dcc.Input(
                             id="input-valeur-pred",
                             type="text",
                             placeholder="Entrez un ticker (ex: AAPL)...",
                             style={"padding": "10px", "width": "100%"} # Use 100% width
                         ), width=8
                     ),
                     dbc.Col(
                         dbc.Button("Prédire Ticker", id="btn-predire", n_clicks=0, color="secondary"),
                         width="auto"
                     )
                 ], align="center", class_name="mb-3"), # Added margin bottom

                 html.Div(id="resultat-prediction", style={
                    "marginTop": "20px",
                    "fontWeight": "bold",
                    "color": "#333"
                 }) # Output for single ticker prediction
            ])
        )
    ])


# --- Callback for Single Ticker Prediction (XGBoost) ---
@callback(
    Output("resultat-prediction", "children"),
    Input("btn-predire", "n_clicks"),
    State("input-valeur-pred", "value"),
    prevent_initial_call=True
)
def res_prediction(n, entree):
    if not entree or not entree.strip():
        return dbc.Alert("Veuillez entrer un ticker.", color="warning")

    ticker = entree.strip().upper() # Standardize ticker input
    print(f"--- Predicting single ticker: {ticker} using XGBoost ---")

    # Create temporary files for this single prediction
    single_ticker_txt = "Data/Tickers/txt/prediction_temp.txt"
    single_ticker_csv = "Data/Tickers/csv/prediction_temp.csv"

    try:
        # Write the single ticker to the temporary txt file
        with open(single_ticker_txt, "w", encoding="utf-8") as f:
            f.write(ticker + "\n")

        # Call the XGBoost prediction function
        df_pred, time_pred = get_prediction(single_ticker_txt, single_ticker_csv)

        # Clean up temporary files
        # try:
        #     os.remove(single_ticker_txt)
        #     os.remove(single_ticker_csv)
        # except OSError as e:
        #     print(f"Warning: Could not remove temporary prediction files: {e}")

        if df_pred.empty:
             return dbc.Alert(f"Impossible d'obtenir une prédiction pour {ticker}. Vérifiez le ticker ou les logs.", color="danger")

        # Filter result for the specific ticker (should be only one row)
        result_row = df_pred[df_pred['Ticker'] == ticker]

        if result_row.empty:
              return dbc.Alert(f"Aucun résultat retourné pour {ticker} après prédiction.", color="warning")

        # Generate display for the single result
        single_result_display = generate_divs_from_df(result_row)

        return html.Div([
            html.P(f"Résultat de la prédiction pour {ticker} ({time_pred}) :", style={"fontWeight": "bold"}),
            html.Div(children=single_result_display, style={
                "padding": "15px", # Reduced padding
                "backgroundColor": "#e9ecef", # Slightly different background
                "margin": "10px 0", # Reduced margin
                "borderRadius": "8px"
            })
        ])

    except Exception as e:
         print(f"Error during single ticker prediction for {ticker}: {e}")
         print(traceback.format_exc())
         return dbc.Alert(f"Erreur lors de la prédiction pour {ticker}. Consulter les logs.", color="danger")


# --- Callback for refreshing portfolio predictions (XGBoost or LSTM) ---
@callback(
    Output('prediction-output', 'children'), # Output division in the layout
    Input('run-prediction', 'n_clicks'),    # Button to trigger update
    State('model-choice', 'value'),         # Dropdown value (xgboost or lstm)
    prevent_initial_call=True               # Trigger only on button click
)
def update_prediction_display(n_clicks, model_choice):
    """
    Callback to update the portfolio prediction display based on selected model.
    """
    print(f"--- Button clicked. Updating portfolio prediction with model: {model_choice} ---")

    portfolio_txt_path = "Data/Tickers/txt/portefeuille.txt"
    # Use different temp CSV files to avoid clashes if processing differs
    portfolio_csv_path_xgb = "Data/Tickers/csv/portefeuille_xgb_temp.csv"
    portfolio_csv_path_lstm = "Data/Tickers/csv/portefeuille_lstm_temp.csv"
    model_base_dir_lstm = "model_results" # Path to LSTM models

    if not model_choice:
         return dbc.Alert("Veuillez sélectionner un modèle.", color="warning")

    try:
        start_time = datetime.now()
        if model_choice == 'xgboost':
            print("Calling get_prediction (XGBoost)...")
            df, date_str = get_prediction(portfolio_txt_path, portfolio_csv_path_xgb)
            model_name = "XGBoost"
        elif model_choice == 'lstm':
            print("Calling get_lstm_prediction (LSTM)...")
            df, date_str = get_lstm_prediction(
                FilePathinput=portfolio_txt_path,
                FilePathoutput=portfolio_csv_path_lstm,
                model_base_dir=model_base_dir_lstm
                # Ensure time_steps here matches the default in get_lstm_prediction if needed
            )
            model_name = "LSTM (Per-Ticker)"
        else:
            # This case should not happen with the dropdown setup, but good practice
            return dbc.Alert(f"Modèle '{model_choice}' non reconnu.", color="danger")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"Prediction function ({model_name}) completed in {duration:.2f} seconds.")
        print(f"DataFrame shape: {df.shape if df is not None else 'None'}, Date: {date_str}")

        # --- Result Handling ---
        if df is None or df.empty:
             print(f"Warning: Prediction function returned None or empty DataFrame for {model_name}.")
             return dbc.Alert(f"Aucune prédiction retournée par le modèle {model_name}. Vérifiez les données d'entrée et les logs.", color="info")

        # Generate display using the helper function
        results_display = generate_divs_from_df(df) # generate_divs_from_df handles error strings if present in df

        # Check if results_display is empty (might happen if df was empty or generate_divs failed)
        if not results_display:
             return dbc.Alert(f"Erreur lors de la génération de l'affichage pour {model_name}.", color="warning")


        return html.Div([
            html.P(f"Prédictions du Portefeuille ({date_str}) via {model_name}:", style={"textAlign":"center","fontSize":"16px","fontWeight":"bold"}),
             html.Div(children=results_display, style={
                 "padding": "15px",
                 "backgroundColor": "#f8f9fa",
                 "margin": "10px 0",
                 "borderRadius": "8px"
             })
        ])

    except FileNotFoundError as fnf_err:
         print(f"!!! File not found error during portfolio prediction update: {fnf_err}")
         print(traceback.format_exc())
         return dbc.Alert(f"Erreur de fichier non trouvé lors de la mise à jour. Vérifiez les chemins ({portfolio_txt_path}, CSV outputs, {model_base_dir_lstm}) et les permissions.", color="danger")
    except Exception as e:
         # Log the full error for debugging
         print(f"!!! Error updating portfolio prediction with {model_choice}:")
         print(traceback.format_exc()) # Print full traceback
         return dbc.Alert(f"Une erreur majeure est survenue lors de la mise à jour ({model_choice}): {str(e)}. Consulter les logs du serveur.", color="danger")


# --- Helper Function to Display Results ---
def generate_divs_from_df(df):
    res = {1: "Hausse 📈", 0: "Baisse📉"}
    divs = []
    for _, row in df.iterrows():
        pred = row.get('Prediction', None)
        conf = row.get('Confidence', None)
        dernier_prix = row.get('Dernier_Prix', None)
        var_j1 = row.get('Var_J1_Pct', None)
        if isinstance(pred, (int, float)) and pred in res:
            pred_str = res[pred]
        else:
            pred_str = str(pred)
        # Affichage : pour XGBoost (0 <= conf <= 1) en pourcentage, sinon score brut
        if conf is not None and pd.notnull(conf):
            try:
                conf_val = float(conf)
                if 0 <= conf_val <= 1:
                    conf_str = f" (Confiance: {conf_val*100:.2f}%)"
                else:
                    conf_str = f" (Score: {conf_val:.4f})"
            except Exception:
                conf_str = f" (Score: {conf})"
        else:
            conf_str = ""
        prix_str = f"Dernier Prix : {dernier_prix:.2f}" if dernier_prix is not None and pd.notnull(dernier_prix) else ""
        var_str = f"Var. J-1 : {var_j1:+.2f}%" if var_j1 is not None and pd.notnull(var_j1) else ""
        divs.append(html.Div([
            html.P(f"Tickers : {row['Ticker']}"),
            html.P(f"Prédiction : {pred_str}{conf_str}"),
            html.P(prix_str + (" | " if prix_str and var_str else "") + var_str) if prix_str or var_str else None
        ], style={
            "padding": "10px",
            "marginBottom": "10px",
            "border": "1px solid #ccc",
            "borderRadius": "8px",
            "backgroundColor": "#f9f9f9"
        }))
    return divs