from dash import html, dcc, Input, Output, State, callback, callback_context
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify
import pandas as pd
from datetime import datetime
from src.utils.prediction import get_prediction, get_lstm_prediction

# --- Layout principal pour la page de prédiction ---
def prediction_layout():
    # Chargement par défaut (XGBoost)
    try:
        df0, t0 = get_prediction(
            "Data/Tickers/txt/portefeuille.txt",
            "Data/Tickers/csv/portefeuille_xgb_temp.csv"
        )
        cards0 = generate_prediction_cards(df0)
        model_name0 = "XGBoost"
    except Exception:
        df0, t0, cards0, model_name0 = pd.DataFrame(), "N/A", [], "N/A"

    return html.Div([
        # Sélection du modèle
        html.Div(
            className="model-selection glass-card mb-4",
            children=[
                html.H3("Select Model", className="section-title font-jetbrains-mono"),
                dbc.RadioItems(
                    id='model-choice',
                    options=[
                        {'label': 'XGBoost (General)', 'value': 'xgboost'},
                        {'label': 'LSTM (Ticker-Specific)', 'value': 'lstm'}
                    ],
                    value='xgboost',
                    inline=True,
                    className="model-radio"
                ),
                dbc.Button("Run Predictions", id='run-prediction', color="primary", className="mt-3")
            ]
        ),

        # Conteneur des résultats
        html.Div(
            id='prediction-output',
            className="glass-card",
            children=[
                html.H3(f"Predictions ({t0}) via {model_name0}", className="section-title font-jetbrains-mono mb-4"),
                dbc.Row(cards0, className="g-3")
            ]
        ),

        # Prédiction pour un ticker unique
        html.Div(
            className="single-ticker-prediction glass-card mt-5",
            children=[
                html.H3("Single Ticker Prediction", className="section-title font-jetbrains-mono"),
                dbc.InputGroup([
                    dbc.Input(id="input-valeur-pred", placeholder="Enter ticker..."),
                    dbc.Button("Predict", id="btn-predire", color="secondary")
                ], className="mb-3"),
                html.Div(id="resultat-prediction")
            ]
        )
    ])

# --- Génération des cartes de prédiction ---
def generate_prediction_cards(df):
    if df is None or df.empty:
        return [dbc.Col(dbc.Alert("No prediction data available.", color="warning"), width=12)]

    cards = []
    for _, row in df.iterrows():
        ticker = row.get('Ticker', 'N/A')
        pred = row.get('Prediction', 'N/A')
        conf = row.get('Confidence', None)
        price = row.get('Dernier_Prix', None)
        change = row.get('Var_J1_Pct', None)

        # Formatage des valeurs
        pred_text = "Bullish 📈" if pred == 1 else "Bearish 📉"
        pred_color = "text-gain" if pred == 1 else "text-loss"
        conf_text = f"{conf*100:.1f}%" if pd.notnull(conf) else "N/A"
        price_text = f"${price:.2f}" if pd.notnull(price) else "N/A"
        change_text = ("+" if change and change > 0 else "") + f"{change:.2f}%" if pd.notnull(change) else "N/A"
        change_color = "text-gain" if change and change > 0 else "text-loss" if change and change < 0 else ""

        card = dbc.Card(
            className="prediction-card tiltable",
            children=[
                dbc.CardHeader(html.H5(ticker, className="font-jetbrains-mono mb-0")),
                dbc.CardBody([
                    html.P([html.Strong("Confidence: "), html.Span(conf_text)], className="mb-2"),
                    html.P([html.Strong("Prediction: "), html.Span(pred_text, className=pred_color)], className="mb-2"),
                    html.P([html.Strong("Last Price: "), html.Span(price_text, className="font-jetbrains-mono")], className="mb-2"),
                    html.P([html.Strong("Change: "), html.Span(change_text, className=change_color)], className="mb-0")
                ])
            ]
        )
        cards.append(dbc.Col(card, xs=12, sm=6, md=4, lg=3))
    return cards

# --- Callback pour rafraîchir les résultats du portefeuille ---
@callback(
    Output('prediction-output', 'children'),
    Input('run-prediction', 'n_clicks'),
    State('model-choice', 'value'),
    prevent_initial_call=True
)
def update_prediction_display(n_clicks, model_choice):
    path_txt = "Data/Tickers/txt/portefeuille.txt"
    path_xgb = "Data/Tickers/csv/portefeuille_xgb_temp.csv"
    path_lstm = "Data/Tickers/csv/portefeuille_lstm_temp.csv"
    if model_choice == 'xgboost':
        df, date_str = get_prediction(path_txt, path_xgb)
        model_name = "XGBoost"
    else:
        df, date_str = get_lstm_prediction(path_txt, path_lstm, model_base_dir="model_results")
        model_name = "LSTM"

    cards = generate_prediction_cards(df)
    return html.Div([
        html.H3(f"Predictions ({date_str}) via {model_name}", className="section-title font-jetbrains-mono mb-4"),
        dbc.Row(cards, className="g-3")
    ])

# --- Callback pour la prédiction d'un seul ticker ---
@callback(
    Output("resultat-prediction", "children"),
    Input("btn-predire", "n_clicks"),
    State("input-valeur-pred", "value"),
    prevent_initial_call=True
)
def res_prediction(n_clicks, value):
    if not value or not value.strip():
        return dbc.Alert("Please enter a ticker.", color="warning")
    ticker = value.strip().upper()
    txt_file = "Data/Tickers/txt/prediction_temp.txt"
    csv_file = "Data/Tickers/csv/prediction_temp.csv"
    with open(txt_file, "w") as f:
        f.write(ticker + "\n")
    df_pred, time_pred = get_prediction(txt_file, csv_file)
    df_row = df_pred[df_pred['Ticker'] == ticker]
    if df_row.empty:
        return dbc.Alert(f"No result for {ticker}.", color="danger")
    card = generate_prediction_cards(df_row)[0]
    return html.Div([
        html.H5(f"{ticker} ({time_pred})", className="section-title font-jetbrains-mono"),
        card
    ])
