from dash import html, dcc, callback, Output, Input, State, ctx, ALL
import plotly.express as px
import pandas as pd
from src.utils.data_process import get_tickers

# Layout principal pour la page Portefeuille
def get_portfolio():
    try:
        with open("Data/Tickers/txt/portefeuille.txt", "r", encoding="utf-8") as f:
            tickers = [line.strip() for line in f if line.strip()]
        return tickers
    except FileNotFoundError:
        print("⚠️ portefeuille.txt not found.")
        return []
    except Exception as e:
        print(f"Error reading portefeuille.txt: {e}")
        return []
    
    
def portefeuille_layout(portfolio):
    return html.Div(
        className="glass-card",
        children=[
            html.H3("Mon Portefeuille", className="section-title font-jetbrains-mono"),
            html.Div(
                className="portfolio-main-content",
                children=[
                    # Colonne de gauche: liste de tickers + formulaire
                    html.Div(
                        className="ticker-buttons-container",
                        children=[
                            html.Div(
                                className="ticker-buttons",
                                children=[
                                    html.Button(
                                        t,
                                        id={"type": "btn-portefeuille", "index": t},
                                        className="ticker-button tiltable btn btn-outline-light text-start mb-2"
                                    ) for t in portfolio
                                ]
                            ),
                            html.Div(
                                className="add-ticker-form d-flex align-items-center mt-3",
                                children=[
                                    dcc.Input(
                                        id="input-mot-cle",
                                        type="text",
                                        placeholder="Enter ticker...",
                                        className="ticker-input form-control me-2"
                                    ),
                                    html.Button("Add", id="btn-ajouter", className="btn-add btn btn-success me-2"),
                                    html.Button("Remove", id="btn-supprimer", className="btn-remove btn btn-danger")
                                ]
                            )
                        ]
                    ),
                    # Colonne de droite: graphique du ticker sélectionné
                    html.Div(
                        className="price-chart-container",
                        children=[
                            html.H4(id="chart-title", className="font-jetbrains-mono mt-0 mb-3"),
                            dcc.Graph(
                                id="price-chart",
                                className="price-chart",
                                figure={}
                            )
                        ]
                    )
                ]
            )
        ]
    )

# Callback: mise à jour du graphique en fonction du bouton cliqué
@callback(
    Output("price-chart", "figure"),
    Output("chart-title", "children"),
    Input({"type": "btn-portefeuille", "index": ALL}, "n_clicks"),
    State({"type": "btn-portefeuille", "index": ALL}, "id"),
    prevent_initial_call=True
)
def update_price_chart(n_clicks_list, ids_list):
    triggered = ctx.triggered_id
    if not triggered:
        return {}, "Select a ticker"
    ticker = triggered["index"]
    # Lecture des données du portefeuille enregistrées
    try:
        df = pd.read_csv("Data/Tickers/csv/portefeuille.csv", parse_dates=["Datetime"])
    except Exception:
        return {}, f"No data file"
    # Filtrage selon format
    if "Price" in df.columns:
        df_t = df[(df["Ticker"] == ticker) & (df["Price"] == "Close")]
    else:
        df_t = df[df["Ticker"] == ticker]
    if df_t.empty:
        return {}, f"No data for {ticker}"
    df_t = df_t.sort_values("Datetime").tail(30)
    # Choix de la colonne de prix
    if "Value" in df_t.columns:
        ycol = "Value"
    elif "Close" in df_t.columns:
        ycol = "Close"
    elif "Adj Close" in df_t.columns:
        ycol = "Adj Close"
    else:
        return {}, f"No price column for {ticker}"
    fig = px.line(df_t, x="Datetime", y=ycol, title="")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20,r=20,t=20,b=20))
    return fig, ticker

# Callback: ajout / suppression de tickers dans le portefeuille
@callback(
    Output("input-mot-cle", "value", allow_duplicate=True),
    Input("btn-ajouter", "n_clicks"),
    Input("btn-supprimer", "n_clicks"),
    State("input-mot-cle", "value"),
    prevent_initial_call=True
)
def modify_portfolio(n_clicks_add, n_clicks_remove, value):
    if not value or not value.strip():
        return ""
    action = ctx.triggered_id
    tickers = set(get_tickers())
    ticker = value.strip().upper()
    if action == "btn-ajouter":
        tickers.add(ticker)
    elif action == "btn-supprimer":
        tickers.discard(ticker)
    # Écriture dans le fichier
    with open("Data/Tickers/txt/portefeuille.txt", "w", encoding="utf-8") as f:
        for t in sorted(tickers):
            f.write(t + "\n")
    return ""

