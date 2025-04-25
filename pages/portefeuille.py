from dash import html
from dash import html, dcc, callback, Output, Input, State, ctx
from dash import MATCH, ALL
import plotly.express as px
import pandas as pd
from src.utils.data_process import get_tickers
import yfinance as yf

def portefeuille_layout(portefolio):
    boutons = [
        html.Button(
            i,
            id={"type": "btn-portefeuille", "index": i},
            n_clicks=0,
            style={"marginBottom": "10px", "width": "100%"}
        )
        for i in portefolio
    ]
    return html.Div([html.Div(
            style={
                "display": "flex",
                "gap": "10px",
                "margin": "20px 40px 0 40px"
            },
            children=[
                dcc.Input(
                    id="input-mot-cle",
                    type="text",
                    placeholder="Entrer un mot...",
                    style={"width": "300px", "padding": "8px"}
                ),
                html.Button("Ajouter", id="btn-ajouter", n_clicks=0),
                html.Button("Supprimer", id="btn-supprimer", n_clicks=0)
            ]
        ),
        html.Div(
        style={
            "display": "flex",
            "justifyContent": "space-between",  # Pour espacer les blocs de manière égale
            "padding": "40px",
            "minHeight": "100vh",
            "backgroundColor": "#f9f9f9"
        },
        children=[
            html.Div(
                style={
                    "backgroundColor": "white",
                    "padding": "30px",
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
                    "borderRadius": "10px",
                    "minWidth": "45%",  # Taille du bloc gauche
                    "minHeight": "300px"
                },
                children=boutons
            ),
            html.Div(id='contenu-dynamique',
                style={
                    "backgroundColor": "white",
                    "padding": "30px",
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
                    "borderRadius": "10px",
                    "minWidth": "45%",  # Taille du bloc droit
                    "minHeight": "300px"
                },
                children=[
                    html.H2("Bloc 2 : Détails ou graphiques avancés", style={"marginBottom": "20px"}),
                    html.P("Ajoute ici un tableau de positions, un graphique d’évolution, ou toute autre info.")
                ]
            )
        ]
    )])

@callback(
    Output("contenu-dynamique", "children"),
    Input({"type": "btn-portefeuille", "index": ALL}, "n_clicks"),
    State({"type": "btn-portefeuille", "index": ALL}, "id")
)
def update_bloc_droit(n_clicks_list, ids):
    triggered_id = ctx.triggered_id
    if not triggered_id:
        return html.P("Clique sur un bouton pour voir le contenu.")
    i = triggered_id["index"]
    try:
        df = pd.read_csv("Data/Tickers/csv/portefeuille.csv")
        # Correction : si le fichier est au format wide, le transformer en long
        if 'Price' not in df.columns and 'Open' in df.columns:
            # On suppose format wide, on le melt
            id_vars = ['Datetime', 'Ticker'] if 'Ticker' in df.columns else ['Datetime']
            value_vars = [col for col in df.columns if col not in id_vars]
            df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='Price', value_name='Value')
        df = df[df['Ticker'] == i]
        df = df[df["Price"] == 'Close']
        df = df.tail(30)
        fig = px.line(df, x='Datetime', y='Value', title=f"{i}")
        return dcc.Graph(figure=fig)
    except Exception as e:
        return html.Div([
            html.H3(f"Erreur lors du chargement de {i}"),
            html.Pre(str(e))
        ])

FICHIER_TXT = "Data/Tickers/txt/portefeuille.txt"

@callback(
    Output("input-mot-cle", "value", allow_duplicate=True),
    Input("btn-ajouter", "n_clicks"),
    Input("btn-supprimer", "n_clicks"),
    State("input-mot-cle", "value"),
    prevent_initial_call=True
)
def modifier_txt(n_clicks_ajouter, n_clicks_supprimer, mot):
    if not mot or mot.strip() == "":
        return None
    mot = mot.strip()
    action = ctx.triggered_id
    try:
        with open(FICHIER_TXT, "r", encoding="utf-8") as f:
            mots = set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        mots = set()
    if action == "btn-ajouter":
        mots.add(mot)
    elif action == "btn-supprimer":
        mots.discard(mot)
    with open(FICHIER_TXT, "w", encoding="utf-8") as f:
        for m in sorted(mots):
            f.write(m + "\n")
    # Générer le csv au format long attendu (Ticker, Datetime, Price, Value)
    tickers = list(mots)
    if tickers:
        start_date = pd.Timestamp.today() - pd.Timedelta(days=729)
        end_date = pd.Timestamp.today()
        data = yf.download(tickers, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), group_by='ticker', auto_adjust=False)
        if len(tickers) == 1:
            # yfinance retourne un DataFrame simple pour un seul ticker
            data.columns = pd.MultiIndex.from_product([tickers, data.columns])
        data.index.name = 'Datetime'
        data = data.stack(level=0).reset_index()
        data = data.melt(id_vars=['Datetime', 'level_1'], var_name='Price', value_name='Value')
        data.rename(columns={'level_1': 'Ticker'}, inplace=True)
        data = data[['Datetime', 'Ticker', 'Price', 'Value']]
        data.to_csv("Data/Tickers/csv/portefeuille.csv", index=False)
    else:
        pd.DataFrame(columns=['Datetime', 'Ticker', 'Price', 'Value']).to_csv("Data/Tickers/csv/portefeuille.csv", index=False)
    return ""