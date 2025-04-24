from dash import html, dcc, Input, Output, State, callback
from src.utils.prediction import get_prediction
import pandas as pd
def prediction_layout():
    df,time=get_prediction("Data/Tickers/txt/portefeuille.txt","Data/Tickers/csv/portefeuille.csv")

    temp=generate_divs_from_df(df)
    return html.Div([html.P(time,style={"textAlign":"center","fontSize":"20px","fontWeight":"bold"}),
        # Bloc supérieur : texte libre
        html.Div(children=temp, style={
            "padding": "30px",
            "backgroundColor": "#f2f2f2",
            "margin": "20px",
            "borderRadius": "10px"
        }),
        # Bloc secondaire : input + bouton + affichage du résultat
        html.Div([
            html.Div([
                dcc.Input(
                    id="input-valeur-pred",
                    type="text",
                    placeholder="Entrez une valeur ici...",
                    style={"padding": "10px", "width": "300px"}
                ),
                html.Button("Prédire", id="btn-predire", n_clicks=0, style={"marginLeft": "10px"})
            ], style={"display": "flex", "alignItems": "center"}),

            html.Br(),

            html.Div(id="resultat-prediction", style={
                "marginTop": "20px",
                "fontWeight": "bold",
                "color": "#333"
            })
        ], style={
            "padding": "30px",
            "backgroundColor": "#ffffff",
            "margin": "20px",
            "borderRadius": "10px",
            "boxShadow": "0 2px 6px rgba(0,0,0,0.1)"
        })
    ])

@callback(
    Output("resultat-prediction", "children"),
    Input("btn-predire", "n_clicks"),
    State("input-valeur-pred", "value"),
    prevent_initial_call=True
)
def res_prediction(n, entree):
    if not entree:
        return "Veuillez entrer une valeur."
    with open("Data/Tickers/txt/prediction.txt", "w", encoding="utf-8") as f:
        f.write(entree.strip())
    df,time=get_prediction("Data/Tickers/txt/prediction.txt","Data/Tickers/csv/prediction.csv")
    temp=generate_divs_from_df(df)
    return html.Div([
        html.P("Résultat de la prédiction :", style={"fontWeight": "bold"}),
        html.Div(children=temp, style={
            "padding": "30px",
            "backgroundColor": "#f2f2f2",
            "margin": "20px",
            "borderRadius": "10px"
        })
    ])

def generate_divs_from_df(df):
    res={1:"Hausse 📈",0:"Baisse📉"}
    return [
        html.Div([
            html.P(f"Tickers : {row['Ticker']}"),
            html.P(f"Prédiction : {res[row['Prediction']]}")
        ], style={
            "padding": "10px",
            "marginBottom": "10px",
            "border": "1px solid #ccc",
            "borderRadius": "8px",
            "backgroundColor": "#f9f9f9"
        })
        for _, row in df.iterrows()
    ]