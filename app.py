from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from pages.portefeuille import portefeuille_layout
from pages.predict import prediction_layout
from src.utils.data_process import get_portfolio
app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Nav([
        dcc.Link('Mon Portefeuille', href='/', className='nav-link', style={'marginRight': '20px'}),]),

    html.Nav([
        dcc.Link('Prédiction', href='/prediction', className='nav-link', style={'marginRight': '20px'}),
        dcc.Link('Etat financier', href='/etat_financier', className='nav-link', style={'marginRight': '20px'}),
        dcc.Link('Sentiments', href='/sentiments', className='nav-link')
    ], style={'padding': '20px', 'backgroundColor': '#f0f0f0'}),

    html.Div(id='page-content', style={'padding': '40px'})
])

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        return portefeuille_layout(get_portfolio())
    if pathname == '/prediction':
        return prediction_layout()
    elif pathname == '/contact':
        return None
    return None

if __name__ == '__main__':
    app.run(debug=True)
