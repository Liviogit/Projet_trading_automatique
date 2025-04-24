from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify

# Assume these imports exist and work for loading page layouts/data
from pages.portefeuille import portefeuille_layout
from pages.predict import prediction_layout
from src.utils.data_process import get_portfolio

# Initialize the Dash app with Bootstrap for responsive design
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

# Define the main layout with proper structure
app.layout = html.Div(
    className="app-container",
    children=[
        # URL Location component
        dcc.Location(id='url', refresh=False),
        
        # Header with logo and navigation
        html.Header(
            className="header",
            children=[
                # Logo and brand section
                html.Div(
                    className="brand",
                    children=[
                        html.Img(src="/assets/logo.png", className="logo"),
                        html.H1("FinDash", className="brand-name")
                    ]
                ),
                
                # Navigation for desktop
                html.Nav(
                    className="nav-desktop",
                    children=[
                        html.Ul(
                            className="nav-links",
                            children=[
                                html.Li(dcc.Link([DashIconify(icon="mdi:chart-box-outline", className="nav-icon"), "Mon Portefeuille"], href='/', className="nav-link")),
                                html.Li(dcc.Link([DashIconify(icon="mdi:chart-line", className="nav-icon"), "Prédiction"], href='/prediction', className="nav-link")),
                                html.Li(dcc.Link([DashIconify(icon="mdi:finance", className="nav-icon"), "Etat financier"], href='/etat_financier', className="nav-link")),
                                html.Li(dcc.Link([DashIconify(icon="mdi:comment-text-outline", className="nav-icon"), "Sentiments"], href='/sentiments', className="nav-link")),
                            ]
                        )
                    ]
                ),
                
                # Mobile menu button
                html.Button(
                    className="mobile-menu-button",
                    children=[DashIconify(icon="mdi:menu", width=24)],
                    id="mobile-menu-button"
                ),
            ]
        ),
        
        # Mobile navigation (hidden by default)
        html.Nav(
            className="nav-mobile",
            id="mobile-nav",
            style={"display": "none"},
            children=[
                html.Ul(
                    className="mobile-nav-links",
                    children=[
                        html.Li(dcc.Link([DashIconify(icon="mdi:chart-box-outline", className="nav-icon"), "Mon Portefeuille"], href='/', className="mobile-nav-link")),
                        html.Li(dcc.Link([DashIconify(icon="mdi:chart-line", className="nav-icon"), "Prédiction"], href='/prediction', className="mobile-nav-link")),
                        html.Li(dcc.Link([DashIconify(icon="mdi:finance", className="nav-icon"), "Etat financier"], href='/etat_financier', className="mobile-nav-link")),
                        html.Li(dcc.Link([DashIconify(icon="mdi:comment-text-outline", className="nav-icon"), "Sentiments"], href='/sentiments', className="mobile-nav-link")),
                    ]
                )
            ]
        ),
        
        # Main content area with a container
        html.Main(
            className="main-content",
            children=[
                html.Div(
                    id='page-content',
                    className="content-container"
                )
            ]
        ),
        
        # Footer
        html.Footer(
            className="footer",
            children=[
                html.P("© 2025 FinDash Financial Analytics", className="copyright"),
                html.Div(
                    className="footer-links",
                    children=[
                        html.A("Privacy Policy", href="#", className="footer-link"),
                        html.A("Terms of Service", href="#", className="footer-link"),
                        html.A("Contact", href="#", className="footer-link"),
                    ]
                )
            ]
        )
    ]
)

# Callback to update page content based on URL
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    # Update active link class via JavaScript (handled in custom.js)
    
    if pathname == '/':
        # Assume get_portfolio() returns data needed by portefeuille_layout
        portfolio_data = get_portfolio()
        return html.Div([
            html.Div(className="page-header", children=[
                html.H2("Mon Portefeuille", className="page-title"),
                html.P("Vue d'ensemble de vos investissements", className="page-description")
            ]),
            portefeuille_layout(portfolio_data)
        ])
    elif pathname == '/prediction':
        return html.Div([
            html.Div(className="page-header", children=[
                html.H2("Prédiction", className="page-title"),
                html.P("Analyse prédictive des tendances du marché", className="page-description")
            ]),
            prediction_layout()
        ])
    elif pathname == '/etat_financier':
        return html.Div([
            html.Div(className="page-header", children=[
                html.H2("Etat Financier", className="page-title"),
                html.P("Aperçu détaillé de votre situation financière", className="page-description")
            ]),
            html.Div(className="content-section", children=[
                html.H1("Etat Financier (Contenu à venir)")
            ])
        ])
    elif pathname == '/sentiments':
        return html.Div([
            html.Div(className="page-header", children=[
                html.H2("Analyse des Sentiments", className="page-title"),
                html.P("Tendances et opinions du marché", className="page-description")
            ]),
            html.Div(className="content-section", children=[
                html.H1("Analyse des Sentiments (Contenu à venir)")
            ])
        ])
    # Default case for unknown paths
    return html.Div([
        html.Div(className="error-container", children=[
            DashIconify(icon="mdi:alert-circle-outline", width=64, className="error-icon"),
            html.H1("404: Page non trouvée", className="error-title"),
            html.P(f"Le chemin {pathname} n'a pas été reconnu...", className="error-message"),
            dcc.Link("Retour à l'accueil", href="/", className="error-link")
        ])
    ])

# Callback for mobile menu toggle
@app.callback(
    Output("mobile-nav", "style"),
    Input("mobile-menu-button", "n_clicks"),
    State("mobile-nav", "style"),
    prevent_initial_call=True
)
def toggle_mobile_nav(n_clicks, current_style):
    if current_style.get("display") == "none":
        return {"display": "block"}
    return {"display": "none"}

# Run the app
if __name__ == '__main__':
    app.run(debug=True)