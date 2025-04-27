from dash import Dash, html, dcc, Input, Output, State, callback, callback_context
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify
import yfinance as yf
from src.utils.data_process import get_portfolio
from pages.portefeuille import portefeuille_layout
from pages.predict import prediction_layout

# -------------  Bootstrap / Theme -----------------
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.DARKLY, "/assets/style.css"],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

# -------------  CAC‑40 helpers --------------------
CAC40_FALLBACK = [
    "AI.PA", "AIR.PA", "ALO.PA", "ALU.PA", "BNP.PA", "CA.PA", "ACA.PA", "CAP.PA",
    "CS.PA", "BN.PA", "EN.PA", "EL.PA", "ENGI.PA", "KER.PA", "LR.PA", "MC.PA",
    "ML.PA", "ORA.PA", "OR.PA", "PUB.PA", "RI.PA", "RNO.PA", "SAF.PA", "SGO.PA",
    "SAN.PA", "SU.PA", "GLE.PA", "SW.PA", "STLA", "TTE", "UG.PA", "VIE.PA",
    "DG.PA", "FR.PA", "HO.PA", "VIV.PA", "MT.AS", "AC.PA", "AIQ.PA", "DSY.PA"
]

def get_cac40_tickers():
    """Return list of tickers (from file or fallback)."""
    try:
        with open("Data/ticker/cac40_tickers.txt", "r", encoding="utf-8") as f:
            tickers = [t.strip() for t in f if t.strip()]
            return tickers if tickers else CAC40_FALLBACK
    except FileNotFoundError:
        return CAC40_FALLBACK

def get_ticker_data():
    """Fetch last‑day price & day‑over‑day change for each CAC40 ticker."""
    tickers = get_cac40_tickers()
    df = yf.download(tickers, period="2d", interval="1d", progress=False, auto_adjust=False)["Close"]
    if df.empty or len(df) < 2:
        return []
    latest, prev = df.iloc[-1], df.iloc[-2]
    data = []
    for symbol in tickers:
        price, prev_price = latest.get(symbol), prev.get(symbol)
        if price is None or prev_price is None:
            continue
        change = (price / prev_price - 1) * 100 if prev_price else 0
        try:
            name = yf.Ticker(symbol).info.get("shortName", symbol)
        except Exception:
            name = symbol
        data.append({"symbol": symbol, "name": name, "price": price, "change": change})
    return data

def create_ticker_tape():
    """Build a slower‑scrolling CAC‑40 ticker tape showing company **names only**."""
    rows = get_ticker_data()
    if not rows:
        return html.Div("No data", className="ticker-tape-empty")

    items = []
    for r in rows:
        cls  = "text-gain" if r["change"] >= 0 else "text-loss"
        sign = "+" if r["change"] >= 0 else ""
        items.append(
            html.Li([
                html.Span(r["name"],                     className="ticker-symbol me-1"),
                html.Span(f"€{r['price']:.2f}",           className="ticker-price me-1"),
                html.Span(f"{sign}{r['change']:.2f}%",    className=f"ticker-change {cls}")
            ], className="ticker-item")
        )

    # Duplicate list (marquee illusion) & slow the animation to 30 s
    ul_kwargs = {"className": "ticker-list", "style": {"animationDuration": "50s"}}
    return html.Div([
        html.Ul(items, **ul_kwargs),
        html.Ul(items, **ul_kwargs)
    ], className="ticker-tape")

# -------------  Modals ----------------------------
command_palette = dbc.Modal([
    dbc.ModalHeader("Command Palette"),
    dbc.ModalBody([
        dbc.Input(placeholder="Type a command…", className="mb-3"),
        dbc.ListGroup([
            dbc.ListGroupItem([DashIconify(icon="mdi:chart-box-outline"), " Portfolio"], href="/"),
            dbc.ListGroupItem([DashIconify(icon="mdi:chart-line"),      " Prediction"], href="/prediction"),
            dbc.ListGroupItem([DashIconify(icon="mdi:finance"),        " Financials"], href="/etat_financier"),
            dbc.ListGroupItem([DashIconify(icon="mdi:comment-text-outline"), " Sentiments"], href="/sentiments")
        ])
    ])
], id="command-palette-modal", is_open=False, centered=True, size="lg")

# -------------  Layout ----------------------------
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    command_palette,

    html.Header(
        className="header",
        children=[
            html.H1("FinDash", className="brand-name font-jetbrains-mono me-4"),
            create_ticker_tape(),
            # Removed the command palette button
        ]
    ),

    html.Div([
        html.Aside([
            html.Button(DashIconify(icon="mdi:menu"), id="sidebar-toggle", className="sidebar-toggle mb-3"),
            dbc.Nav([
                dbc.NavLink([DashIconify(icon="mdi:chart-box-outline"), html.Span(" Portfolio",  className="ms-2")], href="/",            active="exact", className="mb-2"),
                dbc.NavLink([DashIconify(icon="mdi:chart-line"),       html.Span(" Prediction", className="ms-2")], href="/prediction", active="exact", className="mb-2"),
                dbc.NavLink([DashIconify(icon="mdi:finance"),         html.Span(" Financials", className="ms-2")], href="/etat_financier", active="exact", className="mb-2"),
                dbc.NavLink([DashIconify(icon="mdi:comment-text-outline"), html.Span(" Sentiments",  className="ms-2")], href="/sentiments",    active="exact", className="mb-2")
            ], vertical=True, pills=True)
        ], id="sidebar", className="sidebar"),

        html.Main(html.Div(id="page-content", className="content-container"), className="main-content")
    ], className="main-layout"),

    html.Footer([
        html.P("FinDash © 2025", className="copyright font-jetbrains-mono"),
        html.Div([
            html.A("Privacy", href="#", className="footer-link me-3"),
            html.A("Terms",   href="#", className="footer-link me-3"),
            html.A("Contact", href="#", className="footer-link")
        ], className="footer-links")
    ], className="footer")
])

# -------------  Callbacks -------------------------
@callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(path):
    if path == "/":
        return html.Div([
            html.Div([
                html.H2("Mon Portefeuille", className="page-title font-jetbrains-mono"),
                html.P("Overview of your investments", className="page-description")
            ], className="page-header"),
            portefeuille_layout(get_portfolio())
        ])
    if path == "/prediction":
        return html.Div([
            html.Div([
                html.H2("Predictions", className="page-title font-jetbrains-mono"),
                html.P("AI‑powered market insights", className="page-description")
            ], className="page-header"),
            prediction_layout()
        ])
    return html.Div(html.H3("404: Page Not Found"), className="error-container")

@callback(Output("sidebar", "className"), Input("sidebar-toggle", "n_clicks"), State("sidebar", "className"), prevent_initial_call=True)
def toggle_sidebar(n, cls):
    return "sidebar collapsed" if "collapsed" not in cls else "sidebar"

# -------------  Run ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
