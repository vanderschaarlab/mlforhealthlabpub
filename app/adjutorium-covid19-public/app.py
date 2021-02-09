import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "Adjutorium COVID-19"
app.config.suppress_callback_exceptions = True


template = dict(
    layout=go.Layout(colorway=px.colors.qualitative.Dark2)
)
