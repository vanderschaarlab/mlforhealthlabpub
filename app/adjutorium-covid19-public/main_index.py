import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import exec_summary, simulation, about, forecast


# Create the **header** with logo and navigation buttons
# -------------------------------------------------------

brand_img = html.Img(src='assets/mainLogo.png', id="brand-logo", style={"height": "35px"})
lab_website = "http://www.vanderschaar-lab.com/NewWebsite/COVID.html"

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("About", href='/', style={"color": "White", "font-size": '130%', "margin-left": "5px"})),
        dbc.NavItem(dbc.NavLink("Statistics", href="/exec", style={"color": "White", "font-size": '130%', "margin-left": "5px"})),
        dbc.NavItem(dbc.NavLink("Forecast", href="/forecast", style={"color": "White", "font-size": '130%', "margin-left": "5px"})),
        dbc.NavItem(dbc.NavLink("Simulation", href="/simulation", style={"color": "White", "font-size": '130%', "margin-left": "5px"})),
        dbc.NavItem(dbc.NavLink("Prognosis", href="/prediction", style={"color": "White", "font-size": '130%', "margin-left": "5px"})),

        # dbc.NavItem(dbc.NavLink(brand_img, href=lab_website, external_link=True)),
    ],
    brand="Adjutorium COVID-19",
    brand_href="#",
    brand_style={
        "font-size": 28
    },
    color="primary",
    dark=True,
    fluid=True,
    fixed='top',
    sticky='top',
)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return about.layout
    elif pathname == '/exec':
        return exec_summary.layout
    elif pathname == '/forecast':
        return forecast.layout
    elif pathname == '/simulation':
        return simulation.layout
    elif pathname == '/prediction':
        return 'Coming Soon'
    else:
        return '404'


if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
