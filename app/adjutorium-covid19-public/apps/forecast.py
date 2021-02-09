import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pds
from datetime import datetime as dt
import datetime
import plotly.graph_objs as go
import plotly.express as px

from app import app, template


update_date = dt(2020, 4, 4)

d_trust = pds.read_csv('assets/d_trust.csv')
d_trust['hospitaladmissiondate'] = pds.to_datetime(d_trust['hospitaladmissiondate'], errors='coerce')

icu_forecast = pds.read_csv('assets/icu_risk_forecast.csv')
icu_forecast['hospitaladmissiondate'] = pds.to_datetime(icu_forecast['hospitaladmissiondate'], errors='coerce')

death_forecast = pds.read_csv('assets/death_risk_forecast.csv')
death_forecast['hospitaladmissiondate'] = pds.to_datetime(death_forecast['hospitaladmissiondate'], errors='coerce')

dat_t = pds.read_csv('assets/trusts.csv')
trust_name = dat_t['Var1'].values
TRUST_OPTIONS = []
for i in range(len(trust_name)):
    d = {
        'label': trust_name[i],
        'value': trust_name[i]
    }
    TRUST_OPTIONS.append(d)
TRUST_VALUES = 'NATIONAL'

trust_select = dbc.FormGroup(
    [
        dbc.Label("Select NHS Trust", html_for='trust-forecast'),
        dcc.Dropdown(
            id='trust-forecast',
            options=TRUST_OPTIONS,
            value=TRUST_VALUES,
            multi=False,
        )

    ]
)

date_forecast_picker = dbc.FormGroup(
    [
        dbc.Label("Select Forecast Start Date", html_for='date_forecast'),
        dcc.DatePickerSingle(
            id='date_forecast',
            min_date_allowed=dt(2020, 3, 10),
            max_date_allowed=dt(2020, 12, 31),
            initial_visible_month=dt(2020, 3, 5),
            date=dt(2020, 3, 17).date(),
            day_size=45
        ),
    ]
)


date_range_picker = dbc.FormGroup(
    [
        dbc.Label("Select Hospital Admission Dates", html_for='date_range_forecast'),
        dcc.DatePickerRange(
            id='date_range_forecast',
            min_date_allowed=dt(2020, 3, 10),
            max_date_allowed=dt(2020, 12, 31),
            initial_visible_month=dt(2020, 3, 5),
            start_date=dt(2020, 3, 10).date(),
            end_date=dt(2020, 3, 31).date(),
            day_size=45
        ),
    ]
)

trust_date_selector = html.Div(
    [
        html.Div(dbc.Row(dbc.Col(
            html.Header(html.H2("ICU Admission and Mortality Forecast")),
            style={'marginBottom': 25, 'marginTop': 25, 'text-align': 'center'})
        )),

        dbc.Row([
            dbc.Col(trust_select),
            dbc.Col(date_range_picker),
            dbc.Col(date_forecast_picker),
        ]),

    ]  # , style={"margin-left": "5ex"},
)

# Create forecast based on national average
# --------------------------------------------------

occpupancy_display = html.Div(
    [

        html.Div(dbc.Row(dbc.Col(
            html.P("* Forecasts are based on historical data."),
            style={'marginBottom': 5, 'marginTop': 5, 'text-align': 'left', "color": 'Blue'})
        )),
        dbc.Row(
            dbc.Col(dcc.Graph(id="icu_line"), width=12),
        ),
        dbc.Row(
            dbc.Col(dcc.Graph(id="mortality_line"), width=12),
        ),
    ]
)


layout = html.Div([
    trust_date_selector,
    occpupancy_display,
])


def update_mortality_line(df_slice, df_death):

    df_plot = df_slice[['hospitaladmissiondate', 'is_2d_mortality', 'is_7d_mortality', 'newly_admitted']]

    df_7d_observed = df_plot[df_plot['hospitaladmissiondate'] - update_date < datetime.timedelta(days=-7)]

    df_2d_observed = df_plot[df_plot['hospitaladmissiondate'] - update_date < datetime.timedelta(days=-2)]

    df_7d_pred = df_plot[df_plot['hospitaladmissiondate'] - update_date >= datetime.timedelta(days=-8)]
    df_2d_pred = df_plot[df_plot['hospitaladmissiondate'] - update_date >= datetime.timedelta(days=-3)]


    # update_date
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_2d_observed['hospitaladmissiondate'].dt.date,
                             y=df_2d_observed['is_2d_mortality'].values / df_2d_observed['newly_admitted'].values,
                             mode='lines+markers',
                             name='Mortality within 2 Days'))

    fig.add_trace(go.Scatter(x=df_7d_observed['hospitaladmissiondate'].dt.date,
                             y=df_7d_observed['is_7d_mortality'].values / df_7d_observed['newly_admitted'].values,
                             mode='lines+markers',
                             name='Mortality within 7 Days'))
    fig.add_trace(go.Scatter(x=df_death['hospitaladmissiondate'].dt.date,
                             y=df_death['1'].values,
                             mode='lines+markers',
                             line=dict(dash='dash'),
                             name='Mortality within 2 Days (Forecast)'))

    fig.add_trace(go.Scatter(x=df_death['hospitaladmissiondate'].dt.date,
                             y=df_death['6'].values,
                             mode='lines+markers',
                             line=dict(dash='dash'),
                             name='Mortality within 7 Days (Forecast)'))

    fig.update_layout(
        title="2D and 7D Mortality Rate",
        paper_bgcolor='White',
        plot_bgcolor='White',
        hovermode="x",
    )
    fig.update_yaxes(title_text='Mortality Rate', tickformat="%")  # range=[0., .3],
    fig.update_xaxes(title_text='Patient Admission Date')
    fig.update_layout(template=template)

    return fig


def update_icu_line(df_slice, df_icu):

    df_plot = df_slice[['hospitaladmissiondate', 'is_2d_icu', 'is_7d_icu', 'newly_admitted']]
    df_7d_observed = df_plot[df_plot['hospitaladmissiondate'] - update_date < datetime.timedelta(days=-7)]
    # df_7d_pred = df_plot[df_plot['hospitaladmissiondate'] - update_date >= datetime.timedelta(days=-8)]

    df_2d_observed = df_plot[df_plot['hospitaladmissiondate'] - update_date < datetime.timedelta(days=-2)]
    # df_2d_pred = df_plot[df_plot['hospitaladmissiondate'] - update_date >= datetime.timedelta(days=-3)]

    # update_date
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_2d_observed['hospitaladmissiondate'].dt.date,
                             y=df_2d_observed['is_2d_icu'].values / df_2d_observed['newly_admitted'].values,
                             mode='lines+markers',
                             name='ICU Admission within 2 Days'))

    fig.add_trace(go.Scatter(x=df_7d_observed['hospitaladmissiondate'].dt.date,
                             y=df_7d_observed['is_7d_icu'].values / df_7d_observed['newly_admitted'].values,
                             mode='lines+markers',
                             name='ICU Admission within 7 Days'))
    fig.add_trace(go.Scatter(x=df_icu['hospitaladmissiondate'].dt.date,
                             y=df_icu['1'].values,
                             mode='lines+markers',
                             line=dict(dash='dash'),
                             name='ICU Admission within 2 Days (Forecast)'))
    fig.add_trace(go.Scatter(x=df_icu['hospitaladmissiondate'].dt.date,
                             y=df_icu['6'].values,
                             mode='lines+markers',
                             line=dict(dash='dash'),
                             name='ICU Admission within 7 Days (Forecast)'))

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df_plot['hospitaladmissiondate'].values,
    #                          y=df_plot['is_2d_icu'].values / df_plot['newly_admitted'].values,
    #                          mode='lines+markers',
    #                          name='ICU Admission within 2 Days'))
    # fig.add_trace(go.Scatter(x=df_plot['hospitaladmissiondate'].values,
    #                          y=df_plot['is_7d_icu'].values / df_plot['newly_admitted'].values,
    #                          mode='lines+markers',
    #                          name='ICU Admission within 7 Days'))

    # fig = px.line(df2, x="year", y="lifeExp", color='country')

    fig.update_layout(
        title="Proportion of 2D and 7D ICU Admission",
        paper_bgcolor='White',
        plot_bgcolor='White',
        hovermode="x",
    )
    fig.update_yaxes(title_text="Proportion of ICU Admission", tickformat="%")  # range=[0., .8],
    fig.update_xaxes(title_text='Patient Admission Date')
    fig.update_layout(template=template)

    return fig

@app.callback(
    [
        Output("mortality_line", "figure"),
        Output("icu_line", "figure"),
    ],
    [
        Input("trust-forecast", "value"),
        Input("date_range_forecast", "start_date"),
        Input("date_range_forecast", "end_date"),
        Input("date_forecast", "date"),
    ])
def update_forecast(trust, start_date, end_date, date_forecast):
    df_slice = d_trust[(d_trust['trustname'] == trust) &
                       (d_trust['hospitaladmissiondate'].dt.strftime('%Y-%m-%d') >= str(start_date)) &
                       (d_trust['hospitaladmissiondate'].dt.strftime('%Y-%m-%d') <= str(end_date))]

    df_icu = icu_forecast[
                       (icu_forecast['hospitaladmissiondate'].dt.strftime('%Y-%m-%d') >= str(date_forecast)) &
                       (icu_forecast['hospitaladmissiondate'].dt.strftime('%Y-%m-%d') <= str(end_date))]

    df_death = death_forecast[
                       (death_forecast['hospitaladmissiondate'].dt.strftime('%Y-%m-%d') >= str(date_forecast)) &
                       (death_forecast['hospitaladmissiondate'].dt.strftime('%Y-%m-%d') <= str(end_date))]

    mortality_line = update_mortality_line(df_slice, df_death)
    icu_line = update_icu_line(df_slice, df_icu)
    return mortality_line, icu_line