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


update_date = dt(2020, 3, 31)
d_trust = pds.read_csv('assets/d_trust.csv')
d_trust['hospitaladmissiondate'] = pds.to_datetime(d_trust['hospitaladmissiondate'], errors='coerce')

icu_forecast = pds.read_csv('assets/icu_risk_forecast.csv')
icu_forecast['hospitaladmissiondate'] = pds.to_datetime(icu_forecast['hospitaladmissiondate'], errors='coerce')

death_forecast = pds.read_csv('assets/death_risk_forecast.csv')
death_forecast['hospitaladmissiondate'] = pds.to_datetime(death_forecast['hospitaladmissiondate'], errors='coerce')

d_patient_age = pds.read_csv('assets/d_patient_age.csv')
d_patient_age['hospitaladmissiondate'] = pds.to_datetime(d_patient_age['hospitaladmissiondate'], errors='coerce')

d_icu_stay_trust = pds.read_csv('assets/d_icu_stay_trust.csv')
d_ts = pds.read_csv('assets/d_ts.csv')

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
        dbc.Label("Select NHS Trust", html_for='trust'),
        dcc.Dropdown(
            id='trust',
            options=TRUST_OPTIONS,
            value=TRUST_VALUES,
            multi=False,
        )

    ]
)

date_range_picker = dbc.FormGroup(
    [
        dbc.Label("Select Hospital Admission Dates", html_for='date_range'),
        dcc.DatePickerRange(
            id='date_range',
            min_date_allowed=dt(2020, 3, 10),
            max_date_allowed=dt(2020, 12, 31),
            initial_visible_month=dt(2020, 3, 5),
            start_date=dt(2020, 3, 10).date(),
            end_date=dt(2020, 3, 31).date(),
            day_size=45
        ),
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

trust_date_selector = html.Div(
    [
        html.Div(dbc.Row(dbc.Col(
            html.Header(html.H2("Executive Summary")),
            style={'marginBottom': 25, 'marginTop': 25, 'text-align': 'center'})
        )),

        dbc.Row([
            dbc.Col(trust_select),
            dbc.Col(date_range_picker),
        ]),

    ]  # , style={"margin-left": "5ex"},
)

# Create the data display for app body
# --------------------------------------------------


stats_summary_display = html.Div(
    [

        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="age_hist"), width=6),
                dbc.Col(dcc.Graph(id="gender_pi"), width=3),
                dbc.Col(dcc.Graph(id="admission_pi"), width=3),
            ],
            no_gutters=True
        ),

        dbc.Row(
            [
                dbc.Col(html.Div([dcc.Graph(id="num_comor_bar")]), width=6),
                dbc.Col(html.Div([dcc.Graph(id="comor_bar")]), width=6),
            ],
            no_gutters=True
        ),
        dbc.Row(
            [
                dbc.Col(html.Div([dcc.Graph(id="add_line")])),
            ],
            no_gutters=True
        ),
        dbc.Row(
            dbc.Col(dcc.Graph(id="icu_bar"), width=12),
        ),
    ]
)

# Create forecast based on national average
# --------------------------------------------------

occpupancy_display = html.Div(
    [
        html.Div(dbc.Row(dbc.Col(
            html.Header(html.H2("ICU Admission and Mortality Forecast")),
            style={'marginBottom': 25, 'marginTop': 25, 'text-align': 'center'})
        )),
        html.Div(dbc.Row(dbc.Col(date_forecast_picker))),

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
    stats_summary_display,
    # occpupancy_display,
])


# Call backs
# --------------------------------------------------

@app.callback(
    Output("age_hist", "figure"),
    [Input("trust", "value"),
        Input("date_range", "start_date"),
        Input("date_range", "end_date"),
     ])
def update_age_hist(trust, start_date, end_date):
    if trust == 'NATIONAL':
        df_slice = d_patient_age
    else:
        df_slice = d_patient_age[d_patient_age['trustname'] == trust]

    df_slice = df_slice[df_slice["ageyear"] > 0]
    df_slice = df_slice[(df_slice['hospitaladmissiondate'].dt.strftime('%Y-%m-%d') >= str(start_date)) &
                        (df_slice['hospitaladmissiondate'].dt.strftime('%Y-%m-%d') <= str(end_date))]
    med = df_slice.ageyear.median()
    fig = px.histogram(df_slice, x="ageyear", title="Distribution of Age", nbins=20,
                       color_discrete_sequence=px.colors.qualitative.Dark2)
    fig.update_layout(
        paper_bgcolor='White',
        plot_bgcolor='White',
        shapes=[{'line': {'color': 'Red', 'dash': 'dash', 'width': 1},
                 'type': 'line',
                 'x0': med,
                 'x1': med,
                 'xref': 'x',
                 'y0': 0.,
                 'y1': 1,
                 'yref': 'paper'}],
        annotations=[
            dict(
                x=med + 5,
                y=1.05,
                xref="x",
                yref="paper",
                text="Median: {} years".format(med),
                showarrow=False,
                ax=0,
                ay=0
            )
        ]
    )
    fig.update_yaxes(title_text='Number of Patients')
    fig.update_xaxes(title_text='Patient Age')

    # fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='Grey')

    return fig


def update_gender_pi(df_slice):
    m = df_slice['n_male'].sum()
    f = df_slice['n_female'].sum()

    labels = ['Male', 'Female']
    values = [m, f]

    # Use `hole` to create a donut-like pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])
    fig.update_layout(
        title="Sex",
    )
    fig.update_layout(template=template)

    return fig



def update_admission_pi(df_slice):
    with_comor = df_slice['comorbidity_gt1'].sum()
    no_comor = df_slice['newly_admitted'].sum() - with_comor

    labels = ['With Comorbidity', 'No Comorbidity']
    values = [with_comor, no_comor]

    # Use `hole` to create a donut-like pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])
    fig.update_layout(
        title="Patient with Comorbidities"
    )
    fig.update_layout(template=template)

    return fig


def update_num_comor_bar(df_slice):
    cols = ['c0', 'c1', 'c2', 'c3', 'c4', 'c_gt4']
    n_total = df_slice['newly_admitted'].sum()

    vals = df_slice[cols].sum().values
    text = ['{:.1%}'.format(x / n_total) for x in vals]

    fig = go.Figure(go.Bar(
        x=['0', '1', '2', '3', '4', '>4'],
        y=vals,
        text=text,
        textposition='auto',
    ))
    fig.update_layout(
        title="Distribution of Number of Comorbidities",
        paper_bgcolor='White',
        plot_bgcolor='White'
    )
    fig.update_yaxes(title_text='Number of Patients')
    fig.update_xaxes(title_text='Number of Comorbidities')
    fig.update_layout(template=template)

    # fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='Grey')
    return fig


def update_comor_bar(df_slice):
    cols = [
        'chonicrepiratory',
        'asthmarequiring',
        'chronicheart',
        'chronicrenal',
        'chronicliver',
        'isdiabetes',
        'immunosuppressiondisease',
        'obesityclinical',
        'hypertension',
        'pregnancy']

    names = [
        'C. Respiratory',
        'Asthma',
        'C. Heart',
        'C. Renal',
        'C. Liver',
        'Diabetes',
        'Immunosuppression',
        'Obesity',
        'Hypertension',
        'Pregnancy'
    ]
    n_total = df_slice['newly_admitted'].sum()
    vals = df_slice[cols].sum().values

    list_sorted = sorted(zip(vals, names))
    vals_s = [x for x, _ in list_sorted]
    names_s = [y for _, y in list_sorted]
    text = ['{:.1%}'.format(x / n_total) for x in vals_s]

    fig = go.Figure(go.Bar(
        x=vals_s,
        y=names_s,
        text=text,
        textposition='auto',
        orientation='h'
    ))
    fig.update_layout(
        title="Prevalence of Specific Comorbidity",
        paper_bgcolor='White',
        plot_bgcolor='White'
    )
    # fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='Grey')

    fig.update_xaxes(title_text='Number of Patients')
    fig.update_layout(template=template)

    return fig


@app.callback(
    Output("icu_bar", "figure"),
    [Input("trust", "value"),
     ])
def update_icu_bar(trust):
    df_slice = d_icu_stay_trust[d_icu_stay_trust['trustname'] == trust]
    df_slice = df_slice.rename(columns={'length_of_stay_type': 'ICU Admission Time'})
    if df_slice.shape[0] == 0:
        fig = px.bar(df_slice, x="dt", y="n", color_discrete_sequence=px.colors.qualitative.Dark2)
    else:
        fig = px.bar(df_slice, x="dt", y="n", color='ICU Admission Time',
                     color_discrete_sequence=px.colors.qualitative.Dark2)

    fig.update_layout(
        barmode='stack',
        title="Daily ICU Occupancy by ICU Admission Time",
        paper_bgcolor='White',
        plot_bgcolor='White'
    )
    fig.update_yaxes(title_text='Number of Patients')
    fig.update_xaxes(title_text='Date')
    fig.update_layout(template=template)

    return fig




@app.callback(
    Output("add_line", "figure"),
    [Input("trust", "value")
     ])
def update_add_line(trust):
    df_slice = d_ts[d_ts['trustname'] == trust]

    df_plot = df_slice[['dt', 'n_admission', 'n_discharge', 'n_death']]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot['dt'].values,
                             y=df_plot['n_admission'].values,
                             mode='lines+markers',
                             name='Newly Admitted to Hospital'))
    fig.add_trace(go.Scatter(x=df_plot['dt'].values,
                             y=df_plot['n_death'].values,
                             mode='lines+markers',
                             name='New Mortality'))
    fig.add_trace(go.Scatter(x=df_plot['dt'].values,
                             y=df_plot['n_discharge'].values,
                             mode='lines+markers',
                             name='Newly Discharged from Hospital'))

    # fig = px.line(df2, x="year", y="lifeExp", color='country')

    fig.update_layout(
        title="Daily New Hospital Admission, Mortality, and Discharge",
        paper_bgcolor='White',
        plot_bgcolor='White',
        hovermode="x"
    )
    fig.update_yaxes(title_text="Number of Patients")  # range=[0., .8],
    fig.update_xaxes(title_text='Date')
    fig.update_layout(template=template)

    return fig


@app.callback(
    [
        Output("gender_pi", "figure"),
        Output("admission_pi", "figure"),
        Output("num_comor_bar", "figure"),
        Output("comor_bar", "figure"),
    ],
    [
        Input("trust", "value"),
        Input("date_range", "start_date"),
        Input("date_range", "end_date"),
    ])
def update_d_trust(trust, start_date, end_date):
    df_slice = d_trust[(d_trust['trustname'] == trust) &
                       (d_trust['hospitaladmissiondate'].dt.strftime('%Y-%m-%d') >= str(start_date)) &
                       (d_trust['hospitaladmissiondate'].dt.strftime('%Y-%m-%d') <= str(end_date))]
    gender_pi = update_gender_pi(df_slice)
    admission_pi = update_admission_pi(df_slice)
    num_comor_bar = update_num_comor_bar(df_slice)
    comor_bar = update_comor_bar(df_slice)
    return gender_pi, admission_pi, num_comor_bar, comor_bar
#
# @app.callback(
#     [
#         Output("mortality_line", "figure"),
#         Output("icu_line", "figure"),
#     ],
#     [
#         Input("trust", "value"),
#         Input("date_range", "start_date"),
#         Input("date_range", "end_date"),
#         Input("date_forecast", "date"),
#     ])
# def update_forecast(trust, start_date, end_date, date_forecast):
#     df_slice = d_trust[(d_trust['trustname'] == trust) &
#                        (d_trust['hospitaladmissiondate'].dt.strftime('%Y-%m-%d') >= str(start_date)) &
#                        (d_trust['hospitaladmissiondate'].dt.strftime('%Y-%m-%d') <= str(end_date))]
#
#     df_icu = icu_forecast[
#                        (icu_forecast['hospitaladmissiondate'].dt.strftime('%Y-%m-%d') >= str(date_forecast)) &
#                        (icu_forecast['hospitaladmissiondate'].dt.strftime('%Y-%m-%d') <= str(end_date))]
#
#     df_death = death_forecast[
#                        (death_forecast['hospitaladmissiondate'].dt.strftime('%Y-%m-%d') >= str(date_forecast)) &
#                        (death_forecast['hospitaladmissiondate'].dt.strftime('%Y-%m-%d') <= str(end_date))]
#
#     mortality_line = update_mortality_line(df_slice, df_death)
#     icu_line = update_icu_line(df_slice, df_icu)
#     return mortality_line, icu_line
