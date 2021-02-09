import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pds
import scipy.stats as stats

from app import app, template
import helper


# Loading required resources


def get_proba(prob_ICU):
    prob = prob_ICU[:, 1:] - prob_ICU[:, :-1]

    prob = np.concatenate((prob_ICU[:, 0:1], prob), axis=1)
    return prob


df_icu = pds.read_csv('assets/icu_proba.csv', index_col=0)
df_death = pds.read_csv('assets/death_proba.csv', index_col=0)
df_discharge = pds.read_csv('assets/discharge_proba.csv', index_col=0)

df_enc = pds.read_csv('assets/samp_prob.csv')


prob_icu = get_proba(df_icu.values)
prob_death = get_proba(df_death.values)
prob_discharge = get_proba(df_discharge.values)

age_param = (0.005168220626061383, -3024.5774292434444, 3091.093418522737)

# Create forecast based on national average
# --------------------------------------------------

resource_form = dbc.Row(
    [
        dbc.Col(
            dbc.FormGroup(
                [
                    dbc.Label("Surge Capacity New ICU Beds", html_for="beds_input"),
                    dbc.Input(
                        type="number",
                        id="beds_input",
                        min=5,
                        max=5000,
                        step=1,
                        value=100,
                    ),
                ]
            ),
            width=3,
        ),
        dbc.Col(
            dbc.FormGroup(
                [
                    dbc.Label("Number of Newly Hospitalized Patients (Day 1)", html_for="patient_daily_input"),
                    dbc.Input(
                        type="number",
                        id="patient_daily_input",
                        min=5,
                        max=5000,
                        step=1,
                        value=50,
                    ),
                ]
            ),

            width=3,
        ),
        dbc.Col(
            dbc.FormGroup(
                [
                    dbc.Label("Trend for Patient Arrival", html_for="patient_schedule_input"),
                    dcc.Dropdown(
                        id="patient_schedule_input",
                        options=[
                            {"label": "Constant", "value": "1"},
                            {"label": "Exponential", "value": "2"},
                        ],
                        value="1",
                    ),
                ]
            ),
            width=3,
        ),
        dbc.Col(
            dbc.Fade(

                dbc.FormGroup(
                    [
                        dbc.Label("Exponential Rate", html_for="patient_rate_input"),
                        dbc.Input(
                            id="patient_rate_input",
                            type="number",
                            value=1.1,
                            min=0.5,
                            max=1.2,
                            step=0.01
                        ),
                    ]
                ),
                id="fade",
                is_in=False,
                appear=False,
            ),
            width=3,
        ),
    ],
    form=True,
)

patient_form = dbc.Row(
    [
        dbc.Col(
            dbc.FormGroup(
                [
                    dbc.Label("Median Age", html_for="age_input"),
                    dbc.Input(
                        type="number",
                        id="age_input",
                        min=45,
                        max=85,
                        step=1,
                        value=68,
                    ),
                ]
            ),
            width=3,
        ),
        dbc.Col(
            helper.get_input_form(
                label="Percentage Male",
                id="sex_input",
                min=10,
                max=90,
                step=1,
                value=63,
            ),
            width=3,
        ),
    ],
    form=True,
)

comor_form_row1 = dbc.Row(
    [
        dbc.Col(
            helper.get_input_form(
                label="Hypertension %",
                id="hypertension_input",
                min=1,
                max=90,
                step=1,
                value=14
            ),
        ),
        dbc.Col(
            helper.get_input_form(
                label="Diabetes %",
                id="diabetes_input",
                min=1,
                max=90,
                step=1,
                value=11
            ),
        ),
        dbc.Col(
            helper.get_input_form(
                label="Asthma %",
                id="asthma_input",
                min=1,
                max=90,
                step=1,
                value=7
            ),
        ),
        dbc.Col(
            helper.get_input_form(
                label="Obesity %",
                id="obesity_input",
                min=1,
                max=90,
                step=1,
                value=4
            ),
        ),
        dbc.Col(
            helper.get_input_form(
                label="Immunosuppression %",
                id="immuno_input",
                min=1,
                max=90,
                step=1,
                value=2
            ),
        ),

    ],
    form=True,
)

comor_form_row2 = dbc.Row(
    [
        dbc.Col(
            helper.get_input_form(
                label="C. Respiratory %",
                id="respiratory_input",
                min=1,
                max=90,
                step=1,
                value=6,
                icon_url=app.get_asset_url("organs/003-lungs.png")
            ),
        ),
        dbc.Col(
            helper.get_input_form(
                label="C. Heart %",
                id="heart_input",
                min=1,
                max=90,
                step=1,
                value=8,
                icon_url=app.get_asset_url("organs/023-heart.png")
            ),
        ),
        dbc.Col(
            helper.get_input_form(
                label="C. Renal %",
                id="renal_input",
                min=1,
                max=90,
                step=1,
                value=3,
                icon_url=app.get_asset_url("organs/008-kidneys-1.png")
            ),
        ),
        dbc.Col(
            helper.get_input_form(
                label="C. Liver %",
                id="liver_input",
                min=1,
                max=90,
                step=1,
                value=1,
                icon_url=app.get_asset_url("organs/028-liver.png")
            ),
        ),
        dbc.Col(
            helper.get_input_form(
                label="Pregnancy %",
                id="pregnancy_input",
                min=1,
                max=90,
                step=1,
                value=1,
                icon_url=app.get_asset_url("organs/018-fetus.png")
            ),
        ),
    ],
    form=True,
)

button = dbc.Row(
    dbc.Col(dbc.Button("Run", id="simulation_submit_button", color="primary", className="mr-2", size="lg"))
)

# sim_result = dbc.Row(
#     [
#         dbc.Col(html.Div([dcc.Graph(id="sim_line")])),
#     ],
# )

sim_result = html.Div([
    dbc.Row(
        [
            dbc.Col(html.Div([dcc.Graph(id="sim_line")])),
        ],
    ),
    dbc.Row(
        [
            dbc.Col(html.Div([dcc.Graph(id="death_line")])),
        ],
    ),
    dbc.Row(
        [
            dbc.Col(html.Div([dcc.Graph(id="discharge_line")])),
        ],
    ),
])

simulation_input = html.Div(
    [
        html.Div(dbc.Row(dbc.Col(
            html.Header(html.H2("In-Silico Demand Simulation (Advanced)")),
            style={'marginBottom': 25, 'marginTop': 25, 'text-align': 'center'})
        )),
        html.Div(dbc.Row(dbc.Col(
            html.Header(html.H3("Resource & Demand Config")),
            style={'marginBottom': 5, 'marginTop': 0, 'text-align': 'left'})
        )),
        resource_form,
        html.Div(dbc.Row(dbc.Col(
            html.Header(html.H3("Patient Config")),
            style={'marginBottom': 5, 'marginTop': 5, 'text-align': 'left'})
        )),
        html.Div(dbc.Row(dbc.Col(
            html.P("* Default values are national average."),
            style={'marginBottom': 5, 'marginTop': 5, 'text-align': 'left', "color": 'Blue'})
        )),
        patient_form,
        comor_form_row1,
        comor_form_row2,
        button,
        sim_result
    ]
)

layout = html.Div([
    simulation_input,
])


@app.callback(
    Output("fade", "is_in"),
    [Input("patient_schedule_input", "value")],
)
def toggle_fade(v):
    if v == "2":
        # Button has never been clicked
        return True
    return False


def get_prob(data_enc,
             age_param,
             age_input,
             sex_input,
             hypertension_input,
             diabetes_input,
             asthma_input,
             obesity_input,
             immuno_input,
             respiratory_input,
             heart_input,
             renal_input,
             liver_input,
             pregnancy_input):
    prob_vec = np.array([sex_input, hypertension_input, diabetes_input,
                         asthma_input, obesity_input, immuno_input,
                         respiratory_input, heart_input, renal_input,
                         liver_input, pregnancy_input]) * 1. / 100

    fit_alpha, fit_loc, fit_beta = age_param
    fit_loc += (age_input - 68)

    age_log_prob = stats.lognorm.logpdf(data_enc['ageyear'].values, fit_alpha, fit_loc, fit_beta)
    binary_data = data_enc.values[:, 2:-1]

    binary_log_prob = binary_data * np.log(prob_vec) + (1 - binary_data) * np.log(1 - prob_vec)
    binary_log_prob = np.sum(binary_log_prob, axis=1)
    total_log_prob = age_log_prob + binary_log_prob

    samp_prob = np.exp((total_log_prob - data_enc['samp_log_lik']).values)
    samp_prob = samp_prob / np.sum(samp_prob)
    return samp_prob


# example output
@app.callback([Output('sim_line', 'figure'),
               Output('death_line', 'figure'),
               Output('discharge_line', 'figure')],
              [Input('simulation_submit_button', 'n_clicks')],
              [
                  State('beds_input', 'value'),
                  State('patient_daily_input', 'value'),
                  State('patient_schedule_input', 'value'),
                  State('patient_rate_input', 'value'),
                  State('age_input', 'value'),
                  State('sex_input', 'value'),
                  State('hypertension_input', 'value'),
                  State('diabetes_input', 'value'),
                  State('asthma_input', 'value'),
                  State('obesity_input', 'value'),
                  State('immuno_input', 'value'),
                  State('respiratory_input', 'value'),
                  State('heart_input', 'value'),
                  State('renal_input', 'value'),
                  State('liver_input', 'value'),
                  State('pregnancy_input', 'value'),
              ])
def update_output(
        n_clicks,
        beds_input,
        patient_daily_input,
        patient_schedule_input,
        patient_rate_input,
        age_input,
        sex_input,
        hypertension_input,
        diabetes_input,
        asthma_input,
        obesity_input,
        immuno_input,
        respiratory_input,
        heart_input,
        renal_input,
        liver_input,
        pregnancy_input,
):
    if n_clicks is None:
        return ""
    else:
        probs = get_prob(df_enc,
                         age_param,
                         age_input,
                         sex_input,
                         hypertension_input,
                         diabetes_input,
                         asthma_input,
                         obesity_input,
                         immuno_input,
                         respiratory_input,
                         heart_input,
                         renal_input,
                         liver_input,
                         pregnancy_input)

        n_arr = get_patients(patient_daily_input, patient_schedule_input, patient_rate_input)
        n_itr = 100
        icu_sum, death_sum, discharge_sum = one_sim(n_itr, n_arr, prob_icu, prob_death, prob_discharge, probs)

        days = list(range(1, 16))

        fig_icu = get_figure(days,
                             *icu_sum,
                             main_label='Occupancy (Mean)',
                             title="Simulated ICU Occupancy",
                             yaxis="Number of Occupied ICU Beds")
        fig_icu.update_layout(
            shapes=[{'line': {'color': 'grey', 'dash': 'dash', 'width': 1},
                     'type': 'line',
                     'x0': 0.,
                     'x1': 1.,
                     'xref': 'paper',
                     'y0': beds_input,
                     'y1': beds_input,
                     'yref': 'y'}]
        )
        fig_death = get_figure(days,
                               *death_sum,
                               main_label='Total Death (Mean)',
                               title="Simulated Death Toll",
                               yaxis="Number of Deaths")
        fig_discharge = get_figure(days,
                                   *discharge_sum,
                                   main_label='Total Discharge (Mean)',
                                   title="Simulated Cumulative Discharge",
                                   yaxis="Number of Discharge")

    return fig_icu, fig_death, fig_discharge


def get_figure(days, m, upp, lor, main_label='Occupancy (Mean)', title="Simulated ICU Occupancy",
               yaxis="Number of Occupied ICU Beds"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days,
                             y=m,
                             mode='lines+markers',
                             name=main_label))

    fig.add_trace(go.Scatter(x=days,
                             y=upp,
                             mode='lines',
                             line=dict(dash='dot'),
                             name='95% Quantile'))
    fig.add_trace(go.Scatter(x=days,
                             y=lor,
                             mode='lines',
                             line=dict(dash='dot'),
                             name='5% Quantile'))
    fig.update_layout(
        title=title,
        paper_bgcolor='White',
        plot_bgcolor='White',
        hovermode="x",
    )
    fig.update_yaxes(title_text=yaxis)
    fig.update_xaxes(title_text='Days')
    fig.update_layout(template=template)
    return fig


def samp_one_person(p_icu, p_death, p_discharge, probs):
    inds = np.random.choice(p_icu.shape[0], p=probs, size=p_icu.shape[0], replace=True)
    p_icu = p_icu[inds, :]
    p_death = p_death[inds, :]
    p_discharge = p_discharge[inds, :]

    icu = np.random.binomial(1, p_icu)
    death = np.random.binomial(1, p_death)
    discharge = np.random.binomial(1, p_discharge)

    icu = np.cumsum(icu, axis=1) > 0
    death = np.cumsum(death, axis=1) > 0
    discharge = np.cumsum(discharge, axis=1) > 0

    icu_stay = icu.copy()
    icu_stay[death] = False
    icu_stay[discharge] = False

    return icu_stay * 1., death * 1., discharge * 1.


def get_one_icu_line(n_person, one_icu, one_death, one_discharge):
    n_person = n_person.astype(np.int)
    n_day = len(n_person)
    sums_icu = np.zeros(n_day)
    sums_death = np.zeros(n_day)
    sums_discharge = np.zeros(n_day)
    s = 0
    for i in range(n_day):
        row_sum = np.sum(one_icu[s:(s + int(n_person[i])), :], axis=0)
        sums_icu[i:] = sums_icu[i:] + row_sum[:(n_day - i)]

        row_sum = np.sum(one_death[s:(s + int(n_person[i])), :], axis=0)
        sums_death[i:] = sums_death[i:] + row_sum[:(n_day - i)]

        row_sum = np.sum(one_discharge[s:(s + int(n_person[i])), :], axis=0)
        sums_discharge[i:] = sums_discharge[i:] + row_sum[:(n_day - i)]

        s += n_person[i]
    return sums_icu, sums_death, sums_discharge


def create_summary(samps_icu):
    m = np.mean(samps_icu, axis=0)
    upp = np.quantile(samps_icu, 0.95, axis=0)
    lor = np.quantile(samps_icu, 0.05, axis=0)
    return m, upp, lor


def one_sim(n_itr, n_arr, prob_icu, prob_death, prob_discharge, probs):
    samps_icu = np.zeros((n_itr, 15))
    samps_death = np.zeros((n_itr, 15))
    samps_discharge = np.zeros((n_itr, 15))

    for i in range(n_itr):
        one_icu, one_death, one_discharge = samp_one_person(prob_icu, prob_death, prob_discharge, probs)
        icu, death, discharge = get_one_icu_line(n_arr, one_icu, one_death, one_discharge)

        samps_icu[i, :] = icu
        samps_death[i, :] = death
        samps_discharge[i, :] = discharge

    icu_sum = create_summary(samps_icu)
    death_sum = create_summary(samps_death)
    discharge_sum = create_summary(samps_discharge)
    return icu_sum, death_sum, discharge_sum


def get_patients(patient_daily_input, patient_schedule_input, patient_rate_input):
    arr = np.ones(15) * patient_daily_input
    if patient_schedule_input == "1":
        return arr

    for i in range(1, 15):
        arr[i] = arr[i - 1] * patient_rate_input
    arr = arr.astype(np.int)
    return arr
