
# Import required libraries
import os
import pickle
import copy
import datetime as dt
import math

import requests
import pandas as pd
from flask import Flask
import dash
import dash_daq as daq
import dash_table
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import dash_dangerously_set_inner_html
import numpy as np
import os
from os import path

from model.base_model import *
from model.R0forecast import * 
import train_R0forecaster

         
external_styles = [
{
    "href": "https://fonts.googleapis.com/css2?family=Open+Sans+Condensed:ital,wght@0,300;0,700;1,300&display=swap",
    "rel": "stylesheet"
},

{
    "href": "https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@100;300;400;500;700;900&display=swap",
    "rel": "stylesheet"
},

{
    "href": "https://fonts.googleapis.com/css2?family=Ubuntu&display=swap",
    "rel": "stylesheet"
}

]

app               = dash.Dash(__name__, external_stylesheets=external_styles)
server            = app.server
app.title         = "COVID-19 PIP"

POP_UP            = app.get_asset_url("transparent_PIP_logo.png") 

# Define theme color codes

LIGHT_PINK        = "#FF60AA"
DARK_GRAY         = "#323232"
GRAY              = "#808080"
CYAN              = "#95E3FA"

PURPLE_COLOR      = "#AF1CF7"
DARK_PINK         = "#CA1A57"

model_dates       = "/2020-10-08"

#-------------------------------------------------------
'''
Helper functions for style formating and data processing

List of helper functions >>
---------------------------
_get_input_HTML_format :: returns cell formating for 
                          html/dcc numerical input 

_get_radioItems_HTML_format :: returns a radio items 
                               list for display
'''
#-------------------------------------------------------

# TO DO: Attack rate plot

'''
COUNTRIES         = ["United States", "United Kingdom", "Italy", "Germany", "Spain", 
                     "Australia", "Brazil", "Canada", "Sweden", "Norway",  "Finland", 
                     "Estonia", "Egypt", "Japan", "Croatia"]
'''

COUNTRIES        = ["United Kingdom"] #["United States", "United Kingdom", "Italy", "Germany", "Brazil", "Japan"] #, "Egypt"]

# load models and data for all countries 

if path.exists(os.getcwd() + "/PIPmodels/global_models"):

  global_models  = pickle.load(open(os.getcwd() + "/PIPmodels/global_models", 'rb'))

else:

  global_models  = dict.fromkeys(COUNTRIES)

  for country in COUNTRIES:

    global_models[country] = pickle.load(open(os.getcwd() + model_dates + "/models/" + country, 'rb'))

  pickle.dump(global_models, open(os.getcwd() + "/PIPmodels/global_models", 'wb'))

if path.exists(os.getcwd() + "/PIPmodels/country_data"+"_"+str(dt.date.today())):

  country_data   = pickle.load(open(os.getcwd() + "/PIPmodels/country_data"+"_"+str(dt.date.today()), 'rb'))

else:

  country_data = get_COVID_DELVE_data(COUNTRIES)

  pickle.dump(country_data, open(os.getcwd() + "/PIPmodels/country_data", 'wb'))

 
if path.exists(os.getcwd() + "/PIPmodels/projections"+"_"+str(dt.date.today())):

  global_projections = pickle.load(open(os.getcwd() + "/PIPmodels/projections"+"_"+str(dt.date.today()), 'rb'))

else:

  global_projections = dict.fromkeys(COUNTRIES)

  for country in COUNTRIES:

    global_projections[country] = pickle.load(open(os.getcwd() + model_dates + "/projections/" + country, 'rb'))

  pickle.dump(global_models, open(os.getcwd() + "/PIPmodels/global_projections", 'wb'))


npi_model         = pickle.load(open(os.getcwd() + "/PIPmodels/R0Forecaster", 'rb'))

TARGETS           = ["Daily Deaths", "Cumulative Deaths", "Reproduction Number"]

COUNTRY_LIST      = [{'label': COUNTRIES[k], 'value': COUNTRIES[k], "style":{"margin-top":"-.3em", "align": "center"}} for k in range(len(COUNTRIES))]

#TARGET_LIST       = [{'label': TARGETS[k], 'value': k, "style":{"margin-top":"-.3em", "align": "center"}} for k in range(len(TARGETS))]
TARGET_LIST       = [{'label': TARGETS[0], 'value': 0, "style":{"margin-top":"-.3em", "align": "center"}},
                     {'label': TARGETS[2], 'value': 2, "style":{"margin-top":"-.3em", "align": "center"}}]


BOX_SHADOW        = "1px 2px 3px 4px #ccc" 
MARGIN_INPUT      = "20px"
PANEL_COLOR       = "#FBF8F8"

TITLE_STYLE       = {"marginBottom": ".25em", "margin-top": "1em", "margin-left": MARGIN_INPUT, "color":DARK_GRAY, "font-weight": "bold", 
                     "font-size": "12", "font-family": "Noto Sans JP"}
SUBTITLE_STYLE    = {"color":DARK_PINK, "font-size": 13}
SUBTITLE_STYLE_   = {"margin-top":"10px", "color":DARK_PINK, "font-size": 13}
PANEL_TEXT_STYLE  = {"marginBottom": ".25em", "margin-top": "0em", "margin-left": MARGIN_INPUT, "color":GRAY, "font-size": "11px", 
                     "font-style": "italic", "font-family":"Noto Sans JP"}
PANEL_TEXT_STYLE2 = {"marginBottom": ".25em", "margin-top": "0em", "margin-left": MARGIN_INPUT, "color":GRAY, "font-size": "12px", 
                     "font-family":"Noto Sans JP"}
PANEL_TEXT_STYLE3 = {"marginBottom": ".25em", "margin-top": "0em", "margin-left": MARGIN_INPUT, "color":GRAY, "font-size": "12px", 
                     "font-family":"Noto Sans JP", "font-weight":"bold"} 
PANEL_TEXT_STYLE4 = {"marginBottom": ".25em", "margin-top": "0em", "margin-left": MARGIN_INPUT, "margin-right": MARGIN_INPUT, "color":GRAY, 
                     "font-size": "12px", "font-family":"Noto Sans JP", "font-weight":"bold"}                     
PANEL_TEXT_STYLE_ = {"marginBottom": "0em", "margin-top": "0em", "color":DARK_GRAY, "font-size": "13px", "font-family":"Open Sans Condensed"}

CAPTION_STYLE     = {"color":"#4E4646", "font-size": 10}
BULLET_STYLE_0    = {"color":"#4E4646", "text-shadow":"#4E4646", "background-color":"#4E4646", "border-radius": "10%", "font-size": 10, "width":"7px", "margin-right":"10px"}
BULLET_STYLE_1    = {"color":"#4F27EC", "text-shadow":"#4F27EC", "background-color":"#4F27EC", "border-radius": "10%", "font-size": 10, "width":"7px", "margin-right":"10px"}
BULLET_STYLE_2    = {"color":"#AF1CF7", "text-shadow":"#AF1CF7", "background-color":"#AF1CF7", "border-radius": "10%", "font-size": 10, "width":"7px", "margin-right":"10px"}
BULLET_STYLE_3    = {"color":"#F71C93", "text-shadow":"#F71C93", "background-color":"#F71C93", "border-radius": "10%", "font-size": 10, "width":"7px", "margin-right":"10px"}

name_style        = dict({"color": "#4E4646", 'fontSize': 13, "width": "150px", "marginBottom": ".5em", "textAlign": "left", "font-family": "Noto Sans JP"})
name_style_       = dict({"color": "#4E4646", 'fontSize': 13, "width": "250px", "marginBottom": ".5em", "textAlign": "left", "font-family": "Noto Sans JP"})
input_style       = dict({"width": "100px", "height": "30px", "columnCount": 1, "textAlign": "center", "marginBottom": "1em", "font-size":12, "border-color":LIGHT_PINK})
form_style        = dict({'width' : '10%', 'margin' : '0 auto'})
radio_style       = dict({"width": "150px", "color": "#524E4E", "columnCount": 3, "display": "inline-block", "font-size":11, "border-color":LIGHT_PINK})
radio_style_short = dict({"width": "110px", "color": "#524E4E", "columnCount": 3, "display": "inline-block", "font-size":11})
radio_style_long  = dict({"width": "450px", "color": GRAY, "columnCount": 6, "display": "inline-block", "font-size":11, "font-family": "Noto Sans JP"})
name_style_long   = dict({"color": "#4E4646", 'fontSize': 13, "width": "450px", "columnCount": 3, "marginBottom": ".5em", "textAlign": "left"})
radio_style_her2  = dict({"width": "150px", "color": "#524E4E", "columnCount": 3, "display": "inline-block", "font-size":11})
name_style_her2   = dict({"color": "#4E4646", 'fontSize': 13, "width": "120px", "columnCount": 1, "marginBottom": ".5em", "textAlign": "left"})


npi_variables     = ["npi_workplace_closing", "npi_school_closing", "npi_cancel_public_events",  
                     "npi_gatherings_restrictions", "npi_close_public_transport", "npi_stay_at_home", 
                     "npi_internal_movement_restrictions", "npi_international_travel_controls", "npi_masks"]


def _get_input_HTML_format(name, ID, name_style, input_range, input_step, placeholder, input_style):

    _html_input = html.P(children=[html.Div(name, style=name_style), 
                                      dcc.Input(placeholder=placeholder, type='number', 
                                        min=input_range[0], max=input_range[1], step=input_step, 
                                        style=input_style, id=ID)])

    return _html_input


def _get_radioItems_HTML_format(name, ID, name_style, options, radio_style):

    _html_radioItem = html.P(children=[html.Div(name, style=name_style), 
                                       dcc.RadioItems(options=options, value=1, style=radio_style, id=ID)])

    return _html_radioItem


def _get_toggle_switch(name, name_style, color_style, ID):

    _html_toggle    = html.P(children=[html.Div(name, style=name_style),
                                       daq.ToggleSwitch(color=color_style, size=30, value=True,  
                                                        label=['No', 'Yes'], style={"font-size":9, "font-family": "Noto Sans JP", "color":GRAY}, id=ID)], 
                                                        style={"width": "100px", "font-size":9})

    return _html_toggle

def HORIZONTAL_SPACE(space_size):
    
    return dbc.Row(dbc.Col(html.Div(" ", style={"marginBottom": str(space_size) + "em"})))

def VERTICAL_SPACE(space_size):

    return dbc.Col(html.Div(" "), style={"width": str(space_size) + "px"})

#-------------------------------------------------------
'''
App layout components

List of layout components >>
---------------------------
HEADER :: Logo display and navigation buttons on the app
          header area 

PATIENT_INFO_FORM :: form that reads patient information
                     to compute and display risk
'''
#-------------------------------------------------------


# Create the **header** with logo and navigation buttons
#-------------------------------------------------------

LEARN_BUTTON    = html.A(dbc.Button("Learn More", style={"bgcolor": "gray"}), href="https://www.vanderschaar-lab.com/policy-impact-predictor-for-covid-19/", className="two columns")
WEBSITE_BUTTON  = html.A(dbc.Button("Go back to website", style={"bgcolor": "gray"}), href="https://www.vanderschaar-lab.com/policy-impact-predictor-for-covid-19/", className="two columns")
FEEDBACK_BUTTON = html.A(dbc.Button("Send Feedback", style={"bgcolor": "gray"}), href="https://www.vanderschaar-lab.com/contact-us/", className="two columns")
GITHUB_BUTTON   = html.A(dbc.Button("GitHub", style={"bgcolor": "gray"}), href="https://www.vanderschaar-lab.com/contact-us/", className="two columns")
UPDATE_BUTTON   = dbc.Button("Reset to Current Policy", style={"bgcolor": "gray"}, id="updatebutton")


HEADER  = html.Div([

      html.Div(
        [ 
        
        dbc.Row([dbc.Col(html.Img(src=app.get_asset_url("logo.png"), id="adjutorium-logo", style={"height": "100px", 'textAlign': 'left',
                                                                                                  "width": "auto",})),
                 VERTICAL_SPACE(325),   
                 dbc.Col(LEARN_BUTTON),
                 VERTICAL_SPACE(20),
                 dbc.Col(WEBSITE_BUTTON),
                 VERTICAL_SPACE(20),
                 dbc.Col(FEEDBACK_BUTTON), 
                 VERTICAL_SPACE(20),
                 dbc.Col(GITHUB_BUTTON)]),         
       
        ], style={"margin-left":"5ex"}, className="header"),

    ],

)


# Create the *Patient Information form* for app body
# -------------------------------------------------- 

# Input, name & HTML form styling dictionaries


COUNTRY_DROPMENU  = dcc.Dropdown(id='country', options= COUNTRY_LIST, value="United Kingdom",  
                                 placeholder="  ", style={"width":"150px", "height": "30px", "font-size": 11, "border-color":GRAY, "color":GRAY,
                                                          "font-color":GRAY, "margin-top":"-.1em", "textAlign": "left", "font-family": "Noto Sans JP", 
                                                          "vertical-align":"top", "display": "inline-block"}) 

REGION_DROPMENU   = dcc.Dropdown(id='region', options= COUNTRY_LIST, disabled=True, 
                                 placeholder="  ", style={"width":"150px", "height": "30px", "font-size": 11, "border-color":GRAY, "color":GRAY,
                                                          "font-color":GRAY, "margin-top":"-.1em", "textAlign": "left", "font-family": "Noto Sans JP", 
                                                          "vertical-align":"top", "display": "inline-block"}) 

TARGET_DROPMENU   = dcc.Dropdown(id='target', options= TARGET_LIST, value=0, 
                                 placeholder="  ", style={"width":"150px", "height": "30px", "font-size": 11, "border-color":GRAY, "color":GRAY,
                                                          "font-color":GRAY, "margin-top":"-.1em", "textAlign": "left", "font-family": "Noto Sans JP", 
                                                          "vertical-align":"top", "display": "inline-block"}) 

HORIZON_SLIDER    = dcc.Slider(id='horizonslider', marks={7: "1w", 30: "1m", 60: "2m", 90: "3m"}, min=7, 
                               max=90, value=30, step=1, updatemode="drag", tooltip={"always_visible":False})


MASK_SLIDER       = dcc.RadioItems(id='maskslider',
                                   options=[{'label': 'No policy measures', 'value': 0}, 
                                            {'label': 'Recommended', 'value': 1},
                                            {'label': 'Limited mandate', 'value': 2},
                                            {'label': 'Universal', 'value': 3}], value=1, 
                                   labelStyle={"display": "inline-block", "font-size": 11,
                                               "font-family": "Noto Sans JP", "color":GRAY, "width":"50%"},
                                   inputStyle={"color":CYAN}) 


SOCIAL_DIST_OPT   = dcc.Checklist(id='socialdistance',
                                  options=[{'label': 'Workplace closure', 'value': 0},
                                           {'label': 'Public events cancellation', 'value': 1},
                                           {'label': 'Public transport closure', 'value': 2},
                                           {'label': 'Gatherings restrictions', 'value': 3},
                                           {'label': 'Shelter-in-place' , 'value': 4},
                                           #{'label': 'Internal movement restrictions' , 'value': 5},
                                           {'label': 'Travel restrictions' , 'value': 6}],
                                  value=[0],
                                  labelStyle={"display": "inline-block", "font-size": 11,
                                              "font-family": "Noto Sans JP", "color":GRAY, "width":"50%"}) 


DISPLAY_LIST_2    = dcc.Checklist(options=[{'label': 'Show PIP model fit', 'value': 1}],
                                  labelStyle={"font-size": 11, "font-family": "Noto Sans JP", "color":GRAY, 'display': 'inline-block'},
                                  id="pipfit") 


DISPLAY_LIST_3    = dcc.Checklist(options=[{'label': 'Show confidence intervals', 'value': 1}], 
                                  value=[1],
                                  labelStyle={"font-size": 11, "font-family": "Noto Sans JP", "color":GRAY, 'display': 'inline-block'},
                                  id="confidenceint") 


DISPLAY_LIST_4    = dcc.Checklist(options=[{'label': 'Logarithmic scale', 'value': 1}], 
                                  labelStyle={"font-size": 11, "font-family": "Noto Sans JP", "color":GRAY, 'display': 'inline-block'},
                                  id="logarithmic") 

Num_days          = (dt.date.today() - dt.date(2020, 1, 1)).days 
BEGIN_DATE        = dcc.Slider(id='dateslider', marks={0: "Jan 1st, 2020", Num_days: "Today"}, 
                               min=0, max=Num_days, value=0, step=1, updatemode="drag", tooltip={"always_visible":False})

HORIZON_NOTE      = "*w = week, m = month." 
REQUEST_NOTE      = "Select a geographical location and the required forecast." 
REQUEST_NOTE_2    = "Select the non-pharmaceutical interventions (NPIs) to be applied in the geographical area selected above." 

COUNTRY_SELECT    = html.P(children=[html.Div("Country", style=name_style), COUNTRY_DROPMENU])
REGION_SELECT     = html.P(children=[html.Div("Region", style=name_style), REGION_DROPMENU])
TARGET_SELECT     = html.P(children=[html.Div("Forecast Target", style=name_style), TARGET_DROPMENU])
HORIZON_SELECT    = html.P(children=[html.Div("Forecast Days*", style=name_style), HORIZON_SLIDER])
MASK_SELECT       = html.P(children=[html.Div("Mask Policy", style=name_style), MASK_SLIDER])
SOCIAL_SELECT     = html.P(children=[html.Div("Social Distancing Measures", style=name_style_), SOCIAL_DIST_OPT])
BEGIN_SELECT      = html.P(children=[html.Div("View from", style=PANEL_TEXT_STYLE4), BEGIN_DATE]) 
SCHOOL_CLOSURE    = _get_toggle_switch(name="School Closure ", name_style=name_style, color_style=CYAN, ID="school_closure")


PATIENT_INFO_FORM = html.Div(
    [

      html.Div(
        [ 
        
        dbc.Row(dbc.Col(html.Div("Forecast Settings", style={"marginBottom": "0.5em", "margin-top": "1em", "margin-left": MARGIN_INPUT, 
                                                               "color":DARK_GRAY, "font-weight": "bold", "font-size": "11", "font-family": 'Noto Sans JP'}))),
        dbc.Row(dbc.Col(html.Div(REQUEST_NOTE, style=PANEL_TEXT_STYLE2))),
        HORIZONTAL_SPACE(1),
        dbc.Row(
            [
                dbc.Col(COUNTRY_SELECT), 
                VERTICAL_SPACE(40),
                dbc.Col(REGION_SELECT),  
                
            ], style={"margin-left": "40px"}
               ),
        HORIZONTAL_SPACE(1),
        dbc.Row(
            [   dbc.Col(TARGET_SELECT),
                VERTICAL_SPACE(40),
                dbc.Col(HORIZON_SELECT), 
            ], style={"margin-left": "40px"}
               ),
        HORIZONTAL_SPACE(.5),
        dbc.Row([VERTICAL_SPACE(200), dbc.Col(html.Div(HORIZON_NOTE, style=PANEL_TEXT_STYLE))]),
        HORIZONTAL_SPACE(1),

        ],  style={"box-shadow": BOX_SHADOW, "margin": MARGIN_INPUT, "background-color": PANEL_COLOR, "width": "450px"}), 

    html.Div(
        [     
        dbc.Row(dbc.Col(html.Div("Policy Scenario", style={"marginBottom": "1em", "margin-top": "1em", "margin-left": MARGIN_INPUT,
                                                                    "color":DARK_GRAY, "font-weight": "bold", "font-size": "11", "font-family":'Noto Sans JP'}))), 
        dbc.Row(dbc.Col(html.Div(REQUEST_NOTE_2, style=PANEL_TEXT_STYLE2))),
        HORIZONTAL_SPACE(1),
        dbc.Row(
            [   
                VERTICAL_SPACE(40),
                dbc.Col(SCHOOL_CLOSURE), 
                VERTICAL_SPACE(60),
                dbc.Col(MASK_SELECT),
            ], style={"margin-left": MARGIN_INPUT}
               ),
        HORIZONTAL_SPACE(.5),
        dbc.Row(
            [   
                VERTICAL_SPACE(25),
                dbc.Col(SOCIAL_SELECT), 
            ], style={"margin-left": MARGIN_INPUT}
               ),
        HORIZONTAL_SPACE(1),
        dbc.Row(
            [   
                VERTICAL_SPACE(80),
                dbc.Col(UPDATE_BUTTON),
            ], style={"margin-left": MARGIN_INPUT}
               ),
        HORIZONTAL_SPACE(1.5),
        ], style={"box-shadow": BOX_SHADOW, "margin": MARGIN_INPUT, "background-color": PANEL_COLOR, "width": "450px"}),

    ],

)

# Create the results display panel

CAUTION_STATEMENT = "Disclaimer: PIP uses machine learning to predict the most likely trajectory of COVID-19 deaths based on current knowledge and data, but will not provide 100% accurate predictions. Click on the 'Learn more' button to read our model's assumptions and limitations."


RESULTS_DISPLAY   = html.Div(
    [

      html.Div( 
        [  

        dbc.Row(dbc.Col(html.Div("COVID-19 Forecasts", style=TITLE_STYLE))),
        dbc.Row(dbc.Col(html.Div(CAUTION_STATEMENT, style=PANEL_TEXT_STYLE2))),
        HORIZONTAL_SPACE(.5),
        dbc.Row([dbc.Col(html.Div("Display Options", style=PANEL_TEXT_STYLE3)), VERTICAL_SPACE(10), DISPLAY_LIST_2,
                 VERTICAL_SPACE(10), DISPLAY_LIST_3, VERTICAL_SPACE(10), DISPLAY_LIST_4, VERTICAL_SPACE(60), BEGIN_SELECT]), 
        HORIZONTAL_SPACE(4), # used to be 2
        dbc.Row(html.Div(dcc.Graph(id="covid_19_forecasts", config={'displayModeBar': False}), style={"marginBottom": ".5em", "margin-top": "0em", "margin-left": MARGIN_INPUT})),                          
        HORIZONTAL_SPACE(5.5), # used to be 1.25
        ],  style={"box-shadow": BOX_SHADOW, "margin": MARGIN_INPUT, "background-color": PANEL_COLOR, "width": "800px"}),

    ]

)

#-----------------------------------------------------
'''
APP Layout: contains the app header, the information
form and the displayed graphs
'''
#-----------------------------------------------------

          #<h2 class="modal__title" id="modal-1-title">
          #  Micromodal
          #</h2>

popup = html.Div([
    dash_dangerously_set_inner_html.DangerouslySetInnerHTML('''
         <div class="modal micromodal-slide" id="modal-1" aria-hidden="true">
    <div class="modal__overlay" tabindex="-1" data-micromodal-close>
      <div class="modal__container" role="dialog" aria-modal="true" aria-labelledby="modal-1-title">
        <header class="modal__header">
          <div class="image-wrapper">
            <img src="assets/transparent_PIP_logo.png" style="width:100%;" alt="image">
          </div>
          <button class="modal__close" aria-label="Close modal" data-micromodal-close></button>
        </header>
        <main class="modal__content" id="modal-1-content">
          <p>
            PIP is an online tool that uses machine learning to predict the impact of non-pharmaceutical policy measures on the future trajectory of COVID-19 deaths. The model is designed and trained based on current knowledge and data, and does not provide 100% accurate predictions. Please make sure to discuss the projections of PIP with your local health officials and experts. Visit our <a href="https://www.vanderschaar-lab.com/policy-impact-predictor-for-covid-19/"> website </a> to learn more about our model's assumptions and limitations.
          </p>
        </main>
        <footer class="modal__footer">
          <button class="modal__btn" style="font-size:12px" data-micromodal-close aria-label="Close this dialog window">Start using PIP now!</button>
        </footer>
      </div>
    </div>
  </div>
    '''),
]) 

app.layout = html.Div([popup, HEADER, html.Div([PATIENT_INFO_FORM, RESULTS_DISPLAY], className="row app-center")])
 
@app.callback(
    Output("covid_19_forecasts", "figure"),
    [Input("target", "value"), Input("horizonslider", "value"), Input("maskslider", "value"), Input("country", "value"),
     Input("pipfit", "value"), Input("confidenceint", "value"), Input("dateslider", "value"), Input("socialdistance", "value"),
     Input("school_closure", "value"), Input("logarithmic", "value")]) 


def update_risk_score(target, horizonslider, maskslider, country, pipfit, confidenceint, dateslider, socialdistance, school_closure, logarithmic):


    """
    Set X and Y axes based on input callbacks             

    """

    SHOW_PIP_FIT      = False
    SHOW_LOG          = False
    SHOW_CONFIDENCE   = True

    if type(logarithmic)==list:

      if len(logarithmic) > 0:

        SHOW_LOG  = True

    if type(pipfit)==list:

      if len(pipfit) > 0:

        SHOW_PIP_FIT  = True


    if type(confidenceint) !=list or len(confidenceint)==0:   
      
      SHOW_CONFIDENCE = False    

    Y_AXIS_NAME       = TARGETS[target] 
    TODAY_DATE        = dt.datetime.today()
    BEGIN_YEAR        = dt.datetime(2020, 1, 1)
    DAYS_TILL_TODAY   = (TODAY_DATE - BEGIN_YEAR).days
    END_DATE          = TODAY_DATE + dt.timedelta(days=horizonslider)
    START_DATE        = BEGIN_YEAR + dt.timedelta(days=dateslider)
    DATE_RANGE        = pd.date_range(start=START_DATE, end=END_DATE) 
    TOTAL_NUM_DAYS    = len(DATE_RANGE)
    TRUE_DEATHS_DATES = pd.date_range(start=START_DATE, end=TODAY_DATE)
    FORECAST_DATES    = pd.date_range(start=TODAY_DATE + dt.timedelta(days=1), end=END_DATE)
    MAX_HORIZON       = 120 
    PLOT_RATIO        = 0.1

    predictive_model  = global_models[country]
    country_DELVE_dat = country_data[country]
    deaths_true       = country_DELVE_dat["Daily deaths"]
    NPI_data          = country_data[country]["NPI data"]

    deaths_true[deaths_true < 0] = 0
    deaths_smooth                = smooth_curve_1d(deaths_true)
    cumulative_deaths            = np.cumsum(deaths_true)

    if maskslider==0:

      deaths_pred, _, R_t = predictive_model.predict(DAYS_TILL_TODAY + MAX_HORIZON, R0_forecast=1*np.ones(MAX_HORIZON))
      deaths_forecast     = deaths_pred[DAYS_TILL_TODAY-1:DAYS_TILL_TODAY + horizonslider-1]
      PIP_MODEL_FIT       = deaths_pred[:DAYS_TILL_TODAY-1] 

    else:
      
      deaths_forecast     = global_projections[country][0][DAYS_TILL_TODAY-1:DAYS_TILL_TODAY + horizonslider-1]
      PIP_MODEL_FIT       = global_projections[country][0][:DAYS_TILL_TODAY-1] 

    #####
    deaths_forecast       = global_projections[country][0][DAYS_TILL_TODAY-1:DAYS_TILL_TODAY + horizonslider-1]
    PIP_MODEL_FIT         = global_projections[country][0][:DAYS_TILL_TODAY-1]    
    #####

    R0_t_forecast         = global_projections[country][3][dateslider:DAYS_TILL_TODAY + horizonslider-1]
    deaths_CI_l           = global_projections[country][2][:horizonslider]
    deaths_CI_u           = global_projections[country][1][:horizonslider] 

    #deaths_CI             = 50 * np.ones(len(deaths_CI_u))

    #deaths_forecast_u     = deaths_forecast + deaths_CI
    #deaths_forecast_l     = deaths_forecast - deaths_CI


    # -------------------------------------------------------------------------------------------------------------------------------------------

    """
    Compute projections

    """

    npi_vars              = ["npi_workplace_closing", "npi_school_closing", "npi_cancel_public_events",  
                             "npi_gatherings_restrictions", "npi_close_public_transport", "npi_stay_at_home", 
                             "npi_internal_movement_restrictions", "npi_international_travel_controls", "npi_masks"]

    npi_policy                                       = dict.fromkeys(npi_vars)                         

    npi_policy['npi_workplace_closing']              = (np.sum(np.array(socialdistance)==0) > 0) * 2 + 1           #3 
    npi_policy['npi_cancel_public_events']           = (np.sum(np.array(socialdistance)==1) > 0) * 1 + 1           #2 
    npi_policy['npi_school_closing']                 = ((school_closure==True) | (school_closure=="Yes")) * 1 + 2  #3
    npi_policy['npi_close_public_transport']         = (np.sum(np.array(socialdistance)==2) > 0) * 2               #2
    npi_policy['npi_gatherings_restrictions']        = (np.sum(np.array(socialdistance)==3) > 0) * 4               #4

    npi_policy['npi_stay_at_home']                   = (np.sum(np.array(socialdistance)==4) > 0) * 3               #3
    #npi_policy['npi_internal_movement_restrictions'] = (np.sum(np.array(socialdistance)==5) > 0) * 2              #2
    npi_policy['npi_internal_movement_restrictions'] = 2
    npi_policy['npi_international_travel_controls']  = (np.sum(np.array(socialdistance)==6) > 0) * 2 + 2           #4  

    npi_policy['npi_masks']                          = maskslider * .5                                             #3
    npi_policy['stringency']                         = 0   

    (y_pred, y_pred_u, y_pred_l), (R0_frc, R0_frc_u, R0_frc_l) = npi_model.projection(days=MAX_HORIZON, npi_policy=npi_policy, country=country)

    #####deaths_forecast       = y_pred[DAYS_TILL_TODAY - 1 : DAYS_TILL_TODAY + horizonslider - 1]
    #####R0_t_forecast         = R0_frc[dateslider : DAYS_TILL_TODAY + horizonslider - 1] #smooth_curve_1d(R0_frc[dateslider : DAYS_TILL_TODAY + horizonslider - 1])
    cum_death_forecast    = np.cumsum(deaths_forecast) + np.sum(deaths_true)
    #####deaths_forecast_u     = y_pred_u[DAYS_TILL_TODAY - 1 : DAYS_TILL_TODAY + horizonslider - 1]
    #####deaths_forecast_l     = y_pred_l[DAYS_TILL_TODAY - 1 : DAYS_TILL_TODAY + horizonslider - 1]

    R0_factor             = np.mean(R0_frc[DAYS_TILL_TODAY+1:DAYS_TILL_TODAY+10]) / np.mean(R0_frc[DAYS_TILL_TODAY-10:DAYS_TILL_TODAY])

    current_R0            = global_projections[country][3][DAYS_TILL_TODAY-1]

    deaths_pred, _, R_t   = predictive_model.predict(DAYS_TILL_TODAY + MAX_HORIZON, R0_forecast=current_R0 * R0_factor * np.ones(MAX_HORIZON))
    deaths_forecast       = deaths_pred[DAYS_TILL_TODAY-1:DAYS_TILL_TODAY + horizonslider-1]
    R0_t_forecast         = R_t 


    # CIs

    deaths_pred_u, _, R_u = predictive_model.predict(DAYS_TILL_TODAY + MAX_HORIZON, R0_forecast=current_R0 * R0_factor * np.ones(MAX_HORIZON) + 0.1)
    deaths_pred_l, _, R_l = predictive_model.predict(DAYS_TILL_TODAY + MAX_HORIZON, R0_forecast=current_R0 * R0_factor * np.ones(MAX_HORIZON) - 0.1)

    deaths_forecast_u     = deaths_pred_u[DAYS_TILL_TODAY - 1 : DAYS_TILL_TODAY + horizonslider - 1]
    deaths_forecast_l     = deaths_pred_l[DAYS_TILL_TODAY - 1 : DAYS_TILL_TODAY + horizonslider - 1]

    deaths_forecast_l[deaths_forecast_l < 0] = 0 

    # -------------------------------------------------------------------------------------------------------------------------------------------

    if target==0:

      Y_MAX_VAL           = np.maximum(np.max(deaths_smooth), np.max(deaths_forecast[:DAYS_TILL_TODAY + horizonslider - 1])) #y_pred
      Y_MAX_VAL           = Y_MAX_VAL * (1 + PLOT_RATIO)

    elif target==1:
      
      Y_MAX_VAL           = np.max(cum_death_forecast) + np.max(deaths_CI_u) 
      Y_MAX_VAL           = Y_MAX_VAL * (1 + PLOT_RATIO)

    elif target==2:
      
      Y_MAX_VAL           = 6 


    if SHOW_LOG:

      deaths_forecast     = np.log10(deaths_forecast + 1)
      deaths_forecast_u   = np.log10(deaths_forecast_u + 1)
      deaths_forecast_l   = np.log10(deaths_forecast_l + 1)
      deaths_smooth       = np.log10(deaths_smooth + 1)
      deaths_true         = np.log10(deaths_true + 1)
      PIP_MODEL_FIT       = np.log10(PIP_MODEL_FIT + 1)
      Y_MAX_VAL           = np.log10(Y_MAX_VAL)

      if Y_AXIS_NAME=="Daily Deaths":

        Y_AXIS_NAME   = "Daily Deaths (Log-scale)"

    LINE_WIDTH        = 2
    LINE_WIDTH_       = 3
    _OPACITY_1        = 0.2
    _OPACITY_2        = 0.3
    _OPACITY_3        = 0.4
    COLOR_1           = ("#4F27EC", "rgba(79, 39, 236, " + str(_OPACITY_1)+")")
    COLOR_2           = ("#AF1CF7", "rgba(175, 28, 247, " + str(_OPACITY_2)+")")
    COLOR_3           = ("#F5B7B1", "rgba(245, 183, 177, " + str(_OPACITY_3)+")")   

    LINE_STYLE_0      = {"color":"#2C2B2D", "width":LINE_WIDTH, "dash": "dot"}
    LINE_STYLE_1      = {"color":COLOR_1[0], "width":LINE_WIDTH}
    LINE_STYLE_2      = {"color":COLOR_2[0], "width":LINE_WIDTH}
    LINE_STYLE_3      = {"color":COLOR_3[0], "width":LINE_WIDTH}
    LINE_STYLE_4      = {"color":GRAY, "width":LINE_WIDTH, "dash": "dot"}

    TRUE_DEATH_STYLE  = {"color":"red", "opacity":.25}
    PIP_FIT_STYLE     = {"color":PURPLE_COLOR, "symbol":"cross", "opacity":.5}
    SMTH_DEATH_STYLE  = {"color":"red", "width":LINE_WIDTH, "dash": "dot"}
    R0_STYLE          = {"color":PURPLE_COLOR, "width":LINE_WIDTH_}
    R0_STYLE_         = {"color":PURPLE_COLOR, "width":LINE_WIDTH_, "dash": "dot"}
    FORECAST_STYLE    = {"color":"black", "width":LINE_WIDTH_, "dash": "dot"}
    FORECAST_STYLE_   = {"color":COLOR_3[1], "width":LINE_WIDTH}

    pip_fit_dict      = {"x":TRUE_DEATHS_DATES, "y": PIP_MODEL_FIT[dateslider:], "mode":"markers", "marker":PIP_FIT_STYLE, 
                         "name": "PIP Model Fit"}

    deaths_true_dict  = {"x":TRUE_DEATHS_DATES, "y": deaths_true[dateslider:], "mode":"markers", "marker":TRUE_DEATH_STYLE, 
                         "name": "Daily Deaths"}

    death_smooth_dict = {"x":TRUE_DEATHS_DATES, "y": deaths_smooth[dateslider:], "mode":"lines", "line":SMTH_DEATH_STYLE, 
                         "name": "7-day Average Deaths"} 

    death_frcst_dict  = {"x":FORECAST_DATES, "y": deaths_forecast, "mode":"lines", "line":FORECAST_STYLE, 
                         "name": "Deaths Forecast"} 

    death_frcst_dictu = {"x":FORECAST_DATES, "y": deaths_forecast_u, "mode":"lines", "line":FORECAST_STYLE_, 
                         "fill":"tonextx", "fillcolor":COLOR_3[1], "name": "Deaths Forecast  (Upper)"}  

    death_frcst_dictl = {"x":FORECAST_DATES, "y": deaths_forecast_l, "mode":"lines", "line":FORECAST_STYLE_, 
                         "name": "Deaths Forecast (Lower)"}  

    cum_frcst_dictu   = {"x":FORECAST_DATES, "y": np.sum(deaths_true) + np.cumsum(deaths_CI_u), "mode":"lines", "line":FORECAST_STYLE_, 
                         "fill":"tonexty", "fillcolor":COLOR_3[1], "name": "Cumulative Deaths (Upper)"}  

    cum_frcst_dictl   = {"x":FORECAST_DATES, "y": np.sum(deaths_true) + np.cumsum(deaths_CI_l), "mode":"lines", "line":FORECAST_STYLE_, 
                         "name": "Cumulative Deaths (Lower)"}                                                                

    cum_frcst_dict    = {"x":FORECAST_DATES, "y": cum_death_forecast, "mode":"lines", "line":FORECAST_STYLE, 
                         "name": "Cumulative Deaths Forecast"}                                           

    cum_death_dict    = {"x":TRUE_DEATHS_DATES, "y": cumulative_deaths[dateslider:], "mode":"lines", "line":SMTH_DEATH_STYLE, 
                         "name": "Cumulative Deaths"} 

    R0_frcst_dict     = {"x":TRUE_DEATHS_DATES, "y": R0_t_forecast[dateslider:DAYS_TILL_TODAY], "mode":"lines", "line":R0_STYLE, 
                         "name": "R0_t"} 

    R0_pred_dict      = {"x":FORECAST_DATES, "y": R0_t_forecast[DAYS_TILL_TODAY-1:DAYS_TILL_TODAY + horizonslider-1], "mode":"lines", "line":R0_STYLE_, 
                         "name": "R0_t"}                      

    today_line        = {"x":[dt.date.today() for k in range(int(Y_MAX_VAL))], "y": np.linspace(0, int(Y_MAX_VAL), int(Y_MAX_VAL)), "mode":"lines", "line":LINE_STYLE_4, 
                         "name": "Forecast day"}                         

    if target==0:

      DATA_DICT       = [deaths_true_dict, death_smooth_dict, death_frcst_dict]

      if SHOW_CONFIDENCE:

        DATA_DICT     = DATA_DICT + [death_frcst_dictl, death_frcst_dictu]

      if SHOW_PIP_FIT:

        DATA_DICT     = DATA_DICT + [pip_fit_dict]

      DATA_DICT       = DATA_DICT + [today_line]  

    elif target==1:
      
      DATA_DICT       = [cum_death_dict, cum_frcst_dict]

      if SHOW_CONFIDENCE:

        DATA_DICT     = DATA_DICT + [cum_frcst_dictu, cum_frcst_dictl]

      DATA_DICT       = DATA_DICT + [today_line]  

    elif target==2:

      DATA_DICT       = [R0_frcst_dict, R0_pred_dict, today_line]

    plot_dict = {
        "data": DATA_DICT,
        "showlegend": False, 
        "layout": {
            "legend":{"x":-10, "y":0, "bgcolor": "rgba(0,0,0,0)", "font-size":8},
            "showlegend": False, 
            "font-size":11,
            "width":775,
            "height":383,
            "plot_bgcolor":PANEL_COLOR,
            "paper_bgcolor":PANEL_COLOR,
            "margin":dict(l=60, r=50, t=30, b=40),
            "fill":"toself", "fillcolor":"violet",
            "title":"<b> Confirmed deaths: </b>" + " "+ str(format(int(np.sum(deaths_true)), ",")) + " (as of today)  | <b> Projected total deaths: </b>" + " "+ str(format(int(np.ceil(cum_death_forecast[-1])), ",")) + "  by  " + END_DATE.strftime("%b %d, %Y"), 
            "titlefont":dict(size=13, color=GRAY, family="Noto Sans JP"),
            "xaxis":go.layout.XAxis(title_text="<b> Date </b>", type="date", tickvals=DATE_RANGE, dtick=10, tickmode="auto", 
                                    zeroline=False, titlefont=dict(size=12, color=GRAY, family="Noto Sans JP")),
            "yaxis":go.layout.YAxis(title_text="<b> " + Y_AXIS_NAME + " </b>", tickmode="auto", range=[0, Y_MAX_VAL], 
                                    titlefont=dict(size=12, color=GRAY, family="Noto Sans JP"))} 
    }

    return plot_dict


@app.callback(
    Output("socialdistance", "value"), 
    [Input("country", "value"), Input("updatebutton", "n_clicks")]) 


def update_NPIs(country, updatebutton):

  social_dist_measure = ["npi_workplace_closing", "npi_cancel_public_events", "npi_close_public_transport", 
                         "npi_gatherings_restrictions", "npi_stay_at_home", "npi_internal_movement_restrictions", 
                         "npi_international_travel_controls"]

  country_NPI_data    = country_data[country]["NPI data"][social_dist_measure].fillna(method="ffill")
  NPI_selections      = np.where(np.array(country_NPI_data)[-1, :]>0)[0]

  if (type(updatebutton)==list) | (type(updatebutton)==int):

    if updatebutton > 0:

      NPI_selections  = np.where(np.array(country_NPI_data)[-1, :]>0)[0]

  return list(NPI_selections)


@app.callback(
    Output("maskslider", "value"), 
    [Input("country", "value"), Input("updatebutton", "n_clicks")]) 

def update_mask_info(country, updatebutton):

  country_NPI_data    = country_data[country]["NPI data"]["npi_masks"].fillna(method="ffill")
  mask_selection      = int(np.array(country_NPI_data)[-1])

  if (type(updatebutton)==list) | (type(updatebutton)==int):

    if updatebutton > 0:

      mask_selection  = int(np.array(country_NPI_data)[-1])


  return mask_selection  

@app.callback(
    Output("school_closure", "value"), 
    [Input("country", "value"), Input("updatebutton", "n_clicks")]) 

def update_school_info(country, updatebutton):

  school_options      = ["No", "Yes"]
  country_NPI_data    = country_data[country]["NPI data"]["npi_school_closing"].fillna(method="ffill")
  school_closure      = school_options [(np.array(country_NPI_data)[-1] > 0) * 1]

  if (type(updatebutton)==list) | (type(updatebutton)==int):

    if updatebutton > 0:

      school_closure  = school_options [(np.array(country_NPI_data)[-1] > 0) * 1]


  return school_closure    
   

#-----------------------------------------------------
'''
Main

'''
#-----------------------------------------------------
if __name__ == '__main__':
    
    app.server.run(debug=True, threaded=True)

