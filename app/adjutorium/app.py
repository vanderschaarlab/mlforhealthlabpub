# Import required libraries
import os
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
import dash_table
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

from ADJUTORIUM_v1 import *


app = dash.Dash(__name__)
server = app.server
app.title = "Adjutorium Breast Cancer"

# Define theme color codes

LIGHT_PINK = "#FF60AA"

gadjutorium_paper = "https://www.nature.com/articles/s42256-021-00353-8"
gadjutorium_website = "https://www.vanderschaar-lab.com/adjutorium"
gadjutorium_feedback = "https://www.vanderschaar-lab.com/contact-us/"


# -------------------------------------------------------
"""
Helper functions for style formating and data processing

List of helper functions >>
---------------------------
_get_input_HTML_format :: returns cell formating for
                          html/dcc numerical input

_get_radioItems_HTML_format :: returns a radio items
                               list for display
"""
# -------------------------------------------------------


def _get_input_HTML_format(
    name, ID, name_style, input_range, input_step, placeholder, input_style
):

    _html_input = html.P(
        children=[
            html.Div(name, style=name_style),
            dcc.Input(
                placeholder=placeholder,
                type="number",
                min=input_range[0],
                max=input_range[1],
                step=input_step,
                style=input_style,
                id=ID,
            ),
        ]
    )

    return _html_input


def _get_radioItems_HTML_format(name, ID, name_style, options, radio_style):

    _html_radioItem = html.P(
        children=[
            html.Div(name, style=name_style),
            dcc.RadioItems(options=options, value=1, style=radio_style, id=ID),
        ]
    )

    return _html_radioItem


def _get_toggle_switch(name, name_style, color_style, ID):

    _html_toggle = html.P(
        children=[
            html.Div(name, style=name_style),
            daq.ToggleSwitch(
                color=color_style,
                size=30,
                value=False,
                label=["No", "Yes"],
                style={"font-size": 11},
                id=ID,
            ),
        ],
        style={"width": "100px", "font-size": 11},
    )

    return _html_toggle


def HORIZONTAL_SPACE(space_size):

    return dbc.Row(
        dbc.Col(html.Div(" ", style={"marginBottom": str(space_size) + "em"}))
    )


def VERTICAL_SPACE(space_size):

    return dbc.Col(html.Div(" "), style={"width": str(space_size) + "px"})


# -------------------------------------------------------
"""
App layout components

List of layout components >>
---------------------------
HEADER :: Logo display and navigation buttons on the app
          header area

PATIENT_INFO_FORM :: form that reads patient information
                     to compute and display risk
"""
# -------------------------------------------------------


# Create the **header** with logo and navigation buttons
# -------------------------------------------------------


LEARN_BUTTON = html.A(
    dbc.Button("NATURE MACHINE INTELLIGENCE PAPER", style={"bgcolor": "gray"}),
    href=gadjutorium_paper,
    className="two columns",
)
WEBSITE_BUTTON = html.A(
    dbc.Button("Website", style={"bgcolor": "gray"}),
    href=gadjutorium_website,
    className="two columns",
)
FEEDBACK_BUTTON = html.A(
    dbc.Button("Send Feedback", style={"bgcolor": "gray"}),
    href=gadjutorium_feedback,
    className="two columns",
)

HEADER = html.Div(
    [
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Img(
                                src=app.get_asset_url("logo.png"),
                                id="adjutorium-logo",
                                style={
                                    "height": "90px",
                                    "width": "auto",
                                },
                            )
                        ),
                        VERTICAL_SPACE(400),
                        dbc.Col(
                            LEARN_BUTTON,
                            style={
                                "margin-top": "2rem",
                            },
                        ),
                        dbc.Col(
                            WEBSITE_BUTTON,
                            style={
                                "margin-top": "2rem",
                            },
                        ),
                        dbc.Col(
                            FEEDBACK_BUTTON,
                            style={
                                "margin-top": "2rem",
                            },
                        ),
                    ],
                ),
            ],
            style={
                "display": "flex",
                "justify-content": "center",
                "align-items": "center",
            },
        ),  # style={"box-shadow": BOX_SHADOW, "margin": MARGIN_INPUT, "background-color": PANEL_COLOR, "width": "450px"}),
    ],
)

# Create the *Patient information form* for app body
# --------------------------------------------------

# Input, name & HTML form styling dictionaries

BOX_SHADOW = "1px 2px 3px 4px #ccc"  # "3px 3px 5px 6px #ccc" #"0 0 0 4px #ff0030, 2px 1px 6px 4px rgba(10, 10, 0, 0.5)"
MARGIN_INPUT = "20px"
PANEL_COLOR = "#FBF8F8"

name_style = dict(
    {
        "color": "#4E4646",
        "fontSize": 13,
        "width": "150px",
        "marginBottom": ".5em",
        "textAlign": "left",
        "font": "HelveticaNeue",
    }
)
input_style = dict(
    {
        "width": "100px",
        "height": "30px",
        "columnCount": 1,
        "textAlign": "center",
        "marginBottom": "1em",
        "font-size": 12,
        "border-color": LIGHT_PINK,
    }
)
form_style = dict({"width": "10%", "margin": "0 auto"})
radio_style = dict(
    {
        "width": "150px",
        "color": "#524E4E",
        "columnCount": 3,
        "display": "inline-block",
        "font-size": 11,
        "border-color": LIGHT_PINK,
    }
)
radio_style_short = dict(
    {
        "width": "110px",
        "color": "#524E4E",
        "columnCount": 3,
        "display": "inline-block",
        "font-size": 11,
    }
)
radio_style_long = dict(
    {
        "width": "450px",
        "color": "#524E4E",
        "columnCount": 6,
        "display": "inline-block",
        "font-size": 11,
    }
)
name_style_long = dict(
    {
        "color": "#4E4646",
        "fontSize": 13,
        "width": "450px",
        "columnCount": 3,
        "marginBottom": ".5em",
        "textAlign": "left",
    }
)
radio_style_her2 = dict(
    {
        "width": "150px",
        "color": "#524E4E",
        "columnCount": 3,
        "display": "inline-block",
        "font-size": 11,
    }
)
name_style_her2 = dict(
    {
        "color": "#4E4646",
        "fontSize": 13,
        "width": "120px",
        "columnCount": 1,
        "marginBottom": ".5em",
        "textAlign": "left",
    }
)

GRADE_OPTIONS = [
    {"label": "1", "value": 1},
    {"label": "2", "value": 2},
    {"label": "3", "value": 3},
]
ER_OPTIONS = [{"label": "Positive", "value": 1}, {"label": "Negative", "value": 0}]
HER2_OPTIONS = [
    {"label": "Positive", "value": 1},
    {"label": "Negative", "value": 0},
]
SCREEN_OPTIONS = [
    {"label": "Screening", "value": 1},
    {"label": "Symptoms", "value": 0},
]


AGE_DIAGNOSIS = _get_input_HTML_format(
    name="Age at diagnosis",
    ID="age",
    name_style=name_style,
    input_range=(30, 90),
    input_step=1,
    placeholder="---",
    input_style=input_style,
)

TUMOUR_SIZE = _get_input_HTML_format(
    name="Tumor size (mm)",
    ID="size",
    name_style=name_style,
    input_range=(0, 90),
    input_step=1,
    placeholder="---",
    input_style=input_style,
)

POSITIVE_NODES = _get_input_HTML_format(
    name="Positive nodes",
    ID="nodes",
    name_style=name_style,
    input_range=(0, 40),
    input_step=1,
    placeholder="---",
    input_style=input_style,
)

TUMOUR_GRADE = _get_radioItems_HTML_format(
    name="Tumor grade",
    ID="grade",
    name_style=name_style,
    options=GRADE_OPTIONS,
    radio_style=radio_style_short,
)

ER_STATUS = _get_radioItems_HTML_format(
    name="ER status",
    ID="ER",
    name_style=name_style,
    options=ER_OPTIONS,
    radio_style=radio_style,
)

HER2_STATUS = _get_radioItems_HTML_format(
    name="HER2 status",
    ID="HER2",
    name_style=name_style_her2,
    options=HER2_OPTIONS,
    radio_style=radio_style_her2,
)

SCREEN_MODE = _get_radioItems_HTML_format(
    name="Tumor detected by",
    ID="SCR",
    name_style=name_style_long,
    options=SCREEN_OPTIONS,
    radio_style=radio_style_long,
)

CHEMOTHERAPY = _get_toggle_switch(
    name="Chemotherapy", name_style=name_style, color_style="#F71C93", ID="CT"
)
HORMOTHERAPY = _get_toggle_switch(
    name="Hormone therapy", name_style=name_style, color_style="#AF1CF7", ID="HT"
)


HORIZON_OPTS = [
    {
        "label": str(k + 1),
        "value": k + 1,
        "style": {"margin-top": "-.5em", "align": "center"},
    }
    for k in range(10)
]

HORIZON_DROPMENU = dcc.Dropdown(
    id="t_horizon",
    options=HORIZON_OPTS,
    placeholder="  ",
    value="10",
    style={
        "width": "35px",
        "height": "25px",
        "font-size": 10,
        "margin-left": "4px",
        "margin-right": "10px",
        "align": "left",
        "margin-top": "-.1em",
        "textAlign": "left",
        "vertical-align": "top",
        "display": "inline-block",
        "border-color": LIGHT_PINK,
    },
)


PATIENT_INFO_FORM = html.Div(
    [
        html.Div(
            [
                dbc.Row(
                    dbc.Col(
                        html.Div(
                            "Patient information",
                            style={
                                "marginBottom": "1em",
                                "margin-top": "1em",
                                "margin-left": MARGIN_INPUT,
                                "color": "#CA1A57",
                                "font-size": "12",
                            },
                        )
                    )
                ),
                HORIZONTAL_SPACE(1),
                dbc.Row(
                    [
                        dbc.Col(AGE_DIAGNOSIS),
                        dbc.Col(TUMOUR_SIZE),
                        dbc.Col(POSITIVE_NODES),
                    ],
                    style={"margin-left": MARGIN_INPUT},
                ),
                HORIZONTAL_SPACE(2),
                dbc.Row(
                    [
                        dbc.Col(TUMOUR_GRADE),
                        dbc.Col(SCREEN_MODE),
                    ],
                    style={"margin-left": MARGIN_INPUT},
                ),
                HORIZONTAL_SPACE(2),
                dbc.Row(
                    [
                        dbc.Col(ER_STATUS),
                        dbc.Col(HER2_STATUS),
                    ],
                    style={"margin-left": MARGIN_INPUT},
                ),
                HORIZONTAL_SPACE(2),
            ],
            style={
                "box-shadow": BOX_SHADOW,
                "margin": MARGIN_INPUT,
                "background-color": PANEL_COLOR,
                "width": "450px",
            },
        ),
        html.Div(
            [
                dbc.Row(
                    dbc.Col(
                        html.Div(
                            "Adjuvant therapy options",
                            style={
                                "marginBottom": "1em",
                                "margin-top": "1em",
                                "margin-left": MARGIN_INPUT,
                                "color": "#CA1A57",
                                "font-size": "12",
                            },
                        )
                    )
                ),
                HORIZONTAL_SPACE(1),
                dbc.Row(
                    [
                        VERTICAL_SPACE(60),
                        dbc.Col(CHEMOTHERAPY),
                        VERTICAL_SPACE(80),
                        dbc.Col(HORMOTHERAPY),
                    ],
                    style={"margin-left": MARGIN_INPUT},
                ),
                HORIZONTAL_SPACE(1),
            ],
            style={
                "box-shadow": BOX_SHADOW,
                "margin": MARGIN_INPUT,
                "background-color": PANEL_COLOR,
                "width": "450px",
            },
        ),
    ],
)

# Create the results display panel
PURPLE_COLOR = "#AF1CF7"
DARK_PINK = "#CA1A57"

CAUTION_STATEMENT = "This demonstrator has been created solely to support the academic publication of Adjutorium, a new machine learning research methodology for survival benefit prediction in breast cancer. It is not suitable for use as a medical device and should not be used to make clinical decisions on the management of individual patients. To learn more about what Adjutorium does and how it works, please use the links above to view our dedicated webpage and/or the Nature Machine Intelligence article introducing Adjutorium."

TITLE_STYLE = {
    "marginBottom": ".25em",
    "margin-top": "1em",
    "margin-left": MARGIN_INPUT,
    "color": "#CA1A57",
    "font-size": "12",
}
SUBTITLE_STYLE = {
    "color": DARK_PINK,
    "font-size": 13,
    "justify-content": "center",
    "align-items": "center",
}
SUBTITLE_STYLE_ = {"margin-top": "10px", "color": DARK_PINK, "font-size": 13}
PANEL_TEXT_STYLE = {
    "marginBottom": ".25em",
    "margin-top": "0em",
    "margin-left": MARGIN_INPUT,
    "color": "#524E4E",
    "font-size": "12px",
}
PANEL_TEXT_STYLE_ = {
    "marginBottom": "0em",
    "margin-top": "0em",
    "color": "#524E4E",
    "font-size": "12px",
}

CAPTION_STYLE = {"color": "#4E4646", "font-size": 10}
BULLET_STYLE_0 = {
    "color": "#4E4646",
    "text-shadow": "#4E4646",
    "background-color": "#4E4646",
    "border-radius": "10%",
    "font-size": 10,
    "width": "7px",
    "margin-right": "10px",
}
BULLET_STYLE_1 = {
    "color": "#4F27EC",
    "text-shadow": "#4F27EC",
    "background-color": "#4F27EC",
    "border-radius": "10%",
    "font-size": 10,
    "width": "7px",
    "margin-right": "10px",
}
BULLET_STYLE_2 = {
    "color": "#AF1CF7",
    "text-shadow": "#AF1CF7",
    "background-color": "#AF1CF7",
    "border-radius": "10%",
    "font-size": 10,
    "width": "7px",
    "margin-right": "10px",
}
BULLET_STYLE_3 = {
    "color": "#F71C93",
    "text-shadow": "#F71C93",
    "background-color": "#F71C93",
    "border-radius": "10%",
    "font-size": 10,
    "width": "7px",
    "margin-right": "10px",
}


CAPTION_0 = html.Div(
    [
        html.Div(" . ", style=BULLET_STYLE_0),
        html.Div(
            "Survival rate excluding deaths from breast cancer", style=CAPTION_STYLE
        ),
    ],
    className="row",
)
CAPTION_1 = html.Div(
    [
        html.Div(" . ", style=BULLET_STYLE_1),
        html.Div("Surgery only", style=CAPTION_STYLE),
    ],
    className="row",
)
CAPTION_2 = html.Div(
    [
        html.Div(" . ", style=BULLET_STYLE_2),
        html.Div("Additional benefit of hormone therapy", style=CAPTION_STYLE),
    ],
    className="row",
)
CAPTION_3 = html.Div(
    [
        html.Div(" . ", style=BULLET_STYLE_3),
        html.Div("Additional benefit of chemotherapy", style=CAPTION_STYLE),
    ],
    className="row",
)

CAPTION_ = [CAPTION_0, CAPTION_1, CAPTION_2, CAPTION_3]

TABLE_RESULT = dash_table.DataTable(
    id="table",
    columns=[
        {"name": "Treatment", "id": "treatment"},
        {"name": "Additional Benefit", "id": "addBenefit"},
        {"name": "Overall Survival %", "id": "overall_survival"},
    ],
    style_cell={
        "font_family": "Arial",
        "font_size": 10,
        "text_align": "left",
        "width": "100px",
    },
    style_as_list_view=True,
)

MESSAGE_AREA = dcc.Textarea(
    placeholder="Enter a value...",
    value="This is a TextArea component",
    style={
        "width": "100%",
        "marginBottom": "0em",
        "margin-top": "0em",
        "color": "#524E4E",
        "font-size": "12px",
    },
)

RESULTS_DISPLAY = html.Div(
    [
        html.Div(
            [
                dbc.Row(dbc.Col(html.Div("Prediction results", style=TITLE_STYLE))),
                dbc.Row(dbc.Col(html.Div(CAUTION_STATEMENT, style=PANEL_TEXT_STYLE))),
                HORIZONTAL_SPACE(1),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    html.Div(
                                        "10-year patient survival profile",
                                        style=SUBTITLE_STYLE,
                                    ),
                                    dcc.Graph(id="survival_curve"),
                                ],
                                style={
                                    "marginBottom": ".5em",
                                    "margin-top": "0em",
                                    "margin-left": MARGIN_INPUT,
                                    "justify-content": "center",
                                    "align-items": "center",
                                },
                            )
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    CAPTION_,
                                    style={"margin-top": "5px", "margin-bottom": "3px"},
                                ),
                                html.Div(
                                    "Adjuvant therapy benefit", style=SUBTITLE_STYLE_
                                ),
                                html.Div(
                                    [
                                        "This table shows the expected survival chance for",
                                        HORIZON_DROPMENU,
                                        "years",
                                    ],
                                    style=PANEL_TEXT_STYLE_,
                                    className="row",
                                ),
                                html.Div(
                                    "after diagnosis based on the patient information.",
                                    style=PANEL_TEXT_STYLE_,
                                ),
                                html.Div(TABLE_RESULT, style={"margin-top": "1ex"}),
                                html.Div(id="messagereport"),
                            ]
                        ),
                    ]
                ),
            ],
            style={
                "box-shadow": BOX_SHADOW,
                "margin": MARGIN_INPUT,
                "background-color": PANEL_COLOR,
                "width": "800px",
            },
        ),
    ]
)

# -----------------------------------------------------
"""
APP Layout: contains the app header, the information
form and the displayed graphs
"""
# -----------------------------------------------------

app.layout = html.Div(
    [
        dcc.ConfirmDialog(
            id="confirm",
            message="This demonstrator has been created solely to support the academic publication of Adjutorium, a new machine learning research methodology for survival benefit prediction in breast cancer. It is not suitable for use as a medical device and should not be used to make clinical decisions on the management of individual patients.",
            displayed=True,
        ),
        HEADER,
        html.Div(
            [PATIENT_INFO_FORM, RESULTS_DISPLAY],
            className="row",
            style={
                "align-items": "center",
                "justify-content": "center",
                "padding-left": "3%",
                "padding-right": "3%",
            },
        ),
    ]
)


# -----------------------------------------------------
#'''
# Callbacks
#'''
# -----------------------------------------------------


@app.callback(
    Output("survival_curve", "figure"),
    [
        Input("age", "value"),
        Input("size", "value"),
        Input("nodes", "value"),
        Input("grade", "value"),
        Input("ER", "value"),
        Input("HER2", "value"),
        Input("SCR", "value"),
        Input("CT", "value"),
        Input("HT", "value"),
    ],
)
def update_risk_score(age, size, nodes, grade, ER, HER2, SCR, CT, HT):

    if (age is None) or (size is None) or (nodes is None):

        surv_oth = [-1] * 10
        surv_all_surg = [-1] * 10
        surv_all_h = [-1] * 10
        surv_all_c = [-1] * 10
        surv_all_hc = [-1] * 10

    else:
        _, surv_all_surg, surv_oth = Adjutorium_one_patient(
            [age, grade, ER, HER2, size, nodes, SCR, 0, 0]
        )
        _, surv_all_c, _ = Adjutorium_one_patient(
            [age, grade, ER, HER2, size, nodes, SCR, 1, 0]
        )
        _, surv_all_h, _ = Adjutorium_one_patient(
            [age, grade, ER, HER2, size, nodes, SCR, 0, 1]
        )
        _, surv_all_hc, _ = Adjutorium_one_patient(
            [age, grade, ER, HER2, size, nodes, SCR, 1, 1]
        )

    LINE_WIDTH = 2
    _OPACITY_1 = 0.2
    _OPACITY_2 = 0.3
    _OPACITY_3 = 0.3
    COLOR_1 = ("#4F27EC", "rgba(79, 39, 236, " + str(_OPACITY_1) + ")")
    COLOR_2 = ("#AF1CF7", "rgba(175, 28, 247, " + str(_OPACITY_2) + ")")
    COLOR_3 = ("#F71C93", "rgba(247, 28, 147, " + str(_OPACITY_3) + ")")

    LINE_STYLE_0 = {"color": "#2C2B2D", "width": LINE_WIDTH, "dash": "dot"}
    LINE_STYLE_1 = {"color": COLOR_1[0], "width": LINE_WIDTH}
    LINE_STYLE_2 = {"color": COLOR_2[0], "width": LINE_WIDTH}
    LINE_STYLE_3 = {"color": COLOR_3[0], "width": LINE_WIDTH}

    time_horizon = np.linspace(1, 10, 10)

    data_non_bc = {
        "x": time_horizon,
        "y": surv_oth,
        "mode": "lines",
        "line": LINE_STYLE_0,
        "name": "Non-breast cancer survival rate",
    }
    data_surgery_only = {
        "x": time_horizon,
        "y": surv_all_surg,
        "mode": "lines",
        "fill": "tozeroy",
        "fillcolor": COLOR_1[1],
        "line": LINE_STYLE_1,
        "name": "Surgery only",
    }
    data_surgery_h = {
        "x": time_horizon,
        "y": surv_all_h,
        "mode": "lines",
        "fill": "tonexty",
        "fillcolor": COLOR_2[1],
        "line": LINE_STYLE_2,
        "name": "Hormone therapy",
    }
    data_surgery_c = {
        "x": time_horizon,
        "y": surv_all_c,
        "mode": "lines",
        "fill": "tonexty",
        "fillcolor": COLOR_3[1],
        "line": LINE_STYLE_3,
        "name": "Chemotherapy",
    }
    data_surgery_hc = {
        "x": time_horizon,
        "y": surv_all_hc,
        "mode": "lines",
        "fill": "tonexty",
        "fillcolor": COLOR_3[1],
        "line": LINE_STYLE_3,
        "name": "Chemotherapy and hormone therapy",
    }

    if (CT == False) and (HT == False):

        DATA_DICT = [data_non_bc, data_surgery_only]

    elif (CT == True) and (HT == False):

        DATA_DICT = [data_non_bc, data_surgery_only, data_surgery_c]

    elif (CT == False) and (HT == True):

        DATA_DICT = [data_non_bc, data_surgery_only, data_surgery_h]

    elif (CT == True) and (HT == True):

        DATA_DICT = [data_non_bc, data_surgery_only, data_surgery_h, data_surgery_hc]

    plot_dict = {
        "data": DATA_DICT,
        "showlegend": False,
        "layout": {
            "legend": {"x": -10, "y": 0, "bgcolor": "rgba(0,0,0,0)", "font-size": 8},
            "showlegend": False,
            "font-size": 11,
            "width": 400,
            "height": 383,
            "plot_bgcolor": PANEL_COLOR,
            "paper_bgcolor": PANEL_COLOR,
            "margin": dict(l=60, r=50, t=30, b=40),
            "fill": "toself",
            "fillcolor": "violet",
            "xaxis": go.layout.XAxis(
                title_text="Years since diagnosis",
                tickvals=time_horizon,
                zeroline=False,
                titlefont=dict(size=12, family="Helvetica"),
            ),
            "yaxis": go.layout.YAxis(
                title_text="Percentage of women surviving",
                tickvals=[0, 20, 40, 60, 80, 100],
                ticktext=["   0%", "   20%", "   40%", "   60%", "   80%", "   100%"],
                range=[0, 100],
                titlefont=dict(size=12, family="Helvetica"),
            ),
        },
    }

    return plot_dict


@app.callback(
    Output("table", "data"),
    [
        Input("age", "value"),
        Input("size", "value"),
        Input("nodes", "value"),
        Input("grade", "value"),
        Input("ER", "value"),
        Input("HER2", "value"),
        Input("SCR", "value"),
        Input("CT", "value"),
        Input("HT", "value"),
        Input("t_horizon", "value"),
    ],
)
def update_table(age, size, nodes, grade, ER, HER2, SCR, CT, HT, t_horizon):

    if (age is None) or (size is None) or (nodes is None):

        surv_oth = [-1] * 10
        surv_all_surg = [-1] * 10
        surv_all_h = [-1] * 10
        surv_all_c = [-1] * 10
        surv_all_hc = [-1] * 10

    else:

        _, surv_all_surg, surv_oth = Adjutorium_one_patient(
            [age, grade, ER, HER2, size, nodes, SCR, 0, 0]
        )
        _, surv_all_c, _ = Adjutorium_one_patient(
            [age, grade, ER, HER2, size, nodes, SCR, 1, 0]
        )
        _, surv_all_h, _ = Adjutorium_one_patient(
            [age, grade, ER, HER2, size, nodes, SCR, 0, 1]
        )
        _, surv_all_hc, _ = Adjutorium_one_patient(
            [age, grade, ER, HER2, size, nodes, SCR, 1, 1]
        )

    if (age is None) or (size is None) or (nodes is None) or (t_horizon is None):

        data = [
            {
                "treatment": "Surgery only",
                "addBenefit": "---",
                "overall_survival": "---",
            }
        ]

    else:

        surg_only_surv = int(np.round(surv_all_surg[int(t_horizon) - 1]))
        surg_h_surv = int(np.round(surv_all_h[int(t_horizon) - 1]))
        surg_c_surv = int(np.round(surv_all_c[int(t_horizon) - 1]))
        surg_hc_surv = int(np.round(surv_all_hc[int(t_horizon) - 1]))

        ht_benefit = int(np.round(surg_h_surv - surg_only_surv))
        ct_benefit = int(np.round(surg_c_surv - surg_only_surv))
        ct_ht_benefit = int(np.round(surg_hc_surv - surg_only_surv))

        if (CT == False) and (HT == False):

            data = [
                {
                    "treatment": "Surgery only",
                    "addBenefit": "---",
                    "overall_survival": str(surg_only_surv) + "%",
                }
            ]

        elif (CT == True) and (HT == False):

            data = [
                {
                    "treatment": "Surgery only",
                    "addBenefit": "---",
                    "overall_survival": str(surg_only_surv) + "%",
                },
                {
                    "treatment": "+ Chemotherapy",
                    "addBenefit": str(ct_benefit) + "%",
                    "overall_survival": str(surg_c_surv) + "%",
                },
            ]

        elif (CT == False) and (HT == True):

            data = [
                {
                    "treatment": "Surgery only",
                    "addBenefit": "---",
                    "overall_survival": str(surg_only_surv) + "%",
                },
                {
                    "treatment": "+ Hormone therapy",
                    "addBenefit": str(ht_benefit) + "%",
                    "overall_survival": str(surg_h_surv) + "%",
                },
            ]

        elif (CT == True) and (HT == True):

            data = [
                {
                    "treatment": "Surgery only",
                    "addBenefit": "---",
                    "overall_survival": str(surg_only_surv) + "%",
                },
                {
                    "treatment": "+ Hormone therapy",
                    "addBenefit": str(ht_benefit) + "%",
                    "overall_survival": str(surg_h_surv) + "%",
                },
                {
                    "treatment": "+ Chemotherapy",
                    "addBenefit": str(ct_ht_benefit) + "%",
                    "overall_survival": str(surg_hc_surv) + "%",
                },
            ]

    return data


@app.callback(
    Output("messagereport", "children"),
    [
        Input("age", "value"),
        Input("size", "value"),
        Input("nodes", "value"),
        Input("grade", "value"),
        Input("ER", "value"),
        Input("HER2", "value"),
        Input("SCR", "value"),
        Input("CT", "value"),
        Input("HT", "value"),
        Input("t_horizon", "value"),
    ],
)
def update_message(age, size, nodes, grade, ER, HER2, SCR, CT, HT, t_horizon):

    _PANEL_TEXT_STYLE_ = {
        "marginBottom": "0em",
        "margin-top": "0em",
        "margin-right": "15px",
        "color": "#524E4E",
        "font-size": "12px",
    }

    if (age is None) or (size is None) or (nodes is None):

        surv_oth = [-1] * 10
        surv_all_surg = [-1] * 10
        surv_all_h = [-1] * 10
        surv_all_c = [-1] * 10
        surv_all_hc = [-1] * 10

    else:

        surv_br_surg, surv_all_surg, surv_oth = Adjutorium_one_patient(
            [age, grade, ER, HER2, size, nodes, SCR, 0, 0]
        )
        surv_br_c, surv_all_c, _ = Adjutorium_one_patient(
            [age, grade, ER, HER2, size, nodes, SCR, 1, 0]
        )
        surv_br_h, surv_all_h, _ = Adjutorium_one_patient(
            [age, grade, ER, HER2, size, nodes, SCR, 0, 1]
        )
        surv_br_hc, surv_all_hc, _ = Adjutorium_one_patient(
            [age, grade, ER, HER2, size, nodes, SCR, 1, 1]
        )

    if (age is None) or (size is None) or (nodes is None) or (t_horizon is None):

        data = html.Div(
            "To predict an individual's survival benefit from adjuvant therapy options, enter their information in the 'patient information' box and select one or more therapy options.",
            style=_PANEL_TEXT_STYLE_,
        )

    else:

        surg_only_surv = int(np.round(surv_all_surg[int(t_horizon) - 1]))
        surg_h_surv = int(np.round(surv_all_h[int(t_horizon) - 1]))
        surg_c_surv = int(np.round(surv_all_c[int(t_horizon) - 1]))
        surg_hc_surv = int(np.round(surv_all_hc[int(t_horizon) - 1]))

        surg_only_surv_br = int(np.round(surv_br_surg[int(t_horizon) - 1]))
        surg_h_surv_br = int(np.round(surv_br_h[int(t_horizon) - 1]))
        surg_c_surv_br = int(np.round(surv_br_c[int(t_horizon) - 1]))
        surg_hc_surv_br = int(np.round(surv_br_hc[int(t_horizon) - 1]))

        ht_benefit = int(np.round(surg_h_surv - surg_only_surv))
        ct_benefit = int(np.round(surg_c_surv - surg_only_surv))
        ct_ht_benefit = int(np.round(surg_hc_surv - surg_only_surv))

        DEATH_BREASTCAN_surg = 100 - surg_only_surv_br
        DEATH_OTHER_surg = 100 - surg_only_surv - (100 - surg_only_surv_br)

        main_string = (
            "For every 100 women with features similar to this patient, "
            + str(surg_only_surv)
            + " survive with surgery alone, "
            + str(DEATH_BREASTCAN_surg)
            + " die due to breast cancer,"
        )

        if (CT == False) and (HT == False):

            main_string = (
                main_string
                + " and "
                + str(DEATH_OTHER_surg)
                + " die due to other causes."
            )

        elif (CT == True) and (HT == False):

            if ct_benefit == 1:

                designator_ = ", and 1 extra woman survive due to chemotherapy."

            elif ct_benefit == 0:

                designator_ = ", and no extra women survive due to chemotherapy."

            elif ct_benefit > 1:

                designator_ = (
                    ", and "
                    + str(ct_benefit)
                    + " extra women survive due to chemotherapy."
                )

            main_string = (
                main_string
                + " "
                + str(DEATH_OTHER_surg)
                + " die due to other causes"
                + designator_
            )

        elif (CT == False) and (HT == True):

            if ht_benefit == 1:

                designator_ = ", and 1 extra woman survive due to hormone therapy."

            elif ht_benefit == 0:

                designator_ = ", and no extra women survive due to hormone therapy."

            elif ht_benefit > 1:

                designator_ = (
                    ", and "
                    + str(ht_benefit)
                    + " extra women survive due to hormone therapy."
                )

            main_string = (
                main_string
                + " "
                + str(DEATH_OTHER_surg)
                + " die due to other causes"
                + designator_
            )

        elif (CT == True) and (HT == True):

            if ht_benefit == 1:

                designator_ht = (
                    ", and 1 extra woman survive due to hormone therapy, and "
                )

            elif ht_benefit == 0:

                designator_ht = (
                    ", and no extra women survive due to hormone therapy, and "
                )

            elif ht_benefit > 1:

                designator_ht = (
                    ", and "
                    + str(ht_benefit)
                    + " extra women survive due to hormone therapy, and "
                )

            if ct_ht_benefit - ht_benefit == 1:

                designator_ct = "1 extra woman survive due to chemotherapy."

            elif ct_ht_benefit - ht_benefit == 0:

                designator_ct = "no extra women survive due to chemotherapy."

            elif ct_ht_benefit - ht_benefit > 1:

                designator_ct = (
                    str(ct_ht_benefit - ht_benefit)
                    + " extra women survive due to chemotherapy."
                )

            main_string = (
                main_string
                + " "
                + str(DEATH_OTHER_surg)
                + " die due to other causes"
                + designator_ht
                + designator_ct
            )

        data = html.Div(main_string, style=_PANEL_TEXT_STYLE_)

    return data


# -----------------------------------------------------
"""
Main
"""
# -----------------------------------------------------
if __name__ == "__main__":

    app.server.run(
        debug=False,
        threaded=True,
        host="0.0.0.0",
        port=os.getenv("PORT") if os.getenv("PORT") else 4999,
    )


# ---------------------------------------------------------------------
# Questions:
# What is my survival profile?
# How can adj therapy help?
# What happened to patients like me?
# ---------------------------------------------------------------------
