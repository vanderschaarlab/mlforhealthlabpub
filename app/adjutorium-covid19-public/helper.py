import dash_daq as daq
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

# -------------------------------------------------------
'''
Helper functions for style formating and data processing

List of helper functions >>
---------------------------
_get_input_HTML_format :: returns cell formating for 
                          html/dcc numerical input 

_get_radioItems_HTML_format :: returns a radio items 
                               list for display
'''


# -------------------------------------------------------

def _get_input_HTML_format(name, ID, name_style, input_range, input_step, placeholder, input_style):
    _html_input = html.P(children=[html.Div(name, style=name_style),
                                   dcc.Input(placeholder=placeholder, type='number',
                                             min=input_range[0], max=input_range[1], step=input_step,
                                             style=input_style, id=ID)])

    return _html_input


def _get_input_dropdown_format(name, ID, name_style, options, value, input_style, multi=True):
    i = dcc.Dropdown(
        id=ID,
        options=options,
        value=value,
        multi=multi,
        style=input_style
    )
    _html_input = html.P(children=[html.Div(name, style=name_style), i])

    return _html_input


def _get_radioItems_HTML_format(name, ID, name_style, options, radio_style):
    _html_radioItem = html.P(children=[html.Div(name, style=name_style),
                                       dcc.RadioItems(options=options, value=1, style=radio_style, id=ID)])

    return _html_radioItem


def _get_toggle_switch(name, name_style, color_style, ID, label=None):
    if label is None:
        label = ['No', 'Yes']

    _html_toggle = html.P(children=[html.Div(name, style=name_style),
                                    daq.ToggleSwitch(color=color_style, size=30, value=False,
                                                     label=label, style={"font-size": 11}, id=ID)],
                          style={"width": "100px", "font-size": 11})

    return _html_toggle


def HORIZONTAL_SPACE(space_size):
    return dbc.Row(dbc.Col(html.Div(" ", style={"marginBottom": str(space_size) + "em"})))


def VERTICAL_SPACE(space_size):
    return dbc.Col(html.Div(" "), style={"width": str(space_size) + "px"})


def get_input_form(label, id, min, max, step, value, icon_url=None):
    if icon_url is None:
        a = dbc.FormGroup(
            [
                dbc.Label(label, html_for=id),
                dbc.InputGroup(
                    [
                        dbc.Input(
                            type="number",
                            id=id,
                            min=min,
                            max=max,
                            step=step,
                            value=value
                        ),
                        dbc.InputGroupAddon("%", addon_type="append"),
                    ]
                ),

            ]
        )
    else:
        img = html.Img(src=icon_url, id=id+"-logo", style={"height": "30px", 'textAlign': 'left',
                                                                  "margin-right":"5px",
                                                                  "width": "auto"}),

        a = dbc.FormGroup(
            [
                dbc.Label(label, html_for=id),
                dbc.InputGroup(
                    [
                        dbc.InputGroupAddon(img, addon_type="prepend"),
                        dbc.Input(
                            type="number",
                            id=id,
                            min=min,
                            max=max,
                            step=step,
                            value=value
                        ),
                        dbc.InputGroupAddon("%", addon_type="append"),
                    ]
                ),

            ]
        )
    return a
