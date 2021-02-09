import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app

link = '''https://www.linkedin.com/pulse/partnering-nhs-digital-public-health-england-mihaela-van-der-schaar/'''

p1 = """
Adjutorium is an AI-powered tool that accurately predicts how COVID-19 will impact resource needs (ventilators, ICU beds, etc.) at the individual patient level and the hospital level, thereby giving a reliable picture of future resource usage and enabling healthcare professionals to make well-informed decisions about how these scarce resources can be used to achieve the maximum benefit.

"""

p2 = """
Social policies can certainly help take the strain off healthcare systems around the world. But there’s no guarantee that certain individual hospitals won’t still be stretched well beyond capacity. Additionally, these measures themselves may not be properly observed by everyone, or may be relaxed slowly over time. It’s important to ensure that hospitals remain armed with information that will help them manage peaks in demand for resources like ICU beds or ventilators.

"""

p3 = """
life-or-death choices will be made regarding the use of scarce resources like ventilators and ICU beds. If you are managing or working in a hospital, it would be incredibly helpful (but it’s currently not possible) to have a highly reliable picture of the likely usage status of these resources over time. 
"""

p4 = """
This is what too many healthcare professionals around the world are currently worrying about:
"""

p5 = """
We can help answer these questions by being smart about how we use existing data on hospital admissions, ICU admissions, use of ventilators, patient outcomes (e.g. discharge, mortality), and more. If we have access to high-quality datasets containing such information, we can use machine learning to answer questions such as:"""

q_list = html.P([
    html.Li("Which patients are most likely to need ventilators within a week?"),
    html.Li("How many free ICU beds is this hospital likely to have in a week?"),
    html.Li("Which of these two patients will get the most benefit from going on a ventilator today?"),
])

p6 = """
While these questions can reliably be answered using the machine learning techniques we’ve developed, I cannot emphasize enough that the decisions themselves will, of course, still be made by healthcare professionals on the basis of their organization’s priorities and policies.
"""

p7 = "Here’s how a machine learning model can help answer questions in a way that’s useful to healthcare professions:"

p8 = "As you can see, patients are given risk scores based on their likelihood of ICU admission or ventilator usage. These are then aggregated across the hospital to give a picture of future demand on resources."

p9 = """The simulations and forecast are based on individualized risk prediction that takes into account various known COVID-19 risk factors.
These risk factors are subject to substantial uncertainty and missingness. 
Our model has not been externally validated and should not be used for clinical purposes. The findings should be interpreted with caution.
"""

p10 = """All data shown in this demo are synthetic and for illustrative purposes only."""

Author = """This tool is developed by: Zhaozhi Qian (1), Ahmed M. Alaa (2), Mihaela van der Schaar (1), Ari Ercole (3)"""

Affiliation = """
(1) University of Cambridge Centre for Mathematical Sciences, Wilberforce Rd, Cambridge CB3 0WA, UK
(2) University of California, Los Angeles, CA 90095, USA
(3) University of Cambridge Division of Anaesthesia, Addenbrooke's Hospital, Hills Road, Cambridge CB2 0QQ, UK
"""

layout = html.Div([
    html.H3('About'),
    html.Div(html.P([html.Strong('A detailed introduction with video demonstration is available '),
                     dcc.Link('here', href=link), '.'])),

    html.Div(html.P(p1)),
    html.Div(html.Strong('Isn’t flattening the curve enough?')),
    html.Div(html.P(p2)),
    html.Div([html.P(html.Img(src=app.get_asset_url("flat_curve.png"), id="flat_curve-logo",
                              style={"height": "250px", 'textAlign': 'center',
                                     "width": "auto", "margin-bottom": "25px", }),

                     )]),
    html.Div(html.P(p3)),
    html.Div(html.P(p4)),
    html.Div([html.P(html.Img(src=app.get_asset_url("flat_curve2.png"), id="flat_curve-logo2",
                              style={"height": "300px", 'textAlign': 'center',
                                     "width": "auto", "margin-bottom": "25px", }),

                     )]),
    html.Div(html.P(p5)),
    html.Div(q_list, style={"margin-left": "25px", }),
    html.Div(html.P(p6)),
    html.Div(html.P(p7)),
    html.Div([html.P(html.Img(src=app.get_asset_url("flat_curve3.png"), id="flat_curve-logo2",
                              style={"height": "300px", 'textAlign': 'center',
                                     "width": "auto", "margin-bottom": "25px", }),

                     )]),
    html.Div(html.P(p8)),
    html.H3('Disclaimer'),
    html.Div(html.P(p9)),
    html.Div(html.Strong(p10)),

    html.H3('Credits'),
    html.Div(html.P(Author)),
    html.Div(html.P(Affiliation)),

]
)
