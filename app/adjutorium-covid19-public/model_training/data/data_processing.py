import pandas as pd
import numpy as np

from datetime import datetime

import sys
import warnings

if not sys.warnoptions:
  warnings.simplefilter("ignore")

# date_format         = "%Y-%m-%d"
date_format = "%d/%m/%Y"

personal_features = ['ageyear', 'ethnicity', 'sex']

laboratory_details = ['estimateddate', 'notknownonset', 'infectionswabdate', 'labtestdate', 'typeofspecimen',
                      'otherspecimentype', 'covid19', 'influenzaah1n1pdm2009',
                      'influenzaah3n2', 'influenzab', 'influenzaanonsubtyped',
                      'influenzaaunsubtypable', 'rsv', 'otherresult', 'otherdetails']

hospital_details = ['admittedfrom', 'dateadmittedicu', 'hoursadmittedicu', 'minutesadmittedicu', 'dateleavingicu',
                    'othercomplication', 'sbother', 'admittedhospital', 'sbdate', 'ventilatedwhilstadmitted',
                    'admissionflu', 'otherhospital', 'admissioncovid19', 'ispneumoniacomplication',
                    'isardscomplication',
                    'isunknowncomplication', 'isothercoinfectionscomplication', 'isothercomplication',
                    'issecondarybacterialpneumoniacom', 'ventilatedwhilstadmitteddays', 'patientecmo',
                    'wasthepatientadmittedtoicu', 'organismname', 'daysecmo', 'hospitaladmissiondate',
                    'hospitaladmissionhours', 'hospitaladmissionminutes', 'admissionrsv', 'respiratorysupportnone',
                    'oxygenviacannulaeormask', 'highflownasaloxygen', 'noninvasiveventilation',
                    'invasivemechanicalventilation', 'respiratorysupportecmo']

antiviral_treatment = ['anticovid19treatment']

risk_factors = ['chonicrepiratory', 'chonicrepiratorycondition', 'asthmarequiring', 'asthmarequiringcondition',
                'chronicheart', 'chronicheartcondition', 'chronicrenal', 'chronicrenalcondition', 'chronicliver',
                'chroniclivercondition', 'chronicneurological', 'chronicneurologicalcondition', 'isdiabetes',
                'diabetestype', 'immunosuppressiontreatment', 'immunosuppressiontreatmentcondit',
                'immunosuppressiondisease', 'immunosuppressiondiseaseconditio', 'other', 'othercondition',
                'obesityclinical', 'obesitybmi', 'pregnancy', 'gestationweek', 'travel', 'travelto',
                'traveldateofreturn', 'prematurity', 'hypertension', 'hypertensioncondition', 'travelin14days',
                'travelin14dayscondition', 'worksashealthcareworker', 'contactwithconfirmedcovid19case',
                'contactwithconfirmedcovid19casec']

outcome = ['finaloutcome', 'finaloutcomedate', 'transferdestination', 'outcomeother', 'causeofdeath']

personal_info = ["Age", "Gender", "Obesity", "Pregnancy"]

trust_info = ["trustcode", "trustname"]

ethnicity_info = ["ETHNOS", 'Bangladeshi', 'Black African', 'Black Caribbean', 'British', 'Chinese', 'Indian',
                  'Irish', 'Other Asian', 'Other Black', 'Other White', 'Other mixed', 'Pakistani', 'Unknown ethnicity',
                  'White and Asian', 'White and Black African', 'White and Black Caribbean', 'other ethnicity']

lab_test_info = ["Lab test - onset", "Specimen type: Broncho-alveolar lavage", "Specimen type: Nasal/throat swab",
                 "Specimen type: Nasopharyngeal/nasal aspirate", "Specimen type: Sputum",
                 "Specimen type: Tracheal aspirate", "Specimen type: Unknown"]

comorb_info = ["Chronic Respiratory", "Asthma", "Chronic Heart", "Chronic Renal",
               "Chronic Liver", "Chronic Neurological", "Diabetes", "Immunosuppression Treatment",
               "Immunosuppression Disease", "Other Comorbidities", "Hypertension"]

hosp_info = ["Hosp - Lab test"]

complications_info = ["Pneumonia", "ARDS", "Unknown complication", "Other co-infections complication",
                      "Secondary Bacterial Pneumonia", "Other complication"]

interv_info = ["Invasive Mechanical Ventilation", "Non-invasive Mechanical Ventilation", "Anti-viral Treatment",
               "Oxygen via Cannulae or Mask", "High Flow Nasal Oxygen"]

ICU_info = ["ICU Admission", "Time to ICU", "Time in ICU"]

Outcome_info = ["Dead", "Discharged", "Follow up time"]


def clean_CHESS_data(cohort_data, curr_date):
  feature_dict = {"Personal info": personal_info, "Ethnicity info": ethnicity_info, "Lab test info": lab_test_info,
                  "Comorbidity info": comorb_info, "Hospitalization info": hosp_info,
                  "Complications info": complications_info, "Interventions info": interv_info,
                  "ICU info": ICU_info, "Outcome info": Outcome_info}

  for u in range(len(cohort_data.columns)):
    cohort_data[cohort_data.columns[u].lower()] = cohort_data[cohort_data.columns[u]]

    # personal information

  cohort_data["Age"] = cohort_data["ageyear"] + (cohort_data["agemonth"] / 12)
  cohort_data.loc[cohort_data["Age"] == 0, "Age"] = np.nan
  cohort_data["Gender"] = (cohort_data["sex"] == "Male") * 1
  cohort_data["Obesity"] = (cohort_data["obesityclinical"] == "Yes") * 1
  cohort_data["Pregnancy"] = (cohort_data["pregnancy"] == "Yes") * 1

  cohort_data["Bangladeshi"] = (cohort_data["ethnicity"] == 'Bangladeshi               ') * 1
  cohort_data["Black African"] = (cohort_data["ethnicity"] == 'Black African             ') * 1
  cohort_data["Black Caribbean"] = (cohort_data["ethnicity"] == 'Black Caribbean           ') * 1
  cohort_data["British"] = (cohort_data["ethnicity"] == 'British                   ') * 1
  cohort_data["Chinese"] = (cohort_data["ethnicity"] == 'Chinese                   ') * 1
  cohort_data["Indian"] = (cohort_data["ethnicity"] == 'Indian                    ') * 1
  cohort_data["Irish"] = (cohort_data["ethnicity"] == 'Irish                     ') * 1
  cohort_data["Other Asian"] = (cohort_data["ethnicity"] == 'Other Asian               ') * 1
  cohort_data["Other Black"] = (cohort_data["ethnicity"] == 'Other Black               ') * 1
  cohort_data["Other White"] = (cohort_data["ethnicity"] == 'Other White               ') * 1
  cohort_data["Other mixed"] = (cohort_data["ethnicity"] == 'Other mixed               ') * 1
  cohort_data["Pakistani"] = (cohort_data["ethnicity"] == 'Pakistani                 ') * 1
  cohort_data["White and Asian"] = (cohort_data["ethnicity"] == 'White and Asian           ') * 1
  cohort_data["White and Black African"] = (cohort_data["ethnicity"] == 'White and Black African   ') * 1
  cohort_data["White and Black Caribbean"] = (cohort_data["ethnicity"] == 'White and Black Caribbean ') * 1
  cohort_data["other ethnicity"] = (cohort_data["ethnicity"] == 'other                     ') * 1
  cohort_data["Unknown ethnicity"] = (cohort_data["ethnicity"] == 'Unknown                   ') * 1

  cohort_data.loc[cohort_data["ethnicity"].isnull(), "Unknown ethnicity"] = 1

  # lab test information

  swab_date = cohort_data.loc[~cohort_data["infectionswabdate"].isnull(), "infectionswabdate"].apply(
    lambda x: datetime.strptime(str(x), date_format))
  onset_date = cohort_data.loc[~cohort_data["estimateddate"].isnull(), "estimateddate"].apply(
    lambda x: datetime.strptime(str(x), date_format))
  date_available = (~cohort_data["infectionswabdate"].isnull()) & (~cohort_data["estimateddate"].isnull())

  date_delta = (swab_date.loc[date_available] - onset_date.loc[date_available]).apply(lambda x: x.days)

  cohort_data["Lab test - onset"] = np.nan
  cohort_data.loc[date_available, "Lab test - onset"] = date_delta

  cohort_data["Specimen type: Broncho-alveolar lavage"] = (cohort_data[
                                                             "typeofspecimen"] == "Broncho-alveolar lavage") * 1
  cohort_data["Specimen type: Nasal/throat swab"] = (cohort_data["typeofspecimen"] == "Nasal/throat swab") * 1
  cohort_data["Specimen type: Nasopharyngeal/nasal aspirate"] = (cohort_data[
                                                                   "typeofspecimen"] == "Nasopharyngeal/nasal aspirate") * 1
  cohort_data["Specimen type: Sputum"] = (cohort_data["typeofspecimen"] == "Sputum") * 1
  cohort_data["Specimen type: Tracheal aspirate"] = (cohort_data["typeofspecimen"] == "Tracheal aspirate") * 1
  cohort_data["Specimen type: Unknown"] = (cohort_data["typeofspecimen"] == "Unknown") * 1

  # Comorbidity information

  cohort_data["Chronic Respiratory"] = (cohort_data["chonicrepiratory"] == "Yes") * 1
  cohort_data["Asthma"] = (cohort_data["asthmarequiring"] == "Yes") * 1
  cohort_data["Chronic Heart"] = (cohort_data["chronicheart"] == "Yes") * 1
  cohort_data["Chronic Renal"] = (cohort_data["chronicrenal"] == "Yes") * 1
  cohort_data["Chronic Liver"] = (cohort_data["chronicliver"] == "Yes") * 1
  cohort_data["Chronic Neurological"] = (cohort_data["chronicneurological"] == "Yes") * 1
  cohort_data["Diabetes"] = (cohort_data["isdiabetes"] == "Yes") * 1
  cohort_data["Immunosuppression Treatment"] = (cohort_data["immunosuppressiontreatment"] == "Yes") * 1
  cohort_data["Immunosuppression Disease"] = (cohort_data["immunosuppressiondisease"] == "Yes") * 1
  cohort_data["Other Comorbidities"] = (cohort_data["other"] == "Yes") * 1
  cohort_data["Hypertension"] = (cohort_data["hypertension"] == "Yes") * 1

  # Hosp info

  hosp_date = cohort_data.loc[~cohort_data["hospitaladmissiondate"].isnull(), "hospitaladmissiondate"].apply(
    lambda x: datetime.strptime(str(x), date_format))
  date_available = (~cohort_data["infectionswabdate"].isnull()) & (~cohort_data["hospitaladmissiondate"].isnull())

  date_delta = (hosp_date.loc[date_available] - swab_date.loc[date_available]).apply(lambda x: x.days)

  cohort_data["Hosp - Lab test"] = np.nan
  cohort_data.loc[date_available, "Hosp - Lab test"] = date_delta

  # Complications info

  cohort_data["Pneumonia"] = (cohort_data["ispneumoniacomplication"] == "Yes") * 1
  cohort_data["ARDS"] = (cohort_data["isardscomplication"] == "Yes") * 1
  # cohort_data["Secondary Bacterial Pneumonia"]    = (cohort_data["issecondarybacterialpneumoniacom"]=="Yes") * 1
  cohort_data["Secondary Bacterial Pneumonia"] = (cohort_data["issecondarybacterialpneumoniacomplication"] == "Yes") * 1
  cohort_data["Other co-infections complication"] = (cohort_data["isothercoinfectionscomplication"] == "Yes") * 1
  cohort_data["Other complication"] = (cohort_data["isothercomplication"] == "Yes") * 1
  cohort_data["Unknown complication"] = (cohort_data["isunknowncomplication"] == "Yes") * 1

  # Interventions

  cohort_data["Anti-viral Treatment"] = (cohort_data["anticovid19treatment"] == "Yes") * 1
  cohort_data["Invasive Mechanical Ventilation"] = (cohort_data["invasivemechanicalventilation"] == "Yes") * 1
  cohort_data["Non-invasive Mechanical Ventilation"] = (cohort_data["noninvasiveventilation"] == "Yes") * 1
  cohort_data["Oxygen via Cannulae or Mask"] = (cohort_data["oxygenviacannulaeormask"] == "Yes") * 1
  # cohort_data["ECMO"]                                 = (cohort_data["respiratorysupportecmo"] == "Yes") * 1
  cohort_data["High Flow Nasal Oxygen"] = (cohort_data["highflownasaloxygen"] == "Yes") * 1

  # ICU info

  cohort_data["ICU Admission"] = (cohort_data["wasthepatientadmittedtoicu"] == "Yes") * 1
  cohort_data["Outcome date"] = cohort_data["finaloutcomedate"]

  cohort_data.loc[cohort_data["finaloutcomedate"].isnull(), "Outcome date"] = curr_date
  cohort_data.loc[cohort_data["dateadmittedicu"].isnull(), "dateadmittedicu"] = curr_date
  cohort_data.loc[cohort_data["dateleavingicu"].isnull(), "dateleavingicu"] = curr_date

  cohort_data["Time to ICU"] = np.nan
  cohort_data["Time in ICU"] = np.nan

  ICU_date_in = cohort_data.loc[~cohort_data["hospitaladmissiondate"].isnull(), "dateadmittedicu"].apply(
    lambda x: datetime.strptime(str(x), date_format))
  ICU_date_out = cohort_data.loc[~cohort_data["hospitaladmissiondate"].isnull(), "dateleavingicu"].apply(
    lambda x: datetime.strptime(str(x), date_format))

  cohort_data.loc[~cohort_data["hospitaladmissiondate"].isnull(), "Time to ICU"] = (ICU_date_in - hosp_date).apply(
    lambda x: int(x.days))
  cohort_data.loc[~cohort_data["hospitaladmissiondate"].isnull(), "Time in ICU"] = (ICU_date_out - ICU_date_in).apply(
    lambda x: int(x.days))

  # Outcome info

  cohort_data["Dead"] = (cohort_data["finaloutcome"] == "Death") * 1
  cohort_data["Discharged"] = (cohort_data["finaloutcome"] == "Discharged") * 1

  cohort_data["Follow up time"] = np.nan
  outc_date = cohort_data.loc[~cohort_data["hospitaladmissiondate"].isnull(), "Outcome date"].apply(
    lambda x: datetime.strptime(str(x), date_format))

  cohort_data.loc[~cohort_data["hospitaladmissiondate"].isnull(), "Follow up time"] = (outc_date - hosp_date).apply(
    lambda x: int(x.days))

  variable_names = personal_info + trust_info + ethnicity_info + lab_test_info + comorb_info + hosp_info + complications_info + interv_info + ICU_info + Outcome_info

  return cohort_data.loc[cohort_data["covid19"] == "Yes", variable_names], feature_dict, cohort_data.loc[
    cohort_data["covid19"] == "Yes", ['caseid', 'trustname', 'hospitaladmissiondate', "Follow up time"]]


def get_data(curr_date):
  # chess_data                   = pd.read_csv("data/CHESS_COVID19_CaseReport.csv")
  # chess_data                   = pd.read_csv("data/chess_deidentified.csv")

  chess_data = pd.read_csv("data/chess_April_14.csv")
  chess_data.loc[chess_data["InfectionSwabDate"] == "2020-04-02", "InfectionSwabDate"] = "02/04/2020"
  chess_data.loc[chess_data["HospitalAdmissionDate"] == "2020-03-26", "HospitalAdmissionDate"] = "26/03/2020"
  chess_data.drop_duplicates(["CaseId"], keep="first")

  ethnic_link = pd.read_csv("data/ethnicity_linked.csv").drop_duplicates(["case_id"], keep="first")
  data = chess_data.join(ethnic_link.set_index("case_id"), on="CaseId")

  cleaned_data, variable_names, aux_data = clean_CHESS_data(data, curr_date)

  return apply_inclusion_criteria(cleaned_data), variable_names, apply_inclusion_criteria(aux_data)


def apply_inclusion_criteria(data):
  data = data.loc[~data["Follow up time"].isnull()].copy()

  return data


