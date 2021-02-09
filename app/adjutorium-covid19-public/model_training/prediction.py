import pandas as pds
from train_mortality_model import *
import pickle


ASSET_PATH = 'data/res/'
MODEL_STAGING_PATH = ''

with open(MODEL_STAGING_PATH + "adjutorium_mortality", "rb") as f:
  death_model = pickle.load(f)

with open(MODEL_STAGING_PATH + "adjutorium_discharge", "rb") as f:
  discharge_model = pickle.load(f)

with open(MODEL_STAGING_PATH + "adjutorium_icu", "rb") as f:
  icu_model = pickle.load(f)

model_dict = {
  'death': death_model,
  'discharge': discharge_model,
  'icu': icu_model
}

for target, model in model_dict.items():
  _, X, _, _, _, data = prepare_CHESS_data(data_collection_date="14/4/2020",
                                           feature_groups=["Personal info", "Comorbidity info"],
                                           imputer=model.imputer)

  curve = np.array(predict_batch(model, X))

  res = pds.DataFrame(data=curve)
  res['caseid'] = data['caseid']  # this is the caseid field from CHESS data

  # export individual curves
  res.to_csv(ASSET_PATH + '{}_proba.csv'.format(target))

  # export daily aggregate
  res['hospitaladmissiondate'] = data[
    'hospitaladmissiondate']  # this is the hospitaladmissiondate field from CHESS data

  res['hospitaladmissiondate'] = pd.to_datetime(res.hospitaladmissiondate, dayfirst=True).dt.strftime('%Y-%m-%d')
  res['trustname'] = data['trustname']

  df_risk = res[['hospitaladmissiondate', 1, 6]]
  df_risk = df_risk.groupby(['hospitaladmissiondate'], as_index=False).mean()
  df_risk = df_risk[df_risk.hospitaladmissiondate >= '2020-03-10']

  df_risk_trust = res[['hospitaladmissiondate', 'trustname', 1, 6]]
  df_risk_trust = df_risk_trust.groupby(['hospitaladmissiondate', 'trustname'], as_index=False).mean()
  df_risk_trust = df_risk_trust[df_risk_trust.hospitaladmissiondate >= '2020-03-10']

  # df_risk['hospitaladmissiondate'] = pd.to_datetime(df_risk.hospitaladmissiondate, dayfirst=True).dt.strftime('%Y-%m-%d')
  # df_risk_trust['hospitaladmissiondate'] = pd.to_datetime(df_risk_trust.hospitaladmissiondate, dayfirst=True).dt.strftime('%Y-%m-%d')

  df_risk.to_csv(ASSET_PATH + '{}_risk_forecast.csv'.format(target))
  df_risk_trust.to_csv(ASSET_PATH + '{}_trust_risk_forecast.csv'.format(target))
