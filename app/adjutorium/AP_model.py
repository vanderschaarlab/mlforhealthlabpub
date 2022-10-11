from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import pickle


class AP_model:
    def __init__(self, breast_cancer_models_dict, other_cause_models_dict):

        self.breast_cancer_models_dict = breast_cancer_models_dict
        self.other_cause_models_dict = other_cause_models_dict

        self.RR_HT_ERP = np.log(0.69)
        self.RR_HT_ERN = np.log(1)
        self.RR_CT = np.log(0.786)

        self.time_horizons = list(range(1, 11))
        self.year_dict = [
            str(self.time_horizons[u]) + " years"
            for u in range(len(self.time_horizons))
        ]

    def predict(self, X):

        PREDICT_features_exp = [
            "AGE",
            "GRADE 1",
            "GRADE 2",
            "GRADE 3",
            "TUMOURSIZE",
            "NODESINVOLVED",
            "ER_STATUS",
            "HER2_STATUS",
            "SCREENDETECTED",
        ]

        X_pred = X[PREDICT_features_exp]
        surv_curve = []

        for u in range(len(self.time_horizons)):

            other_cause_risk = self.other_cause_models_dict[
                self.year_dict[u]
            ].predict_proba(X_pred)[:, 0]

            CT_only = (X["CT_FLAG"] == "Y") & (X["HT_FLAG"] != "Y")
            HT_only = (X["CT_FLAG"] != "Y") & (X["HT_FLAG"] == "Y")
            HT_CT = (X["CT_FLAG"] == "Y") & (X["HT_FLAG"] == "Y")
            no_treat = (X["CT_FLAG"] != "Y") & (X["HT_FLAG"] != "Y")

            treatment_benefits_P = (
                CT_only * np.exp(self.RR_CT)
                + HT_only * np.exp(self.RR_HT_ERP)
                + HT_CT * np.exp(self.RR_CT + self.RR_HT_ERP)
                + no_treat
            )
            treatment_benefits_N = (
                CT_only * np.exp(self.RR_CT)
                + HT_only * np.exp(self.RR_HT_ERN)
                + HT_CT * np.exp(self.RR_CT + self.RR_HT_ERN)
                + no_treat
            )

            treatment_benefits = treatment_benefits_P * X[
                "ER_STATUS"
            ] + treatment_benefits_N * (1 - X["ER_STATUS"])

            surv_curve.append(
                1
                - self.breast_cancer_models_dict[self.year_dict[u]].predict_proba(
                    X_pred
                )[:, 0]
                ** treatment_benefits
            )

        return np.array(surv_curve)
