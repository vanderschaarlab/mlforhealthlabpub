# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pandas as pd


def Adjutorium(features):

    return np.apply_along_axis(Adjutorium_one_patient, axis=1, arr=features)


def compute_patient_specific_hazard(
    Adjutorium_coefficients, Adjutorium_coefficients_norm, X, features
):

    hazard = 0

    for feature in features:

        hazard = hazard + Adjutorium_coefficients[feature] * (
            X[feature] - Adjutorium_coefficients_norm[feature]
        )

    S = np.exp(hazard)

    return S


def Adjutorium_one_patient(features):

    """Define individual patient features and prediction horizon
    ---------------------------------------------------------

    Input features: Age, Grade, ER, HER2, Tumor size, Lymph nodes, Screening status, Chemotherapy, Hormone therapy
    --------------

    """

    T = 10

    feature_list = [
        "AGE**2",
        "log(AGE)",
        "GRADE",
        "TUMOURSIZE",
        "log(NODESINVOLVED)",
        "HER2_STATUS",
        "SCREENDETECTED",
        "NODES * TUMOR_SIZE",
        "TUMOR_SIZE * SCREEN_DETECTED",
        "NODES * SCREEN",
        "GRADE * SCREEN",
        "log(TUMOR_SIZE)",
        "TUMOR_SIZE**.5",
        "CT",
        "HT",
        "CT_HT",
    ]

    feature_dict = dict.fromkeys(feature_list)

    """ Adjuvant therapy parameters
        ---------------------------
        
        Log-rank rate ratios from the Early Breast Cancer Collaborative Trialist Group (EBCCTG)
        
        (1) EBCTCG. "Effects of chemotherapy and hormonal therapy for early breast cancer on recurrence 
            and 15-year survival: an overview of the randomized trials." Lancet, 365(9472):1687-717, 2005.
        
        (2) Early Breast Cancer Trialists' Collaborative Group. "Comparisons between different polychemotherapy 
            regimens for early breast cancer: meta-analyses of long-term outcome among 100 000 women in 123 randomised 
            trials." The Lancet 379.9814 (2012): 432-444.
        
    """

    RR_ERP_CT = np.log(0.786)
    RR_ERP_HT = np.log(0.69)
    RR_ERP_CT_HT = np.log(0.786) + np.log(0.69)

    RR_ERN_CT = np.log(0.786)
    RR_ERN_HT = 0
    RR_ERN_CT_HT = np.log(0.786)

    """ Read input features
        
    """

    ER_STATUS = features[2]

    feature_dict["AGE**2"] = features[0] ** 2 / 100
    feature_dict["log(AGE)"] = np.log(features[0])
    feature_dict["GRADE"] = features[1]
    feature_dict["HER2_STATUS"] = features[3]
    feature_dict["TUMOURSIZE"] = features[4]
    feature_dict["log(NODESINVOLVED)"] = np.log(features[5] + 1)
    feature_dict["SCREENDETECTED"] = features[6]
    feature_dict["CT"] = (1 - features[8]) * features[7]
    feature_dict["HT"] = (1 - features[7]) * features[8]
    feature_dict["CT_HT"] = features[7] * features[8]

    feature_dict["NODES * TUMOR_SIZE"] = features[5] * feature_dict["TUMOURSIZE"] / 100
    feature_dict["TUMOR_SIZE * SCREEN_DETECTED"] = (
        feature_dict["TUMOURSIZE"] * feature_dict["SCREENDETECTED"]
    )
    feature_dict["NODES * SCREEN"] = features[5] * feature_dict["SCREENDETECTED"]
    feature_dict["GRADE * SCREEN"] = (
        feature_dict["GRADE"] * feature_dict["SCREENDETECTED"]
    )
    feature_dict["log(TUMOR_SIZE)"] = np.log(feature_dict["TUMOURSIZE"] + 1)
    feature_dict["TUMOR_SIZE**.5"] = feature_dict["TUMOURSIZE"] ** 0.5

    """ Baseline hazards
        
    """

    t = np.array(list(range(1, T + 1)))

    P_ERP = [
        3.13770642e-07,
        -5.54774139e-05,
        7.95109460e-04,
        8.19640380e-03,
        -3.27774646e-04,
    ]
    P_ERN = [
        1.07169721e-05,
        -2.56601609e-04,
        -3.23196738e-05,
        4.64272316e-02,
        -1.27902165e-02,
    ]
    P_OC = [
        -9.75757728e-07,
        6.25680299e-05,
        -1.65800215e-04,
        4.87673870e-03,
        -1.28465222e-03,
    ]

    bazeline_hazard_ERP = (
        4 * P_ERP[0] * (t ** 3)
        + 3 * P_ERP[1] * (t ** 2)
        + 2 * P_ERP[2] * (t ** 1)
        + P_ERP[3]
    )
    bazeline_hazard_ERN = (
        4 * P_ERN[0] * (t ** 3)
        + 3 * P_ERN[1] * (t ** 2)
        + 2 * P_ERN[2] * (t ** 1)
        + P_ERN[3]
    )
    bazeline_hazard_OC = (
        4 * P_OC[0] * (t ** 3)
        + 3 * P_OC[1] * (t ** 2)
        + 2 * P_OC[2] * (t ** 1)
        + P_OC[3]
    )

    """ Breast-cancer specific and other cause mortality hazards
        
    """

    br_coeff_ERP = dict(
        {
            "AGE**2": 0.08729967829482473,
            "CT": RR_ERP_CT,
            "CT_HT": RR_ERP_CT_HT,
            "GRADE": 0.556177775988343,
            "GRADE * SCREEN": -0.22616287125513823,
            "HER2_STATUS": 0.09137790577080077,
            "HT": RR_ERP_HT,
            "NODES * SCREEN": 0.008760358564454108,
            "NODES * TUMOR_SIZE": -0.019885245602335587,
            "SCREENDETECTED": 0.6970339850910485,
            "TUMOR_SIZE * SCREEN_DETECTED": -0.009051080318133335,
            "TUMOR_SIZE**.5": 1.1285809554658954,
            "TUMOURSIZE": -0.05687098620631947,
            "log(AGE)": -3.6021064884916996,
            "log(NODESINVOLVED)": 0.4711085268274988,
            "log(TUMOR_SIZE)": -0.5889207106773614,
        }
    )

    br_coeff_ERN = dict(
        {
            "AGE**2": 0.06461429259180375,
            "CT": RR_ERN_CT,
            "CT_HT": RR_ERN_CT_HT,
            "GRADE": 0.2626799845889968,
            "GRADE * SCREEN": -0.25024448960617557,
            "HER2_STATUS": -0.27298071930423246,
            "HT": RR_ERN_HT,
            "NODES * SCREEN": 0.05259408190919204,
            "NODES * TUMOR_SIZE": 0.003955582254687177,
            "SCREENDETECTED": 0.8142697772638426,
            "TUMOR_SIZE * SCREEN_DETECTED": -0.011559110647963774,
            "TUMOR_SIZE**.5": 1.4099301308100884,
            "TUMOURSIZE": -0.05781110927659145,
            "log(AGE)": -2.92483157477463,
            "log(NODESINVOLVED)": 0.31110286575081136,
            "log(TUMOR_SIZE)": -1.3264406622028775,
        }
    )

    br_coeff_OC = dict({"AGE**2": 0.07403501883766975})

    br_norm_ERP = dict(
        {
            "AGE**2": 39.331771876557745,
            "CT": 0.1402999954683464,
            "CT_HT": 0.10699234150541533,
            "GRADE": 1.8222912040603616,
            "GRADE * SCREEN": 1.1768930982915666,
            "HER2_STATUS": 0.1167807132822767,
            "HT": 0.3603797525717134,
            "NODES * SCREEN": 0.41963112339692754,
            "NODES * TUMOR_SIZE": 0.8035791000135863,
            "SCREENDETECTED": 0.6736529659672814,
            "TUMOR_SIZE * SCREEN_DETECTED": 11.38204105678162,
            "TUMOR_SIZE**.5": 4.386156833004787,
            "TUMOURSIZE": 20.972900711469617,
            "log(AGE)": 4.098998792523891,
            "log(NODESINVOLVED)": 0.801562139203854,
            "log(TUMOR_SIZE)": 2.9335235202748167,
        }
    )

    br_norm_ERN = dict(
        {
            "AGE**2": 36.4583734171449,
            "CT": 0.39646533661291794,
            "CT_HT": 0.14070328869454313,
            "GRADE": 2.890543864443837,
            "GRADE * SCREEN": 0.8173909082627312,
            "HER2_STATUS": 0.8985606267650542,
            "HT": 0.12271112325772068,
            "NODES * SCREEN": 0.2139473444474811,
            "NODES * TUMOR_SIZE": 1.1310458230846294,
            "SCREENDETECTED": 0.27612280222282953,
            "TUMOR_SIZE * SCREEN_DETECTED": 5.403889951717227,
            "TUMOR_SIZE**.5": 5.013837316523504,
            "TUMOURSIZE": 26.88494124077617,
            "log(AGE)": 4.051770734095522,
            "log(NODESINVOLVED)": 1.055155060811392,
            "log(TUMOR_SIZE)": 3.201575735841454,
        }
    )

    br_norm_OC = dict({"AGE**2": 38.37724776056207})

    if ER_STATUS == 0:

        br_hazard = compute_patient_specific_hazard(
            Adjutorium_coefficients=br_coeff_ERN,
            Adjutorium_coefficients_norm=br_norm_ERN,
            X=feature_dict,
            features=feature_list,
        )
    else:

        br_hazard = compute_patient_specific_hazard(
            Adjutorium_coefficients=br_coeff_ERP,
            Adjutorium_coefficients_norm=br_norm_ERP,
            X=feature_dict,
            features=feature_list,
        )

    Oth_hazard = compute_patient_specific_hazard(
        Adjutorium_coefficients=br_coeff_OC,
        Adjutorium_coefficients_norm=br_norm_OC,
        X=feature_dict,
        features=["AGE**2"],
    )

    """ Breast-cancer specific and other cause cumulative survival
        
    """

    br_cum_survival = np.exp(
        -1
        * np.cumsum(
            (ER_STATUS * bazeline_hazard_ERP + (1 - ER_STATUS) * bazeline_hazard_ERN)
            * br_hazard
        )
    )
    oth_cum_survival = np.exp(-1 * np.cumsum(bazeline_hazard_OC * Oth_hazard))

    # All cause mortality
    m_cum_all = 1 - br_cum_survival * oth_cum_survival
    s_cum_all = 1 - m_cum_all

    # Proportion of all cause mortality that is breast cancer
    prop_br = (1 - br_cum_survival) / ((1 - oth_cum_survival) + (1 - br_cum_survival))
    prop_oth = (1 - oth_cum_survival) / ((1 - oth_cum_survival) + (1 - br_cum_survival))

    # Predicted cumulative breast cancer, non-breast cancer specific and all-cause mortality
    pred_m_br = prop_br * m_cum_all
    pred_m_oth = prop_oth * m_cum_all
    pred_all = pred_m_br + pred_m_oth

    # Survival curves

    surv_br = 100 - 100 * pred_m_br
    surv_all = 100 - 100 * pred_all
    surv_oth = 100 - 100 * pred_m_oth

    return surv_br, surv_all, surv_oth
