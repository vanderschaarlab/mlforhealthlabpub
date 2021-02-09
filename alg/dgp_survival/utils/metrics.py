# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np

from sksurv.metrics import cumulative_dynamic_auc, concordance_index_ipcw

from lifelines.utils import concordance_index


def evaluate_performance(T_train, c_train, T_test, c_test, prediction, time_horizon, 
                         num_causes=2, cause_names=["Cause 1", "Cause 2"]):

    Harell_c_index     = []
    UNO_c_index        = []
    dynamic_auc        = []

    for _ in range(num_causes):

        y_train = np.array([((c_train.loc[c_train.index[k]]== _ + 1), T_train.loc[T_train.index[k]]) for k in range(len(T_train))], dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
        y_test  = np.array([((c_test.loc[c_test.index[k]]== _ + 1), T_test.loc[T_test.index[k]]) for k in range(len(T_test))], dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

        Harell_c_index.append(concordance_index(T_test, prediction[_ + 1], event_observed=(c_test==(_+1))*1))
        tau = max(y_train['Survival_in_days'])
        ci_tau = concordance_index_ipcw(y_train, y_test, 1 - prediction[_ + 1], tau=tau)[0]

        UNO_c_index.append(ci_tau)
        try:
            dynamic_auc_val = cumulative_dynamic_auc(y_train, y_test, 1 - prediction[_ + 1], times=[time_horizon])[0][0]
        except ValueError:
            print('*warning: exception while calculating dynamic_auc, dynamic_auc is not calculated*')
            dynamic_auc_val = "-"
        dynamic_auc.append(dynamic_auc_val)
        print("--- Cause: {} -> [C-index: {:0.4f} ] [Dynamic AUC-ROC: {} ]".format(
            cause_names[_],
            UNO_c_index[-1],
            '{:0.4f}'.format(dynamic_auc[-1]) if dynamic_auc[-1] != "-"  else "-"))
