

# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# ---------------------------------------------------------
# Helper functions and utilities for deep learning models
# ---------------------------------------------------------

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

from models.sequential import *
from utils.make_data import * 


def plot_1D_uncertainty(results, Y_test, data_index):
    
    plt.fill_between(list(range(len(results["Lower limit"][data_index]))), 
                     results["Lower limit"][data_index].reshape(-1,), 
                     results["Upper limit"][data_index].reshape(-1,), color="r", alpha=0.25)

    plt.plot(results["Lower limit"][data_index], linestyle=":", linewidth=3, color="r")
    plt.plot(results["Upper limit"][data_index], linestyle=":", linewidth=3, color="r")

    plt.plot(Y_test[data_index], linestyle="--", linewidth=2, color="black")
    plt.plot(results["Point predictions"][data_index], linewidth=3, color="r", Marker="o")


def evaluate_performance(model, X_test, Y_test, coverage=.9, error_threshold=1):
    
    if type(model) is RNN_uncertainty_wrapper:
    
        y_pred, y_l_approx, y_u_approx  = model.predict(X_test, coverage=coverage)
        
    elif type(model) is QRNN:
        
        y_u_approx, y_l_approx          = model.predict(X_test)
        y_pred                          = [(y_l_approx[k] + y_u_approx[k])/2 for k in range(len(y_u_approx))]
    
    elif type(model) is DPRNN:
        
        y_pred, y_std                   = model.predict(X_test, alpha=1-coverage)
        y_u_approx                      = [y_pred[k] + y_std[k] for k in range(len(y_pred))]
        y_l_approx                      = [y_pred[k] - y_std[k] for k in range(len(y_pred))]
        
    
    results                         = dict({"Point predictions": None,
                                            "Confidence intervals": None,
                                            "Errors": None,
                                            "Upper limit": None,
                                            "Lower limit": None,
                                            "Coverage indicators": None, 
                                            "Coverage": None,
                                            "AUC-ROC": None})
    
    results["Point predictions"]    = y_pred
    results["Upper limit"]          = y_u_approx
    results["Lower limit"]          = y_l_approx
    results["Confidence intervals"] = [y_u_approx[k] - y_l_approx[k] for k in range(len(y_u_approx))]
    results["Errors"]               = [np.abs(Y_test[k] - y_pred[k]) for k in range(len(y_u_approx))]
    results["Coverage indicators"]  = [((y_u_approx[k] >= Y_test[k]) * (y_l_approx[k] <= Y_test[k])) * 1 for k in range(len(y_u_approx))]
    results["Coverage"]             = np.mean(np.concatenate(results["Coverage indicators"]))

    if error_threshold == "Auto":

        results["AUC-ROC"]          = roc_auc_score((np.concatenate(results["Errors"]) > np.median(np.concatenate(results["Errors"]))) * 1, 
                                                     np.concatenate(results["Confidence intervals"]))

    else:    

        results["AUC-ROC"]          = roc_auc_score((np.concatenate(results["Errors"]) > error_threshold) * 1, 
                                                     np.concatenate(results["Confidence intervals"]))
    results["CI length"]            = np.mean(np.concatenate(results["Confidence intervals"]))
    
    return results


def collect_synthetic_results(noise_vars, params, coverage=0.9, seq_len=5, n_train_seq=1000, n_test_seq=1000):

    #noise_profs     = [noise_vars[k] * np.ones(seq_len) for k in range(len(noise_vars))]
    noise_profs     = noise_vars * np.ones(seq_len)

    result_dict     = dict({"BJRNN": [], "QRNN": [], "DPRNN": []})
    
    model_type      = [RNN, QRNN, DPRNN]
    model_names     = ["BJRNN", "QRNN", "DPRNN"]
    
    for u in range(len(model_type)):

        X, Y            = create_autoregressive_data(n_samples=n_train_seq, noise_profile=noise_profs,
                                                         seq_len=seq_len, mode="time-dependent")

        RNN_model       = model_type[u](**params)

        print("Training model " + model_names[u] + " with aleatoric noise variance %.4f and %d training sequences" % (noise_vars, n_train_seq))

        RNN_model.fit(X, Y)
            
        if type(RNN_model) is RNN:
                
            RNN_model_  = RNN_uncertainty_wrapper(RNN_model)

        else:

            RNN_model_  = RNN_model

        X_test, Y_test  = create_autoregressive_data(n_samples=n_test_seq, noise_profile=noise_profs,
                                                         seq_len=seq_len, mode="time-dependent")

        result_dict[model_names[u]].append(evaluate_performance(RNN_model_, X_test, Y_test, coverage=coverage, error_threshold="Auto"))

    return result_dict
