
from __future__ import absolute_import, division, print_function

import numpy as np
import pickle

from models.sequential import *
from utils.make_data import *
from utils.mimic_data_processing import *
from utils.parameters import *



def evaluate_performance(preds, y_true):
    
    Errors  = np.abs(preds[0] - y_true)
    acc     = (np.sum((y_true >= 10) * 1 * (preds[2] >= 10) * 1) / np.sum(preds[2] >= 10),
               np.sum((y_true <= 4) * 1 * (preds[1] <= 4) * 1) / np.sum(preds[1] <= 4))

    results = dict({"RMSE": np.mean((preds[0] - y_true)**2), 
                    "Coverage": np.mean(((preds[2] >= y_true) * (y_true >= preds[1])) * 1), 
                    "CI length": np.mean(preds[2] - preds[1]), 
                    "Audit accuracy": acc})
    
    return results


def compute_predictions(model, X_test, Y_test, model_type, coverage=.9):
    
    if model_type=="BJRNN":
        
        y_pred, y_lower, y_upper = model.predict(X_test, coverage=coverage)
        Y_predicts               = np.concatenate([y_pred[k, :Y_test[k].shape[0]] for k in range(len(Y_test))])
        Y_lo                     = np.concatenate([y_lower[k, :Y_test[k].shape[0]] for k in range(len(Y_test))])
        Y_up                     = np.concatenate([y_upper[k, :Y_test[k].shape[0]] for k in range(len(Y_test))])

    elif model_type=="QRNN":
        
        y_test_predictions       = model.predict(X_test)
        Y_up                     = np.concatenate([y_test_predictions[0][k] for k in range(len(y_test_predictions[0]))])
        Y_lo                     = np.concatenate([y_test_predictions[1][k] for k in range(len(y_test_predictions[1]))])
        Y_predicts               = (Y_up + Y_lo) / 2
        
    elif model_type=="DPRNN":
        
        y_test_predictions       = model.predict(X_test)
        Y_predicts               = np.concatenate([y_test_predictions[0][k] for k in range(len(y_test_predictions[0]))])
        y_std                    = np.concatenate([y_test_predictions[1][k] for k in range(len(y_test_predictions[1]))])
        Y_up                     = Y_predicts + y_std
        Y_lo                     = Y_predicts - y_std
        
    return Y_predicts, Y_lo, Y_up    


def run_MIMIC_experiment(model_params, coverage=.9, retrain=False):
    
    baselines     = ["BJRNN", "QRNN", "DPRNN"]
    
    MIMIC_results = dict({"BJRNN": None, 
                          "QRNN": None, 
                          "DPRNN": None})
    
    infile_          = open('data/real_experiment_train','rb')
    result_          = pickle.load(infile_)

    X_train, Y_train = result_[0], result_[1]

    infile_          = open('data/real_experiment_test','rb')
    result_          = pickle.load(infile_)

    X_test, Y_test   = result_[0], result_[1]
    
    
    if retrain:
        
        # Train the BJ-RNN model
        
        RNN_model        = RNN(**model_params)
        
        RNN_model.fit(X_train, Y_train)
        
        RNN_post_hoc     = RNN_uncertainty_wrapper(RNN_model, damp=1e-2)
        
        # Train the QRNN model
        
        QRNN_model       = QRNN(**model_params)

        QRNN_model.fit(X_train, Y_train)
        
        # Train the DPRNN model
        
        DPRNN_model      = DPRNN(**model_params)

        DPRNN_model.fit(X_train, Y_train)
    
    else:
 
        file_bjrnn   = open('saved_models/real_experiment_BJRNN', 'rb')
        file_qrnn    = open('saved_models/real_experiment_QRNN', 'rb')
        file_dprnn   = open('saved_models/real_experiment_DPRNN', 'rb')

        RNN_post_hoc = pickle.load(file_bjrnn)
        QRNN_model   = pickle.load(file_qrnn)
        DPRNN_model  = pickle.load(file_dprnn)
     
    y_true = np.concatenate([Y_test[k] for k in range(len(Y_test))])
    
    preds  = dict({"BJRNN": compute_predictions(RNN_post_hoc, X_test, Y_test, model_type="BJRNN"),
                   "QRNN": compute_predictions(QRNN_model, X_test, Y_test, model_type="QRNN"),
                   "DPRNN": compute_predictions(DPRNN_model, X_test, Y_test, model_type="DPRNN")})
    
    for baseline in baselines:
        
        MIMIC_results[baseline] = evaluate_performance(preds[baseline], y_true)
    
    return MIMIC_results
    


def main():

    mode         = "RNN"
    inputsize    = 5
    epochs       = 10
    nsteps       = 1000
    batchsize    = 100
    maxsteps     = 10
    retrain      = True

    model_params = dict({"INPUT_SIZE":inputsize, 
                         "EPOCH":epochs, 
                         "N_STEPS":nsteps, 
                         "BATCH_SIZE":batchsize, 
                         "MAX_STEPS":maxsteps, 
                         "mode":mode})
    
    MIMIC_results = run_MIMIC_experiment(model_params, retrain=retrain)

    print(MIMIC_results)



if __name__ == '__main__':

    main()
