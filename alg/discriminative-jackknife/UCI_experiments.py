
import numpy as np
import scipy.stats as st
import pickle
import pandas as pd

from models.static import *
from utils.performance import *  

from models.BNNSGLD import *
from models.PBP import *

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

def mean_confidence_interval(data, confidence=0.95):
    
    a     = 1.0 * np.array(data)
    n     = len(a)
    m, se = np.mean(a), st.sem(a)
    h     = se * st.t.ppf((1 + confidence) / 2., n-1)
    
    return m, h

def load_dataset(dataset):
    
    
    if dataset=="Boston":
        
        X, y          = load_boston(return_X_y=True)
        
    elif dataset=="Yacht":   

        yacht_data    = np.loadtxt('data/yacht_hydrodynamics.data', delimiter=',', dtype=str)
        valid_indexes = np.argwhere(np.array([1 - len(np.argwhere(np.array(yacht_data[k].split(' '))=="")) for k in range(len(yacht_data))])==1)
        yacht_data    = np.array([np.array(yacht_data[k][0].split(' ')).astype(float) for k in valid_indexes])
        X             = yacht_data[:,:-1]
        y             = yacht_data[:,-1]

    elif dataset=="Wine":
        
        wine_data     = np.loadtxt('data/wine.data', delimiter=',', dtype=str)
        wine_data     = np.delete(wine_data, np.argwhere((wine_data=='?').sum(0)>0).reshape(-1),1)

        wine_data     = wine_data.astype(float)
        X             = wine_data[:,:-1]
        y             = wine_data[:,-1]

    elif dataset=="Energy":  
        
        energy_data   = pd.read_csv("data/ENB2012_data.csv")

        X_energy      = np.array(energy_data[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]])
        Y_energy      = np.array(energy_data["Y1"])

        y             = Y_energy[np.where((np.sum(~np.isnan(X_energy), axis=1) > 0)==1)[0]]
        X             = X_energy[np.where((np.sum(~np.isnan(X_energy), axis=1) > 0)==1)[0], :]

    elif dataset=="Naval":
    
        naval_data    = np.loadtxt('data/naval.txt')

        # remove predictors with missing values
        #naval_data       = np.delete(naval_data, 
        #np.argwhere((naval_data=='?').sum(0)>0).reshape(-1),1)

        naval_data    = naval_data.astype(float)
        X             = naval_data[:,:-1]
        y             = naval_data[:,-1]
        
    elif dataset=="kin8nm":
        
        kin8nm_data   = np.array(pd.read_csv("data/dataset_2175_kin8nm.csv"))
        X             = kin8nm_data[:,:-1]
        y             = kin8nm_data[:,-1]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    
    return X_train, X_test, y_train, y_test


def compute_AUPRC(Errors, err_percentile, CI):
    
    auc_roc_perf_ = roc_auc_score((Errors > np.percentile(Errors, err_percentile)) * 1, CI)
    auc_roc_perf_ = average_precision_score((Errors > np.percentile(Errors, err_percentile)) * 1, CI)
    
    if auc_roc_perf_ <= 0.5:
        
        auc_roc_perf = 1 - auc_roc_perf_
    
    else:    
        
        auc_roc_perf = auc_roc_perf_ 
    
    return auc_roc_perf


def run_baseline(X_train, y_train, X_test, params, train_params, baseline="DJ1", damp=1e-4, coverage=.9):
    
    z_critical           = st.norm.ppf(coverage)
    
    if baseline=="DJ":
        
        model            = DNN(**params) 
        model.fit(X_train, y_train, **train_params)
        
        DNN_posthoc      = DNN_uncertainty_wrapper(model, damp=damp)
        y_pred, y_l, y_u = DNN_posthoc.predict(X_test, coverage=coverage) 
    
    elif baseline=="MCDP":
    
        model            = MCDP_DNN(**params) 
        model.fit(X_train, y_train,**train_params)
        
        y_MCDP           = model.predict(X_test, alpha=1-coverage)
        y_pred, y_l, y_u = y_MCDP[0], y_MCDP[0] - y_MCDP[1], y_MCDP[0] + y_MCDP[1] 
        
    elif baseline=="BNN":
        
        y_pred, y_std    = BNN_sgld(X_train, y_train.reshape((-1,1)), 
                                    torch.tensor(X_test), input_dim=X_train.shape[1])
        y_l, y_u         = y_pred - z_critical * y_std, y_pred + z_critical * y_std
        
    elif baseline=="PBP":
        
        pbp_model        = Bayes_backprop(input_dim=X_train.shape[1])
        pbp_model.fit(X_train, y_train.reshape((-1,1)))
    
        y_pred, y_u, y_l = pbp_model.predict(X_test, alpha=1-coverage)
    
    elif baseline=="DE":
        
        y_pred, y_std    = Deep_ensemble(X_train, y_train, X_test, params, n_ensemble=5, train_frac=0.8)
        y_l, y_u         = y_pred - z_critical * y_std, y_pred + z_critical * y_std
    
    return y_pred, y_l, y_u
    


# In[ ]:

def run_experiments(baselines, datasets, params, train_params, N_exp=100, damp=1e-4, coverage=.9, err_percentile=90):
    
    results = dict.fromkeys(baselines)
    
    for baseline in baselines:
        
        results[baseline] = dict.fromkeys(datasets)
        
        for dataset in datasets:
        
            results[baseline][dataset] = dict({"AUPRC":[], "Coverage":[], "MSE":[]})
        
    for dataset in datasets:
        
        print("Running experiments on dataset: ", dataset)
        
        for _ in range(N_exp):
            
            print(_)

            X_train, X_test, y_train, y_test = load_dataset(dataset)
        
            for baseline in baselines:
            
                y_pred, y_l, y_u = run_baseline(X_train, y_train, X_test, params, train_params, baseline=baseline, 
                                                damp=damp, coverage=coverage)
                CI               = y_u - y_l 
                Errors           = np.abs(y_test.reshape((-1, 1)) - y_pred.reshape((-1, 1)))
                
                results[baseline][dataset]["AUPRC"].append(compute_AUPRC(Errors, err_percentile, CI))
                results[baseline][dataset]["Coverage"].append(np.mean((y_test >= y_l) * (y_test <= y_u)))
                results[baseline][dataset]["MSE"].append(np.mean(Errors**2))
     
    return results       



def main():
    
    retrain      = True
    
    n_epochs     = 1000
    n_dim        = 13
    activtn      = "ReLU"  # "Tanh"      
    num_hidden   = 100
    num_layers   = 1
    verbosity    = False
    learn_rate   = 0.01

    params       = dict({"n_dim":n_dim, 
                         "activation":activtn, 
                         "num_hidden":num_hidden,
                         "num_layers":num_layers})

    train_params = dict({"num_iter":n_epochs, 
                         "verbosity":verbosity, 
                         "learning_rate":learn_rate})
    
    
    baselines    = ["BNN", "DE", "PBP", "DJ", "MCDP"]  
    datasets     = ["Yacht", "Boston"]         # "kin8nm", "Energy", 
    
    if retrain:
        
        results   = run_experiments(baselines, datasets, params, train_params, N_exp=10, damp=1e-2, coverage=.9, err_percentile=90)
        
        for dataset in datasets:
            
            print(dataset)
        
            for baseline in baselines:
    
                print(baseline)
    
                print(mean_confidence_interval(results[baseline][dataset]["AUPRC"]), 
                      np.mean(results[baseline][dataset]["Coverage"]),
                      np.mean(results[baseline][dataset]["MSE"]))
        
    else:  
        
        for dataset in datasets:
            
            print(dataset)
            
            results = pickle.load(open('saved_models/DJ_Experiment_3_'+dataset+'_results', 'rb'))
            
            for baseline in baselines:
    
                print(baseline)
    
                print(mean_confidence_interval(results[baseline][dataset]["AUPRC"]), 
                      np.mean(results[baseline][dataset]["Coverage"]),
                      np.mean(results[baseline][dataset]["MSE"]))



if __name__ == '__main__':
    
    main()




