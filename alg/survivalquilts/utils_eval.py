import numpy as np
from lifelines import KaplanMeierFitter
from sksurv.metrics import concordance_index_ipcw

##### WEIGHTED C-INDEX & BRIER-SCORE
def CensoringProb(Y, T):

    T = T.reshape([-1]) # (N,) - np array
    Y = Y.reshape([-1]) # (N,) - np array

    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=(Y==0).astype(int))  # censoring prob = survival probability of event "censoring"
    G = np.asarray(kmf.survival_function_.reset_index()).transpose()
    G[1, G[1, :] == 0] = G[1, G[1, :] != 0][-1]  #fill 0 with ZoH (to prevent nan values)
    
    return G

def weighted_brier_score(T_train, Y_train, Prediction, T_test, Y_test, Time):
    G = CensoringProb(Y_train, T_train)
    N = len(Prediction)

    W = np.zeros(len(Y_test))
    Y_tilde = (T_test > Time).astype(float)

    for i in range(N):
        tmp_idx1 = np.where(G[0,:] >= T_test[i])[0]
        tmp_idx2 = np.where(G[0,:] >= Time)[0]

        if len(tmp_idx1) == 0:
            G1 = G[1, -1]
        else:
            G1 = G[1, tmp_idx1[0]]

        if len(tmp_idx2) == 0:
            G2 = G[1, -1]
        else:
            G2 = G[1, tmp_idx2[0]]
        W[i] = (1. - Y_tilde[i])*float(Y_test[i])/G1 + Y_tilde[i]/G2

    y_true = ((T_test <= Time) * Y_test).astype(float)

    return np.mean(W*(Y_tilde - (1.-Prediction))**2)


def calc_metrics(tr_t_, tr_y_, te_t_, te_y_, preds, eval_time):
    train_y_ = [(tr_y_.iloc[i,0], tr_t_.iloc[i,0]) for i in range(len(tr_y_))]
    train_y_ = np.array(train_y_, dtype=[('status', 'bool'),('time','<f8')])
    
    test_y_ = [(te_y_.iloc[i,0], te_t_.iloc[i,0]) for i in range(len(te_y_))]
    test_y_ = np.array(test_y_, dtype=[('status', 'bool'),('time','<f8')])
    
    c_index, _, _, _, _ = concordance_index_ipcw(train_y_, test_y_, preds, int(eval_time))
    brier_score = weighted_brier_score(np.asarray(tr_t_), np.asarray(tr_y_), preds, 
                                       np.asarray(te_t_), np.asarray(te_y_), int(eval_time))
    return c_index, brier_score