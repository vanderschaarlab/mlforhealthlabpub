import json
import numpy as np
import sys
import os
from sklearn.linear_model import LinearRegression
from matching_rep import MatchingRep




def load_data(data_path):
    # read data from json file
    with open(data_path, 'r', encoding='utf8') as f:
        data_dic = json.loads(f.read())

    # separate data into patien fatures, organ features...
    data = np.array(data_dic['paired_data'])
    real_X = data[:, data_dic['rstart']:data_dic['rend']]
    real_O = data[:, data_dic['dstart']:data_dic['dend']]
    X = np.array(data_dic['X'])
    O = np.array(data_dic['O'])
    S = np.array(data_dic['S'])

    # obtain observed outcomes
    # handle both real data and fully obsereved synthetic data
    if 'fully_observed' in data_dic and data_dic['fully_observed']:
        Ys = data[:, data_dic['ystart']:data_dic['yend']]
        o_clus = data[:, -1].astype('int')
        Y = np.sum(Ys*np.eye(data_dic['n_class'])[o_clus], axis=-1)
    else:
        Y = data[:, data_dic['y_col']]
        o_clus = None

    if 'r_cls' in data_dic:
        r_clus = np.array(data_dic['r_cls']).astype('int')
    else:
        r_clus = None

    if 'n_class' in data_dic:
        n_clusters = data_dic['n_class']
    else:
        n_clusters = 5

    return data_dic['n_feature_x'], data_dic['n_feature_o'], n_clusters, X, O, S, real_X, real_O, Y, r_clus, o_clus


def real(X, O, Y, S):
    n_size = X.shape[0]
    death_count = 0
    survival = 0
    benefit = 0
    
    Y = Y.predict(np.hstack([X, O]))

    for i in range(O.shape[0]):
    
        if len(S) < 1:
            break
    
        benefit += max(0, Y[0] - S[0])
        survival += Y[0]
    
        Y = Y[1:]
        S = S[1:]
    
        S -= 1
        death_count += np.sum(np.where(S==0, 1, 0))
        non_zero = np.nonzero(S)
        Y = Y[non_zero]
        S = S[non_zero]
    
    print('='*10, 'real policy', '='*10)
    print('death rate: ', death_count/n_size, ' (', death_count, '/', n_size, ')')
    print('avg survival time: ', survival/n_size, ' (', survival, '/', n_size, ')')
    print('avg benefit: ', benefit/n_size, ' (', benefit, '/', n_size, ')')

    print('='*30)


def FCFS(X, O, Y, S):
    n_size = X.shape[0]
    death_count = 0
    survival = 0
    benefit = 0
    
    for i in range(O.shape[0]):
    
        if len(S) < 1:
            break
        
        x = X[0]
        y = Y.predict(np.hstack([x.reshape((1, -1)), O[i].reshape((1, -1))])).reshape(-1)[0]
        
        survival += y
        benefit += max(0, y-S[0])
    
        X = X[1:]
        S = S[1:]
    
        S -= 1
        death_count += np.sum(np.where(S==0, 1, 0))
        non_zero = np.nonzero(S)
        X = X[non_zero]
        S = S[non_zero]
        
    print('='*10, 'FCFS policy', '='*10)
    print('death rate: ', death_count/n_size, ' (', death_count, '/', n_size, ')')
    print('avg survival time: ', survival/n_size, ' (', survival, '/', n_size, ')')
    print('avg benefit: ', benefit/n_size, ' (', benefit, '/', n_size, ')')


    print('='*30)

def UF(X, O, Y, S, predictor):
    n_size = X.shape[0]
    death_count = 0
    survival = 0
    benefit = 0

    for i in range(O.shape[0]):
    
        if len(S) < 1:
            break
            
        xs = X[:500] # only considers the first 500 patient for efficiency and fairness
        os = np.array([O[i]])[np.zeros(len(xs)).astype('int')]
        
        ys = predictor.predict_y([xs, os]).reshape(-1)
        idx = np.argmax(ys) 
        x = X[idx]
        y = Y.predict(np.hstack([x.reshape((1, -1)), O[i].reshape((1, -1))])).reshape(-1)[0]
        
        survival += y
        benefit += max(0, y-S[idx])
    
        X = np.delete(X, idx, axis=0)
        S = np.delete(S, idx)
    
        S -= 1
        death_count += np.sum(np.where(S==0, 1, 0))
        non_zero = np.nonzero(S)
        X = X[non_zero]
        S = S[non_zero]
    
    print('='*10, 'Utility-First policy', '='*10)
    print('death rate: ', death_count/n_size, ' (', death_count, '/', n_size, ')')
    print('avg survival time: ', survival/n_size, ' (', survival, '/', n_size, ')')
    print('avg benefit: ', benefit/n_size, ' (', benefit, '/', n_size, ')')


    print('='*30)

def BF(X, O, Y, S, predictor):
    n_size = X.shape[0]
    death_count = 0
    survival = 0
    benefit = 0
    
    for i in range(O.shape[0]):
    
        if len(S) < 1:
            break
    
        xs = X[:500] # only considers the first 500 patient for efficiency and fairness
        os = np.array([O[i]])[np.zeros(len(xs)).astype('int')]
        
        ys = predictor.predict_y([xs, os]).reshape(-1)
        potential_benefits = ys - S[:500]
        idx = np.argmax(potential_benefits)
    
        x = X[idx]
        y = Y.predict(np.hstack([x.reshape((1, -1)), O[i].reshape((1, -1))])).reshape(-1)[0]
        
        survival += y
        benefit += max(0, y-S[idx])
    
        X = np.delete(X, idx, axis=0)
        S = np.delete(S, idx)
    
        S -= 1
        death_count += np.sum(np.where(S==0, 1, 0))
        non_zero = np.nonzero(S)
        X = X[non_zero]
        S = S[non_zero]

    print('='*10, 'Benefit-First policy', '='*10)
    print('death rate: ', death_count/n_size, ' (', death_count, '/', n_size, ')')
    print('avg survival time: ', survival/n_size, ' (', survival, '/', n_size, ')')
    print('avg benefit: ', benefit/n_size, ' (', benefit, '/', n_size, ')')


    print('='*30)


def MatchingRepFCFS(X, O, Y, S, r_clus, o_clus, MR):
    n_size = X.shape[0]
    death_count = 0
    survival = 0
    benefit = 0

    assigned_o_clus = np.zeros(X.shape[0])-1
    survivals = np.zeros(X.shape[0])
    for i in range(O.shape[0]):
    
        if len(S) < 1:
            break
        
        soft_clus = MR.predict_clus(O[i].reshape((1, -1))).reshape(-1)
        organ_type = np.argmax(soft_clus)
    
        xs = X[:500] # only considers the first 500 patient for efficiency and fairness
        ys = MR.predict(xs)
        best_type = np.argmax(ys, axis=1)
        idxs = np.array(np.where(best_type==organ_type)).reshape(-1)
        if len(idxs) > 0:
            xs = X[idxs]
            os = np.array([O[i]])[np.zeros(len(xs)).astype('int')]
            ys = MR.predict_y([xs, os]).reshape(-1)
            potential_benefits = ys - S[idxs]
            idx = np.argmax(potential_benefits)
        else:
            # no matches of best organ type
            # run backup policy
            # ---------modify here to achieve combination of other policies--------------
            idx = 0

        assigned_o_clus[idx] = organ_type
        x = X[idx]
        y = Y.predict(np.hstack([x.reshape((1, -1)), O[i].reshape((1, -1))])).reshape(-1)[0]
        survivals[idx] = y
        
        survival += y
        benefit += max(0, y-S[idx])
    
        X = np.delete(X, idx, axis=0)
        S = np.delete(S, idx)
    
        S -= 1
        death_count += np.sum(np.where(S==0, 1, 0))
        non_zero = np.nonzero(S)
        X = X[non_zero]
        S = S[non_zero]

    print('='*10, 'Matching rep. (FCFS)', '='*10)
    print('death rate: ', death_count/n_size, ' (', death_count, '/', n_size, ')')
    print('avg survival time: ', survival/n_size, ' (', survival, '/', n_size, ')')
    print('avg benefit: ', benefit/n_size, ' (', benefit, '/', n_size, ')')

    if o_clus is not None:
        print('-'*5, 'assignment flip', '-'*5)
        t1_idxs = np.array(np.where(r_clus==0)).reshape(-1)
        t1_assignments_true = o_clus[t1_idxs]
        t1_assignments_pred = assigned_o_clus[t1_idxs]
        t1_survival = survivals[t1_idxs]
        assigned_idx = np.nonzero(t1_assignments_pred+1)
        t1_assignments_true = t1_assignments_true[assigned_idx]
        t1_assignments_pred = t1_assignments_pred[assigned_idx]
        t1_survival = t1_survival[assigned_idx]
        diff = t1_assignments_pred - t1_assignments_true
        print('flip ratio: ', np.sum(np.where(diff==0, 0, 1))/t1_assignments_true.shape[0])
        print('type 1 patients avg survival: ', np.mean(t1_survival))

    print('='*30)


def MatchingRepUF(X, O, Y, S, r_clus, o_clus, MR):
    n_size = X.shape[0]
    death_count = 0
    survival = 0
    benefit = 0

    assigned_o_clus = np.zeros(X.shape[0])-1
    survivals = np.zeros(X.shape[0])
    for i in range(O.shape[0]):
    
        if len(S) < 1:
            break
        
        soft_clus = MR.predict_clus(O[i].reshape((1, -1))).reshape(-1)
        organ_type = np.argmax(soft_clus)
    
        xs = X[:500] # only considers the first 500 patient for efficiency and fairness
        ys = MR.predict(xs)
        best_type = np.argmax(ys, axis=1)
        idxs = np.array(np.where(best_type==organ_type)).reshape(-1)
        if len(idxs) > 0:
            xs = X[idxs]
            os = np.array([O[i]])[np.zeros(len(xs)).astype('int')]
            ys = MR.predict_y([xs, os]).reshape(-1)
            potential_benefits = ys - S[idxs]
            idx = np.argmax(potential_benefits)
        else:
            # no matches of best organ type
            # run backup policy
            # ---------modify here to achieve combination of other policies--------------
            xs = X[:500]
            os = np.array([O[i]])[np.zeros(len(xs)).astype('int')]
            ys = MR.predict_y([xs, os]).reshape(-1)
            idx = np.argmax(ys)

        assigned_o_clus[idx] = organ_type
        x = X[idx]
        y = Y.predict(np.hstack([x.reshape((1, -1)), O[i].reshape((1, -1))])).reshape(-1)[0]
        survivals[idx] = y
        
        survival += y
        benefit += max(0, y-S[idx])
    
        X = np.delete(X, idx, axis=0)
        S = np.delete(S, idx)
    
        S -= 1
        death_count += np.sum(np.where(S==0, 1, 0))
        non_zero = np.nonzero(S)
        X = X[non_zero]
        S = S[non_zero]

    print('='*10, 'Matching rep. (UF)', '='*10)
    print('death rate: ', death_count/n_size, ' (', death_count, '/', n_size, ')')
    print('avg survival time: ', survival/n_size, ' (', survival, '/', n_size, ')')
    print('avg benefit: ', benefit/n_size, ' (', benefit, '/', n_size, ')')

    if o_clus is not None:
        print('-'*5, 'assignment flip', '-'*5)
        t1_idxs = np.array(np.where(r_clus==0)).reshape(-1)
        t1_assignments_true = o_clus[t1_idxs]
        t1_assignments_pred = assigned_o_clus[t1_idxs]
        t1_survival = survivals[t1_idxs]
        assigned_idx = np.nonzero(t1_assignments_pred+1)
        t1_assignments_true = t1_assignments_true[assigned_idx]
        t1_assignments_pred = t1_assignments_pred[assigned_idx]
        t1_survival = t1_survival[assigned_idx]
        diff = t1_assignments_pred - t1_assignments_true
        print('flip ratio: ', np.sum(np.where(diff==0, 0, 1))/t1_assignments_true.shape[0])
        print('type 1 patients avg survival: ', np.mean(t1_survival))

    print('='*30)



def MatchingRepBF(X, O, Y, S, r_clus, o_clus, MR):
    n_size = X.shape[0]
    death_count = 0
    survival = 0
    benefit = 0

    assigned_o_clus = np.zeros(X.shape[0])-1
    survivals = np.zeros(X.shape[0])
    for i in range(O.shape[0]):
    
        if len(S) < 1:
            break
        
        soft_clus = MR.predict_clus(O[i].reshape((1, -1))).reshape(-1)
        organ_type = np.argmax(soft_clus)
    
        xs = X[:500] # only considers the first 500 patient for efficiency and fairness
        ys = MR.predict(xs)
        best_type = np.argmax(ys, axis=1)
        idxs = np.array(np.where(best_type==organ_type)).reshape(-1)
        if len(idxs) > 0:
            xs = X[idxs]
            os = np.array([O[i]])[np.zeros(len(xs)).astype('int')]
            ys = MR.predict_y([xs, os]).reshape(-1)
            potential_benefits = ys - S[idxs]
            idx = np.argmax(potential_benefits)
        else:
            # no matches of best organ type
            # run backup policy
            # ---------modify here to achieve combination of other policies--------------
            xs = X[:500]
            os = np.array([O[i]])[np.zeros(len(xs)).astype('int')]
            ys = MR.predict_y([xs, os]).reshape(-1)
            potential_benefits = ys - S[:500]
            idx = np.argmax(potential_benefits)

        assigned_o_clus[idx] = organ_type
        x = X[idx]
        y = Y.predict(np.hstack([x.reshape((1, -1)), O[i].reshape((1, -1))])).reshape(-1)[0]
        survivals[idx] = y
        
        survival += y
        benefit += max(0, y-S[idx])
    
        X = np.delete(X, idx, axis=0)
        S = np.delete(S, idx)
    
        S -= 1
        death_count += np.sum(np.where(S==0, 1, 0))
        non_zero = np.nonzero(S)
        X = X[non_zero]
        S = S[non_zero]

    print('='*10, 'Matching rep. (BF)', '='*10)
    print('death rate: ', death_count/n_size, ' (', death_count, '/', n_size, ')')
    print('avg survival time: ', survival/n_size, ' (', survival, '/', n_size, ')')
    print('avg benefit: ', benefit/n_size, ' (', benefit, '/', n_size, ')')

    if o_clus is not None:
        print('-'*5, 'assignment flip', '-'*5)
        t1_idxs = np.array(np.where(r_clus==0)).reshape(-1)
        t1_assignments_true = o_clus[t1_idxs]
        t1_assignments_pred = assigned_o_clus[t1_idxs]
        t1_survival = survivals[t1_idxs]
        assigned_idx = np.nonzero(t1_assignments_pred+1)
        t1_assignments_true = t1_assignments_true[assigned_idx]
        t1_assignments_pred = t1_assignments_pred[assigned_idx]
        t1_survival = t1_survival[assigned_idx]
        diff = t1_assignments_pred - t1_assignments_true
        print('flip ratio: ', np.sum(np.where(diff==0, 0, 1))/t1_assignments_true.shape[0])
        print('type 1 patients avg survival: ', np.mean(t1_survival))

    print('='*30)

def MatchingRepSim(X, O, Y, S, r_clus, o_clus, MR):
    # MatchingRepFCFS(X, O, Y, S, r_clus, o_clus, MR)
    MatchingRepUF(X, O, Y, S, r_clus, o_clus, MR)
    # MatchingRepBF(X, O, Y, S, r_clus, o_clus, MR)


    
def run_simulation(data_path='./data/gmixbiased.json', model_path=None):   

    if not os.path.isfile(data_path):
        print('ERROR: the data path does not exist')
        return

    n_feature_x, n_feature_o, n_clusters, X, O, S, real_X, real_O, Y, r_clus, o_clus = load_data(data_path)
    # train Matching rep
    MR = MatchingRep(n_feature_x=n_feature_x, n_feature_o=n_feature_o, n_clusters=n_clusters)
    if model_path is not None:
        MR.load_weights(model_path)
    else:
        split = int(X.shape[0]/10*9)
        X_train = real_X[:split]
        X_valid = real_X[split:]
        O_train = real_O[:split]
        O_valid = real_O[split:]
        Y_train = Y[:split]
        Y_valid = Y[split:]
        MR.fit([X_train, O_train], Y_train, validation_data=([X_valid, O_valid], Y_valid), epochs=100, verbose=0)
        MR.load_weights()

    # fit a ground truth outcome estimator
    Y = LinearRegression().fit(np.hstack([real_X, real_O]), Y)
    

    # run simulation with the real policy
    real(real_X, real_O, Y, S)

    # run simulation with First-Come-First-Serve policy
    FCFS(X, O, Y, S)
    # run simulation with Utility-First policy
    UF(X, O, Y, S, MR)
    # run simulation with Benefit-First policy
    BF(X, O, Y, S, MR)
    # run simulation with Matching rep policy
    MatchingRepSim(X, O, Y, S, r_clus, o_clus, MR)



    
    
    


if __name__ == '__main__':
    try:
        data_path = sys.argv[1]
    except:
        data_path = './data/gmixbiased.json'
    
    # run_simulation(data_path, model_path='./model/MatchingRepCheckpoint')
    run_simulation(data_path)

