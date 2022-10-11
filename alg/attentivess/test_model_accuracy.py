

from models.SeqModels import attentive_state_space_model, sequence_prediction, HMM
from data.CF_dataset_processing import data_loader
from sklearn.metrics import roc_auc_score
import scipy.stats
import numpy as np
import itertools
import initpath_alg
initpath_alg.init_sys_path()
import utilmlab



def mean_confidence_interval(data, confidence=0.95):
    
    a     = 1.0 * np.array(data)
    n     = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h     = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    
    return m, h


def scrambled(orig):
    
    dest = orig[:]
    np.random.shuffle(dest)
    
    return dest


def partition(lst, n): 
    
    division = len(lst) / float(n) 
    
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]


def get_feature_loc(feature_names, required_feature_name):
    
    return np.where(np.array([(feature_names[k]==required_feature_name)*1 for k in range(len(feature_names))])==1)[0][0]


def get_valid_onsets(X_in):
    
    onset_loc = np.where(X_in==1)[0]
    
    if len(onset_loc) > 0:
        
        X_out = np.zeros(len(X_in))
        X_out[onset_loc[0]:] = 1
        
    else:   
        
        X_out = X_in
        
        
    return X_out    


def train_test_split(X_observations, test_frac):
    
    num_samples   = len(X_observations)
    
    train_size    = int(np.floor(num_samples * (1 - test_frac)))
    test_size     = num_samples - train_size
    
    train_samples = np.random.choice(range(num_samples), train_size,  replace=False)
    test_samples  = list(set(range(num_samples)) - set(train_samples))
    
    X_train       = [X_observations[u] for u in train_samples]
    X_test        = [X_observations[u] for u in test_samples]
    
    return X_train, X_test 


def get_labels_predictions(all_outcomes, _obs, _X, hmm_preds=None):
    
    
    if hmm_preds is None:
        
        Labels        = dict.fromkeys(all_outcomes)
        predictions   = dict.fromkeys(all_outcomes)

        for outcome in all_outcomes:
    
            outcome_loc    = get_feature_loc(feature_names, outcome)
    
            _obs_pred  = [_obs[k][:len(_obs[k])-1, outcome_loc] for k in range(len(_obs))] 
            _obs_label = [_X[k][1:, outcome_loc] for k in range(len(_obs))] 
            validity_  = np.concatenate([get_valid_onsets(_X[k][:len(_X[k])-1, outcome_loc]) for k in range(len(_X))])
    
    
            predictions[outcome] = np.concatenate(_obs_pred)[list(np.where(validity_==0)[0])]
            Labels[outcome]      = np.concatenate(_obs_label)[list(np.where(validity_==0)[0])]
    
    else:
        
        Labels        = dict.fromkeys(all_outcomes)
        predictions   = dict.fromkeys(all_outcomes)

        for outcome in all_outcomes:
    
            outcome_loc = get_feature_loc(feature_names, outcome)
            _obs_label  = [_X[k][1:, outcome_loc] for k in range(len(_X))]
            valid_locs  = np.concatenate([(np.hstack((np.zeros((1,_X[k].shape[0]-1)), np.ones((1,1)))) + get_valid_onsets(_X[k][:, outcome_loc]).reshape((1,-1))).reshape((-1,)) for k in range(len(_X))])
            validity_   = np.concatenate([get_valid_onsets(_X[k][:len(_X[k])-1, outcome_loc]) for k in range(len(_X))])
            
            predictions[outcome] = hmm_preds[list(np.where(valid_locs==0)[0]), outcome_loc]
            Labels[outcome]      = np.concatenate(_obs_label)[list(np.where(validity_==0)[0])]
            

    return Labels, predictions 



def get_labels_predictions_noonset(all_outcomes, _obs, _X, hmm_preds=None):
    
    
    if hmm_preds is None:
        
        Labels        = dict.fromkeys(all_outcomes)
        predictions   = dict.fromkeys(all_outcomes)

        for outcome in all_outcomes:
    
            outcome_loc    = get_feature_loc(feature_names, outcome)
    
            _obs_pred  = [_obs[k][:len(_obs[k])-1, outcome_loc] for k in range(len(_obs))] 
            _obs_label = [_X[k][1:, outcome_loc] for k in range(len(_obs))] 
    
    
            predictions[outcome] = np.concatenate(_obs_pred)
            Labels[outcome]      = np.concatenate(_obs_label)
    
    else:
        
        Labels        = dict.fromkeys(all_outcomes)
        predictions   = dict.fromkeys(all_outcomes)

        for outcome in all_outcomes:
    
            outcome_loc = get_feature_loc(feature_names, outcome)
            _obs_label  = [_X[k][1:, outcome_loc] for k in range(len(_X))]
            valid_locs  = np.concatenate([np.hstack((np.zeros((1,_X[k].shape[0]-1)), np.ones((1,1)))).reshape((-1,)) for k in range(len(_X))])
    
            predictions[outcome] = hmm_preds[list(np.where(valid_locs==0)[0]), outcome_loc]
            Labels[outcome]      = np.concatenate(_obs_label)
            

    return Labels, predictions 


def get_performance(y_true_dict, y_pred_dict, Outcomes):
    
    perf_dict = dict.fromkeys(Outcomes)
    
    for outcome in Outcomes:
        
        if np.sum(y_true_dict[outcome]) > 0:
            
            perf_dict[outcome] = roc_auc_score(y_true_dict[outcome], y_pred_dict[outcome])
         
        else:
            
            perf_dict[outcome] = 0
            
        if outcome == 'FEV1':
            
            perf_dict[outcome] = np.mean((y_true_dict[outcome].reshape((-1,1)) - y_pred_dict[outcome].reshape((-1,1)))**2)
            
    return perf_dict
        
        
def get_hmm_performance(hmm_preds, Y_test, Outcomes):
    
    perf_dict = dict.fromkeys(Outcomes)
    
    return perf_dict

    
def run_benchmark(X_obs, benchmark_name, num_folds, feature_names, Outcomes):
    
    
    perf_dicts    = []
    num_samples   = len(X_obs)
    
    train_size    = int(np.floor(num_samples * (1 - 1/num_folds)))
    test_size     = num_samples - train_size
    
    test_indexes  = partition(scrambled(list(range(num_samples))), num_folds)

    for _ in range(num_folds):
        
        #train_samples = train_indexes[_]
        
        #test_samples  = list(set(range(num_samples)) - set(train_samples))

        test_samples  = test_indexes[_]
        
        train_samples = list(set(range(num_samples)) - set(test_samples))
        
        X_train       = [X_obs[u] for u in train_samples]
        X_test        = [X_obs[u] for u in test_samples]
        
        perf_dicts.append(run_one_fold(X_train, X_test, benchmark_name, feature_names, Outcomes))
        
     
    perf_dict = dict.fromkeys(Outcomes)
    
    for outcome in Outcomes:
        
        perf_dict[outcome] = mean_confidence_interval([adjust_auc(perf_dicts[k][outcome]) for k in range(len(perf_dicts))])
    
    
    return perf_dict    

def adjust_auc(auc_value):
    
    if auc_value >= 0.5:
        
        auc_value_ = auc_value
        
    else:
        
        auc_value_ = 1 - auc_value
     
    return auc_value_

    
def run_one_fold(X_train, X_test, benchmark_name, feature_names, Outcomes):
    
    
    if benchmark_name == "attentive_state_space_RNNinference":
        
        model = attentive_state_space_model(num_states=3,
                                    maximum_seq_length=7, 
                                    input_dim=90, 
                                    inference_network='RNN', 
                                    rnn_type='LSTM',
                                    num_iterations=50, 
                                    num_epochs=10, 
                                    batch_size=100, 
                                    learning_rate=5*1e-4, 
                                    num_rnn_hidden=100, 
                                    num_out_hidden=100,
                                    verbosity=False)
        
        model.fit(X_train)
        
        predicted_states, expected_observations, attention = model.predict(X_test)
        y_true_dict, y_pred_dict = get_labels_predictions(Outcomes, expected_observations, X_test)
        
        perf_dict = get_performance(y_true_dict, y_pred_dict, Outcomes)
        
    
    
    if benchmark_name == 'attentive_state_space':
        
        
        model = attentive_state_space_model(num_states=3,
                                    maximum_seq_length=7, 
                                    input_dim=90, 
                                    inference_network='Seq2SeqAttention', 
                                    rnn_type='LSTM',
                                    num_iterations=50, 
                                    num_epochs=10, 
                                    batch_size=100, 
                                    learning_rate=5*1e-4, 
                                    num_rnn_hidden=100, 
                                    num_out_hidden=100,
                                    verbosity=False)
        
        model.fit(X_train)
        
        predicted_states, expected_observations, attention = model.predict(X_test)
        y_true_dict, y_pred_dict = get_labels_predictions(Outcomes, expected_observations, X_test)
        
        perf_dict = get_performance(y_true_dict, y_pred_dict, Outcomes)
        
        
    if benchmark_name == 'HMM':
        
        model = HMM(num_states=3)
        model.fit(X_train)
        
        hmm_predictions = model.predict(X_test)
        y_true_dict, y_pred_dict = get_labels_predictions(Outcomes, hmm_predictions, X_test, hmm_preds=hmm_predictions)
        
        perf_dict = get_performance(y_true_dict, y_pred_dict, Outcomes)
        
        
    elif benchmark_name == 'RETAIN':
        
        model  = sequence_prediction(maximum_seq_length=7, 
                             input_dim=90, 
                             model_type='RETAIN',
                             rnn_type='LSTM',
                             input_name="Input", 
                             output_name="Output",
                             num_iterations=50, 
                             num_epochs=10, 
                             batch_size=100, 
                             learning_rate=5*1e-4, 
                             num_rnn_hidden=100, 
                             num_out_hidden=100,
                             verbosity=False)
        
        model.fit(X_train)
        
        rnn_predictions, attention = model.predict(X_test)
        y_true_dict, y_pred_dict = get_labels_predictions(Outcomes, rnn_predictions, X_test)
        
        perf_dict = get_performance(y_true_dict, y_pred_dict, Outcomes)
        
        
    elif benchmark_name == 'RNN':
        
        model  = sequence_prediction(maximum_seq_length=7, 
                             input_dim=90, 
                             model_type='RNN',
                             rnn_type='RNN',
                             input_name="Input", 
                             output_name="Output",
                             num_iterations=50, 
                             num_epochs=10, 
                             batch_size=100, 
                             learning_rate=5*1e-4, 
                             num_rnn_hidden=100, 
                             num_out_hidden=100,
                             verbosity=False)
        
        model.fit(X_train)
        
        rnn_predictions = model.predict(X_test) 
        y_true_dict, y_pred_dict = get_labels_predictions(Outcomes, rnn_predictions, X_test)
        
        perf_dict = get_performance(y_true_dict, y_pred_dict, Outcomes)
        
        
    elif benchmark_name == 'LSTM':
        
        model  = sequence_prediction(maximum_seq_length=7, 
                             input_dim=90, 
                             model_type='RNN',
                             rnn_type='LSTM',
                             input_name="Input", 
                             output_name="Output",
                             num_iterations=50, 
                             num_epochs=10, 
                             batch_size=100, 
                             learning_rate=5*1e-4, 
                             num_rnn_hidden=100, 
                             num_out_hidden=100,
                             verbosity=False)
        
        model.fit(X_train)
        
        rnn_predictions = model.predict(X_test)   
        y_true_dict, y_pred_dict = get_labels_predictions(Outcomes, rnn_predictions, X_test)
        
        perf_dict = get_performance(y_true_dict, y_pred_dict, Outcomes)
    
    
    elif benchmark_name == 'GRU':
        
        model  = sequence_prediction(maximum_seq_length=7, 
                             input_dim=90, 
                             model_type='RNN',
                             rnn_type='GRU',
                             input_name="Input", 
                             output_name="Output",
                             num_iterations=50, 
                             num_epochs=10, 
                             batch_size=100, 
                             learning_rate=5*1e-4, 
                             num_rnn_hidden=100, 
                             num_out_hidden=100,
                             verbosity=False)
        
        model.fit(X_train)
        
        rnn_predictions = model.predict(X_test)  
        y_true_dict, y_pred_dict = get_labels_predictions(Outcomes, rnn_predictions, X_test)
        
        perf_dict = get_performance(y_true_dict, y_pred_dict, Outcomes)
        
        
    return perf_dict


# In[ ]:

if __name__ == "__main__":

    X_observations, feature_names = data_loader(
        '{}/alg/attentivess'.format(utilmlab.get_proj_dir()))

    num_folds = 5
    
    benchmarks = ["HMM",
                  "attentive_state_space",
                  "RNN", 
                  "LSTM",
                  "RETAIN"]

    
    comorbidities = ['Liver Disease', 'Asthma', 'ABPA', 
                     'Hypertension', 'Diabetes', 'Arthropathy', 
                     'Bone fracture','Osteoporosis', 'Osteopenia', 
                     'Cancer', 'Cirrhosis', 'Kidney Stones', 'Depression', 
                     'Hemoptysis','Pancreatitus','Intestinal Obstruction']

    Infections    = ['Burkholderia Cepacia', 'Pseudomonas Aeruginosa', 'Haemophilus Influenza', 
                     'Klebsiella Pneumoniae', 'Ecoli', 'ALCA', 'Aspergillus', 'NTM', 'Gram-Negative',
                     'Xanthomonas', 'Staphylococcus Aureus']
    
    Outcomes      = ["Diabetes", "ABPA", "Depression", "Pancreatitus", "Pseudomonas Aeruginosa"] 

    log_handle    = open('accuracy_results.txt', 'w')  

    
    for benchmark in benchmarks:
        
        print('------------ Working on the %s model ------------\n' % (benchmark))
        log_handle.write('------------ Results for the %s model ------------ \n' % (benchmark))
        
        pref_dict   = run_benchmark(X_observations, benchmark, num_folds, feature_names, Outcomes)
        
        init_string = benchmark + " performance: "

        for outcome in Outcomes:
            
            if outcome != Outcomes[-1]:
                
                init_string += outcome + ": " + "%.3f +/- %.2f, " 
             
            else:
                
                init_string += outcome + ": " + "%.3f +/- %.2f. \n"

    
        print(init_string % tuple(itertools.chain.from_iterable([list(pref_dict[outcome]) for outcome in Outcomes]))) 
        
        log_handle.write(init_string % tuple(itertools.chain.from_iterable([list(pref_dict[outcome]) for outcome in Outcomes])))  


    log_handle.close()  
        


