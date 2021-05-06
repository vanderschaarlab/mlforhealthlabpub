'''
This train DeepHit, and outputs the validation performance for random search.

INPUTS:
    - DATA = (data, time, label)
    - MASK = (mask1, mask2)
    - in_parser: dictionary of hyperparameters
    - out_itr: the training/testing split indicator
    - eval_time: None or a list (e.g. [12, 24, 36]) at which the validation of the network is performed
    - MAX_VALUE: maximum validation value
    - OUT_ITERATION: total number of training/testing splits
    - seed: random seed for training/testing/validation

OUTPUTS:
    - the validation performance of the trained network
    - save the trained network in the folder directed by "in_parser['out_path'] + '/itr_' + str(out_itr)"
'''
import numpy as np
import tensorflow as tf
import random
import os

from termcolor import colored
from sklearn.model_selection import train_test_split
from class_DeepHit import Model_DeepHit
from utils_eval import c_index, brier_score, weighted_c_index, weighted_brier_score



##### USER-DEFINED FUNCTIONS
def log(x):
    return tf.log(x + 1e-8)


def div(x, y):
    return tf.div(x, (y + 1e-8))


def f_get_minibatch(mb_size, x, label, time, mask1, mask2):
    idx = range(np.shape(x)[0])
    idx = random.sample(idx, mb_size)

    x_mb = x[idx, :].astype(np.float32)
    k_mb = label[idx, :].astype(np.float32) # censoring(0)/event(1,2,..) label
    t_mb = time[idx, :].astype(np.float32)
    m1_mb = mask1[idx, :, :].astype(np.float32) #fc_mask
    m2_mb = mask2[idx, :].astype(np.float32) #fc_mask
    return x_mb, k_mb, t_mb, m1_mb, m2_mb


def get_valid_performance(DATA, MASK, in_parser, out_itr, eval_time=None, MAX_VALUE = -99, OUT_ITERATION=5, seed=1234):
    ##### DATA & MASK
    (data, time, label)  = DATA
    (mask1, mask2)       = MASK

    x_dim                       = np.shape(data)[1]
    _, num_Event, num_Category  = np.shape(mask1)  # dim of mask1: [subj, Num_Event, Num_Category]
    
    ACTIVATION_FN               = {'relu': tf.nn.relu, 'elu': tf.nn.elu, 'tanh': tf.nn.tanh}

    ##### HYPER-PARAMETERS
    mb_size                     = in_parser['mb_size']

    iteration                   = in_parser['iteration']

    keep_prob                   = in_parser['keep_prob']
    lr_train                    = in_parser['lr_train']


    alpha                       = in_parser['alpha']  #for log-likelihood loss
    beta                        = in_parser['beta']  #for ranking loss
    gamma                       = in_parser['gamma']  #for RNN-prediction loss
    parameter_name              = 'a' + str('%02.0f' %(10*alpha)) + 'b' + str('%02.0f' %(10*beta)) + 'c' + str('%02.0f' %(10*gamma))

    initial_W                   = tf.contrib.layers.xavier_initializer()


    ##### MAKE DICTIONARIES
    # INPUT DIMENSIONS
    input_dims                  = { 'x_dim'         : x_dim,
                                    'num_Event'     : num_Event,
                                    'num_Category'  : num_Category}

    # NETWORK HYPER-PARMETERS
    network_settings            = { 'h_dim_shared'       : in_parser['h_dim_shared'],
                                    'num_layers_shared'  : in_parser['num_layers_shared'],
                                    'h_dim_CS'           : in_parser['h_dim_CS'],
                                    'num_layers_CS'      : in_parser['num_layers_CS'],
                                    'active_fn'          : ACTIVATION_FN[in_parser['active_fn']],
                                    'initial_W'          : initial_W }


    file_path_final = in_parser['out_path'] + '/itr_' + str(out_itr)

    #change parameters...
    if not os.path.exists(file_path_final + '/models/'):
        os.makedirs(file_path_final + '/models/')


    print (file_path_final + ' (a:' + str(alpha) + ' b:' + str(beta) + ' c:' + str(gamma) + ')' )

    ##### CREATE DEEPFHT NETWORK
    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model = Model_DeepHit(sess, "DeepHit", input_dims, network_settings)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())


    ### TRAINING-TESTING SPLIT
    (tr_data,te_data, tr_time,te_time, tr_label,te_label, 
     tr_mask1,te_mask1, tr_mask2,te_mask2)  = train_test_split(data, time, label, mask1, mask2, test_size=0.20, random_state=seed) 
    

    (tr_data,va_data, tr_time,va_time, tr_label,va_label, 
     tr_mask1,va_mask1, tr_mask2,va_mask2)  = train_test_split(tr_data, tr_time, tr_label, tr_mask1, tr_mask2, test_size=0.20, random_state=seed) 
    
    max_valid = -99
    stop_flag = 0

    if eval_time is None:
        eval_time = [int(np.percentile(tr_time, 25)), int(np.percentile(tr_time, 50)), int(np.percentile(tr_time, 75))]


    ### TRAINING - MAIN
    print( "MAIN TRAINING ...")
    print( "EVALUATION TIMES: " + str(eval_time))

    avg_loss = 0
    for itr in range(iteration):
        if stop_flag > 5: #for faster early stopping
            break
        else:
            x_mb, k_mb, t_mb, m1_mb, m2_mb = f_get_minibatch(mb_size, tr_data, tr_label, tr_time, tr_mask1, tr_mask2)
            DATA = (x_mb, k_mb, t_mb)
            MASK = (m1_mb, m2_mb)
            PARAMETERS = (alpha, beta, gamma)
            _, loss_curr = model.train(DATA, MASK, PARAMETERS, keep_prob, lr_train)
            avg_loss += loss_curr/1000
                
            if (itr+1)%1000 == 0:
                print('|| ITR: ' + str('%04d' % (itr + 1)) + ' | Loss: ' + colored(str('%.4f' %(avg_loss)), 'yellow' , attrs=['bold']))
                avg_loss = 0

            ### VALIDATION  (based on average C-index of our interest)
            if (itr+1)%1000 == 0:
                ### PREDICTION
                pred = model.predict(va_data)

                ### EVALUATION
                va_result1 = np.zeros([num_Event, len(eval_time)])

                for t, t_time in enumerate(eval_time):
                    eval_horizon = int(t_time)

                    if eval_horizon >= num_Category:
                        print('ERROR: evaluation horizon is out of range')
                        va_result1[:, t] = va_result2[:, t] = -1
                    else:
                        risk = np.sum(pred[:,:,:(eval_horizon+1)], axis=2) #risk score until eval_time
                        for k in range(num_Event):
                            # va_result1[k, t] = c_index(risk[:,k], va_time, (va_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
                            va_result1[k, t] = weighted_c_index(tr_time, (tr_label[:,0] == k+1).astype(int), risk[:,k], va_time, (va_label[:,0] == k+1).astype(int), eval_horizon)
                tmp_valid = np.mean(va_result1)


                if tmp_valid >  max_valid:
                    stop_flag = 0
                    max_valid = tmp_valid
                    print( 'updated.... average c-index = ' + str('%.4f' %(tmp_valid)))

                    if max_valid > MAX_VALUE:
                        saver.save(sess, file_path_final + '/models/model_itr_' + str(out_itr))
                else:
                    stop_flag += 1

    return max_valid
