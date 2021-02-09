import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import random
import os, sys
import argparse

from tensorflow.python.ops.rnn import _transpose_batch_time
from sklearn.model_selection import train_test_split

#performance metrics
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix

#user defined
from utils_log import save_logging, load_logging
from data_loader import import_data
from class_AC_TPC import AC_TPC, initialize_embedding


def f_get_minibatch(mb_size, x, y):
    idx = range(np.shape(x)[0])
    idx = random.sample(idx, mb_size)

    x_mb   = x[idx].astype(float)    
    y_mb   = y[idx].astype(float)    

    return x_mb, y_mb

### PERFORMANCE METRICS:
def f_get_prediction_scores(y_true_, y_pred_):
    if np.sum(y_true_) == 0: #no label for running roc_auc_curves
        auroc_ = -1.
        auprc_ = -1.
    else:
        auroc_ = roc_auc_score(y_true_, y_pred_)
        auprc_ = average_precision_score(y_true_, y_pred_)
    return (auroc_, auprc_)


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    c_matrix = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(c_matrix, axis=0)) / np.sum(c_matrix)


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', default=6, help='number of maximum clusters', type=int)

    parser.add_argument('--h_dim_FC', default=50, help='number of hidden nodes in FC-layers', type=int)
    parser.add_argument('--h_dim_RNN', default=50, help='number of hidden nodes in RNN', type=int)

    parser.add_argument('--n_layer_enc', default=1, help='number of layers -- encoder', type=int)
    parser.add_argument('--n_layer_sel', default=2, help='number of layers -- selector', type=int)
    parser.add_argument('--n_layer_pre', default=2, help='number of layers -- predictor', type=int)

    parser.add_argument("--rnn_type", choices=['LSTM','GRU'], default='LSTM', type=str)
    
    parser.add_argument("--lr_rate_init", default=1e-3, type=float)
    parser.add_argument("--lr_rate_clu_1", default=1e-3, type=float)
    parser.add_argument("--lr_rate_clu_2", default=1e-3, type=float)

    parser.add_argument("--itrs_init1", help='initialization for encoder-predictor', default=10000, type=int)
    parser.add_argument("--itrs_init2", help='initialization for selector and embedding', default=5000, type=int)
    parser.add_argument("--itrs_clu", default=2000, type=int)

    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--keep_prob", help='keep probability for dropout', default=0.7, type=float)
    
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--beta', default=0.001, type=float)

    return parser.parse_args()


if __name__ == '__main__':
    # IMPORT DATASET
    data_name              = 'sample'
    data_x, data_y, y_type = import_data(data_name = data_name)

    args                 = init_arg()
    K                    = args.K
    h_dim_FC             = args.h_dim_FC
    h_dim_RNN            = args.h_dim_RNN

    num_layer_encoder    = args.n_layer_enc
    num_layer_selector   = args.n_layer_sel
    num_layer_predictor  = args.n_layer_pre

    rnn_type             = args.rnn_type

    x_dim = np.shape(data_x)[2]
    y_dim = np.shape(data_y)[2]
    z_dim = h_dim_RNN * num_layer_encoder

    max_length = np.shape(data_x)[1]


    seed = 1234

    tr_data_x,te_data_x, tr_data_y,te_data_y = train_test_split(
        data_x, data_y, test_size=0.2, random_state=seed
    )

    tr_data_x,va_data_x, tr_data_y,va_data_y = train_test_split(
        tr_data_x, tr_data_y, test_size=0.2, random_state=seed
    )


    input_dims ={
        'x_dim': x_dim,
        'y_dim': y_dim,
        'y_type': y_type,
        'max_cluster': K,
        'max_length': max_length    
    }

    network_settings ={
        'h_dim_encoder': h_dim_RNN,
        'num_layers_encoder': num_layer_encoder,
        'rnn_type': rnn_type,
        'rnn_activate_fn': tf.nn.tanh,

        'h_dim_selector': h_dim_FC,
        'num_layers_selector': num_layer_selector,
        
        'h_dim_predictor': h_dim_FC,
        'num_layers_predictor': num_layer_predictor,
        
        'fc_activate_fn': tf.nn.relu
    }


    lr_rate    = args.lr_rate_init
    keep_prob  = args.keep_prob
    mb_size    = args.batch_size
    ITERATION  = args.itrs_init1
    check_step = 1000

    save_path = './{}/proposed/init/'.format(data_name)

    if not os.path.exists(save_path + '/models/'):
        os.makedirs(save_path + '/models/')


    print('Initialize Network...')

    tf.reset_default_graph()

    # Turn on xla optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model = AC_TPC(sess, "AC_TPC", input_dims, network_settings)


    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer(), feed_dict={model.E:np.zeros([K, z_dim]).astype(float)})

    avg_loss  = 0
    for itr in range(ITERATION):
        x_mb, y_mb  = f_get_minibatch(mb_size, tr_data_x, tr_data_y)

        _, tmp_loss = model.train_mle(x_mb, y_mb, lr_rate, keep_prob)
        avg_loss   += tmp_loss/check_step

        if (itr+1)%check_step == 0:                
            tmp_y, tmp_m = model.predict_y_hats(va_data_x)

            y_pred = tmp_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]
            y_true = va_data_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]

            AUROC = np.zeros([y_dim])
            AUPRC = np.zeros([y_dim])
            for y_idx in range(y_dim):
                auroc, auprc = f_get_prediction_scores(y_true[:, y_idx], y_pred[:, y_idx])
                AUROC[y_idx] = auroc
                AUPRC[y_idx] = auprc

            print ("ITR {:05d}: loss_2={:.3f} | va_auroc:{:.3f}, va_auprc:{:.3f}".format(
                    itr+1, avg_loss, np.mean(AUROC), np.mean(AUPRC))
                  )        
            avg_loss = 0

    saver.save(sess, save_path + 'models/model_K{}'.format(K))
    save_logging(network_settings, save_path + 'models/network_settings_K{}.txt'.format(K))


    alpha      = args.alpha
    beta       = args.beta

    mb_size    = args.batch_size
    M          = int(tr_data_x.shape[0]/mb_size) #for main algorithm
    keep_prob  = args.keep_prob
    lr_rate1   = args.lr_rate_clu_1
    lr_rate2   = args.lr_rate_clu_2

    save_path = './{}/proposed/trained/'.format(data_name)

    if not os.path.exists(save_path + '/models/'):
        os.makedirs(save_path + '/models/')

    if not os.path.exists(save_path + '/results/'):
        os.makedirs(save_path + '/results/')



    ### LOAD INITIALIZED NETWORK
    load_path = './{}/proposed/init/'.format(data_name)

    tf.reset_default_graph()

    # Turn on xla optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    network_settings = load_logging(load_path + 'models/network_settings_K{}.txt'.format(K))
    z_dim = network_settings['num_layers_encoder'] * network_settings['h_dim_encoder']

    model = AC_TPC(sess, "AC_TPC", input_dims, network_settings)

    saver = tf.train.Saver()
    saver.restore(sess, load_path + 'models/model_K{}'.format(K))


    print('=============================================')
    print('===== INITIALIZING EMBEDDING & SELECTOR =====')
    # K-means over the latent encodings
    e, s_init, tmp_z = initialize_embedding(model, tr_data_x, K)
    e = np.arctanh(e)
    sess.run(model.EE.initializer, feed_dict={model.E:e}) #model.EE = tf.nn.tanh(model.E)

    # update selector wrt initial classes
    ITERATION  = args.itrs_init2
    check_step = 1000

    avg_loss_s = 0
    for itr in range(ITERATION):
        z_mb, s_mb = f_get_minibatch(mb_size, tmp_z, s_init)
        _, tmp_loss_s = model.train_selector(z_mb, s_mb, lr_rate1, k_prob=keep_prob)

        avg_loss_s += tmp_loss_s/check_step
        if (itr+1)%check_step == 0:
            print("ITR:{:04d} | Loss_s:{:.4f}".format(itr+1, avg_loss_s) )
            avg_loss_s = 0

    tmp_ybars = model.predict_yy(np.tanh(e))
    new_e     = np.copy(e)
    print('=============================================')


    print('=============================================')
    print('========== TRAINING MAIN ALGORITHM ==========')
    '''
        L1: predictive clustering loss
        L2: sample-wise entropy loss
        L3: embedding separation loss
    '''

    ITERATION     = args.itrs_clu
    check_step    = 10

    avg_loss_c_L1 = 0
    avg_loss_a_L1 = 0
    avg_loss_a_L2 = 0
    avg_loss_e_L1 = 0 
    avg_loss_e_L3 = 0

    va_avg_loss_L1 = 0
    va_avg_loss_L2 = 0
    va_avg_loss_L3 = 0

    for itr in range(ITERATION):        
        e = np.copy(new_e)

        for _ in range(M):
            x_mb, y_mb = f_get_minibatch(mb_size, tr_data_x, tr_data_y)

            _, tmp_loss_c_L1  = model.train_critic(x_mb, y_mb, lr_rate1, keep_prob)
            avg_loss_c_L1    += tmp_loss_c_L1/(M*check_step)

            x_mb, y_mb = f_get_minibatch(mb_size, tr_data_x, tr_data_y)

            _, tmp_loss_a_L1, tmp_loss_a_L2 = model.train_actor(x_mb, y_mb, alpha, lr_rate2, keep_prob)
            avg_loss_a_L1 += tmp_loss_a_L1/(M*check_step)
            avg_loss_a_L2 += tmp_loss_a_L2/(M*check_step)
            
        for _ in range(M):
            x_mb, y_mb = f_get_minibatch(mb_size, tr_data_x, tr_data_y)

            _, tmp_loss_e_L1, tmp_loss_e_L3 = model.train_embedding(x_mb, y_mb, beta, lr_rate1, keep_prob)
            avg_loss_e_L1  += tmp_loss_e_L1/(M*check_step)
            avg_loss_e_L3  += tmp_loss_e_L3/(M*check_step)

            
        x_mb, y_mb = f_get_minibatch(mb_size, va_data_x, va_data_y)
        tmp_loss_L1, tmp_loss_L2, tmp_loss_L3 = model.get_losses(x_mb, y_mb)
        
        va_avg_loss_L1  += tmp_loss_L1/check_step
        va_avg_loss_L2  += tmp_loss_L2/check_step
        va_avg_loss_L3  += tmp_loss_L3/check_step

        new_e = sess.run(model.embeddings)

        if (itr+1)%check_step == 0:
            tmp_ybars = model.predict_yy(new_e)
            print ("ITR {:04d}: L1_c={:.3f}  L1_a={:.3f}  L1_e={:.3f}  L2={:.3f}  L3={:.3f} || va_L1={:.3f}  va_L2={:.3f}  va_L3={:.3f}".format(
                itr+1, avg_loss_c_L1, avg_loss_a_L1, avg_loss_e_L1, avg_loss_a_L2, avg_loss_e_L3,
                va_avg_loss_L1, va_avg_loss_L2, va_avg_loss_L3
            ))
            avg_loss_c_L1 = 0
            avg_loss_a_L1 = 0
            avg_loss_a_L2 = 0
            avg_loss_e_L1 = 0
            avg_loss_e_L3 = 0
            va_avg_loss_L1 = 0
            va_avg_loss_L2 = 0
            va_avg_loss_L3 = 0
    print('=============================================')


    saver.save(sess, save_path + 'models/model_K{}'.format(K))

    save_logging(network_settings, save_path + 'models/network_settings_K{}.txt'.format(K))
    np.savez(save_path + 'models/embeddings.npz', e=e)


    _, tmp_pi, tmp_m = model.predict_zbars_and_pis_m2(te_data_x)

    tmp_pi = tmp_pi.reshape([-1, K])[tmp_m.reshape([-1]) == 1]

    ncol = nrow = int(np.ceil(np.sqrt(K)))
    plt.figure(figsize=[4*ncol, 2*nrow])
    for k in range(K):
        plt.subplot(ncol, nrow, k+1)
        plt.hist(tmp_pi[:, k])
    plt.suptitle("Clustering assignment probabilities")
    # plt.show()
    plt.savefig(save_path + 'results/figure_clustering_assignments.png')
    plt.close()


    # check selector outputs and intialized classes
    pred_y, tmp_m = model.predict_s_sample(tr_data_x)

    pred_y = pred_y.reshape([-1, 1])[tmp_m.reshape([-1]) == 1]
    print(np.unique(pred_y))

    plt.hist(pred_y[:, 0], bins=15, color='C1', alpha=1.0)
    # plt.show()
    plt.savefig(save_path + 'results/figure_clustering_hist.png')
    plt.close()


    tmp_y, tmp_m = model.predict_y_bars(te_data_x)


    y_pred = tmp_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]
    y_true = te_data_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]


    AUROC = np.zeros([y_dim])
    AUPRC = np.zeros([y_dim])
    for y_idx in range(y_dim):
        auroc, auprc = f_get_prediction_scores(y_true[:, y_idx], y_pred[:, y_idx])
        AUROC[y_idx] = auroc
        AUPRC[y_idx] = auprc

    print('AUROC: {}'.format(AUROC))
    print('AUPRC: {}'.format(AUPRC))

    pred_y, tmp_m = model.predict_s_sample(te_data_x)

    pred_y = (pred_y * tmp_m).reshape([-1, 1])
    pred_y = pred_y[(tmp_m.reshape([-1, 1]) == 1)[:, 0], 0]

    true_y = (te_data_y * np.tile(np.expand_dims(tmp_m, axis=2), [1,1,y_dim])).reshape([-1, y_dim])
    true_y = true_y[(tmp_m.reshape([-1]) == 1)]
    true_y = np.argmax(true_y, axis=1)

    tmp_nmi    = normalized_mutual_info_score(true_y, pred_y)
    tmp_ri     = adjusted_rand_score(true_y, pred_y)
    tmp_purity = purity_score(true_y, pred_y)

    print('NMI:{:.4f}, RI:{:.4f}, PURITY:{:.4f}'.format(tmp_nmi, tmp_ri, tmp_purity))