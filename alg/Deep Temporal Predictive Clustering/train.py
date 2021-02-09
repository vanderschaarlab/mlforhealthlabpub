import tensorflow as tf
import numpy as np
import pandas as pd

import os, sys
import random
import matplotlib.pyplot as plt

from tensorflow.contrib.layers import fully_connected as FC_Net
from tensorflow.python.ops.rnn import _transpose_batch_time

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
# from sklearn.metrics import silhouette_samples


from tqdm import tqdm

import utils_network as utils
import import_data as impt
import class_clustering as clustering

from class_DeepTPC import DeepTPC


def print_lambda_predictions_v2(model_, M_, D_, X_, Mask_, MU_, S_, delta_range_, num_Cluster_, m):
    tmp_y_stacked = model_.f_get_prediction_Y_stacked(M_=M_, D_=D_, X_=X_, Mask_=Mask_, delta_range_=delta_range_)
    tmp_psi_stacked = model_.f_get_prediction_PSI_stacked(MU_=MU_, K_=num_Cluster_, delta_range_=delta_range_)
#     tmp_psi_stacked = f_get_prediction_PSI_stacked(M_=M_, D_=D_, MU_=MU_, K_=num_Cluster_, delta_range_=delta_range_)

    y_lim_max = 0
    
    for k in range(num_Cluster_):
        tmp_y_lim = 1.2 * np.max(np.percentile(tmp_y_stacked[np.where(S_ == k)[0], m+1, :], q=95, axis =0))
        if y_lim_max < tmp_y_lim:
            y_lim_max = tmp_y_lim

    plt.figure(figsize=[15,3])
    for k in range(num_Cluster_):
        plt.subplot(1,num_Cluster_,k+1)
        plt.plot(delta_range_, tmp_psi_stacked[k, m+1, :], color = 'purple')
        plt.plot(delta_range_, np.mean(tmp_y_stacked[np.where(S_ == k)[0], m+1, :], axis =0), color = 'C'+str(k))
        plt.plot(delta_range_, np.percentile(tmp_y_stacked[np.where(S_ == k)[0], m+1, :], q=95, axis =0), color = 'C'+str(k), linestyle='dashed')
        plt.plot(delta_range_, np.percentile(tmp_y_stacked[np.where(S_ == k)[0], m+1, :], q=5, axis =0), color = 'C'+str(k), linestyle='dashed')

        plt.grid()    
        plt.ylim([0, y_lim_max])
    #     plt.plot(delta_range_, np.max(tmp_y_stacked[np.where(s_ == k)[0], m+1, :], axis =0), color = 'C'+str(k), linestyle='dashed')
    #     plt.plot(delta_range_, np.min(tmp_y_stacked[np.where(s_ == k)[0], m+1, :], axis =0), color = 'C'+str(k), linestyle='dashed')

    plt.show()
    plt.close()
    
    

def print_z_pca(Z_, S_, num_Cluster_):
    pca = PCA(n_components=2)
    pca.fit(Z_)

    z_pca = pca.transform(Z_)

    plt.figure(figsize=[4,4])
#         plt.scatter(z_pca[:,0], z_pca[:,1], c=S_, edgecolors='black', alpha=0.5)
    for k in range(num_Cluster_):
        plt.scatter(z_pca[np.where(S_ == k)[0], 0], z_pca[np.where(S_ == k)[0], 1], c='C'+str(k), edgecolors='black', alpha=0.5)
    plt.grid()
    plt.show()
    plt.close()


def train_init(DATA, input_dims, network_settings, train_parameters, init_path):
    ITERATION = train_parameters['ITERATION']
    
    M_ = DATA['M']
    D_ = DATA['D']
    T_ = DATA['T']
    X_ = DATA['X']
    Mask_ = DATA['Mask']

    mb_size_ = train_parameters['mb_size']
    lr_train = train_parameters['lr_train']
    k_prob   = train_parameters['k_prob']
    seed     = train_parameters['seed']

    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model = DeepTPC(sess, "Deep_TPC", input_dims, network_settings)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    (tr_M,va_M, tr_D,va_D, tr_T,va_T, tr_X,va_X, tr_Mask,va_Mask) = train_test_split(
        M_, D_, T_, X_, Mask_, test_size=0.2, random_state=seed
    )

    check_step = 100

    avg_mle_loss = 0
    min_mle_loss =  1e8
    for itr in range(ITERATION):
        idx = range(np.shape(tr_M)[0])
        idx = random.sample(idx, mb_size_)

        M_mb = tr_M[idx, :, :].astype(float)
        T_mb = tr_T[idx, :, :].astype(float)
        D_mb = tr_D[idx, :, :].astype(float)
        X_mb = tr_X[idx, :].astype(float)

        _, mle_loss = model.train_mle(
            M_mb, D_mb, X_mb, lr_train, k_prob
        )

        avg_mle_loss   += mle_loss/check_step

        if (itr+1)%check_step == 0:
            va_mle_loss = model.get_loss_mle(va_M, va_D, va_X)

            tmp_string1 = "ITR {:04d}: || MLE_loss={:.4f} | va_MLE_loss={:.4f}".format(
                itr+1, avg_mle_loss, va_mle_loss
            )
            tmp_string  = tmp_string1

            if va_mle_loss < min_mle_loss:
                min_mle_loss = va_mle_loss
                print('saved...')
                saver.save(sess, init_path + 'MTPP_v7X_init')

            print(tmp_string)
            avg_mle_loss = 0

    saver.restore(sess, init_path + 'MTPP_v7X_init')
    
    return sess, model

            
            
def train_DeepTPC(DATA, input_dims, network_settings, train_parameters1, train_parameters2, init_path):
    ITERATION = train_parameters2['ITERATION']

    M_ = DATA['M']
    D_ = DATA['D']
    T_ = DATA['T']
    X_ = DATA['X']
    Mask_ = DATA['Mask']

    mb_size_ = train_parameters1['mb_size']
    lr_train = train_parameters1['lr_train']
    k_prob   = train_parameters1['k_prob']
    seed     = train_parameters1['seed']
    
    num_Cluster = train_parameters2['num_Cluster']
    alpha = train_parameters2['alpha']
    beta = train_parameters2['beta']
    beta_cluster = train_parameters2['beta_cluster']
    beta_ms = train_parameters2['beta_ms']
    gamma = train_parameters2['gamma']
    
    delta_range = network_settings['delta_range']

    
    save_path = init_path + 'K{}/'.format(num_Cluster)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    
    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model = DeepTPC(sess, "Deep_TPC", input_dims, network_settings)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    (tr_M,va_M, tr_D,va_D, tr_T,va_T, tr_X,va_X, tr_Mask,va_Mask) = train_test_split(
        M_, D_, T_, X_, Mask_, test_size=0.2, random_state=seed
    )

    saver.restore(sess, init_path + 'MTPP_v7X_init')
    
    ### CLUSTER INITIALIZATION
    m_test = np.argmax(np.sum(np.sum(M_, axis=1)[:, 1:], axis=0))

    mu_z, mu_y, tr_S = clustering.kmeans_MTPP_MODIFIED(model, tr_M, tr_D, tr_X, tr_Mask, delta_range, num_Cluster, 1.0, beta_cluster, beta_ms, init_Z=None)
    tr_Z = model.f_get_prediction_Z(tr_M, tr_D, tr_X, tr_Mask)

    print_lambda_predictions_v2(model, tr_M, tr_D, tr_X, tr_Mask, mu_z, tr_S, delta_range, num_Cluster, m=m_test)
    print_z_pca(tr_Z, tr_S,num_Cluster)
    
    ### TRAINING
    check_step = 100
    
    avg_total_loss = 0
    avg_mle_loss = 0
    avg_c_loss = 0

    min_total_loss =  1e8

    for itr in range(ITERATION):
        idx = range(np.shape(tr_M)[0])
        idx = random.sample(idx, mb_size_)

        M_mb = tr_M[idx, :, :].astype(float)
        T_mb = tr_T[idx, :, :].astype(float)
        D_mb = tr_D[idx, :, :].astype(float)
        X_mb = tr_X[idx, :].astype(float)
        Mask_mb = tr_Mask[idx, :].astype(float)

        Z_mb = model.f_get_prediction_Z(M_mb, D_mb, X_mb, Mask_mb)
        S_mb = tr_S[idx]
        mu_z = mu_z

        _, _, total_loss, mle_loss, c_loss, d_z_ho, d_y_ho, d_y_he = model.train_total(
            M_mb, D_mb, X_mb, Mask_mb, S_mb, mu_z, num_Cluster, delta_range, alpha, beta, beta_ms, gamma, lr_train, k_prob
        )

        _, mle_loss = model.train_mle(
            M_mb, D_mb, X_mb, lr_train, k_prob
        )

        avg_total_loss += (total_loss)/check_step
        avg_mle_loss   += mle_loss/check_step
        avg_c_loss     += c_loss/check_step


        if (itr+1)%check_step == 0:
            _, _, va_S = clustering.kmeans_MTPP_TEST(model, va_M, va_D, va_X, va_Mask, mu_z, mu_y, delta_range, num_Cluster, 1.0, beta_cluster, beta_ms)

            va_total_loss = model.get_loss_total(
                va_M, va_D, va_X, va_Mask, va_S, mu_z, num_Cluster, delta_range, alpha, beta, beta_ms, gamma
            )

            tmp_string1 = "K{} || {:04d}: | T_loss={:.4f} | MLE_loss={:.4f} | C_loss={:.4f} | VA_T_loss={:.4f} |".format(
                num_Cluster, itr+1, avg_total_loss, avg_mle_loss, avg_c_loss, va_total_loss
            )
            tmp_string  = tmp_string1

            if va_total_loss < min_total_loss:
                min_total_loss = va_total_loss
                # _, _, te_S = clustering.kmeans_MTPP_TEST(model, te_M, te_D, te_X, te_Mask, mu_z, mu_y, delta_range, num_Cluster, 1.0, beta_cluster, beta_ms)

                print('saved...')
                print_lambda_predictions_v2(model, tr_M, tr_D, tr_X, tr_Mask, mu_z, tr_S, delta_range, num_Cluster, m=m_test)
                print_z_pca(tr_Z, tr_S,num_Cluster)

                saver.save(sess, save_path + 'MTPP_v7X_clustered')
                np.savez(save_path + 'MTPP_v7X_clustered.npz',
                        tr_S = tr_S,
                        # te_S = te_S,
                        mu_z = mu_z,
                        mu_y = mu_y)

            print(tmp_string)
            avg_total_loss = 0
            avg_mle_loss = 0
            avg_c_loss = 0


        if (itr+1)%check_step==0:        
            tr_Z = model.f_get_prediction_Z(tr_M, tr_D, tr_X, tr_Mask)        
            mu_z, mu_y, tr_S = clustering.kmeans_MTPP_MODIFIED(model, tr_M, tr_D, tr_X, tr_Mask, delta_range, num_Cluster, 1.0, beta_cluster, beta_ms, init_Z=mu_z)
            # _, _, te_S = clustering.kmeans_MTPP_TEST(model, te_M, te_D, te_X, te_Mask, mu_z, mu_y, delta_range, num_Cluster, 1.0, beta_cluster, beta_ms)
    
    
    saver.restore(sess, save_path + 'MTPP_v7X_clustered')
    npz = np.load(save_path + 'MTPP_v7X_clustered.npz')
    
    return sess, model, npz