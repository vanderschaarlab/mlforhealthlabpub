import tensorflow as tf
import numpy as np

import os, sys
import random

from tqdm import tqdm

import utils_network as utils
import import_data as impt
import class_clustering as clustering

from class_DeepTPC import DeepTPC

from sklearn.metrics import silhouette_samples
from scipy.stats import expon


def KL_divergence_categorical(P_, Q_):
    return np.sum(P_ * np.log(P_/(Q_+1e-8) + 1e-8))


def JS_divergence_categorical(P_, Q_):
    return 1/2*(KL_divergence_categorical(P_, Q_) + KL_divergence_categorical(Q_, P_))


def get_distance_matrix(empirical_M1, empirical_M2):
    N = np.shape(empirical_M1)[0]
    M = np.shape(empirical_M2)[0]
    distance_M = np.zeros([N, M])
    for n_i in tqdm(range(N)):
        tmp_P = empirical_M1[n_i, :]
        for n_j in range(M):
            tmp_Q = empirical_M2[n_j, :]
            distance_M[n_i, n_j] = JS_divergence_categorical(tmp_P, tmp_Q)
    return distance_M


def get_similarity_sample_vs_sample(dist_matrix, S_, num_Cluster):
    similarity_measure = np.zeros([num_Cluster, num_Cluster])
    for k_i in range(num_Cluster):
        tmp_idx_i = (S_ == k_i)
        N_i = np.sum(tmp_idx_i)
        for k_j in range(num_Cluster):
            tmp_idx_j = (S_ == k_j)
            N_j = np.sum(tmp_idx_j)
            if k_j == k_i:
                similarity_measure[k_i, k_j] = np.mean( np.sum(dist_matrix[tmp_idx_i][:, tmp_idx_j], axis=1) / (N_j - 1) )
            else:
                similarity_measure[k_i, k_j] = np.mean( np.sum(dist_matrix[tmp_idx_i][:, tmp_idx_j], axis=1) / (N_j) )
    return similarity_measure


def get_similarity_sample_vs_centroid(dist_matrix, S_, num_Cluster):
    similarity_measure = np.zeros([num_Cluster, num_Cluster])
    for k_i in range(num_Cluster):
        tmp_idx_i = (S_ == k_i)
        for k_j in range(num_Cluster):        
            similarity_measure[k_i, k_j] = np.mean(dist_matrix[tmp_idx_i][:, k_j])
    return(similarity_measure)


def KL_divergence_exp(lambda_p, lambda_q):
    return np.log(lambda_p + 1e-8) - np.log(lambda_q + 1e-8) + lambda_q/(lambda_p + 1e-8) - 1

def JS_divergence_exp(lambda_p, lambda_q):
    return 1/2*(KL_divergence_exp(lambda_p, lambda_q) + KL_divergence_exp(lambda_q, lambda_p))

def get_dist_matrix_exp(pred_D):
    labmdas = np.zeros(np.shape(pred_D)[0])
    dist_matrix_exp = np.zeros([np.shape(pred_D)[0], np.shape(pred_D)[0]])
    for i in range(len(labmdas)):
        _,  scale = expon.fit(pred_D[i, :, 0], floc=0)
        labmdas[i] = 1./scale

    for i in tqdm(range(np.shape(dist_matrix_exp)[0])):
        for j in range(np.shape(dist_matrix_exp)[0]):
            dist_matrix_exp[i,j] = JS_divergence_exp(labmdas[i], labmdas[j])
    return dist_matrix_exp



def future_prediction_thinning_new(model_, num_Event, Z_, delta_range_, max_delta,tqdm_mode='ON'):
    tmp_y_stacked  = np.zeros([np.shape(Z_)[0], num_Event, len(delta_range_)])

    for l in range(len(delta_range_)):
        tmp_y_stacked[:, :, l] = model_.f_get_prediction_Y_pred(Z_=Z_, D_=delta_range_[[l]])

    lambda_max = np.max(tmp_y_stacked, axis=2)[:, 1:]

    tmp_delta_sum = np.zeros(np.shape(lambda_max))

    if tqdm_mode in ['on', 'ON', 'On']:
        LIST = tqdm(range(np.shape(Z_)[0]))
    else:
        LIST = range(np.shape(Z_)[0])

    for i in LIST:    
        z_in = Z_[[i]]

        for m in range(np.shape(lambda_max)[1]):
            count = 0

            while(1):
                count += 1
                tmp_delta = np.random.exponential(1 / lambda_max[i, m])  # np.random.exponential uses beta = 1/lambda
                tmp_delta_sum[i, m] += tmp_delta

                tmp_u     = np.random.uniform(0,1)

                tmp_lambda = model_.f_get_prediction_Y_pred(Z_=z_in, D_=np.asarray(tmp_delta_sum[i, m]).reshape([-1]))
                tmp_lambda = tmp_lambda[:, 1:]

                if tmp_u * lambda_max[i,m] <= tmp_lambda[0, m]:
                    break
                if tmp_delta_sum[i,m] > max_delta:
                    break

    pred_delta = np.min(tmp_delta_sum, axis=1)
    pred_M     = np.zeros([np.shape(Z_)[0], num_Event])
    tmp_M      = np.argmin(tmp_delta_sum, axis=1)

    for m in range(num_Event - 1):
        pred_M[np.where(tmp_M == m)[0], m+1] = 1

    return (pred_M, pred_delta, tmp_delta_sum)



def get_next_event(model_, Z_, num_Event, check_range, num_sample_=100, Delta_max=365):
    N = np.shape(Z_)[0]

    pred_M     = np.zeros([N, num_sample_, num_Event])
    pred_D     = np.zeros([N, num_sample_, 1])
    pred_D_all = np.zeros([N, num_sample_, num_Event-1])

    for i in tqdm(range(N)):
        pred_M[i, :, :], pred_D[i, :, 0], pred_D_all[i, :, :] = future_prediction_thinning_new(
            model_, num_Event, np.tile(Z_[i], [num_sample_, 1]), check_range, Delta_max, 'OFF'
        )

    return (pred_M, pred_D, pred_D_all)



def evaluate_DeepTPC_cohesion_n_separation(model_, M_, D_, X_, S_, Mask_, mu_z_, mu_y_, delta_range_, num_sample):
    num_Event   = np.shape(M_)[2]
    num_Cluster = len(np.unique(S_))

    Z_ = model_.f_get_prediction_Z(M_, D_, X_, Mask_)

    #get "num_sample" samples for each cluster member using Thinning Alg.
    s_pred_M, s_pred_D, s_pred_Dall = get_next_event(model_, Z_, num_Event, delta_range_, num_sample_=num_sample, Delta_max=365)
    #get "num_sample" samples for each cluster centroid using Thinning Alg.
    c_pred_M, c_pred_D, c_pred_Dall = get_next_event(model_, mu_z_, num_Event, delta_range_, num_sample_=num_sample, Delta_max=365)

    s_empirical_M = np.mean(s_pred_M, axis=1)[:, 1:]
    c_empirical_M = np.mean(c_pred_M, axis=1)[:, 1:]

    s_dist_matrix = get_distance_matrix(s_empirical_M, s_empirical_M)
    c_dist_matrix = get_distance_matrix(s_empirical_M, c_empirical_M)

    s_silhouette_score = silhouette_samples(s_dist_matrix, labels = S_, metric='precomputed')

    similarity_sample_vs_sample = get_similarity_sample_vs_sample(s_dist_matrix, S_, num_Cluster)
    similarity_sample_vs_centroid = get_similarity_sample_vs_centroid(c_dist_matrix, S_, num_Cluster)
    
    s_dist_matrix_exp = get_dist_matrix_exp(s_pred_D)
    s_silhouette_score_exp = silhouette_samples(s_dist_matrix_exp, labels = S_, metric='precomputed')
    
    predictions = {
        's_pred_M': s_pred_M, 
        's_pred_D': s_pred_D, 
        's_pred_Dall': s_pred_Dall,
        'c_pred_M': c_pred_M,
        'c_pred_D': c_pred_D, 
        'c_pred_Dall': c_pred_Dall
    }

    measures = {
        's_empirical_M': s_empirical_M, 
        'c_empirical_M': c_empirical_M, 
        's_dist_matrix': s_dist_matrix,
        'c_dist_matrix': c_dist_matrix,
        's_silhouette_score': s_silhouette_score, 
#         's_dist_matrix_exp': s_dist_matrix_exp,
#         's_silhouette_score_exp': s_silhouette_score_exp,        
        'similarity_sample_vs_sample': similarity_sample_vs_sample,
        'similarity_sample_vs_centroid':similarity_sample_vs_centroid
    }

    return predictions, measures