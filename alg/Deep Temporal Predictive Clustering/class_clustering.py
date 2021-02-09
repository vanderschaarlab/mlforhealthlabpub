import numpy as np

from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer, random_center_initializer


def kmeans_MTPP_MODIFIED(model_, M_, D_, X_, Mask_, delta_range_, num_Cluster, beta_1, beta_2, beta_ms_, init_Z=None):
    # clustering assignment is modified; since this was not able to be duplicated for hold-out samples

    num_Event = np.shape(M_)[2]
    L         = len(delta_range_)
    
    ## DEFINE DISTANCE FUNCTIONS
    def f_get_XX_dist(p1, p2):
        p1 = p1.reshape([-1, z_dim + num_Event * L])
        p2 = p2.reshape([-1, z_dim + num_Event * L])

        pz1, py1 = p1[:, :z_dim], p1[:, z_dim:]
        pz2, py2 = p2[:, :z_dim], p2[:, z_dim:]

        dz = f_get_ZZ_dist(pz1, pz2)
        dy = f_get_YY_dist(py1, py2)

        return beta_1 * dz + beta_2 * dy    

    def f_get_YY_dist(p1, p2):
        tmp_YY = (p1.reshape([-1, num_Event, L]) - p2.reshape([-1, num_Event, L]) )**2
        tmp_result = np.sum(
            (delta_range_[-1] / (L-1) * (np.sum(tmp_YY, axis=2) - tmp_YY[:, :, 0] - tmp_YY[:, :, -1]) )[:, 1:] * beta_ms_, 
            axis=1
        )
        return tmp_result

    def f_get_ZZ_dist(p1, p2):
        return np.sum(np.square(p1 - p2), axis=1)
    
    Z_ = model_.f_get_prediction_Z(M_, D_, X_, Mask_)
    z_dim = np.shape(Z_)[1]
    
#     Y_ = sess.run(YY_stacked, feed_dict={ZZ: Z_, delta_range:delta_range_, mb_size:np.shape(Z_)[0], keep_prob:1.0})
    Y_ = model_.f_get_prediction_YY_stacked(Z_, delta_range_)
    Y_ = Y_.reshape([np.shape(Y_)[0], num_Event * L])
    
    sample = np.concatenate([Z_, Y_], axis=1)
    
    if init_Z is None:
        initial_centers = kmeans_plusplus_initializer(sample, num_Cluster).initialize()
    else:
#         init_Y = sess.run(YY_stacked, feed_dict={ZZ: init_Z, delta_range:delta_range_, mb_size:np.shape(Z_)[0], keep_prob:1.0})
        init_Y = model_.f_get_prediction_YY_stacked(init_Z, delta_range_)
        init_Y = init_Y.reshape([np.shape(init_Y)[0], num_Event * L])
        initial_centers = np.concatenate([init_Z, init_Y], axis=1)
        
    metric = distance_metric(type_metric.USER_DEFINED, func=f_get_XX_dist)
    kmeans_instance = kmeans(sample, initial_centers=initial_centers, metric=metric)

    # run cluster analysis and obtain results
    kmeans_instance.process()    
    MU_ = np.asarray(kmeans_instance.get_centers())
    
    ### assign clusters with minimum distance
    tmp_dist = np.zeros([np.shape(Z_)[0], num_Cluster])
    for i in range(np.shape(sample)[0]):
        tmp_dist[i, :] = f_get_XX_dist(sample[[i]], MU_)

    S_ = np.argmin(tmp_dist, axis=1)
    
    return (MU_[:, :z_dim], MU_[:, z_dim:], S_) #centers [num_Cluster, z_dim] and assignments [None]




def kmeans_MTPP_TEST(model_, M_, D_, X_, Mask_, MU_Z, MU_Y, delta_range_, num_Cluster, beta_1, beta_2, beta_ms_):

    num_Event = np.shape(M_)[2]
    L         = len(delta_range_)

    ## DEFINE DISTANCE FUNCTIONS
    def f_get_XX_dist(p1, p2):
        p1 = p1.reshape([-1, z_dim + num_Event * L])
        p2 = p2.reshape([-1, z_dim + num_Event * L])

        pz1, py1 = p1[:, :z_dim], p1[:, z_dim:]
        pz2, py2 = p2[:, :z_dim], p2[:, z_dim:]

        dz = f_get_ZZ_dist(pz1, pz2)
        dy = f_get_YY_dist(py1, py2)

        return beta_1 * dz + beta_2 * dy    

    def f_get_YY_dist(p1, p2):
        tmp_YY = (p1.reshape([-1, num_Event, L]) - p2.reshape([-1, num_Event, L]) )**2
        tmp_result = np.sum(
            (delta_range_[-1] / (L-1) * (np.sum(tmp_YY, axis=2) - tmp_YY[:, :, 0] - tmp_YY[:, :, -1]) )[:, 1:] * beta_ms_, 
            axis=1
        )
        return tmp_result

    def f_get_ZZ_dist(p1, p2):
        return np.sum(np.square(p1 - p2), axis=1)

    Z_ = model_.f_get_prediction_Z(M_, D_, X_, Mask_)
    z_dim = np.shape(Z_)[1]

#     Y_ = sess.run(YY_stacked, feed_dict={ZZ: Z_, delta_range:delta_range_, mb_size:np.shape(Z_)[0], keep_prob:1.0})
    Y_ = model_.f_get_prediction_YY_stacked(Z_, delta_range_)
    Y_ = Y_.reshape([np.shape(Y_)[0], num_Event * L])

    sample = np.concatenate([Z_, Y_], axis=1)

    ### get centroid (Z and Y)    
    MU_   = np.concatenate([MU_Z, MU_Y], axis=1)
    
    ### assign clusters with minimum distance
    tmp_dist = np.zeros([np.shape(Z_)[0], num_Cluster])
    for i in range(np.shape(sample)[0]):
        tmp_dist[i, :] = f_get_XX_dist(sample[[i]], MU_)

    S_ = np.argmin(tmp_dist, axis=1)

    return (MU_[:, :z_dim], MU_[:, z_dim:], S_) #centers [num_Cluster, z_dim] and assignments [None]