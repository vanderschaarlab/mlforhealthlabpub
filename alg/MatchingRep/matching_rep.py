from sklearn.cluster import KMeans
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Dot, Multiply, Concatenate, BatchNormalization
from keras import backend as K
from keras.utils import np_utils, plot_model
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
import utils
import numpy as np
import os

class MatchingRep():
    """
    Class implements Matching rep. model using NNs

    Parameters
    ----------------------------------------------
    n_feature_x: int
        number of features of each patient
    n_feature_o: int
        number of features of each organ
    n_clusters: int, default 3
        number of clusters classifying into
    AEdims: list(int), default [48, 48, 8]
        hidden dimensions of autoencoder network
    RPdims: list(int), default [48, 96, 10]
        hidden dimensions of representation network
    FCdims: list(int), default [20, 20]
        hidden dimensions of full connection layers before output layer
    """

    def Models(self, act='relu'):
        o = Input(shape=(self.n_feature_o, ), name='organ_input')
        z_o = BatchNormalization()(o)
        # internal layers in encoder
        for i in range(len(self.AEdims)):
            z_o = Dense(self.AEdims[i], activation=act, name='encoder_layer_%d' % i)(z_o)

        # encoder output
        encoded = z_o

        decoded = encoded
        # internal layers in decoder
        for i in range(len(self.AEdims[:-1])):
            decoded = Dense(self.AEdims[-i-1], activation=act, name='decoder_layer_%d' % i)(decoded)

        # decoder output
        rec_o = Dense(self.n_feature_o, name='reconstruction_layer')(decoded)

        # cluster output
        clus = utils.ClusteringLayer(self.n_clusters, name='clustering_layer')(encoded)

        x = Input(shape=(self.n_feature_x, ), name='recipient_input')
        
        z_x = BatchNormalization()(x)
        for i in range(len(self.RPdims)):
            z_x = Dense(self.RPdims[i], activation=act, name='representation_layer_%d' % i)(z_x)
    
        ys = []
        for i in range(self.n_clusters):
            z = z_x
            for j in range(len(self.FCdims)):
                z = Dense(self.FCdims[j], activation=act, name='branch_%d_%d' % (i, j))(z)
            z = Dense(1, activation=act, name='branch_%d' % i)(z)
            ys.append(z)
        
        ys = Concatenate()(ys)
        y = Dot(axes=1)([ys, clus])
        out = Concatenate()([ys, z_x, clus])
    
        return Model(inputs=o, outputs=encoded), Model(inputs=o, outputs=rec_o), Model(inputs=o, outputs=clus), Model(inputs=x, outputs=ys), Model(inputs=[x, o], outputs=y), Model(inputs=[x, o], outputs=out)

    # pre-train autoencoder
    def fit_AE(self, X, optimizer='Adam', batch_size=256):
        self.models['AE'].compile(optimizer=optimizer, loss='mse')
        self.models['AE'].fit(X, X, batch_size=batch_size, epochs=200, verbose=0)
    
    # pre-train clustering network 
    def fit_CLUS(self, X, optimizer='Adam', batch_size=256):
        # initialize clustering layer by kmeans
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        clus = kmeans.fit_predict(self.models['encoder'].predict(X))
        clus_pre = np.copy(clus)
        self.models['CLUS'].get_layer(name='clustering_layer').set_weights([kmeans.cluster_centers_])

        self.models['CLUS'].compile(optimizer=optimizer, loss='kld')

        # train the clusters        
        index = 0
        maxiter = 5000
        update_interval = 100
        index_array = np.arange(X.shape[0])
        conv_threshold = 0.0001
        for i in range(int(maxiter)):
            if i % update_interval == 0:
                q = self.models['CLUS'].predict(X, verbose=0)
                p = utils.target_distribution(q)  # update the auxiliary target distribution p
                
                clus = q.argmax(axis=1)
                change_ratio = np.sum(clus != clus_pre).astype(np.float32) / clus.shape[0]
                clus_pre = np.copy(clus)
                if i > 0 and change_ratio < conv_threshold:
                    print('Reached convergence threshold. Stopping training.')
                    break 

            idx = index_array[index * batch_size: min((index+1) * batch_size, X.shape[0])]
            loss = self.models['CLUS'].train_on_batch(x=X[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= X.shape[0] else 0
            # print(loss)

    # train matching rep. network 
    def fit_MatchingRep(self, X, Y, validation_data=None, optimizer='Adam', batch_size=256, epochs=50, verbose=1):
        self.models['MatchingRep_train'].compile(optimizer=optimizer, loss=utils.MatchingRepLoss)

        # create checkpoint to save the best model
        if not os.path.isdir('./model'):
            os.mkdir('./model')
        checkpointer = ModelCheckpoint(filepath='./model/MatchingRepCheckpoint', verbose=verbose, save_best_only=True, save_weights_only=True)

        # result holder
        history = {}
        history['loss'] = []
        if validation_data is not None:
            history['val_loss'] = []

        iters = max(1, int(epochs/5))
        for _ in range(iters):
            # update target clusters
            q = self.models['CLUS'].predict(X[1], verbose=0)
            p = utils.target_distribution(q)
            target = np.hstack([Y.reshape(-1, 1), p])
        
            if validation_data is not None:
                q_val = self.models['CLUS'].predict(validation_data[0][1], verbose=0)
                p_val = utils.target_distribution(q_val)
                target_val = np.hstack([validation_data[1].reshape(-1, 1), p_val])
                hist = self.models['MatchingRep_train'].fit(X, target, validation_data=(validation_data[0], target_val), batch_size=batch_size, epochs=5, callbacks=[checkpointer], verbose=verbose)
            else:
                hist = self.models['MatchingRep_train'].fit(X, target, batch_size=batch_size, epochs=5, callbacks=[checkpointer], verbose=verbose)
            
            history['loss'] += hist.history['loss']
            if validation_data is not None:
                history['val_loss'] += hist.history['val_loss']

        # self.models['MatchingRep_train'].load_weights('./model/MatchingRepCheckpoint')
        print('done', '='*30)

        return history

    """
    Train the model

    Parameters
    --------------------------------------------------
    X: [2d-array(n_samples, n_feature_x), 2d-array(n_samples, n_feature_o)]
        training data
        the first 2d-array corresponds to patients data
        the second 2d-array corresponds to organ data
    Y: 1d-array
        training target---observed outcomes
    validation_data: ([2d-array, 2d-array], 1d-array), default None
        validation data
    optimizer: optimizer, default 'Adam'
    batch_size: int, default 256
    epochs: int, default 40
    """
    def fit(self, X, Y, validation_data=None, optimizer='Adam', batch_size=256, epochs=50, verbose=1):
        print('start training', '='*30)
        
        print('pre-training auto-encoder')
        self.fit_AE(X[1], optimizer=optimizer, batch_size=batch_size)

        print('pre-training clusters')
        self.fit_CLUS(X[1], optimizer=optimizer, batch_size=batch_size)

        print('start training MatchingRep', '='*20)
        return self.fit_MatchingRep(X, Y, validation_data=validation_data, optimizer=optimizer, batch_size=batch_size, epochs=epochs, verbose=verbose)
    
    # load the best model so far
    def load_weights(self, path='./model/MatchingRepCheckpoint'):
        self.models['MatchingRep_train'].load_weights(path)

    # predict all potential outcomes for each patient
    #   X: 2d-array (n_samples, n_feature_X) 
    def predict(self, X):
        return self.models['MatchingRep'].predict(X)

    # predict soft clustering result for each organ
    #   X: 2d_array (n_samples, n_feature_o)
    def predict_clus(self, X):
        return self.models['CLUS'].predict(X)

    # predict potential outcome for each organ-patient pair
    #   X: [2d-array(n_samples, n_feature_x), 2d-array(n_samples, n_feature_o)]
    def predict_y(self, X):
        return self.models['MatchingRep_y'].predict(X)

    # evaluate the model on precision of estimated outcomes
    #   X: [2d-array(n_samples, n_feature_x), 2d-array(n_samples, n_feature_o)]
    #   Y: 1d-array, ground truth outcomes   
    def evaluate(self, X, Y):
        self.models['MatchingRep_y'].compile(loss='mse')
        return self.models['MatchingRep_y'].evaluate(X, Y)

    # allocate a given organ to a patient in the waiting queue
    # organ is allocated to the first patient that has the organ's type as the best type (Matching rep. (FCFS))
    # code can be modified to achieve other policies (Matching rep. (BF), Matching rep. (UF))
    def allocate_one(self, patients, organ):
        # estimate potential outcomes for all patients
        ys = self.predict(patients)
        # compute the best organ type for each patient
        best_clus = np.argmax(ys, axis=-1)

        # predict the soft clustering result of the given organ
        o_clus = self.predict_clus(organ.reshape((1, -1)))
        # order cluster labels for the organ
        o_clus = np.vstack([o_clus, np.arange(self.n_clusters)]).T
        o_clus = sorted(o_clus, key=lambda x:x[0])
        o_clus.reverse()
        o_clus = list(map(lambda x:int(x[1]), o_clus))

        # allocate the organ to the first patient that has the organ's type as the best type
        # if no such a patient, consider the organ as its second likely type
        for c in o_clus:
            xs = np.array(np.where(best_clus==c)).reshape(-1)
            if len(xs) == 0:
                continue
            # return the index of the patient
            # ---------modify here to achieve other policies--------------
            return xs[0]

        # return -1 if no patients
        return -1

    # initialize an instance of the class
    def __init__(self, n_feature_x, n_feature_o, n_clusters=3,
     AEdims=[48, 48, 8], RPdims=[48, 96, 10], FCdims=[20, 20],
     alpha=1, beta=.1):
        self.n_feature_x = n_feature_x
        self.n_feature_o = n_feature_o
        self.n_clusters = n_clusters
        self.AEdims = AEdims
        self.RPdims = RPdims
        self.FCdims = FCdims
        self.alpha = alpha
        self.beta = beta
        encoder, AE, CLUS, MatchingRep, MatchingRep_y, MatchingRep_train = self.Models()
        self.models = {
            'encoder': encoder,
            'AE': AE,
            'CLUS': CLUS,
            'MatchingRep': MatchingRep,
            'MatchingRep_y': MatchingRep_y,
            'MatchingRep_train': MatchingRep_train
        }