from keras import backend as K
from keras.engine.topology import Layer, InputSpec
import numpy as np
from keras.losses import KLD

class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clustering_layer')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.        
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: data, shape=(n_samples, n_features)
        Return:
            q: soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# target distribution of DEC    
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

# loss function of Matching rep.
def MatchingRepLoss(true, pred, alpha=1, beta=.001):
    # get the number of clusters
    n_clusters = true.shape[1]-1

    # extract arguments
    ys = pred[:, :n_clusters]
    rep = pred[:, n_clusters:-n_clusters]
    c_pred = pred[:, -n_clusters:]
    y_true = true[:, 0]
    c_true = true[:, 1:]
    
    # compute mse loss
    y_pred = K.sum(ys*c_pred, axis=-1)
    y_loss = K.square(y_pred-y_true)
        
    # compute clustering loss
    c_loss = KLD(c_true, c_pred)
    
    # compute representation loss to adversarialy balance representations
    # KLD can be replaced by other probability metrics
    r_loss = 0
    for i in range(n_clusters):
        w = K.reshape(c_true[:, i], shape=(-1, 1))
        weighted = rep * w
        kld = KLD(rep, weighted)
        r_loss += kld
    
    return y_loss + alpha*c_loss + beta*r_loss
