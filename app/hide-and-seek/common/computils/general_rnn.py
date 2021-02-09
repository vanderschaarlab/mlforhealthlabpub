"""General RNN core functions for time-series prediction.

Author: Jinsung Yoon
Contact: jsyoon0823@gmail.com

-----------------------------------------------

(1) binary_cross_entropy_loss: binary cross entropy loss (excluding padded data)
(2) mse_loss: mse loss (excluding padded data)
(3) rnn_sequential: rnn module for GeneralRNN class
(4) GeneralRNN: class of general RNN modules
"""

# Necessary packages
import os
import tensorflow as tf
import numpy as np
import tempfile
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint


def binary_cross_entropy_loss(y_true, y_pred):
    """User defined cross entropy loss.

    Args:
        - y_true: true labels
        - y_pred: predictions

    Returns:
        - loss: computed loss
    """
    # Exclude masked labels
    idx = tf.cast((y_true >= 0), float)
    # Cross entropy loss excluding masked labels
    loss = -(idx * y_true * tf.math.log(y_pred + 1e-6) + idx * (1 - y_true) * tf.math.log(1 - y_pred + 1e-6))
    return loss


def mse_loss(y_true, y_pred):
    """User defined mean squared loss.

    Args:
        - y_true: true labels
        - y_pred: predictions

    Returns:
        - loss: computed loss
    """
    # Exclude masked labels
    idx = tf.cast((y_true >= 0), float)
    # Mean squared loss excluding masked labels
    loss = idx * ((y_true - y_pred) ** 2)
    return loss


def rnn_sequential(model, model_name, h_dim, return_seq, name=None):
    """Add one rnn layer in sequential model.

    Args:
        - model: sequential rnn model
        - model_name: rnn, lstm, or gru
        - h_dim: hidden state dimensions
        - return_seq: True or False
        - name: layer name

    Returns:
        - model: sequential rnn model
    """
    if name == None:
        if model_name == "rnn":
            model.add(layers.SimpleRNN(h_dim, return_sequences=return_seq))
        elif model_name == "lstm":
            model.add(layers.LSTM(h_dim, return_sequences=return_seq))
        elif model_name == "gru":
            model.add(layers.GRU(h_dim, return_sequences=return_seq))
    else:
        if model_name == "rnn":
            model.add(layers.SimpleRNN(h_dim, return_sequences=return_seq, name=name))
        elif model_name == "lstm":
            model.add(layers.LSTM(h_dim, return_sequences=return_seq, name=name))
        elif model_name == "gru":
            model.add(layers.GRU(h_dim, return_sequences=return_seq, name=name))

    return model


class GeneralRNN:
    """RNN predictive model to time-series.

    Attributes:
        - model_parameters:
            - task: classification or regression
            - model_type: 'rnn', 'lstm', or 'gru'
            - h_dim: hidden dimensions
            - n_layer: the number of layers
            - batch_size: the number of samples in each batch
            - epoch: the number of iteration epochs
            - learning_rate: the learning rate of model training
    """

    def __init__(self, model_parameters):

        self.task = model_parameters["task"]
        self.model_type = model_parameters["model_type"]
        self.h_dim = model_parameters["h_dim"]
        self.n_layer = model_parameters["n_layer"]
        self.batch_size = model_parameters["batch_size"]
        self.epoch = model_parameters["epoch"]
        self.learning_rate = model_parameters["learning_rate"]

        assert self.model_type in ["rnn", "lstm", "gru"]

        # Predictor model define
        self.predictor_model = None

    def _build_model(self, x, y):
        """Construct the predictive model using feature and label statistics.

        Args:
            - x: temporal feature
            - y: labels

        Returns:
            - model: predictor model
        """
        # Parameters
        dim = len(x[0, 0, :])
        max_seq_len = len(x[0, :, 0])

        model = tf.keras.Sequential()
        model.add(layers.Masking(mask_value=-1.0, input_shape=(max_seq_len, dim)))

        # Stack multiple layers
        for _ in range(self.n_layer - 1):
            model = rnn_sequential(model, self.model_type, self.h_dim, return_seq=True)

        dim_y = len(y.shape)
        if dim_y == 2:
            return_seq_bool = False
        elif dim_y == 3:
            return_seq_bool = True
        else:
            raise ValueError("Dimension of y {} is not 2 or 3.".format(str(dim_y)))

        model = rnn_sequential(model, self.model_type, self.h_dim, return_seq_bool)
        self.adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

        if self.task == "classification":
            if dim_y == 3:
                model.add(layers.TimeDistributed(layers.Dense(y.shape[-1], activation="sigmoid")))
            elif dim_y == 2:
                model.add(layers.Dense(y.shape[-1], activation="sigmoid"))
            model.compile(loss=binary_cross_entropy_loss, optimizer=self.adam)
        elif self.task == "regression":
            if dim_y == 3:
                model.add(layers.TimeDistributed(layers.Dense(y.shape[-1], activation="linear")))
            elif dim_y == 2:
                model.add(layers.Dense(y.shape[-1], activation="linear"))
            model.compile(loss=mse_loss, optimizer=self.adam, metrics=["mse"])

        return model

    def fit(self, x, y, valid_rate=0.2, verbose=False):
        """Fit the predictor model.

        Args:
            - x: training features
            - y: training labels

        Returns:
            - self.predictor_model: trained predictor model
        """
        idx = np.random.permutation(len(x))
        train_idx = idx[: int(len(idx) * (1 - valid_rate))]
        valid_idx = idx[int(len(idx) * (1 - valid_rate)) :]

        train_x, train_y = x[train_idx], y[train_idx]
        valid_x, valid_y = x[valid_idx], y[valid_idx]

        self.predictor_model = self._build_model(train_x, train_y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_file_name = os.path.join(tmpdir, "model.ckpt")

            # Callback for the best model saving
            save_best = ModelCheckpoint(
                save_file_name,
                monitor="val_loss",
                mode="min",
                verbose=verbose,
                save_best_only=True,
                save_weights_only=True,
            )

            # Check initial weights:
            # print(self.predictor_model.get_weights())

            # Train the model
            self.predictor_model.fit(
                train_x,
                train_y,
                batch_size=self.batch_size,
                epochs=self.epoch,
                validation_data=(valid_x, valid_y),
                callbacks=[save_best],
                verbose=verbose,
            )

            self.predictor_model.load_weights(save_file_name)

        return self.predictor_model

    def predict(self, test_x):
        """Return the temporal and feature importance.

        Args:
            - test_x: testing features

        Returns:
            - test_y_hat: predictions on testing set
        """
        test_y_hat = self.predictor_model.predict(test_x)
        return test_y_hat
