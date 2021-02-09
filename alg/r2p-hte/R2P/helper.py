import numpy as np

from nonconformist.base import RegressorAdapter
from nonconformist.icp import IcpRegressor
from nonconformist.nc import RegressorNc


class Function_RegressorAdapter(RegressorAdapter):
    """ Conditional mean estimator, formulated as neural net
    """

    def __init__(self,
                 model):
        super(Function_RegressorAdapter, self).__init__(model)
        # Instantiate model
        self.model = model

    def fit(self, x, y):
        return

    def predict(self, x):
        y_hat = self.model(x)
        y_hat = y_hat.squeeze()

        return y_hat


class RegressorAdapter_HTE(RegressorAdapter):
    """ Conditional mean estimator, formulated as neural net
        mode= 0 is control 1 is treated
    """

    def __init__(self, model, mode=None):
        super(RegressorAdapter_HTE, self).__init__(model)
        # Instantiate model
        self.model = model
        self.mode = mode

    def fit(self, x, y):
        return

    def predict(self, x):
        if self.mode is 1:
            y_hat = self.model.predict(x)[2]
        elif self.mode is 0:
            y_hat = self.model.predict(x)[1]
        y_hat = y_hat.squeeze()

        return y_hat


class RegressorNc_r2p(RegressorNc):
    def predict(self, x, nc, significance=None, prediction=None):
        n_test = x.shape[0]
        if prediction is None:
            prediction = self.model.predict(x)
        if self.normalizer is not None:
            norm = self.normalizer.score(x) + self.beta
        else:
            norm = np.ones(n_test)

        if significance:
            intervals = np.zeros((x.shape[0], 2))
            err_dist = self.err_func.apply_inverse(nc, significance)
            err_dist = np.hstack([err_dist] * n_test)
            if prediction.ndim > 1:  # CQR
                intervals[:, 0] = prediction[:, 0] - err_dist[0, :]
                intervals[:, 1] = prediction[:, -1] + err_dist[1, :]
            else:  # regular conformal prediction
                err_dist *= norm
                intervals[:, 0] = prediction - err_dist[0, :]
                intervals[:, 1] = prediction + err_dist[1, :]

            return intervals
        else:  # Not tested for CQR
            significance = np.arange(0.01, 1.0, 0.01)
            intervals = np.zeros((x.shape[0], 2, significance.size))

            for i, s in enumerate(significance):
                err_dist = self.err_func.apply_inverse(nc, s)
                err_dist = np.hstack([err_dist] * n_test)
                err_dist *= norm

                intervals[:, 0, i] = prediction - err_dist[0, :]
                intervals[:, 1, i] = prediction + err_dist[0, :]

            return intervals


class IcpRegressor_r2p(IcpRegressor):
    def predict(self, x, significance=None, est_input=None):
        n_significance = (99 if significance is None
                          else np.array(significance).size)

        if n_significance > 1:
            prediction = np.zeros((x.shape[0], 2, n_significance))
        else:
            prediction = np.zeros((x.shape[0], 2))

        condition_map = np.array([self.condition((x[i, :], None))
                                  for i in range(x.shape[0])])

        for condition in self.categories:
            idx = condition_map == condition
            if np.sum(idx) > 0:
                p = self.nc_function.predict(x[idx, :],
                                             self.cal_scores[condition],
                                             significance,
                                             est_input)
                if n_significance > 1:
                    prediction[idx, :, :] = p
                else:
                    prediction[idx, :] = p

        return prediction

    def predict_given_scores(self, x, significance=None, cal_scores=None, est_input=None):
        n_significance = (99 if significance is None
                          else np.array(significance).size)

        if n_significance > 1:
            prediction = np.zeros((x.shape[0], 2, n_significance))
        else:
            prediction = np.zeros((x.shape[0], 2))

        condition_map = np.array([self.condition((x[i, :], None))
                                  for i in range(x.shape[0])])

        for condition in self.categories:
            idx = condition_map == condition
            if np.sum(idx) > 0:
                p = self.nc_function.predict(x[idx, :],
                                             cal_scores[0],
                                             significance,
                                             est_input)
                if n_significance > 1:
                    prediction[idx, :, :] = p
                else:
                    prediction[idx, :] = p

        return prediction
