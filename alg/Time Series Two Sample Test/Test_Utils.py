
import numpy as np
from numpy.linalg import inv, norm, solve
from math import sqrt
from scipy.linalg import sqrtm
import scipy.io
from scipy.stats import chi2, f
from scipy import optimize
from scipy.sparse.linalg import eigsh
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic, ExpSineSquared, WhiteKernel)
from sklearn.gaussian_process.kernels import ConstantKernel as C



def W(f_1,f_2):
    """
    :param f1: GP represented as an array with first argument vector of means and second argument covariance function
    :param f2: GP represented as an array with first argument vector of means and second argument covariance function
    :return: Wassertstein distance between two GPs
    """
    mean_1 = f_1[0]; mean_2 = f_2[0]
    cov_1 = f_1[1]; cov_2 = f_2[1]

    # assert mean_1.shape == mean_2.shape, 'Shape of means not equal'
    assert cov_1.shape == cov_2.shape, 'Shape of covariance matrices not equal'

    temp = np.real(sqrtm(np.matmul(np.matmul(sqrtm(cov_1),cov_2),sqrtm(cov_1))))

    cov_distance = np.trace(cov_1) + np.trace(cov_2) -2*np.trace(temp)
    l2_norm = norm(mean_1 - mean_2)**2

    return(sqrt(np.abs(cov_distance + l2_norm)))


def Wbarycenter(population, weights=None, err = 1e-5):
    """
    :param population: m GP each entry including mean vector and covariance matrix over same inducing points
    :param eta: vector wwith barycentric weights
    :param err: error margin
    :return: Wasserstein barycenter of a population of GPs as a mean and covariance matrix over inducing points
    """

    d = len(population[0][0]) # dimension of each posterior
    m = len(population) # number of trajectories

    if weights == None: weights = (1 / m)*np.ones(m)

    means = np.zeros((m, d))
    covmats = np.zeros((m, d, d))

    for i in range(m):
        means[i] = np.array(population[i][0].flatten())
        covmats[i] = np.array(population[i][1])

    limit = 10**2

    # The Barycenter is the fixed point of the map F
    # Initialize K
    K = covmats[0]
    K_next = F(K, covmats, weights)
    count = 0
    # W(list([0, K]), list([0, K_next]))
    while W(list([0, K]), list([0, K_next]))>err and count < limit:
        K = K_next.copy()
        K_next = F(K_next,covmats,weights)
        count += 1

    if count == limit: print('Barycenter did not converge')

    barycenter_mean = np.average(means,weights=weights,axis = 0)

    return(barycenter_mean, K_next, means, covmats)

def F(K, covmats, weights):
    """
    :param K: state of covariance matrix at current iteration
    :param covmats: 3-d matric with all covariance matrices
    :param weights: weights
    :return: next state of covariance matrix
    """
    sqrtK = np.real(sqrtm(K))
    d = covmats.shape[1]
    m = len(weights)
    output = np.zeros((d,d))
    for i in range(m):
        output += weights[i]*np.real(sqrtm(np.matmul(np.matmul(sqrtK,covmats[i]),sqrtK)))

    return(output)

def cov_mat(x1, x2, a, b):
    return a * np.exp(-b * (x1[:, np.newaxis] - x2)**2)


def reg_cov_mat(x, a, b, c):
    return cov_mat(x, x, a, b) + c * np.eye(x.shape[0])

def _dfunc(dk, cov, Kinv_y):
    return (Kinv_y.dot(dk).dot(Kinv_y) - np.trace(solve(cov, dk))) * 0.5

def learn_hyperparms(ts, ys,
                     a_shape=None, a_mean=None,
                     b_shape=None, b_mean=None,
                     c_shape=None, c_mean=None,
                     mean_shift=True):
    """
    :param ts: array with times
    :param ys: array with values
    :return: next state of covariance matrix
    """

    if a_shape is not None:
        a_scale = a_mean / a_shape

    if b_shape is not None:
        b_scale = b_mean / b_shape

    if c_shape is not None:
        c_scale = c_mean / c_shape

    def neg_mloglik(w):
        a, b, c = w
        f = 0
        df = np.zeros_like(w)
        yss = ys

        if mean_shift:
            yss = yss - np.mean(yss)

        y = yss; t = ts
        n = t.shape[0]
        tsq = (t[:, np.newaxis] - t)**2
        da = np.exp(-b * tsq)
        cov = a * da
        db = -tsq * cov
        cov += c * np.eye(n)
        dc = np.eye(n)
        log_det = np.linalg.slogdet(cov)[1]
        Kinv_y = solve(cov, y)
        f += -0.5 * (y.dot(Kinv_y) + log_det)

        df[0] += _dfunc(da, cov, Kinv_y)
        df[1] += _dfunc(db, cov, Kinv_y)
        df[2] += _dfunc(dc, cov, Kinv_y)

        if a_shape is not None:
            f += (a_shape - 1) * np.log(a) - a / a_scale
            df[0] += (a_shape - 1) / a - 1 / a_scale

        if b_shape is not None:
            f += (b_shape - 1) * np.log(b) - b / b_scale
            df[1] += (b_shape - 1) / b - 1 / b_scale

        if c_shape is not None:
            f += (c_shape - 1) * np.log(c) - c / c_scale
            df[2] += (c_shape - 1) / c - 1 / c_scale

        return -f, -df

    w0 = np.array([1, 200, 1e-1])
    opt_parms = optimize.fmin_l_bfgs_b(neg_mloglik, w0,
            #bounds=[(1e-3, None), (1e-3, None), (1e-3, None)],
            bounds=[(1e-3, None), (1e-5, None), (1e-5, None)],
            factr=1e3, pgtol=1e-07, #disp=1,
            )[0]

    #print neg_mloglik(opt_parms)[1]
    #def _func(x, *args): return neg_mloglik(x, *args)[0]
    #print approx_fprime(opt_parms, _func, 1e-8)

    return opt_parms

def posterior_mean_cov(t_train, y_train, t_test, parms, mean_shift=False):
    """Prediction for Gaussian process regression
    Returns the exact predictive mean and covariance matrix
    Parameters
    ----------
    t_train : array_like, shape (n_training_samples, )
              time points of training samples
    y_train : array_like, shape (n_training_samples, )
              values of corresponding time points of training samples
    t_test : array_like, shape (n_test_samples, )
             time points of test samples
    parms : tuple, length 3
            hyperparameters (a, b, c) for covariance function of Gaussian
            process.
            [K(t)]_ij = a * exp(-b * (t_i - t_j)^2) + c * I(i == j)
    mean_shift : Whether the data is centered or not, Flase means mean = 0
    Returns
    -------
    mean_test : array, shape (n_test_samples, )
                predictive mean
    cov_test : array, shape (n_test_samples, n_test_samples)
               predictive covariance matrix
    """

    #assert t_train.shape[0] == y_train.shape[0]
    a, b, c = parms

    K_train = reg_cov_mat(t_train, a, b, c)
    K_train_test = cov_mat(t_train, t_test, a, b)

    Ktr_inv_Ktt = solve(K_train, K_train_test)

    if mean_shift:
        mu = np.mean(y_train)
        mean_test = mu + Ktr_inv_Ktt.T.dot(y_train - mu)
    else:
        mean_test = Ktr_inv_Ktt.T.dot(y_train)

    cov_test = cov_mat(t_test, t_test, a, b) - K_train_test.T.dot(Ktr_inv_Ktt)

    return mean_test, cov_test

def rbf_kernel(x,y,length_scale=1,var=1):

    rbf = var * RBF(length_scale=length_scale)
    return(rbf(x[:,np.newaxis],y[:,np.newaxis]))

def matern_kernel(x,y,length_scale=1,nu=1.5):

    kernel= Matern(length_scale=length_scale, nu=nu)
    return(kernel(x[:,np.newaxis],y[:,np.newaxis]))

def rational_kernel(x,y,length_scale=1,alpha=1):

    kernel = RationalQuadratic(length_scale=length_scale, alpha=alpha)
    return(kernel(x[:,np.newaxis],y[:,np.newaxis]))

def expsine_kernel(x,y,length_scale=1,periodicity=1):

    kernel = ExpSineSquared(length_scale=length_scale, periodicity=periodicity)
    return(kernel(x[:,np.newaxis],y[:,np.newaxis]))

def dot_kernel(x,y,sigma_0=1):

    kernel = DotProduct(sigma_0=sigma_0)
    return(kernel(x[:,np.newaxis],y[:,np.newaxis]))


def compute_posterior(t,y,u,kernel='rbf',approximation=False):

    if kernel == 'rbf':
        kernel = 1.0*RBF(length_scale_bounds=(1e-3,100.0)) + C(1.0, (1e-3, 1e3)) +\
                 WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e+1))
    if kernel == 'Matern':
        kernel = 1.0*Matern(length_scale_bounds=(1e-3,100.0))+ \
                 WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-10, 1e+1)) + C(1.0, (1e-3, 1e3))
    if kernel == 'RationalQuadratic':
        kernel = 1.0*RationalQuadratic(length_scale_bounds=(1e-1,100.0))+ \
                 WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))+ C(1.0, (1e-3, 1e3))
    if kernel == 'ExpSineSquared':
        kernel = 1.0*ExpSineSquared(length_scale_bounds=(1e-1,100.0))+ \
                 WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))+ C(1.0, (1e-3, 1e3))

    max_t = 1
    #max_t = 2557
    out = []
    for i in range(len(t)):

        if not approximation:
            temp_t = np.array(t[i]) / max_t
            temp_y = np.array(y[i])
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,n_restarts_optimizer=10)
            gp.fit(temp_t[:, np.newaxis], temp_y)
            mean_cov = gp.predict(u[:, np.newaxis], return_cov=True)
            # adjust noise estimation (variance of trajectory) to be equal to max(errors)/2
            #max_error = np.max(abs(gp.predict(temp_t[:, np.newaxis], return_cov=False)-temp_y))
            np.fill_diagonal(mean_cov[1], np.diag(mean_cov[1]))

            #np.fill_diagonal(mean_cov[1],np.diag(mean_cov[1])/2)

        else:
            temp_t = np.array(t[i]) / max_t
            temp_y = np.array(y[i])
            len_u = len(u)
            #extra_u = 2
            #margin = (t_max - t_min) / (len_u - extra_u * 2) * 2
            #u = np.linspace(t_min - margin, t_max + margin, len_u)  # inducing points
            t_test = u
            len_test = len(t_test)
            idx_train, w_train = sparse_w(u, temp_t)
            idx_test, w_test = sparse_w(u, t_test)
            a, b, c = learn_hyperparms(temp_t, temp_y)
            ku = covmat_col0(u, (a, b))
            mu = post_mean(idx_train, w_train, idx_test, w_test, ku, len_u, c, temp_y)
            var = post_cov_ys(idx_train, w_train, idx_test, w_test, ku, len_u, c, np.eye(len_test))
            mean_cov = [mu,var]

        out.append(mean_cov)

    return out


def compute_posterior_approx(ts,ys,num_inducing):
    """
    :param ts: original array of observation times
    :param ys: original array of observation values
    :return: list of tuples, each tuple cntains two arrays: the posterior mean vector over inducing points and
    the poterior covariance
    """
    parameters = learn_hyperparms(ts, ys)

    # define inducing points
    t_min, t_max = ts.min(), ts.max()
    len_u = num_inducing
    extra_u = 2
    margin = (t_max - t_min) / (len_u - extra_u * 2) * 2
    u = np.linspace(t_min - margin, t_max + margin, len_u)  # inducing points

    # ku = covmat_col0(u, (a, b))

    out = []
    for i in range(len(ts)):
        mean_cov = posterior_mean_cov(ts[i], ys[i], u, parameters, mean_shift=True)

        # Aternative way that makes it easier to include approximations
        # post_mean_exact(xs[0], u, (a, b), c, ys[0])
        # post_cov_exact(xs[0], u, (a,b), c)

        out.append(mean_cov)

    return out


def RMD(t1,y1,t2,y2,k=None):
    """
    Robust mean discrcepancy two-sample test for irregularly sampled time series
    :param t1: array of observation times of first population
    :param y1: array of observation values of first population
    :param t2: array of observation times of second population
    :param y2: array of observation values of second population
    :param k: number of principal components to be retained, if none use penalized procedure
    :return: p-value for the hypothesis of equal means
    """
    u = np.linspace(0,1,20)
    post_1, post_2 = compute_posterior(t1, y1, u), compute_posterior(t2, y2, u)

    barycenter_1, barycenter_2 = Wbarycenter(post_1), Wbarycenter(post_2)

    means_1, means_2 = barycenter_1[2], barycenter_2[2]
    covs_1, covs_2 = barycenter_1[3], barycenter_2[3]

    mean_1, cov_1 = barycenter_1[0], barycenter_1[1]
    mean_2, cov_2 = barycenter_2[0], barycenter_2[1]

    n_1, n_2 = len(t1), len(t2)
    N = n_1 + n_2
    cov = (n_1 * cov_1 + n_2 * cov_2) / N

    # find common eigen-decomposition to both samples

    e_values, e_vectors = eigsh(cov, k=cov.shape[0] - 1)
    e_vectors, e_values = e_vectors[::-1], e_values[::-1]


    if k == None:

        penalized_fit = []
        e_values_1, e_vectors_1 = eigsh(cov_1, k=cov_1.shape[0] - 1)
        e_vectors_1, e_values_1 = e_vectors_1[::-1], e_values_1[::-1]
        e_values_2, e_vectors_2 = eigsh(cov_2, k=cov_2.shape[0] - 1)
        e_vectors_2, e_values_2 = e_vectors_2[::-1], e_values_2[::-1]

        for k in range(2,cov.shape[0] - 1):
            proj_1 = np.matmul(np.matmul(means_1, e_vectors[:, :k]), e_vectors[:, :k].T)
            proj_2 = np.matmul(np.matmul(means_2, e_vectors[:, :k]), e_vectors[:, :k].T)
            gof = np.sum((proj_1 - means_1)**2) + np.sum((proj_2 - means_2)**2)
            norm_1 = np.matmul(proj_1, e_vectors_1)**2/e_values_1
            norm_2 = np.matmul(proj_2, e_vectors_2) ** 2 / e_values_2
            pen = 2*np.sum(e_values)*np.sum(norm_1)/n_1 + 2*np.sum(e_values)*np.sum(norm_2)/n_2
            penalized_fit.append(gof + pen)

        k = penalized_fit.index(min(penalized_fit)) + 2

        print('k = ', k)

    if k == 'Cumulative_Variance':

        cum_eig = np.cumsum(e_values) / np.sum(e_values)
        k = np.where(cum_eig > 0.8)[0][0] + 1
        print('k = ', k)



    cov_squared = cov * cov

    #T_stat =  np.dot(mean_2 - mean_1, mean_2 - mean_1) * np.trace(cov) / np.trace(cov_squared)
    T_stat = (n_1*n_2)*np.dot(mean_2 - mean_1, mean_2 - mean_1) * np.sum(e_values[:k]) / np.sum(e_values[:k]**2)/(n_1+n_2)
    #df = np.trace(cov)**2 / np.trace(cov_squared)
    df = (np.sum(e_values[:k]) ** 2) / np.sum(e_values[:k]**2)
    p_value = 1 - chi2.cdf(T_stat, df)

    return float(p_value)