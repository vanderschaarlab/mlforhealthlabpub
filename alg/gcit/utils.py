from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import ks_2samp
from scipy.stats import wilcoxon
import numpy as np
import random
from scipy import stats
import time
from collections import defaultdict
import numpy as np
import warnings
from scipy.stats import rankdata



def same(x):
    return x

def cube(x):
    return np.power(x, 3)

def negexp(x):
    return np.exp(-np.abs(x))


def generate_samples_random(size=1000, sType='CI', dx=1, dy=1, dz=20, nstd=1, fixed_function='linear',
                            debug=False, normalize = True, seed = None, dist_z = 'gaussian'):
    '''Generate CI,I or NI post-nonlinear samples
    1. Z is independent Gaussian or Laplace
    2. X = f1(<a,Z> + b + noise) and Y = f2(<c,Z> + d + noise) in case of CI
    Arguments:
        size : number of samples
        sType: CI, I, or NI
        dx: Dimension of X
        dy: Dimension of Y
        dz: Dimension of Z
        nstd: noise standard deviation
        f1, f2 to be within {x,x^2,x^3,tanh x, e^{-|x|}, cos x}

    Output:
        Samples X, Y, Z
    '''
    if seed == None:
        np.random.seed()
    else:
        np.random.seed(seed)

    if fixed_function == 'linear':
        f1 = same
        f2 = same
    else:
        I1 = random.randint(2, 6)
        I2 = random.randint(2, 6)

        if I1 == 2:
            f1 = np.square
        elif I1 == 3:
            f1 = cube
        elif I1 == 4:
            f1 = np.tanh
        elif I2 == 5:
            f1 = negexp
        else:
            f1 = np.cos

        if I2 == 2:
            f2 = np.square
        elif I2 == 3:
            f2 = cube
        elif I2 == 4:
            f2 = np.tanh
        elif I2 == 5:
            f2 = negexp
        else:
            f2 = np.cos
    if debug:
        print(f1, f2)

    num = size

    if dist_z =='gaussian':
        cov = np.eye(dz)
        mu = np.ones(dz)
        Z = np.random.multivariate_normal(mu, cov, num)
        Z = np.matrix(Z)

    elif dist_z == 'laplace':
        Z = np.random.laplace(loc=0.0, scale=1.0, size=num*dz)
        Z = np.reshape(Z,(num,dz))
        Z = np.matrix(Z)

    Ax = np.random.rand(dz, dx)
    for i in range(dx):
        Ax[:, i] = Ax[:, i] / np.linalg.norm(Ax[:, i], ord=1)
    Ax = np.matrix(Ax)
    Ay = np.random.rand(dz, dy)
    for i in range(dy):
        Ay[:, i] = Ay[:, i] / np.linalg.norm(Ay[:, i], ord=1)
    Ay = np.matrix(Ay)

    Axy = np.random.rand(dx, dy)
    for i in range(dy):
        Axy[:, i] = Axy[:, i] / np.linalg.norm(Axy[:, i], ord=1)
    Axy = np.matrix(Axy)

    temp = Z * Ax
    m = np.mean(np.abs(temp))
    nstd = nstd * m 

    if sType == 'CI':
        X = f1(Z * Ax + nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num))
        Y = f2(Z * Ay + nstd * np.random.multivariate_normal(np.zeros(dy), np.eye(dy), num))
    elif sType == 'I':
        X = f1(nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num))
        Y = f2(nstd * np.random.multivariate_normal(np.zeros(dy), np.eye(dy), num))
    else:
        X = np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num)
        Y = f2(2 * X * Axy + Z * Ay)

    if normalize == True:
        Z = (Z - Z.min()) / (Z.max() - Z.min())
        X = (X - X.min()) / (X.max() - X.min())
        Y = (Y - Y.min()) / (Y.max() - Y.min())

    return np.array(X), np.array(Y), np.array(Z)

def pc_ks(pvals):
    """ Compute the area under power curve and the Kolmogorov-Smirnoff
    test statistic of the hypothesis that pvals come from the uniform
    distribution with support (0, 1).
    """
    if pvals.size == 0:
        return [-1, -1]
    if -1 in pvals or -2 in pvals:
        return [-1, -1]
    pvals = np.sort(pvals)
    cdf = ecdf(pvals)
    auc = 0
    for (pv1, pv2) in zip(pvals[:-1], pvals[1:]):
        auc += integrate.quad(cdf, pv1, pv2)[0]
    auc += integrate.quad(cdf, pvals[-1], 1)[0]
    _, ks = kstest(pvals, 'uniform')
    return auc, ks

def np2r(x):
    """ Convert a numpy array to an R matrix.
    Args:
        x (dim0, dim1): A 2d numpy array.
    Returns:
        x_r: An rpy2 object representing an R matrix isometric to x.
    """
    if 'rpy2' not in sys.modules:
        raise ImportError(("rpy2 is not installed.",
                " Cannot convert a numpy array to an R vector."))
    try:
        dim0, dim1 = x.shape
    except IndexError:
        raise IndexError("Only 2d arrays are supported")
    return R.r.matrix(R.FloatVector(x.flatten()), nrow=dim0, ncol=dim1)

def fdr(truth, pred, axis=None):
    """ Computes False discovery rate
    """
    return ((pred==1) & (truth==0)).sum(axis=axis) / pred.sum(axis=axis).astype(float).clip(1,np.inf)

def tpr(truth, pred, axis=None):
    """ Computes true positive rate
    """
    return ((pred==1) & (truth==1)).sum(axis=axis) / truth.sum(axis=axis).astype(float).clip(1,np.inf)

def true_positives(truth, pred, axis=None):
    """ Computes number of true positive
    """
    return ((pred==1) & (truth==1)).sum(axis=axis)

def false_positives(truth, pred, axis=None):
    """ Computes number of false positive
    """
    return ((pred==1) & (truth==0)).sum(axis=axis)

def bh(p, fdr):
    """ From vector of p-values and desired false positive rate,
    returns significant p-values with Benjamini-Hochberg correction
    """
    p_orders = np.argsort(p)
    discoveries = []
    m = float(len(p_orders))
    for k, s in enumerate(p_orders):
        if p[s] <= (k+1) / m * fdr:
            discoveries.append(s)
        else:
            break
    return np.array(discoveries, dtype=int)



def mmd_squared(X, Y, gamma = 1):

    X = X.reshape((len(X)), 1)
    Y = Y.reshape((len(Y)), 1)

    K_XX = rbf_kernel(X, gamma=gamma)
    K_YY = rbf_kernel(Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)

    n = K_XX.shape[0]
    m = K_YY.shape[0]

    mmd_squared = (np.sum(K_XX)-np.trace(K_XX))/(n*(n-1)) + (np.sum(K_YY)-np.trace(K_YY))/(m*(m-1)) - 2 * np.sum(K_XY) / (m * n)

    return mmd_squared

def correlation(X,Y):
    X = X.reshape((len(X)))
    Y = Y.reshape((len(Y)))
    return np.abs(np.corrcoef(X, Y)[0, 1])

def kolmogorov(X,Y):

    X = X.reshape((len(X)))
    Y = Y.reshape((len(Y)))
    return ks_2samp(X, Y)[0]

def wilcox(X,Y):

    X = X.reshape((len(X)))
    Y = Y.reshape((len(Y)))
    return wilcoxon(X, Y)[0]


'''
X = np.random.normal(0,2,500)
Y = np.random.normal(0,2,500)

kolmogorov(X,Y)
'''

def rdc(x, y, f=np.sin, k=20, s=1/6., n=1):
    """
    Computes the Randomized Dependence Coefficient
    x,y: numpy arrays 1-D or 2-D
         If 1-D, size (samples,)
         If 2-D, size (samples, variables)
    f:   function to use for random projection
    k:   number of random projections to use
    s:   scale parameter
    n:   number of times to compute the RDC and
         return the median (for stability)
    According to the paper, the coefficient should be relatively insensitive to
    the settings of the f, k, and s parameters.
    
    Source: https://github.com/garydoranjr/rdc
    """
    x = x.reshape((len(x)))
    y = y.reshape((len(y)))
    
    if n > 1:
        values = []
        for i in range(n):
            try:
                values.append(rdc(x, y, f, k, s, 1))
            except np.linalg.linalg.LinAlgError: pass
        return np.median(values)

    if len(x.shape) == 1: x = x.reshape((-1, 1))
    if len(y.shape) == 1: y = y.reshape((-1, 1))

    # Copula Transformation
    cx = np.column_stack([rankdata(xc, method='ordinal') for xc in x.T])/float(x.size)
    cy = np.column_stack([rankdata(yc, method='ordinal') for yc in y.T])/float(y.size)

    # Add a vector of ones so that w.x + b is just a dot product
    O = np.ones(cx.shape[0])
    X = np.column_stack([cx, O])
    Y = np.column_stack([cy, O])

    # Random linear projections
    Rx = (s/X.shape[1])*np.random.randn(X.shape[1], k)
    Ry = (s/Y.shape[1])*np.random.randn(Y.shape[1], k)
    X = np.dot(X, Rx)
    Y = np.dot(Y, Ry)

    # Apply non-linear function to random projections
    fX = f(X)
    fY = f(Y)

    # Compute full covariance matrix
    C = np.cov(np.hstack([fX, fY]).T)

    # Due to numerical issues, if k is too large,
    # then rank(fX) < k or rank(fY) < k, so we need
    # to find the largest k such that the eigenvalues
    # (canonical correlations) are real-valued
    k0 = k
    lb = 1
    ub = k
    while True:

        # Compute canonical correlations
        Cxx = C[:k, :k]
        Cyy = C[k0:k0+k, k0:k0+k]
        Cxy = C[:k, k0:k0+k]
        Cyx = C[k0:k0+k, :k]

        eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.pinv(Cxx), Cxy),
                                        np.dot(np.linalg.pinv(Cyy), Cyx)))

        # Binary search if k is too large
        if not (np.all(np.isreal(eigs)) and
                0 <= np.min(eigs) and
                np.max(eigs) <= 1):
            ub -= 1
            k = (ub + lb) // 2
            continue
        if lb == ub: break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) // 2

    return np.sqrt(np.max(eigs))