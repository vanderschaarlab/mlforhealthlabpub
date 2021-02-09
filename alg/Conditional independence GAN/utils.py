from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import ks_2samp
from scipy.stats import wilcoxon
import numpy as np
import random
from scipy import stats





def same(x):
    return x

def cube(x):
    return np.power(x, 3)

def negexp(x):
    return np.exp(-np.abs(x))


def generate_samples_random(size=1000, sType='CI', dx=1, dy=1, dz=20, nstd=0.5, freq=1.0, fixed_function='linear',
                            debug=False, normalize = True, seed = None, dist_z = 'gaussian'):
    '''Generate CI,I or NI post-nonlinear samples
    1. Z is independent Gaussian
    2. X = f1(<a,Z> + b + noise) and Y = f2(<c,Z> + d + noise) in case of CI
    Arguments:
        size : number of samples
        sType: CI,I, or NI
        dx: Dimension of X
        dy: Dimension of Y
        dz: Dimension of Z
        nstd: noise standard deviation
        freq: Freq of cosine function
        f1,f2 an be within {x,x^2,x^3,tanh x, e^{-|x|}, cos x}

    Output:
        allsamples --> complete data-set
    Note that:
    [X = first dx coordinates of allsamples each row is an i.i.d samples]
    [Y = [dx:dx + dy] coordinates of allsamples]
    [Z = [dx+dy:dx+dy+dz] coordinates of all samples]
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
        X = f1(freq * (Z * Ax + nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num)))
        Y = f2(freq * (Z * Ay + nstd * np.random.multivariate_normal(np.zeros(dy), np.eye(dy), num)))
    elif sType == 'I':
        X = f1(freq * (nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num)))
        Y = f2(freq * (nstd * np.random.multivariate_normal(np.zeros(dy), np.eye(dy), num)))
    else:
        X = f1(freq * (np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num)))
        Y = f2(freq * (2 * X * Axy + Z * Ay + nstd * np.random.multivariate_normal(np.zeros(dy), np.eye(dy), num)))

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