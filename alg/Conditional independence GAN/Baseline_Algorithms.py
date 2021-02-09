""" Wrappers for conditional independence tests from the literature.
Install the original R package before running this module.
It is available at https://github.com/ericstrobl/RCIT.
"""

import rpy2.robjects as R
import numpy as np
from rpy2.robjects.packages import importr
importr('RCIT')

def RCIT(x, y, z, **kwargs):
    """ Run the RCIT independence test. Reference: Strobl, Eric V. and Zhang, Kun and Visweswaran, Shyam,
        Approximate Kernel-based Conditional Independence Test for Non-Parametric Causal Discovery,
        arXiv preprint arXiv:1202.3775 (2017).
    Args:
        x (n_samples, x_dim): First variable.
        y (n_samples, y_dim): Second variable.
        z (n_samples, z_dim): Conditioning variable.
        max_time (float): Time limit for the test -- it will terminate
            after that and return p-value -1.
    Returns:
        p (float): The p-value for the null hypothesis
            that x is independent of y.
    """
    x = np2r(x)
    y = np2r(y)
    z = np2r(z)
    res = R.r.RCIT(x, y, z)
    return res[0][0]

def KCIT(x, y, z, **kwargs):
    """ Run the KCIT independence test. Reference: Kernel-based Conditional Independence Test and Application
        in Causal Discovery, Zhang, Kun and Peters, Jonas and Janzing, Dominik and Scholkopf, Bernhard,
        arXiv preprint arXiv:1202.3775, 2012.
    Args:
        x (n_samples, x_dim): First variable.
        y (n_samples, y_dim): Second variable.
        z (n_samples, z_dim): Conditioning variable.
        max_time (float): Time limit for the test -- it will terminate
            after that and return p-value -1.
    Returns:
        p (float): The p-value for the null hypothesis
            that x is independent of y.
    """
    x = np2r(x)
    y = np2r(y)
    z = np2r(z)
    res = R.r.KCIT(x, y, z)
    return res[0]

def RCoT(x, y, z, **kwargs):
    """ Run the RCoT independence test. Reference: Strobl, Eric V. and Zhang, Kun and Visweswaran, Shyam,
        Approximate Kernel-based Conditional Independence Test for Non-Parametric Causal Discovery,
        arXiv preprint arXiv:1202.3775 (2017).
    Args:
        x (n_samples, x_dim): First variable.
        y (n_samples, y_dim): Second variable.
        z (n_samples, z_dim): Conditioning variable.
        max_time (float): Time limit for the test -- it will terminate
            after that and return p-value -1.
    Returns:
        p (float): The p-value for the null hypothesis
            that x is independent of y.
    """
    x = np2r(x)
    y = np2r(y)
    z = np2r(z)
    res = R.r.RCoT(x, y, z)
    return res[0][0]

def CRT(x,y,z,seed=None,statistic = 'corr'):

    if seed == None:
        np.random.seed()
    else:
        np.random.seed(seed)

    dx = 1; dz = z.shape[1]
    nstd = 0.5
    n_samples = 1000
    rho = []

    Ax = np.random.rand(dz, dx)
    for i in range(dx):
        Ax[:, i] = Ax[:, i] / np.linalg.norm(Ax[:, i], ord=1)
    Ax = np.matrix(Ax)

    if statistic == "corr":
        stat = correlation
    if statistic == "mmd":
        stat = mmd_squared
    if statistic == "kolmogorov":
        stat = kolmogorov

    for sample in range(n_samples):
        x_hat = z * Ax + nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), len(x))
        rho.append(stat(x_hat, y))

    p_value = sum(stat(x.reshape(len(x)), y) > rho) / n_samples

    if p_value > 0.975:
        p_value = 1 - p_value

    return (p_value)

# Examples:
#CRT(np.random.uniform(0,1,(1000,1)),np.random.uniform(0,1,(1000,1)),np.random.uniform(0,1,(1000,2)),seed=1)
#KCIT(np.random.uniform(0,1,(1000,1)),np.random.uniform(0,1,(1000,1)),np.random.uniform(0,1,(1000,2)))