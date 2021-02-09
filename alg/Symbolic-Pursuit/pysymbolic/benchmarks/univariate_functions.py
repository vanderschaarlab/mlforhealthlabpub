# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import sys, os, time
import numpy as np
import pandas as pd
from scipy.special import jv
import itertools

from mpmath import *
from sympy import *
#from sympy.printing.theanocode import theano_function
from sympy.utilities.autowrap import ufuncify

import warnings
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")



def exponential_function(X):
	"""
    Benchmark exponential function f(x) = exp(-3x)
	"""
	return np.exp(-3 * X) 
	

def rational_function(X):
	"""
    Benchmark rational function f(x) = x/(x + 1)*2
	"""
	return X/((X+1)**2)

def sinusoidal_function(X):
	"""
    Benchmark sinusoid function f(x) = sin(x)
	"""

	return np.sin(X) 


def bessel_function(X):
	"""
    Benchmark Bessel function of the first kind f(x) = J_0(10*sqrt(X))
	"""

	return jv(0, 10*np.sqrt(X))  
