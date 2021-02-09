# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import sys, os, time
import numpy as np
import pandas as pd
import scipy as sc
import itertools

from mpmath import *
from sympy import *

from scipy.optimize import minimize

import warnings
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from pysymbolic.models.special_functions import MeijerG

def visualize_univariate(xrange, f, label_, color_):

	x        = Symbol('x')

	X        = np.linspace(xrange[0], xrange[1], 1000)
	Y_values = np.array([sympify(str(f)).subs(x,X[k]) for k in range(len(X))]).reshape((-1,1))

	plt.plot(X, Y_values, color=color_, label=label_)
	plt.legend()
