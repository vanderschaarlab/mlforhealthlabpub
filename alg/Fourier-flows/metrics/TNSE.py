# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from __future__ import absolute_import, division, print_function
from sklearn.manifold import TSNE
import numpy as np


def tsne_embedded(X, X_synthetic):

    X_embedded = TSNE(n_components=2).fit_transform(np.array([X_synthetic[k] for k in range(len(X_synthetic))]))
    X_original = TSNE(n_components=2).fit_transform(np.array([X[k] for k in range(len(X))]))

    return X_original, X_embedded
