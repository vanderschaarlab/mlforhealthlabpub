from operator import itemgetter

import matplotlib
from matplotlib import cm as CM

matplotlib.rcParams.update({'font.size': 16.5})
import seaborn as sns
sns.set_context(rc={"lines.linewidth": 3.00})
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.special import expit





def compute_PEHE(T_true, T_est):

    return np.sqrt(np.mean((T_true.reshape((-1, 1)) - T_est.reshape((-1, 1))) ** 2))


def compute_PEHE_removal(T_true, T_est, index):


    T_true1 = np.array(T_true)[[index]]
    T_est1 = np.array(T_est)[[index]]

    return np.sqrt(np.mean((T_true1.reshape((-1, 1)) - T_est1.reshape((-1, 1))) ** 2))




def plot_tnse_embedding(X,Y,W):


    W = np.squeeze(W)
    tsne = TSNE(n_components=2, random_state=0)
    tsne_embedding = tsne.fit_transform(X)


    plt.figure(figsize=[6.5, 6])
    plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=W, alpha=0.5)
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.yticks([])
    plt.xticks([])


    plt.show()



    plt.figure(figsize=[8, 6])
    plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c= Y[:,0], alpha=0.6, cmap=CM.jet)
    cb = plt.colorbar()
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.yticks([])
    plt.xticks([])

    plt.show()


def remove_uncertain_data(var_0_tr, var_1_tr, T_true_train, Y_hat_train):


    var_train = var_0_tr + var_1_tr

    residual = np.square(T_true_train[:,0] - (Y_hat_train[:, 1] - Y_hat_train[:, 0]))

    mean_rmse1, mean_rmse2, indices, indices2, sorted_var = two_error(residual, var_train)

    fraction = int(0.1*np.shape(mean_rmse1)[0])

    return mean_rmse1[fraction], mean_rmse2[fraction]


def two_error(residual, test_var):


    indices, L_sorted = zip(*sorted(enumerate(residual), key=itemgetter(1)))
    residual_order = list(L_sorted)

    mean_rmse1 = []
    for i in range(len(residual_order)):
        alpha = residual_order[: len(residual_order) - i]
        mean_rmse1.append(np.sqrt(np.mean(alpha)))

    indices2, L_sorted = zip(*sorted(enumerate(test_var), key=itemgetter(1)))
    residual_2 = [residual[i] for i in indices2]

    mean_rmse2 = []
    for i in range(len(residual_2)):
        alpha = residual_2[: len(residual_2) - i]
        mean_rmse2.append(np.sqrt(np.mean(alpha)))


    return mean_rmse1, mean_rmse2, indices, indices2, L_sorted


def index_select(test_var, percentage):

    indices2, L_sorted = zip(*sorted(enumerate(test_var), key=itemgetter(1)))

    return indices2[: len(test_var) - int(percentage*len(test_var))]



