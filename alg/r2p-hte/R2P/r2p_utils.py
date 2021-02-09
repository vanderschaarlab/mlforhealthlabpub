import numpy as np


def get_within_var(num_leaves, leaf_results, tau_test):
    var = []
    for i in range(num_leaves):
        var_tmp = np.var(tau_test[leaf_results == i + 1])
        if not np.isnan(var_tmp):
            var.append(var_tmp)

    within_var = np.mean(var)
    return within_var


def get_across_var(num_leaves, leaf_results, tau_test):
    mean = []
    for i in range(num_leaves):
        if len(tau_test[leaf_results == i + 1]) > 0:
            mean.append(np.mean(tau_test[leaf_results == i + 1]))
    mean = np.array(mean)

    across_var = np.var(mean)
    return across_var


def divide_set(rows, y, column, value):
    if isinstance(value, int) or isinstance(value, float):  # for int and float values
        idx1 = rows[:, column] >= value
        idx2 = ~idx1
    else:  # for strings
        idx1 = rows[:, column] == value
        idx2 = ~idx1
    # split features
    list1 = rows[idx1]
    list2 = rows[idx2]
    # split outcome
    y1 = y[idx1]
    y2 = y[idx2]
    return list1, list2, y1, y2, idx1, idx2


def get_num_treat(treatment, min_size=1, treat_split=None):
    if treat_split is not None:
        treat_vect = np.copy(treatment)
        treat = treat_vect > treat_split
        control = treat_vect <= treat_split
        treat_vect[treat] = 1
        treat_vect[control] = 0
    else:
        treat_vect = treatment

    num_treatment = np.sum(treat_vect == 1)
    num_control = np.sum(treat_vect == 0)

    if num_treatment >= min_size and num_control >= min_size:
        min_size_check = True
    else:
        min_size_check = False

    return min_size_check, num_control, num_treatment


def variance(y, treatment, treat_split=None):
    treat_vect = np.copy(treatment)

    if treat_split is not None:
        trt = treat_vect > treat_split
        cont = treat_vect <= treat_split
        treat_vect[trt] = 1
        treat_vect[cont] = 0

    treat = treat_vect == 1
    control = treat_vect == 0

    if y.shape[0] == 0:
        return np.array([np.inf, np.inf])

    yt = y[treat]
    yc = y[control]

    if yt.shape[0] == 0:
        var_t = np.inf
    else:
        var_t = np.var(yt)

    if yc.shape[0] == 0:
        var_c = np.inf
    else:
        var_c = np.var(yc)

    return var_t, var_c


def get_treat_size(t, treat_split=0.5):
    num_treatment = t[t > treat_split].shape[0]
    num_control = t[t <= treat_split].shape[0]

    return num_treatment, num_control
