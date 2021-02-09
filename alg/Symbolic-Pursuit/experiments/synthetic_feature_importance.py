import os
import sys
import csv
import numpy as np  # we use numpy to deal with arrays
import lime.lime_tabular  # lime interpreter
import shap  # shap interpreter
import multiprocessing  # in order to parallelize experiments
sys.path.append(os.path.abspath('..'))
from symbolic_pursuit.models import SymbolicRegressor  # our symbolic model class
from sklearn.metrics import mean_squared_error  # mse
from joblib import Parallel, delayed  # in order to parallelize experiments
from numpy.linalg import norm  # vector and matrices norms
from datetime import datetime  # the time to write the logs

n_cores = multiprocessing.cpu_count()  # Number of cores available for the task
n_exp = 50  # Number of experiments
n_train = 100  # Number of points in the training set for each experiment
n_test = int(0.3 * n_train)  # Number of points in the test set for each experiment
dim_X = 3  # Dimension of the input space
lime_mse = []  # List of MSEs for the LIME feature importance vectors
shap_mse = []  # List of MSEs for the SHAP feature importance vectors
symbolic_mse = []  # List of MSEs for the Symbolic feature importance vectors


def order_weights_LIME(exp_list):
    # Order the weights outputted by LIME
    ordered_weights = [0 for _ in range(dim_X)]
    for tup in exp_list:
        feature_id = int(tup[0].split('x_')[1][0])
        ordered_weights[feature_id - 1] = tup[1]
    return ordered_weights


def run_experiment():
    # Building true importance weight and the associated black-box
    importance_true = np.random.uniform(1, 10, dim_X)
    importance_true = importance_true / norm(importance_true)
    true_weight_list = np.array([importance_true for _ in range(n_test)])

    def f(X):
        return X.dot(importance_true)

    # Building datasets
    X_train = np.random.uniform(0, 1, (n_train, dim_X))
    X_test = np.random.uniform(0, 1, (n_test, dim_X))

    # LIME explainer
    lime_weight_list = []
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train,
                                                            feature_names=["x_" + str(k) for k in range(1, dim_X + 1)],
                                                            class_names=['f'],
                                                            verbose=True,
                                                            mode='regression')
    for k in range(n_test):
        exp = lime_explainer.explain_instance(X_test[k], f, num_features=dim_X)
        lime_weight_list.append(order_weights_LIME(exp.as_list()))
    lime_weight_list = np.array(lime_weight_list)
    lime_weight_list = lime_weight_list / norm(lime_weight_list, axis=1)[:, None]
    lime_weight_avg = lime_weight_list.mean(axis=0)
    lime_error = mean_squared_error(true_weight_list, lime_weight_list)
    lime_error2 = mean_squared_error(lime_weight_avg, importance_true)

    # SHAP explainer
    shap_explainer = shap.KernelExplainer(f, X_train)
    shap_weight_list = shap_explainer.shap_values(X_test, nsamples=100)
    shap_weight_list = np.array(shap_weight_list)
    shap_weight_list = shap_weight_list / norm(shap_weight_list, axis=1)[:, None]
    shap_weight_avg = shap_weight_list.mean(axis=0)
    shap_error = mean_squared_error(true_weight_list, shap_weight_list)
    shap_error2 = mean_squared_error(shap_weight_avg, importance_true)

    # Symbolic explainer
    symbolic_model = SymbolicRegressor()
    symbolic_model.fit(f, X_train)
    symbolic_weight_list = []
    for k in range(n_test):
        symbolic_weight_list.append(symbolic_model.get_feature_importance(X_test[k]))
    symbolic_weight_list = np.array(symbolic_weight_list, dtype=float)
    symbolic_weight_list = symbolic_weight_list / norm(symbolic_weight_list, axis=1)[:, None]
    symbolic_weight_avg = symbolic_weight_list.mean(axis=0)
    symbolic_error = mean_squared_error(true_weight_list, symbolic_weight_list)
    symbolic_error2 = mean_squared_error(symbolic_weight_avg, importance_true)
    # Add the entry inside a CSV
    with open('synthetic_results.csv', 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=' ')
        csv_writer.writerow([lime_error, shap_error, symbolic_error, lime_error2, shap_error2, symbolic_error2])
    return [lime_error, shap_error, symbolic_error, lime_error2, shap_error2, symbolic_error2]


# Run experiments in parallel
error_list = Parallel(n_jobs=n_cores, verbose=50)(delayed(run_experiment)() for _ in range(n_exp))
error_list = np.array(error_list)

# Extracts mses, do stats
lime_mse = error_list[:, 0]
shap_mse = error_list[:, 1]
symbolic_mse = error_list[:, 2]
lime_mse2 = error_list[:, 3]
shap_mse2 = error_list[:, 4]
symbolic_mse2 = error_list[:, 5]

lime_score, lime_std = lime_mse.mean(), lime_mse.std()
shap_score, shap_std = shap_mse.mean(), shap_mse.std()
symbolic_score, symbolic_std = symbolic_mse.mean(), symbolic_mse.std()
lime_score2, lime_std2 = lime_mse2.mean(), lime_mse2.std()
shap_score2, shap_std2 = shap_mse2.mean(), shap_mse2.std()
symbolic_score2, symbolic_std2 = symbolic_mse2.mean(), symbolic_mse2.std()

# Write everything in a text file
with open('synthetic_feature_importance.txt', 'a') as f:
    now_str = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
    f.write(100 * '-' + '\n' + 'Experiment ' + now_str + '\n' + 100 * '-' + '\n')
    f.write('LIME MSE 1 : ' + str(lime_score) + ' +/- ' + str(lime_std) + '\n')
    f.write('SHAP MSE 1 : ' + str(shap_score) + ' +/- ' + str(shap_std) + '\n')
    f.write('Symbolic MSE 1 : ' + str(symbolic_score) + ' +/- ' + str(symbolic_std) + '\n')
    f.write('LIME MSE 2 : ' + str(lime_score2) + ' +/- ' + str(lime_std2) + '\n')
    f.write('SHAP MSE 2: ' + str(shap_score2) + ' +/- ' + str(shap_std2) + '\n')
    f.write('Symbolic MSE 2 : ' + str(symbolic_score2) + ' +/- ' + str(symbolic_std2) + '\n')

