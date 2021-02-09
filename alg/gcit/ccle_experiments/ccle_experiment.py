'''
This code reproduces the real data experiments using the CCLE data.
Preprocessing steps follow the code of W. Tansey at https://github.com/tansey/hrt.
'''

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import initpath_alg
initpath_alg.init_sys_path()
import utilmlab


def add_parent_dir_to_sys_path():
    file = Path(__file__).resolve()
    parent, root = file.parent, file.parents[1]
    sys.path.append(str(root))


add_parent_dir_to_sys_path()
from GCIT import GCIT


def get_data_file_name(basename):
    for fn in [
            basename,
            basename + '.gz',
            'data/' + basename,
            'data/' + basename + '.gz',
            utilmlab.get_proj_dir() + '/alg/gcit/ccle_experiments/data/' + basename,
            utilmlab.get_proj_dir() + '/alg/gcit/ccle_experiments/data/' + basename + '.gz']:
        if os.path.isfile(fn):
            return fn
    print(f'***')
    print(f'*** Error: cannot find file {basename}, see README.md of how'
          'to obtain the required data ***')
    print(f'***')
    assert 0


def load_ccle(drug_target='PLX4720', feature_type='both', normalize=False):
    '''
    :param drug target: specific drug we want to analyse
    :param normalize: normalize data
    :return: genetic features (mutations) as a 2d array for each cancer cell and corresponding drug response measured with Amax
    '''
    if feature_type in ['expression', 'both']:
        # Load gene expression
        expression = pd.read_csv(get_data_file_name('expression.txt'), delimiter='\t', header=2, index_col=1).iloc[:, 1:]
        expression.columns = [c.split(' (ACH')[0] for c in expression.columns]
        features = expression
    if feature_type in ['mutation', 'both']:
        # Load gene mutation
        mutations = pd.read_csv(get_data_file_name('mutation.txt'), delimiter='\t', header=2, index_col=1).iloc[:,1:]
        mutations = mutations.iloc[[c.endswith('_MUT') for c in mutations.index]]
        features = mutations
    if feature_type == 'both':
        # get cells having both expression and mutation data
        both_cells = set(expression.columns) & set(mutations.columns)
        z = {}
        for c in both_cells:
            exp = expression[c].values
            if len(exp.shape) > 1:
                exp = exp[:, 0]
            z[c] = np.concatenate([exp, mutations[c].values])
        both_df = pd.DataFrame(z, index=[c for c in expression.index] + [c for c in mutations.index])
        features = both_df

    print('Genetic features dimension = {} on {} cancer cells'.format(features.shape[0],features.shape[1]))

    # Get per-drug X and y regression targets
    response = pd.read_csv(get_data_file_name('response.csv'), header=0, index_col=[0,2])

    # names of cell lines, there are 504
    cells = response.index.levels[0]
    # names of drugs, there are 24
    drugs = response.index.levels[1]

    X_drugs = [[] for _ in drugs]
    y_drugs = [[] for _ in drugs]

    # subset data to include only cells, mutations and response associated with chosen drug
    for j,drug in enumerate(drugs):
            if drug_target is not None and drug != drug_target:
                continue # return to beginning of the loop
            for i,cell in enumerate(cells):
                if cell not in features.columns or (cell, drug) not in response.index:
                    continue
                # all j empty except index that corresponds to target drug
                # for this j we iteratively append all the mutations on every cell
                X_drugs[j].append(features[cell].values) # store genetic features (mutations and expression) that appear in cells
                y_drugs[j].append(response.loc[(cell,drug), 'Amax']) # store response of the drug
            print('{}: Cell number = {}'.format(drug, len(y_drugs[j])))

    # convert to np array
    X_drugs = [np.array(x_i) for x_i in X_drugs]
    y_drugs = [np.array(y_i) for y_i in y_drugs]

    if normalize:
            X_drugs = [(x_i if (len(x_i) == 0) else (x_i - x_i.mean(axis=0,keepdims=True)) / x_i.std(axis=0).clip(1e-6)) for x_i in X_drugs]
            y_drugs = [(y_i if (len(y_i) == 0 or y_i.std() == 0) else (y_i - y_i.mean()) / y_i.std()) for y_i in y_drugs]

    drug_idx = drugs.get_loc(drug_target)
    # 2d array for features and 1d array for response
    X_drug, y_drug = X_drugs[drug_idx], y_drugs[drug_idx]

    return X_drug, y_drug, features


X_drug, y_drug, features = load_ccle(feature_type='mutation')

def ccle_feature_filter(X, y, threshold=0.1):
    '''
    :param X: features
    :param y: response
    :param threshold: correlation threshold
    :return: logical array with False for all features that do not have at least pearson correlation at threshold with y
    and correlations for all variables
    '''
    corrs = np.array([np.abs(np.corrcoef(x, y)[0,1]) if x.std() > 0 else 0 for x in X.T])
    selected = corrs >= threshold # True/False
    print(selected.sum(), selected.shape, corrs)
    return selected, corrs

ccle_selected, corrs = ccle_feature_filter(X_drug, y_drug, threshold=0.1)

features.index[ccle_selected]
stats.describe(corrs[ccle_selected])

def fit_elastic_net_ccle(X, y, nfolds=3):
    '''
    :param X: features
    :param y: response
    :param nfolds: number of folds for hyperparameter tuning
    :return: fitted elastic net model
    '''
    from sklearn.linear_model import ElasticNetCV
    # The parameter l1_ratio corresponds to alpha in the glmnet R package
    # while alpha corresponds to the lambda parameter in glmnet
    # enet = ElasticNetCV(l1_ratio=np.linspace(0.2, 1.0, 10),
    #                     alphas=np.exp(np.linspace(-6, 5, 250)),
    #                     cv=nfolds)
    enet = ElasticNetCV(l1_ratio=0.2, # It always chooses l1_ratio=0.2
                        alphas=np.exp(np.linspace(-6, 5, 250)),
                        cv=nfolds)
    print('Fitting via CV')
    enet.fit(X,y)
    alpha, l1_ratio = enet.alpha_, enet.l1_ratio_
    print('Chose values: alpha={}, l1_ratio={}'.format(alpha, l1_ratio))
    return enet

elastic_model = fit_elastic_net_ccle(X_drug[:,ccle_selected], y_drug)

def fit_random_forest_ccle(X, y):
    '''
    :param X: features
    :param y: response
    :param nfolds: number of folds for hyperparameter tuning
    :return: fitted elastic net model
    '''
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor()

    rf.fit(X,y)

    return rf

rf_model = fit_random_forest_ccle(X_drug[:,ccle_selected], y_drug)

def plot_ccle_predictions(model, X, y):
    from sklearn.metrics import r2_score
    plt.close()
    y_hat = model.predict(X)
    plt.scatter(y_hat, y, color='blue')
    plt.plot([min(y.min(), y_hat.min()),max(y.max(), y_hat.max())], [min(y.min(), y_hat.min()),max(y.max(), y_hat.max())], color='red', lw=3)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title(' ($r^2$={:.4f})'.format( r2_score(y, y_hat)))
    plt.tight_layout()



plot_ccle_predictions(elastic_model,X_drug[:,ccle_selected],y_drug)

def print_top_features(model):
    # model_weights = np.mean([m.coef_ for m in model.models], axis=0)
    if model == rf_model:
        model_weights = model.feature_importances_
    else:
        model_weights = model.coef_

    ccle_features = features[ccle_selected]

    print('Top by fit:')
    for idx, top in enumerate(np.argsort(np.abs(model_weights))[::-1]):
        print('{}. {}: {:.4f}'.format(idx+1, ccle_features.index[top], model_weights[top]))

print_top_features(rf_model)
print_top_features(elastic_model)

def run_test_ccle(X,Y):

    pval = []
    for x_index in range(X.shape[1]):
        z = np.delete(X, x_index, axis=1)
        x = X[:,x_index]
        x = x.reshape((len(x), 1))
        Y = Y.reshape((len(Y),1))
        # now run test
        pval.append(GCIT(x,Y,z))

    ccle_features = features[ccle_selected]

    print('Top by fit:')
    for idx, top in enumerate(np.argsort(np.abs(pval))):
        print('{}. {}: {:.4f}'.format(idx+1, ccle_features.index[top], pval[top]))

run_test_ccle(X_drug[:,ccle_selected],y_drug)
