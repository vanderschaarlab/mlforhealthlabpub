import logging
import os
import fnmatch
import re
from subprocess import call
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score


def init_logger(odir='.', log_fn='log.txt', use_show=True, log_level=None):
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.basicConfig(level=logging.WARNING, format='%(message)s')
    log_fn = "{}/{}".format(odir if len(odir) else ".", log_fn)
    if os.path.isfile(log_fn):
        os.remove(log_fn)
    handler = logging.FileHandler(log_fn)
    if log_level is None:
        log_level = logging.INFO if use_show else logging.WARNING
    handler.setLevel(log_level)
    logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def get_proj_dir():
    return os.path.abspath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        os.pardir))


def get_data_dir():
    return os.path.join(get_proj_dir(), 'data')


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def find_file_dir(dir, fn_mask):
    file_lst = []
    for dir_name, sub_dir, f_list in os.walk(dir):
        for file_name in f_list:
            if fnmatch.fnmatch(file_name, fn_mask):
                file_lst.append(os.path.join(dir_name, file_name))
    return file_lst


def col_with_nan(df):
    col_nan = []
    for el in df.columns:
        if sum(df[el].isnull()):
            col_nan.append(el)
    return col_nan


def exe_cmd(logger, cmd, assert_on_error=True):
    logger.info('cmd:{}'.format(cmd))
    cmd = re.sub(' +', ' ', cmd).rstrip().lstrip()
    cmd_lst = cmd.split(' ')
    rval = call(cmd_lst)
    if assert_on_error:
        assert rval == 0


def count_properties(a):
    d = Counter(a)
    rval_d = dict()
    sum_val = sum(d.values())
    for el in d.keys():
        rval_d['{}_ratio'.format(el)] = d[el]/float(sum_val)
    kys = d.keys()
    for el in kys:
        rval_d[el] = d[el]
    return rval_d


def df_cat_to_one_hot(
        df,
        th=0.05,
        is_verbose=0,
        dummy_na=True,
        labels=None,
        is_cat_one_hot=True):

    '''
    one hot encoding of columns of a dataframe if number of values (set)
    is less than a fraction of the number of samples
    '''

    nsampe = len(df)
    prop_col = dict()
    prop_col['dfcolumns'] = df.columns
    prop_col['is_cat_one_hot'] = is_cat_one_hot
    df_one_hot = pd.DataFrame()
    for colnm in df.columns:
        ncat = len(set(df[colnm].dropna()))
        ratio = ncat/float(nsampe)
        is_cat = ratio < th and (
            colnm not in labels if labels is not None else True)  # is_cat is False if in labels
        prop_col[colnm] = {
            'cat': is_cat
        }
        if is_cat and is_cat_one_hot:
            cat_val = list(set(df[colnm].dropna()))
            df_tmp = pd.get_dummies(df[colnm], dummy_na=dummy_na)
            col_nan = df_tmp[np.nan].values.reshape(-1, 1) \
                      if np.nan in df_tmp.columns else None
            colnms = ['{}_{}'.format(colnm, el) for el in cat_val]
            df_tmp = df_tmp[cat_val]  # drop na column
            if col_nan is not None:
                df_tmp = np.where(
                    col_nan.repeat(ncat, axis=1),
                    np.nan,
                    df_tmp.values)
                df_tmp = pd.DataFrame(df_tmp, columns=cat_val)
            if is_verbose:
                print('cols:{} {} cat:{}'.format(
                    colnms, df_tmp.columns, cat_val))
            df_one_hot[colnms] = df_tmp
            prop_col[colnm]['columns'] = df_tmp.columns
            prop_col[colnm]['columns_df'] = colnms
        else:
            prop_col[colnm]['columns'] = colnm
            df_one_hot[colnm] = df[colnm]
    prop_col['dfcol_one_hot'] = list(df_one_hot.columns)

    cat_lst = [prop_col[el]['cat'] for el in prop_col['dfcolumns']]
    if is_verbose:
        print('cat:{} (#{}) {}'.format(
            sum(cat_lst),
            len(cat_lst),
            sum(cat_lst)/float(len(cat_lst))))
    return df_one_hot, prop_col


def df_one_hot_to_cat(df_one_hot, prop_one_hot_col, labels=None):
    '''
    decodes a one hot encoded dataframe, see method df_cat_to_one_hot(...).
    '''

    df_dst = pd.DataFrame()

    for colnm in prop_one_hot_col['dfcolumns']:
        if prop_one_hot_col[colnm]['cat']:
            if prop_one_hot_col['is_cat_one_hot']:
                c0 = prop_one_hot_col[colnm]['columns_df']
                c1 = list(prop_one_hot_col[colnm]['columns'])
                assert len(c0) == len(c1)
                df_tmp = pd.DataFrame(
                    df_one_hot[prop_one_hot_col[colnm]['columns_df']])
                df_tmp.columns = c1
                df_dst[colnm] = df_tmp.idxmax(1)
            else:
                df_dst[colnm] = np.round(df_one_hot[colnm])
        else:
            df_dst[colnm] = df_one_hot[colnm]

    return df_dst


def df_get_num_na(df):
    return int(sum(np.ravel(np.isnan(df))))


def introduce_missing(data, p_miss):

    nsample = len(data)
    nfeatures = data.shape[1]

    p_miss_vec = p_miss * np.ones((nfeatures, 1))

    mask = np.zeros(data.shape)

    for i in range(data.shape[1]):
        A = np.random.uniform(0., 1., size=[nsample, ])
        B = A > p_miss_vec[i]
        mask[:, i] = 1.*B

    X_missing = np.where(mask < 1, np.nan, data)
    return X_missing


def log_meminfo():
    fn = '/proc/meminfo'
    if os.path.isfile(fn):
        f = open(fn, "r")
        for ln in f:   
            logger.info('{}'.format(ln))
        f.close()


def get_y_pred_proba_hlpr(y_pred_proba, nclasses):
    '''
    returns the column with the class of interest (typically the last
    column), this is classifier/datatype dependent
    '''
    if isinstance(y_pred_proba, tuple):
        y_pred_proba_tmp = y_pred_proba[1]
    elif nclasses <= 2 and isinstance(y_pred_proba, (np.ndarray, np.generic)):
        y_pred_proba_tmp = y_pred_proba if len(y_pred_proba.shape) < 2 else y_pred_proba[:, 1]
    else:
        y_pred_proba_tmp = y_pred_proba
    return y_pred_proba_tmp
        

def evaluate_auc(y_test, y_pred_proba, classes=None):

    nnan = sum(np.ravel(np.isnan(y_pred_proba)))

    if nnan:

        logger.info('nan in preds when calculating score, return low score')
        return 0.5, 0

    else:
               
        n_classes = len(set(np.ravel(y_test)))

        y_pred_proba_tmp = get_y_pred_proba_hlpr(y_pred_proba, n_classes)

        if n_classes > 2:

            logger.debug('+evaluate_auc {} {}'.format(y_test.shape, y_pred_proba_tmp.shape))

            fpr = dict()
            tpr = dict()
            precision = dict()
            recall = dict()
            average_precision = dict()
            thresholds = dict()
            roc_auc = dict()
            prc_auc = dict()

            if classes == None:
                classes = sorted(set(np.ravel(y_test)))
                logger.info(
                    'warning: classes is none and more than two '
                    ' (#{}), classes assumed to be an ordered set:{}'.format(
                        n_classes, classes))

            y_test = label_binarize(y_test, classes=classes)
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_proba_tmp[:, i])
                precision[i], recall[i], thresholds[i] = precision_recall_curve(
                    y_test[:, i],
                    y_pred_proba_tmp[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                average_precision[i] = average_precision_score(
                    y_test[:, i],
                    y_pred_proba_tmp[:, i])

            fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred_proba_tmp.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            precision["micro"], recall["micro"], _ = precision_recall_curve(
                y_test.ravel(),
                y_pred_proba_tmp.ravel())

            average_precision["micro"] = average_precision_score(
                y_test,
                y_pred_proba_tmp,
                average="micro")

            aucroc = roc_auc["micro"]
            aucprc = average_precision["micro"]
        else:

            aucroc = roc_auc_score(
                np.ravel(y_test),
                y_pred_proba_tmp)
            aucprc = average_precision_score(
                np.ravel(y_test),
                y_pred_proba_tmp)

    return aucroc, aucprc


def get_hostname():
    return os.environ['HOSTNAME'] if 'HOSTNAME' in os.environ else 'unknown'


def get_df_compression(fn):
    return 'gzip' if fn.endswith('.gz') else None


logger = logging.getLogger()


