import argparse
import pandas as pd
import numpy as np
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def get_values_mask(val, mask):
    lst = list()
    assert val.shape == mask.shape
    for row in range(val.shape[0]):
        for col in range(val.shape[1]):
            if mask[row, col]:
                lst.append(val[row, col])
    return lst


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help='input csv file  (with missing values)')
    parser.add_argument("-o", help='json file with analysis results')
    parser.add_argument("--ref", help='input csv file without missing values')
    parser.add_argument(
        "--imputed",
        help='input csv file were missing values are imputed')
    parser.add_argument(
        "--target",
        help='column name with target values, this column will '
        'be excluded from analysis')
    parser.add_argument("--description", default='-')
    return parser.parse_args()


def get_num_na(df):
    return int(sum(np.ravel(np.isnan(df))))


def df_ana_diff(df0, df1):
    is_equal = df0.equals(df1)
    count_el_diff = 0
    if not is_equal:
        count_el_diff = sum(np.ravel(df0 != df1))
    return {
        'equal': bool(is_equal),
        'ratio': float(count_el_diff/float(df0.size)),
        'ndiff': int(count_el_diff)
    }


if __name__ == '__main__':

    args = init_arg()
    fn_json = args.o
    assert fn_json is not None
    odir = os.path.dirname(fn_json)
    odir = odir if len(odir) else '.'
    df = pd.read_csv(args.i)   
    df_ref = pd.read_csv(args.ref)
    df_imputed = pd.read_csv(args.imputed)
    label = args.target

    features_imputed = [
        el.replace('.', ' ').lstrip() for el in df_imputed.columns]
    features_ref = [
        el.replace('.', ' ').lstrip() for el in df_ref.columns]

    df_imputed = pd.DataFrame(
        df_imputed.values, columns=features_imputed)
    df_ref = pd.DataFrame(
        df_ref.values, columns=features_ref)

    diff_features = set(features_imputed) - set(features_ref)

    if len(diff_features):
        print('diff {}'.format(diff_features))

    if label is not None:
        assert label in features_imputed
        assert label in features_ref
        features_imputed.remove(label)
        features_ref.remove(label)

    mask = np.where(np.isnan(df[features_ref].values), True, False)
    print(features_imputed, df.columns)
    df_imputed_calc = pd.DataFrame(
        df_imputed[features_imputed].values, columns=features_imputed)
    df_ref_calc = pd.DataFrame(
        df_ref[features_ref].values, columns=features_ref)
    # scale/normalize dataset
    range_scaler = (0, 1)
    scaler = MinMaxScaler(feature_range=range_scaler)
    scaler.fit(df_ref_calc[features_ref].values)

    df_ref_calc = pd.DataFrame(
        scaler.transform(
            df_ref_calc[features_ref]),
        columns=features_ref)
    df_imputed_calc = pd.DataFrame(
        scaler.transform(
            df_imputed_calc[features_imputed]),
        columns=features_imputed)

    num_na_imputed = get_num_na(df_imputed_calc)

    df_imputed_calc = df_imputed_calc.fillna(0)

    nmissing = sum(np.ravel(mask))

    if not nmissing:
        print('warning: no missing values found')

    a = sum(sum(np.where(mask, (df_ref_calc - df_imputed_calc) * (
        df_ref_calc - df_imputed_calc), 0)))

    rmse_final = float(np.sqrt(a/float(nmissing))) if nmissing else 0.0
    print('rmse: {:0.3f} {:0.3f} {:0.3f}'.format(
        rmse_final, a, nmissing/float(mask.size)))

    result_d = {
        'nmissing': get_num_na(df[features_ref]),
        'rmse': rmse_final,
        'nsample': int(df_imputed_calc.size),
        'num_na_imputed': num_na_imputed,
        'description': args.description
    }

    missing_mask_01 = np.where(np.isnan(df[features_ref].values), 1.0, 0.0)
    testM = missing_mask_01
    testX = df_ref_calc.values
    Recover_testX = df_imputed_calc.values
    diff_a = df_ref_calc.values - Recover_testX
    if nmissing:
        plt.subplot(2, 1, 1)
        ax = sns.distplot(
            get_values_mask(diff_a, missing_mask_01),
            hist=False, kde=True, kde_kws={'linewidth': 1}, label='diff')
        ax.set_title('rmse : {:0.3f}'.format(result_d['rmse']))
        plt.subplot(2, 1, 2)
        sns.distplot(
            get_values_mask(testX, missing_mask_01),
            hist=False, kde=True, kde_kws={'linewidth': 1}, label='testX')
        sns.distplot(
            get_values_mask(Recover_testX, missing_mask_01),
            hist=False, kde=True, kde_kws={'linewidth': 1}, label='recoverX')
        plt.xlabel('proba')
        plt.legend()
        plt.grid()
        plt.savefig('{}/plot.png'.format(odir))
    result_ana_diff = df_ana_diff(df_ref, df_imputed)
    result_d['ana_diff'] = result_ana_diff
    print(result_d['rmse'])
    with open(fn_json, "w") as fp:
        json.dump(result_d, fp)
