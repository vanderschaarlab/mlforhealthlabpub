import argparse
import pandas as pd
import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import initpath_alg
initpath_alg.init_sys_path()
import utilmlab
import data_loader_mlab


def array2str(a):
    s = ''
    for idx, el in enumerate(a):
        s += (' ' if idx > 0 else '') + '{:0.3f}'.format(el)
    return s


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i")
    parser.add_argument("-n", type=int, default=0)
    parser.add_argument("--show", type=int, default=0)
    parser.add_argument("-osample")
    parser.add_argument("-oglobal")
    parser.add_argument(
        "-isstd", type=int,
        default=0,
        help="include standard deviation in plot")
    return parser.parse_args()


if __name__ == '__main__':

    args = init_arg()

    fn_csv = args.i
    fn_plot_global = args.oglobal
    fn_plot_sample = args.osample
    nsample = args.n
    is_std = args.isstd
    
    df = pd.read_csv(fn_csv)
    if nsample:
        df = df[:nsample]

    lst = [(el0, el1, el2) for el0, el1, el2 in zip(
        df.columns,
        np.mean(df, axis=0),
        np.std(df, axis=0))]
    lst = sorted(lst, key=lambda el: el[1], reverse=True)
    feature_mean = [el[1] for el in lst]
    feature_std = [el[2] for el in lst]
    feature_nm = [el[0][:6] for el in lst]
    if is_std:
        df_bar_plot = pd.DataFrame({
            'feature_mean_std': feature_mean + feature_std,
            'feature_is_std': [
                0 for el in feature_mean] + [1 for el in feature_std],
            'feature_nm': feature_nm + feature_nm})
        ax = sns.barplot(
            x="feature_mean_std",
            y="feature_nm",
            data=df_bar_plot,
            palette="Blues_d",
            hue="feature_is_std")
    else:
        df_bar_plot = pd.DataFrame({
            'feature_mean': feature_mean,
            'feature_nm': feature_nm})
        ax = sns.barplot(
            x="feature_mean",
            y="feature_nm",
            data=df_bar_plot,
            palette="Blues_d")

    plt.savefig(fn_plot_global)
    fig, ax = plt.subplots()
    ax.matshow(df.values, cmap=plt.cm.Blues)
    fig = plt.gcf()
    dpi = 5
    plot_size = max(10, len(df.columns)/dpi), len(df)/dpi
    fig.set_size_inches(plot_size)
    plt.savefig(fn_plot_sample)
    if args.show:
        plt.show()
