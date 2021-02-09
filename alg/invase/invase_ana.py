import argparse
import pandas as pd
import os
import sys
import numpy as np
import json
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
    parser.add_argument("-o")
    return parser.parse_args()


if __name__ == '__main__':

    args = init_arg()

    fn_csv = args.i
    fn_json = args.o

    df = pd.read_csv(fn_csv)
    
    # fig, ax = plt.subplots()
    # ax.matshow(df.values, cmap=plt.cm.Blues)
    # plt.show()

    logger = utilmlab.init_logger(os.path.dirname(fn_csv))

    lst = list()

    lst = [(el0, el1, el2) for el0, el1, el2 in zip(
        df.columns,
        np.mean(df, axis=0),
        np.std(df, axis=0))]
    lst = sorted(lst, key=lambda el: el[1])
    feature_lst = [el[0] for el in lst if el[1] > 0.5]
    for idx, el in enumerate(lst):
        print('{} {:30s} mean:{:0.3f} std:{:0.3f}'.format(idx, el[0], el[1], el[2]))
    d = {
        'features': [el[0] for el in lst],
        'score': [el[1] for el in lst],
        'score_std': [el[2] for el in lst],
        'feature_score': lst
    }
    if fn_json is not None:
        with open(fn_json, "w") as fp:
            json.dump(d, fp)
