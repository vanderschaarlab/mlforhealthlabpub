import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i")
    parser.add_argument("-o")
    parser.add_argument("--xname")
    parser.add_argument("--yname")
    return parser.parse_args()


if __name__ == '__main__':

    args = init_arg()

    xname = args.xname
    yname = args.yname
    
    idir = args.i
    fn_fdr = '{}/{}_{}_FDR.csv'.format(idir, xname, yname)
    fn_tpr = '{}/{}_{}_TPR.csv'.format(idir, xname, yname)

    df_fdr = pd.read_csv(fn_fdr)
    df_tpr = pd.read_csv(fn_tpr)
    col_lst = list(df_fdr.columns)
    fdr_lst = []
    tpr_lst = []
    for col in col_lst:
        fdr_lst.append(np.mean(df_fdr[col]))
        tpr_lst.append(np.mean(df_tpr[col]))

    plt.plot(tpr_lst, label='tpr')
    plt.plot(fdr_lst, label='fdr')
    plt.legend()
    plt.title('column wise mean of tpr and fdr for {} {}'.format(xname, yname))
    if args.o is None:
        plt.show()
    else:
        plt.savefig(args.o)
