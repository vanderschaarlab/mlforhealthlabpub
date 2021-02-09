'''
Example/Test script for invase.
'''
import sys
import os
import json
import argparse
import initpath_alg
from pathlib import Path
initpath_alg.init_sys_path()
import utilmlab
import data_loader_mlab


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--it", default=10000, type=int)
    parser.add_argument('--exe', help='python interpreter to use')
    parser.add_argument('--projdir')
    return parser.parse_args()


if __name__ == '__main__':

    args = init_arg()

    if args.exe is not None:
        python_exe = args.exe
    else:
        python_exe = 'python' if sys.version_info[0] < 3 else 'python3'
    nepoch = args.it
    version = 4
    alg = 'invase'

    proj_dir = utilmlab.get_proj_dir() \
        if args.projdir is None else args.projdir

    resdir = '{}/result/invase/v_{}/h_{}/e_{}'.format(
        proj_dir,
        version,
        os.environ['HOSTNAME'] if 'HOSTNAME' in os.environ else 'unknown',
        nepoch)

    utilmlab.ensure_dir(resdir)

    logger = utilmlab.init_logger(resdir)

    script = Path('{}/alg/invase/invase.py'.format(proj_dir))
    script_ana = Path('{}/alg/invase/invase_ana.py'.format(proj_dir))
    script_plot = Path('{}/alg/invase/invase_plot.py'.format(proj_dir))

    for dataset in ['csv', 'bc', 'spambase']:

        odir = '{}/dataset_{}'.format(resdir, dataset)

        utilmlab.ensure_dir(odir)

        fn_feature_score = '{}/feature_score.csv.gz'.format(odir)
        fn_json = '{}/feature_score.csv.json'.format(odir)
        fn_plot_sample = '{}/sample.png'.format(odir)
        fn_plot_global = '{}/global.png'.format(odir)

        if dataset == 'csv':
            fn_csv = '{}/spambase.csv.gz'.format(
                utilmlab.get_data_dir())
            utilmlab.exe_cmd(
                logger,
                '{} {} -i {} --target label --it {} -o {}'.format(
                    python_exe, script, fn_csv, nepoch, fn_feature_score))
        else:
            if not data_loader_mlab.is_available(dataset):
                continue
            utilmlab.exe_cmd(
                logger,
                '{} {} --dataset {} --it {} -o {}'.format(
                    python_exe, script, dataset, nepoch, fn_feature_score))

        utilmlab.exe_cmd(logger,'{} {} -i {} -o {}'.format(
            python_exe, script_ana, fn_feature_score, fn_json))

        utilmlab.exe_cmd(logger,'{} {} -i {} -oglobal {} -osample {}'.format(
            python_exe, script_plot, fn_feature_score,
            fn_plot_global, fn_plot_sample))

    result_lst = []

    for dataset in [
            'Syn1',
            'Syn2',
            'Syn3',
            'Syn4',
            'Syn5',
            'Syn6']:

        odir = '{}/dataset_{}'.format(resdir, dataset)

        utilmlab.ensure_dir(odir)
        fn_feature_score = '{}/feature_score.csv.gz'.format(odir)

        utilmlab.exe_cmd(logger,'{} {} --dataset {} --it {} -o {}'.format(
            python_exe, script, dataset, nepoch, fn_feature_score))

        fn = '{}/result.json'.format(odir)
        with open(fn) as f:
            result_d = json.load(f)
        result_desc = 'tpr_mean:{} fdr_mean:{}'.format(
            result_d['tpr_mean'],
            result_d['fdr_mean'])
        result_lst.append((dataset, result_desc))
        logger.info('{}'.format(result_lst[-1]))

    logger.info('\n\nOverview result {}:\n'.format(alg))

    for el in result_lst:
        logger.info('{:20s} {}'.format(el[0], el[1]))
