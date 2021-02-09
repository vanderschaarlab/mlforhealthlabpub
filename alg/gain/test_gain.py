'''
Example/Test script for gain. Executes gain for two datasets: Spam, Letter
'''

import os
import argparse
import json
import sys
import initpath_alg
from pathlib import Path
initpath_alg.init_sys_path()
import utilmlab
import data_loader_mlab


def set_filenames(odir):
    utilmlab.ensure_dir(odir)
    fn_csv = '{}/x.csv'.format(odir)
    fn_missing_csv = '{}/x_missing.csv'.format(odir)
    fn_imputed_csv = '{}/x_imputed.csv'.format(odir)
    return fn_csv, fn_missing_csv, fn_imputed_csv


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', help='python interpreter to use')
    parser.add_argument('--it', default=5000, type=int)
    parser.add_argument('--verify', default=1, type=int)
    parser.add_argument('--projdir')
    return parser.parse_args()


if __name__ == '__main__':

    args = init_arg()

    if args.exe is not None:
        python_exe = args.exe
    else:
        python_exe = 'python' if sys.version_info[0] < 3 else 'python3'

    niter = args.it
    version = 5
    is_verify = args.verify

    proj_dir = utilmlab.get_proj_dir() \
        if args.projdir is None else args.projdir

    resdir = '{}/result/gain/v_{}/h_{}'.format(
        proj_dir,
        version,
        os.environ['HOSTNAME'] if 'HOSTNAME' in os.environ else 'unknown')

    utilmlab.ensure_dir(resdir)

    logger = utilmlab.init_logger(resdir, 'log_test_gain.txt')

    dataset = 'bc'

    odir = '{}/misc/dataset_{}'.format(resdir, dataset)

    fn_csv, fn_missing_csv, fn_imputed_csv = set_filenames(odir)

    script_create_missing = Path('{}/alg/gain/create_missing.py'.format(
        proj_dir))
    script = Path('{}/alg/gain/gain.py'.format(proj_dir))
    script_ana = Path('{}/alg/gain/gain_ana.py'.format(proj_dir))

    for islabel in [0, 1]:
        for autocat in [0, 1, 2]:
            utilmlab.exe_cmd(
                logger,
                '{} {} --dataset {} -o {} '
                '--oref {} --istarget {}'.format(
                    python_exe,
                    script_create_missing,
                    dataset,
                    fn_missing_csv,
                    fn_csv,
                    islabel))

            utilmlab.exe_cmd(
                logger,
                '{} {} -i {} {} '
                '-o {} --it {} --testall 1 --autocategorical {}'.format(
                    python_exe,
                    script,
                    fn_missing_csv,
                    '--target target' if islabel else '',
                    fn_imputed_csv,
                    niter,
                    autocat))

    result_lst = []

    dataset_prop = [
        ('spambase', None),
        ('spambase', 'label'),
        ('bc', None),
        ('spam', None),
        ('letter-recognition', None),
        ('letter-recognition', 'lettr'),
        ('letter', None)
    ]

    for dataset, label in dataset_prop:

        if not data_loader_mlab.is_available(dataset):
            logger.info('skipping dataset {}'.format(dataset))
            continue

        odir = '{}/canned_0/dataset_{}'.format(resdir, dataset)

        fn_csv, fn_missing_csv, fn_imputed_csv = set_filenames(odir)

        label_arg = '--target {}'.format(label) if label is not None else ''

        utilmlab.exe_cmd(
            logger,
            '{} {} --dataset {} -o {} --oref {} --istarget {} '
            '--normalize01 0'.format(
                python_exe,
                script_create_missing,
                dataset,
                fn_missing_csv,
                fn_csv,
                1 if label is not None else 0
            ))

        utilmlab.exe_cmd(
            logger,
            '{} {} -i {} --ref {} -o {} '
            '--it {} --testall 1 {}'.format(
                python_exe,
                script,
                fn_missing_csv,
                fn_csv,
                fn_imputed_csv,
                niter,
                label_arg
            ))

        fn_result = '{}/result_ana.json'.format(odir)
        utilmlab.exe_cmd(
            logger,
            '{} {} -i {} --ref {} --imputed {} -o {} {}'.format(
                python_exe,
                script_ana,
                fn_missing_csv,
                fn_csv,
                fn_imputed_csv,
                fn_result,
                label_arg))

        with open(fn_result) as f:
            result_d = json.load(f)
        result_lst.append((dataset, result_d['rmse']))
        logger.info('{}'.format(result_lst[-1]))

    logger.info('\n\nOverview result gain:\n')

    result_ref = {
        'spambase': 0.055,
        'spam': 0.055,
        'letter': 0.125,
        'letter-recognition': 0.125,
    }
    is_ok = True
    is_ok_th = 0.01
    for el in result_lst:
        ds = el[0]
        diff_desc = ''
        if ds in result_ref.keys():
            diff_result = abs(el[1]-result_ref[ds])
            if diff_result > is_ok_th:
                is_ok = False
            diff_desc = '{} {}'.format(
                diff_result, '(out of spec)' if diff_result > is_ok_th else '')
        logger.info('{:20s} {} {}'.format(
            el[0],
            'rmse {:0.3f}'.format(el[1]),
            diff_desc))
    if is_verify:
        assert is_ok
