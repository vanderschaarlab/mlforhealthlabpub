'''
Example/Test script for autoprognosis
'''
from subprocess import call
import sys
import os
import argparse
import re
import json
from pathlib import Path
import initpath_ap
initpath_ap.init_sys_path()
import utilmlab
import tempfile
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_covtype
import pandas as pd


def set_filenames(odir):
    utilmlab.ensure_dir(odir)
    fn_csv = '{}/data.csv.gz'.format(odir)
    return fn_csv


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', help='python interpreter to use')
    parser.add_argument('--projdir')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--it', type=int, default=20)
    parser.add_argument('--nstage', type=int, default=0)
    return parser.parse_args()


def get_temp_filename():
    f_tmp = tempfile.NamedTemporaryFile()  # need a temporally filename
    f_tmp.close()
    return f_tmp.name


if __name__ == '__main__':

    args = init_arg()

    fn_model = get_temp_filename()

    verbose = args.verbose
    niter = args.it
    nstage = args.nstage

    if args.exe is not None:
        python_exe = args.exe
    else:
        python_exe = 'python' if sys.version_info[0] < 3 else 'python3'

    version = 1

    resdir = '{}/result/autoprognosis/v_{}/h_{}'.format(
        utilmlab.get_proj_dir(),
        version,
        os.environ['HOSTNAME'] if 'HOSTNAME' in os.environ else 'unknown')

    utilmlab.ensure_dir(resdir)

    logger = utilmlab.init_logger(resdir, 'log_test_autoprognosis.txt')

    result_all_d = dict()

    for dataset, nsample, sep in [
            ('bc', 0, ','), 
            ('cover', 5000, ';')]:
    
        odir = '{}/misc/dataset_{}'.format(resdir, dataset)

        fn_csv = set_filenames(odir)

        proj_dir = utilmlab.get_proj_dir() if args.projdir is None \
            else args.projdir
        script = Path('{}/alg/autoprognosis/autoprognosis.py'.format(proj_dir))
        script_report = Path(
            '{}/alg/autoprognosis/autoprognosis_report.py'.format(
                proj_dir))
        script_predict = Path(
            '{}/alg/autoprognosis/autoprognosis_predict.py'.format(
                proj_dir))

        if dataset == 'bc':
            # from sklearn: A copy of UCI ML Breast
            # Cancer Wisconsin (Diagnostic) dataset
            x, y = load_breast_cancer(return_X_y=True)
        elif dataset == 'cover':
            x, y = fetch_covtype(return_X_y=True)
        else:
            assert 0
        lbl = 'target'
        df = pd.DataFrame(x)
        df[lbl] = y
        df.to_csv(fn_csv, index=False, compression='gzip', sep=sep)

        utilmlab.exe_cmd(
            logger,
            '{} {} -i {} --target {} -o {} --verbose {} --it {} -n {} --separator {}'.format(
                python_exe,
                script,
                fn_csv,
                lbl,
                resdir,
                verbose,
                niter,
                nsample,
                sep
            ))

        utilmlab.exe_cmd(
            logger,
            '{} {} -i {} -o {} --verbose {} --it {} --target {} --model {} --separator {}'
            ' --nstage {} -n {}'.format(
                python_exe,
                script,
                fn_csv,
                resdir,
                verbose,
                niter,
                lbl,
                fn_model,
                sep,
                nstage,
                nsample,
            ))

        fn_evaluate = '{}/eva.json'.format(resdir)
        fn_ocsv = '{}/predict.csv'.format(resdir)
        utilmlab.exe_cmd(
            logger,
            '{} {} -i {} --target {} --model {} --evaluate {} -o {} --separator {}'.format(
                python_exe,
                script_predict,
                fn_csv,
                lbl,
                fn_model,
                fn_evaluate,
                fn_ocsv,
                sep
            ))

        fn = '{}/result.json'.format(resdir)
        with open(fn) as f:
            result_d = json.load(f)
        print(result_d)
        os.unlink(fn_model)

        fn = '{}/report.json'.format(resdir)
        with open(fn) as f:
            report_d = json.load(f)
        print(report_d)
