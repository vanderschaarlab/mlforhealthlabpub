'''
Example/Test script for ASAC
'''
import sys
import os
import argparse
import json
from pathlib import Path
import initpath_alg
initpath_alg.init_sys_path()
import utilmlab


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', help='python interpreter to use')
    parser.add_argument('--it', default=5000, type=int)
    parser.add_argument('--projdir')
    parser.add_argument("-n", "--numexp", default=10, type=int)
    return parser.parse_args()


if __name__ == '__main__':

    args = init_arg()

    numexp = args.numexp

    proj_dir = utilmlab.get_proj_dir() \
        if args.projdir is None else args.projdir

    alg = 'asac'

    if args.exe is not None:
        python_exe = args.exe
    else:
        python_exe = 'python' if sys.version_info[0] < 3 else 'python3'

    niter = args.it
    version = 1

    resdir = '{}/result/{}/v_{}/h_{}'.format(
        proj_dir,
        alg,
        version,
        os.environ['HOSTNAME'] if 'HOSTNAME' in os.environ else 'unknown')

    utilmlab.ensure_dir(resdir)

    logger = utilmlab.init_logger(resdir, 'log_test_{}.txt'.format(alg))

    result_lst = []

    script = Path('{}/alg/asac/Main_Synthetic_Exp1.py'.format(proj_dir))

    utilmlab.exe_cmd(
        logger,
        '{} {} {}'.format(
            python_exe,
            script,
            '--it {} -n {}'.format(niter, numexp)))
