'''
Example/Test script for pategan.
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
    parser.add_argument('--projdir')
    parser.add_argument('-o')
    return parser.parse_args()


if __name__ == '__main__':

    args = init_arg()

    proj_dir = utilmlab.get_proj_dir() \
        if args.projdir is None else args.projdir

    if not os.path.isfile(
            '{}/alg/gcit/ccle_experiments/data/mutation.txt.gz'.format(proj_dir)):
        print('warning: data files for ccle_experiments not found')
        sys.exit(0)
        
    alg = 'gcit'
    version = 1
    if args.exe is not None:
        python_exe = args.exe
    else:
        python_exe = 'python' if sys.version_info[0] < 3 else 'python3'

    if args.o is None:
        resdir = '{}/result/{}/v_{}/h_{}'.format(
            proj_dir,
            alg,
            version,
            os.environ['HOSTNAME'] if 'HOSTNAME' in os.environ else 'unknown')
    else:
        resdir = args.o

    utilmlab.ensure_dir(resdir)

    logger = utilmlab.init_logger(resdir, 'log_test_{}.txt'.format(alg))

    script = Path('{}/alg/gcit/ccle_experiments/'
                  'ccle_experiment.py'.format(proj_dir))

    odir = resdir
    
    utilmlab.exe_cmd(
        logger,
        '{} {}'.format(
            python_exe,
            script))
