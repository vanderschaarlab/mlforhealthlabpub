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
    parser.add_argument('--it', default=10000, type=int)
    parser.add_argument("--itout", type=int)
    parser.add_argument("--itrs", type=int)
    parser.add_argument('--projdir')
    parser.add_argument('-o')
    return parser.parse_args()


if __name__ == '__main__':

    args = init_arg()

    proj_dir = utilmlab.get_proj_dir() \
        if args.projdir is None else args.projdir

    alg = 'deephit'

    if args.exe is not None:
        python_exe = args.exe
    else:
        python_exe = 'python' if sys.version_info[0] < 3 else 'python3'

    niter = args.it
    version = 1

    if args.o is None:
        odir = '{}/result/{}/v_{}/h_{}'.format(
            proj_dir,
            alg,
            version,
            os.environ['HOSTNAME'] if 'HOSTNAME' in os.environ else 'unknown')
    else:
        odir = args.o

    utilmlab.ensure_dir(odir)

    logger = utilmlab.init_logger(odir, 'log_test_{}.txt'.format(alg))

    result_lst = []

    dataset_lst = [
        {
            'dataset': 'SYNTHETIC',
        }
    ]

    script = Path('{}/alg/deephit/main_RandomSearch.py'.format(proj_dir))
    script_ana = Path('{}/alg/deephit/summarize_results.py'.format(proj_dir))

    for args_d in dataset_lst:
        cmd_arg = ' '.join([
            '--{} {}'.format(el, args_d[el]) for el in args_d.keys()])

        dataset = args_d['dataset']
        
        odir = '{}/dataset_{}'.format(
            odir,
            dataset)

        utilmlab.ensure_dir(odir)

        utilmlab.exe_cmd(
            logger,
            '{} {} {} {} {} {} {}'.format(
                python_exe,
                script,
                cmd_arg,
                '--it {}'.format(niter),
                '-o {}'.format(odir),
                '--itout {}'.format(
                    args.itout) if args.itout is not None else '',
                '--itrs {}'.format(
                    args.itrs) if args.itrs is not None else ''))

        utilmlab.exe_cmd(
            logger,
            '{} {} {} {} {}'.format(
                python_exe,
                script_ana,
                cmd_arg,
                '-o {}'.format(odir),
                '--itout {}'.format(
                    args.itout) if args.itout is not None else ''))
        
