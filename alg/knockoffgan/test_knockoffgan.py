'''
Test script knockoffGAN: it executes the tutorial notebook
'''
import sys
import os
import argparse
from pathlib import Path
import pandas as pd
import json
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_covtype
import initpath_alg
initpath_alg.init_sys_path()
import utilmlab    


def set_filenames(odir):
    utilmlab.ensure_dir(odir)
    fn_csv = '{}/data.csv.gz'.format(odir)
    return fn_csv


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', help='python interpreter to use')
    parser.add_argument('--projdir')
    parser.add_argument('--it', default=2000, type=int)
    parser.add_argument('--replication', default=20, type=int)
    return parser.parse_args()


if __name__ == '__main__':

    args = init_arg()

    if args.exe is not None:
        python_exe = args.exe
    else:
        python_exe = 'python' if sys.version_info[0] < 3 else 'python3'

    proj_dir = utilmlab.get_proj_dir() \
        if args.projdir is None else args.projdir

    alg = 'knockoffgan'
    version = 1
    niter = args.it
    nreplication = args.replication

    resdir = '{}/result/{}/v_{}/h_{}'.format(
        proj_dir,
        alg,
        version,
        os.environ['HOSTNAME'] if 'HOSTNAME' in os.environ else 'unknown')

    utilmlab.ensure_dir(resdir)

    logger = utilmlab.init_logger(
        resdir,
        'log_test_knockoffgan_{}.txt'.format(utilmlab.get_hostname()))

    script = Path('{}/alg/knockoffgan/knockoffgan.r'.format(proj_dir))

    for dataset, nsample, sep, generate_error in [
            ('bc', 0, ',', 0),
            ('cover', 5000, ',', 1)]:

        odir = '{}/misc/dataset_{}'.format(resdir, dataset)

        fn_csv = set_filenames(odir)

        if dataset == 'bc':
            x, y = load_breast_cancer(return_X_y=True)
        elif dataset == 'cover':
            x, y = fetch_covtype(return_X_y=True)
        else:
            assert 0

        lbl = 'target'
        df = pd.DataFrame(x)
        df[lbl] = y
        df.to_csv(
            fn_csv,
            index=False,
            compression='gzip',
            sep=sep)

        try:
            utilmlab.exe_cmd(
                logger,
                'Rscript {} -i {} --target {} --exe {} --it {} '
                ' --replication {} --projdir {}' .format(
                    script,
                    fn_csv,
                    lbl,
                    python_exe,
                    niter,
                    nreplication,
                    proj_dir
                ),
                assert_on_error=not generate_error  # assert if an error is not expected
            )
        except:
            if generate_error:
                logger.info('expected error generated')
                pass
            assert 0

    fn_data_csv = '{}/data.csv'.format(resdir)
    fn_json = '{}/generated_data_properties.json'.format(resdir)

    utilmlab.exe_cmd(
        logger,
        'Rscript {}/alg/knockoffgan/gen_data.r -o {} --target {} '
        ' --ojson {}'.format(proj_dir, fn_data_csv, lbl, fn_json))

    with open(fn_json, "r") as fp:
        features_gen_data = json.load(fp)
    logger.info('relevant variables:{}'.format(
        features_gen_data['features_selected']))

    false_discovery_rate = 0.1
    stat = "glm"  # Importance statistics based on glmnet_coefdiff (glm)
    fn_json_ko = '{}/result_knockoff_gan.json'.format(resdir)
    utilmlab.exe_cmd(
        logger,
        'Rscript {} -i {} --target {} --it {} --fdr {} --replication {} '
        ' -o {} --stat {} --exe {} --projdir {}'.format(
            script,
            fn_data_csv,
            lbl,
            niter,
            false_discovery_rate,
            nreplication,
            fn_json_ko,
            stat,
            python_exe,
            proj_dir))

    with open(fn_json_ko, 'r') as fp:
        result = json.load(fp)

    agree_set = set(result['features_selected']).intersection(set(
        features_gen_data['features_selected']))
    disagree_set = set(result['features_selected']) - set(
        features_gen_data['features_selected'])

    logger.info('relevant explanatory variables:{}\n'.format(
        result['features_selected']))
    logger.info('agreement generated and detectect explanatory '
                'variables:{}) {}'.format(
                    len(agree_set), agree_set))
    logger.info('disagreement: {}'.format(
        disagree_set if len(disagree_set) else '-'))

    assert len(disagree_set)  <= (false_discovery_rate * len(features_gen_data['features_selected']) + 1)
    assert len(agree_set)  == len(features_gen_data['features_selected'])
    logger.info('pass')

    logger.info('-*-')
